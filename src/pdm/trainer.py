import logging
from time import time
import os
import json
import shutil

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from pdm.data_processing import DataProcessor
from pdm.hyperoptimizer import Hyperoptimizer
from pdm.evaluator import Evaluator
from pdm.utils import set_seeds

class Trainer:
    def __init__(self,cfg: DictConfig, dp: DataProcessor):
        self.cfg = cfg
        self.dp = dp
        self.unit_identifier = cfg.data.unit_identifier
        self.time_identifier = cfg.data.time_identifier
        self.target_feature = cfg.data.target_feature
        self.sampling_n = cfg.data.sampling_n
        self.seed = cfg.seed
        self.save_model = cfg.training_config.save_model
        self.hyperoptimize = cfg.training_config.hyperoptimize
        self.base_params = {'verbosity': 0,
                            'objective': 'survival:aft',
                            'eval_metric': 'aft-nloglik',
                            'tree_method': 'hist'}

        # Set random seed for reproducibility
        set_seeds(self.seed)

    def run(self) -> None:
        """Trains the survival model using XGBoost with AFT loss function.

        The function performs the following steps:
        1. Preprocess the data using the DataProcessor class.
        2. Create labels for interval censoring based on the target feature and time identifier.
        3. Separate the target feature from the feature matrix.
        4. If hyperparameter optimization is enabled, run the HyperOptimizer class to find the best parameters.
        5. Create DMatrix objects for training, validation, and test sets.
        6. Train the XGBoost model using the AFT loss function.
        7. Plot the training and validation losses.
        8. Calculate predictions for the test set.
        9. Calculate the IPCW Concordance Index for the training and test sets.
        10. Plot survival curves for the test set.
        11. Calculate the integrated Brier score for the model.
        12. Save the performance metrics as a JSON file.
        13. If save_model is True, save the trained model and configuration information.
        14. Zip the directory containing the model and configuration files.
        15. Remove the unzipped directory.
        """
        df, train_idx, val_idx, test_idx, scaler = self.dp.preprocess()

        # Create labels for xgboost interval censoring
        if self.sampling_n == 1:  # simple cross-sectional data
            y_lower_bound = pd.Series(df[self.time_identifier].copy(), name='y_lower_bound')
            y_upper_bound = pd.Series(df[self.time_identifier].copy(), name='y_upper_bound')
            y_upper_bound = y_upper_bound.where(df[self.target_feature] == 1, +np.inf)  # where condition true, use orig value, else +inf
        else:  # panel data -> interval censored
            y_lower_bound = pd.Series(df.groupby(self.unit_identifier)[self.time_identifier].shift(1).fillna(0), name='y_lower_bound')  # missing values are left-censored and set to 0
            y_upper_bound = pd.Series(df[self.time_identifier].copy(), name='y_upper_bound')
            y_upper_bound = y_upper_bound.where(df[self.target_feature] == 1, +np.inf)

        # 6. Separate target feature from X matrix
        df = df.set_index(self.unit_identifier)  # note - this has no effect on positional indexing below
        target_features = [self.time_identifier, self.target_feature]
        target = df[target_features].copy()
        target[self.target_feature] = target[self.target_feature].astype(bool)
        X = df.drop(columns=self.target_feature)


        if self.hyperoptimize:

            hyperopt = Hyperoptimizer(self.cfg, self.base_params, train_idx, y_upper_bound, y_lower_bound, X, target)
            params = hyperopt.run()

        else:

            # Set default parameters
            default_params = {
                'learning_rate': 0.05,
                'aft_loss_distribution': 'normal',
                'aft_loss_distribution_scale': 1.0,
                'max_depth': 5,
                'lambda': 0.5,
                'alpha': 0.05
            }
            params = {**self.base_params, **default_params}

        # Create DMatrix for training, validation and test sets
        dtrain = xgb.DMatrix(X.iloc[train_idx],
                            label_lower_bound=y_lower_bound[train_idx],
                            label_upper_bound=y_upper_bound[train_idx],
                            enable_categorical=True)

        dvalid = xgb.DMatrix(X.iloc[val_idx],
                    label_lower_bound=y_lower_bound[val_idx],
                    label_upper_bound=y_upper_bound[val_idx],
                    enable_categorical=True)

        dtest = xgb.DMatrix(X.iloc[test_idx],
            label_lower_bound=y_lower_bound[test_idx],
            label_upper_bound=y_upper_bound[test_idx],
            enable_categorical=True)

        # Train gradient boosted trees using AFT loss
        logging.info("Training model...")
        current = time()
        evals_result = {}
        bst = xgb.train(params,
                        dtrain,
                        num_boost_round=10_000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,
                        evals_result=evals_result)

        logging.info("Total model training time: %.2f seconds. \n", round(time() - current, 2))

        # Calculate and save the evaluation results
        evaluator = Evaluator(train_mode=True,
                              output_path=HydraConfig.get().runtime.output_dir,
                              params=params,
                              eval_results=evals_result,
                              pred_train=bst.predict(dtrain),
                              pred_test=bst.predict(dtest),
                              ytrain=target.iloc[train_idx],
                              ytest=target.iloc[test_idx],
                              )
        evaluator.run()


        # [Optional] Save model & configuration
        if self.save_model:

            # Create the subdirectory for saving data and model artifacts
            data_model_artifacts_dir = os.path.join(HydraConfig.get().runtime.output_dir, 'data_model_artifacts')
            os.makedirs(data_model_artifacts_dir, exist_ok=True)

            # Save the trained model
            bst.save_model(os.path.join(data_model_artifacts_dir, "xgboost_model.json"))

            # Save model & data configuration information
            cfg = OmegaConf.to_container(self.cfg, resolve=True)
            cfg['training_config'].update(params)
            OmegaConf.save(OmegaConf.create(cfg), f=os.path.join(data_model_artifacts_dir, "config.yaml"), resolve=True)

            # Save the MinMaxScaler object
            _ = joblib.dump(scaler, os.path.join(data_model_artifacts_dir, 'minmax_scaler.joblib'))

            # Save which features were used, since some might've been dropped
            features_used = df.reset_index().columns.tolist()
            with open(os.path.join(data_model_artifacts_dir, 'feature_list.json'), 'w', encoding='utf-8') as f:
                json.dump(features_used, f)

            # Zip the directory, removing unzipped directory
            _ = shutil.make_archive(data_model_artifacts_dir, 'zip', data_model_artifacts_dir)
            shutil.rmtree(data_model_artifacts_dir)
