import os
import zipfile
import tempfile

import xgboost as xgb
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from data_processing import DataProcessor
from evaluator import Evaluator

class Predictor:
    def __init__(self, cfg: DictConfig, dp: DataProcessor):
        self.cfg = cfg
        self.dp = dp
        self.unit_identifier = cfg.data.unit_identifier
        self.time_identifier = cfg.data.time_identifier
        self.target_feature = cfg.data.target_feature
        self.sampling_n = cfg.data.sampling_n
        self.model_path = cfg.predict_config.model_path
        self.seed = cfg.seed

    def run(self) -> None:
        """_summary_
        """
        df, model_config = self.dp.transform()

        # Create labels for xgboost interval censoring
        if self.sampling_n == 1:  # simple cross-sectional data
            y_lower_bound = pd.Series(df[self.time_identifier].copy(), name='y_lower_bound')
            y_upper_bound = pd.Series(df[self.time_identifier].copy(), name='y_upper_bound')
            y_upper_bound = y_upper_bound.where(df[self.target_feature] == 1, +np.inf)  # where condition true, use orig value, else +inf
        else:  # panel data -> interval censored
            y_lower_bound = pd.Series(df.groupby(self.unit_identifier)[self.time_identifier].shift(1).fillna(0), name='y_lower_bound')  # missing values are left-censored and set to 0
            y_upper_bound = pd.Series(df[self.time_identifier].copy(), name='y_upper_bound')
            y_upper_bound = y_upper_bound.where(df[self.target_feature] == 1, +np.inf)

        # Separate target feature from X matrix
        df = df.set_index(self.unit_identifier)  # note - this has no effect on positional indexing below
        target_features = [self.time_identifier, self.target_feature]
        target = df[target_features].copy()
        target[self.target_feature] = target[self.target_feature].astype(bool)
        X = df.drop(columns=self.target_feature)

        # Create DMatrix for prediction df
        dpred = xgb.DMatrix(X,
                            label_lower_bound=y_lower_bound,
                            label_upper_bound=y_upper_bound,
                            enable_categorical=True)

        # Load the model
        with zipfile.ZipFile(self.model_path, "r") as zipf:
            with zipf.open("xgboost_model.json") as f:
                with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
                    tmp.write(f.read())
                    tmp.flush()  # Ensure all data is written
                    bst = xgb.Booster()
                    bst.load_model(tmp.name)

        predictions = bst.predict(dpred)

        # Calculate and save the evaluation results
        evaluator = Evaluator(train_mode=False,
                              output_path=HydraConfig.get().runtime.output_dir,
                              params=model_config['training_config'],
                              eval_results={},
                              pred_train=np.ndarray,
                              pred_test=predictions,
                              ytrain=pd.DataFrame(),
                              ytest=target,
                              )
        evaluator.run()

        # Output the predictions
        output_path = HydraConfig.get().runtime.output_dir
        os.makedirs(output_path, exist_ok=True)
