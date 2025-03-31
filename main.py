import os
import logging
from time import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sksurv.metrics import concordance_index_ipcw

from utils import set_seeds, plot_losses
from data_processing import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['HYDRA_FULL_ERROR'] = '1'  # better error trace


matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.5

@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a model using centralized learning

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    set_seeds(cfg.seed)

    # 1. Print parsed config
    logging.info(OmegaConf.to_yaml(cfg))

    # 2. Read and preprocess data
    df = pd.read_csv(cfg.data_path, compression='infer')

    # 3. Initialize data processor class
    dp = DataProcessor(
        unit_identifier = cfg.unit_identifier,
        time_identifier  = cfg.time_identifier,
        target_feature = cfg.target_feature,
        test_size = cfg.test_size,
        seed = cfg.seed,
        lag_length = cfg.lag_length,
        sampling_n = cfg.sampling_n
    )

    # 4. Preprocess data
    df, train_idx, val_idx, test_idx = dp.preprocess(df)


    # 5. Create labels for xgboost interval censoring
    assert cfg.sampling_n >= 1, "Sampling n must be greater than or equal to 1."
    if cfg.sampling_n == 1:  # simple cross-sectional data
        y_lower_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_lower_bound')
        y_upper_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_upper_bound')
        y_upper_bound = y_upper_bound.where(df[cfg.target_feature] == 1, +np.inf)  # where condition true, use orig value, else +inf
    else:  # panel data -> interval censored
        y_lower_bound = pd.Series(df.groupby(cfg.unit_identifier)[cfg.time_identifier].shift(1).fillna(0), name='y_lower_bound')  # missing values are left-censored and set to 0
        y_upper_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_upper_bound')
        y_upper_bound = y_upper_bound.where(df[cfg.target_feature] == 1, +np.inf)

    # Separate target feature from X matrix
    df = df.set_index(cfg.unit_identifier)  # note - this has no effect on positional indexing below
    target_features = [cfg.time_identifier, cfg.target_feature]
    target = df[target_features].copy()
    target[cfg.target_feature] = target[cfg.target_feature].astype(bool)
    X = df.drop(columns=target_features)


    # 5. Train model
    if cfg.hyperoptimize:
        raise NotImplementedError("Hyperoptimization is not yet implemented.")

    else:

        dtrain = xgb.DMatrix(X.iloc[train_idx],
                            label_lower_bound=y_lower_bound[train_idx],
                            label_upper_bound=y_upper_bound[train_idx],
                            enable_categorical=True)
        ytrain = target.iloc[train_idx].set_index(cfg.target_feature).to_records()

        dvalid = xgb.DMatrix(X.iloc[val_idx],
                    label_lower_bound=y_lower_bound[val_idx],
                    label_upper_bound=y_upper_bound[val_idx],
                    enable_categorical=True)

        dtest = xgb.DMatrix(X.iloc[test_idx],
            label_lower_bound=y_lower_bound[test_idx],
            label_upper_bound=y_upper_bound[test_idx],
            enable_categorical=True)
        ytest = target.iloc[test_idx].set_index(cfg.target_feature).to_records()

        # Set default parameters
        params = {
            'verbosity': 0,
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'tree_method': 'hist',
            'learning_rate': 0.05,
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 1.0,
            'max_depth': 5,
            'lambda': 0.5,
            'alpha': 0.05
        }

        # Train gradient boosted trees using AFT loss
        current = time()
        evals_result = {}
        bst = xgb.train(params,
                        dtrain,
                        num_boost_round=10_000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,
                        evals_result=evals_result)

        logging.info("Total model training time: %.2f seconds. \n", round(time() - current, 2))

        # Plot the training and validation losses
        plot_losses(evals_result, HydraConfig.get().runtime.output_dir)

        # Estimated median survival time, we treat as risk score, where higher risk scores indicate higher risk of failure
        pred_test = -bst.predict(dtest)
        pred_train = -bst.predict(dtrain)

        # Calculate IPCW Concordance Index
        c_indices = {}
        c_indices['train'] = concordance_index_ipcw(survival_train=ytrain, survival_test=ytrain,
                                              estimate=pred_train, tau=target.iloc[train_idx][cfg.time_identifier].max())[0]
        c_indices['test'] = concordance_index_ipcw(survival_train=ytrain, survival_test=ytest,
                                              estimate=pred_test, tau=target.iloc[train_idx][cfg.time_identifier].max())[0]

        logging.info("Concordance Index (train): %.4f", c_indices['train'])
        logging.info("Concordance Index (test): %.4f", c_indices['test'])




if __name__ == '__main__':

    main() # pylint: disable=E1120
