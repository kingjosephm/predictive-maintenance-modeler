import os
import logging
from time import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from utils import set_seeds, plot_losses
from data_processing import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['HYDRA_FULL_ERROR'] = '1'  # better error trace

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
    if cfg.sampling_n == 0:  # simple cross-sectional data
        y_lower_bound = df[cfg.time_identifier].copy()
        y_upper_bound = np.where(df[cfg.target_feature] == 1, df[cfg.time_identifier], +np.inf)
    else:  # panel data
        y_lower_bound = np.array(df.groupby(cfg.unit_identifier)[cfg.time_identifier].shift(1).fillna(0))
        y_upper_bound = np.where(df[cfg.target_feature] == 1, df[cfg.time_identifier], +np.inf)

    # Separate target feature from X matrix
    df = df.set_index(cfg.unit_identifier)
    target_features = [cfg.time_identifier, cfg.target_feature]
    target = df[target_features].copy()
    X = df.drop(columns=target_features)


    # 5. Train model
    if cfg.hyperoptimize:
        raise NotImplementedError("Hyperoptimization is not yet implemented.")

    else:

        dtrain = xgb.DMatrix(X.iloc[train_idx],
                            label_lower_bound=y_lower_bound[train_idx],
                            label_upper_bound=y_upper_bound[train_idx],
                            enable_categorical=True)

        dvalid = xgb.DMatrix(X.iloc[val_idx],
                    label_lower_bound=y_lower_bound[val_idx],
                    label_upper_bound=y_upper_bound[val_idx],
                    enable_categorical=True)

        # Set default parameters
        params = {
            'verbosity': 0,
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'tree_method': 'hist',
            'learning_rate': 0.03,
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
                        num_boost_round=10000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,
                        evals_result=evals_result)

        logging.info("Total model training time: %.2f seconds. \n", round(time() - current, 2))

        # Plot the training and validation losses
        plot_losses(evals_result, HydraConfig.get().runtime.output_dir)


if __name__ == '__main__':

    main() # pylint: disable=E1120
