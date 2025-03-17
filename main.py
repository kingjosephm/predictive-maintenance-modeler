import os
from joblib import dump
import logging
import json

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from data_processing import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['HYDRA_FULL_ERROR'] = '1'  # better error trace

@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a model using centralized learning

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    
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
        model = cfg.model,
        lag_length = cfg.lag_length,
        sampling_n = cfg.sampling_n,
        oversample = cfg.oversample
    )
    
    # 5. Preprocess data
    df = dp.preprocess(df)
    
    # 6. Split data
    #data = dp.split_transform(df)  # Tuple(X_train, X_test, y_train, y_test)
    
    # 7. Instantiate appropriate model class
    # TODO
        
    # # 8. Fit model
    # model.fit()
    
    # # 9. Evaluate model
    # metrics = model.evaluate(uno=True, federated=False, brier=True if cfg.model in ['cox', 'gbm'] else False)
    # logging.info(f"Performance metrics: \n {metrics}")
    # with open(os.path.join(HydraConfig.get().runtime.output_dir, 'metrics.json'), 'w') as j:
    #     json.dump(metrics, j)
    
    # # 10. Save weights for federated learning [optional]
    # if cfg.restrict_to_subset and cfg.model in ['svm', 'cox']:  # only save weights for SVM and Cox and if restricted to subset
    #     logging.info("Saving model weights for federated learning to directory './model_weights'.")
    #     dump(model.model, os.path.join('model_weights', f'{cfg.model}.joblib'))
        
    # # 11. Save performance metrics and plot thereof
    # model.plot_forecast(model_type=cfg.model,
    #                     target_feature=cfg.target_feature,
    #                     time_identifier=cfg.time_identifier,
    #                     output_path=HydraConfig.get().runtime.output_dir)

if __name__ == '__main__':
    
    main() # pylint: disable=E1120
