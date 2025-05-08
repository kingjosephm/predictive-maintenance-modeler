import os
from time import time
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pipeline import Pipeline
from utils import set_seeds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['HYDRA_FULL_ERROR'] = '1'  # better error trace

@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entry point: instantiate and run the pipeline"""

    start_time = time()

    set_seeds(cfg.seed)

    logging.info(OmegaConf.to_yaml(cfg))

    pipeline = Pipeline(cfg)
    pipeline.run()

    logging.info("Total run time: %.2f seconds. \n", round(time() - start_time, 2))

if __name__ == "__main__":
    main()  # pylint: disable=E1120