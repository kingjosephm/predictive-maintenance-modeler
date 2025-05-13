from omegaconf import DictConfig

from pdm.data_processing import DataProcessor
from pdm.trainer import Trainer
from pdm.predictor import Predictor

class Pipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dp = DataProcessor(self.cfg)
        self.trainer = Trainer(self.cfg, self.dp)
        self.predictor = Predictor(self.cfg, self.dp)

    def run(self):
        if self.cfg.mode == 'train':
            self.trainer.run()
        elif self.cfg.mode == 'predict':
            self.predictor.run()
        else:
            raise ValueError(f"Unknown mode {self.cfg.mode}")