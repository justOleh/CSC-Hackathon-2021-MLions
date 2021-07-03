from src.post_processors.abstract import AbstractPostProcessor
import torch
import cv2 as cv
import numpy as np


class BlurRemoval(AbstractPostProcessor):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/blur_removal.yml'):
        super().__init__(input_path, output_path, config_path)

        self.transforms = None
        self.classes = ['clear', 'blur']
        model = self._def_model()
        self.model = self._load_model(model, self.config['weights'])

    def _def_model(self) -> torch.Model:
        pass

    def _preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        pass

    def _load_model(self, model: torch.Model, weights_path: str) -> torch.Model:
        pass

    def _model_inference(self, model: torch.Model, img: np.ndarray) -> str:
        return 'clear'
        # return 'blur'

    def process(self):
        pass