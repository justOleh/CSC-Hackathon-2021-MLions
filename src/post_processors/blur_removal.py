from src.post_processors.abstract import AbstractPostProcessor
import torch
import cv2 as cv
import numpy as np


class InferenceModel:
    transforms = None
    classes = ['clear', 'blur']

    def __init__(self, model_weights_path: str):
        self.model = self._def_model()
        self.model.load_state_dict()

    def __call__(self, img: np.ndarray) -> str:
        img_norm = self._preprocess_img(img)
        logits = self.model(img_norm)
        class_id = torch.argmax(logits)
        return self.classes[class_id]

    def _def_model(self) -> torch.Model:
        pass

    def _preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        pass


class BlurRemoval(AbstractPostProcessor):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/blur_removal.yml'):
        super().__init__(input_path, output_path, config_path)

        self.model = InferenceModel(self.config['weights'])

    def process(self):
        pass