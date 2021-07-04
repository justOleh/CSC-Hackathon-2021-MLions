from torchvision import datasets, models, transforms
import torch
from PIL import Image
import cv2 as cv
import numpy as np

from src.post_processors.abstract import AbstractPostProcessor


class InferenceModel:
    transforms = transforms.Compose([
             Image.fromarray,
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
    classes = ['clear', 'blur']

    def __init__(self, model_weights_path: str):
        self.model = self._def_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

    def __call__(self, img: np.ndarray) -> str:
        img_norm = self._preprocess_img(img)
        img_norm = img_norm.to(self.device)
        logits = self.model(img_norm)
        class_id = torch.argmax(logits)
        return self.classes[class_id]

    def _def_model(self) -> torch.nn.Module:
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, 2)
        return model_ft

    def _preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        return self.transforms(img)


class BlurRemoval(AbstractPostProcessor):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/blur_removal.yml'):
        super().__init__(input_path, output_path, config_path)

        self.model = InferenceModel(self.config['weights'])

    def process(self):
        pass