from torchvision import models, transforms
import torch
from PIL import Image
import cv2 as cv
import numpy as np
import os
# from typing import []

from src.post_processors.abstract import AbstractPostProcessor


class InferenceModel:
    transforms_f = transforms.Compose([
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

    def __call__(self, img: [np.ndarray, str]) -> str:
        # if (type(img) is str):
        #     img = cv.imread(img)
        img_norm = self._preprocess_img(img)
        img_norm = img_norm.to(self.device)
        logits = self.model(img_norm)
        class_id = torch.argmax(logits).numpy()
        return self.classes[class_id]

    def _def_model(self) -> torch.nn.Module:
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, 2)
        return model_ft

    def _preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(self.transforms_f(img), dim=0)


class BlurRemoval(AbstractPostProcessor):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/blur_removal.yml'):
        super().__init__(input_path, output_path, config_path)
        self.res_folder_name = self.input_path / 'blurred'
        os.makedirs(self.res_folder_name, exist_ok=True)

        self.model = InferenceModel(self.config['weights'])

    def process(self):
        images_path = [file for file in self.input_path.glob("**/*")
                       if 'png' in str(file) or 'jpg' in str(file) or 'jpeg' in str(file)]
        for idx, image_path in enumerate(images_path):
            img = cv.imread(str(image_path))
            img_class = self.model(img)
            if img_class == 'blur':
                os.rename(image_path, self.res_folder_name/image_path.name)