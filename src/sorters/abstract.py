from typing import Optional, List
from pathlib import Path

from abc import ABC, abstractmethod
import yaml
from PIL import Image


class AbstractSorter(ABC):

    def __init__(self, input_path: str, output_path: str, config_path: Optional[str] = None):
        self.input_path = input_path
        self.file_names = self.get_file_names(self.input_path)
        self.images = self.load_images(self.input_path, self.file_names)
        self.output_path = output_path
        self.config_path = config_path
        if config_path is not None:
            self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path) as file:
            config = yaml.load(file)
            return config

    @staticmethod
    def get_file_names(input_path: str) -> List[str]:
        return [str(file.name) for file in Path(input_path).glob('*')]

    @staticmethod
    def load_images(input_path, file_names: List[str]):
        images = [Image.open(Path(input_path) / file) for file in file_names]
        return images

    @abstractmethod
    def process(self):
        pass
