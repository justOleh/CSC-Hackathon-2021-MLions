from typing import Optional, List
from pathlib import Path

from abc import ABC, abstractmethod
import yaml


class AbstractSorter(ABC):

    def __init__(self, input_path: str, output_path: str, config_path: Optional[str] = None):
        self.input_path = input_path
        self.file_names = self.get_file_names(self.input_path)
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
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

    @abstractmethod
    def process(self):
        pass
