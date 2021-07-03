from abc import ABC, abstractmethod
import yaml


class AbstractSorter(ABC):

    def __init__(self, input_path: str, output_path: str, config_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.config_path = config_path
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> dict:
        with open(config_path) as file:
            config = yaml.load(file)
            return config

    @abstractmethod
    def process(self):
        pass


