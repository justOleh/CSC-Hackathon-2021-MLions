from abc import ABC, abstractmethod


class AbstractSorter(ABC):

    def __init__(self, input_path: str, output_path: str, config: dict):
        pass

    @abstractmethod
    def process(self):
        pass


