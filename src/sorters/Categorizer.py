from src.sorters.AbstractSorter import AbstractSorter


class Categorizer(AbstractSorter):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/categorizer.yml'):
        super().__init__(input_path, output_path, config_path)

    def process(self):
        pass