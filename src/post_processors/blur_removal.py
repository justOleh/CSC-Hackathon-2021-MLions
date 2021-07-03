from src.post_processors.abstract import AbstractPostProcessor


class BlurRemoval(AbstractPostProcessor):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/blur_removal.yml'):
        super().__init__(input_path, output_path, config_path)

    def process(self):
        pass