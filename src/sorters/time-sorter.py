from pathlib import Path
import shutil

from PIL import Image

from src.sorters.abstract import AbstractSorter
from utils.meta_data import get_date_taken, get_exif_data


class TimeSorter(AbstractSorter):
    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path)

    def process(self):
        images_time = []
        for file_name in self.file_names:
            image = Image.open(Path(self.input_path) / file_name)
            time = get_date_taken(get_exif_data(image))
            images_time.append(time)

        sorted_file_names = sorted(zip(images_time, self.file_names))
        for _, file_name in sorted_file_names:
            shutil.copy(Path(self.input_path) / file_name, self.output_path)


if __name__ == '__main__':
    time_sorter = TimeSorter('../../data/image_samples', '../../data/test')
    time_sorter.process()
