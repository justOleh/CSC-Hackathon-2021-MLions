from pathlib import Path
import shutil

from PIL import Image

from src.sorters.abstract import AbstractSorter
from utils.meta_data import get_date_taken, get_exif_data


def int_to_str(x):
    coded_str = ''
    max_power = 5
    for i in range(max_power - 1, -1, -1):
        c = chr(ord('a') + (x // 26 ** i) % 26)
        coded_str += c
    return coded_str


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
        i = 0
        for time, file_name in sorted_file_names:
            image = Image.open(Path(self.input_path) / file_name)
            image.save(Path(self.output_path) / f'{int_to_str(i)}_{file_name}')
            i += 1


if __name__ == '__main__':
    time_sorter = TimeSorter('../../data/image_samples/test_duplicates', '../../data/test_time')
    time_sorter.process()
