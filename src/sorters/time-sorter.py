from pathlib import Path

from src.sorters.abstract import AbstractSorter
from utils.meta_data import get_date_taken, get_exif_data


class TimeSorter(AbstractSorter):
    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path)

    def process(self):
        def comparator(x):
            time = get_date_taken(get_exif_data(x[0]))
            return time

        sorted_data = sorted(zip(self.images, self.file_names), key=comparator)
        for image, file_name in sorted_data:
            image.save(Path(self.output_path) / file_name)


if __name__ == '__main__':
    time_sorter = TimeSorter('../../data/image_samples', '../../data/test')
    time_sorter.process()
