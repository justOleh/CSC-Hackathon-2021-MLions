from pathlib import Path
import shutil

import imagehash
import numpy as np
from PIL import Image

from src.sorters.abstract import AbstractSorter


class DuplicateSorter(AbstractSorter):
    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path)

    def process(self):
        images_hashes = []
        for file_name in self.file_names:
            image = Image.open(Path(self.input_path) / file_name)
            hash = imagehash.phash(image)
            images_hashes.append(hash)

        threshold = 18

        image_num = len(images_hashes)
        used = np.zeros(shape=image_num)
        for i in range(image_num):
            duplicates_images_idx = [i]
            if used[i] != 0:
                continue
            used[i] = 1
            for j in range(image_num):
                if used[j] == 0:
                    if images_hashes[i] - images_hashes[j] < threshold:
                        duplicates_images_idx.append(j)
                        used[j] = 1

            if len(duplicates_images_idx) == 1:
                shutil.copy(Path(self.input_path) / self.file_names[i], self.output_path)
            else:
                duplicates_path = Path(self.output_path) / f'{self.file_names[i]}_duplicates'
                duplicates_path.mkdir(parents=True, exist_ok=True)
                for idx in duplicates_images_idx:
                    shutil.copy(Path(self.input_path) / self.file_names[idx], duplicates_path)


if __name__ == '__main__':
    time_sorter = DuplicateSorter('../../data/image_samples/test_duplicates', '../../data/test_duplicates')
    time_sorter.process()
