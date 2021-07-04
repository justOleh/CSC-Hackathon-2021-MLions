import argparse

from src.sorters import segmentator
from src.sorters.categorizer import Categorizer
from src.sorters.segmentator import Segmentator
from src.sorters.people_finder import PeopleFinder
from src.sorters.dummy import DummySorter
from src.sorters.time_sorter import TimeSorter
from src.sorters.duplicate_sorter import DuplicateSorter

from src.post_processors.blur_removal import BlurRemoval
from src.post_processors.dummy import DummyPostProcessor

sorter_classes = {
    'categorize': Categorizer,
    'segment': Segmentator,
    'find_people': PeopleFinder,
    'time': TimeSorter,
    'None': DummySorter,
}

post_processing_classes = {
    'duplicates': DuplicateSorter,
    'blur': BlurRemoval,
    'None': DummyPostProcessor,
}


def main(args: dict):
    print('Initialization...')
    sorter_class = sorter_classes[args['mode']]
    post_processing_class = post_processing_classes[args['post_process']]
    sorter = sorter_class(args['input_path'], args['output_path'])
    post_processor = post_processing_class(args['output_path'], args['output_path'])

    print('Start processing')
    sorter.process()
    print('Main method finished')
    post_processor.process()
    print('Post processing finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', required=True, choices=['categorize', 'segment', 'find_people', 'time', 'None'],
                        help='indicates which of the main methods should be used')
    parser.add_argument('--post_process', required=False, default='None', choices=['duplicates', 'blur', 'None'])

    args = parser.parse_args()
    args = vars(args)

    main(args)