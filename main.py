import argparse
from src.sorters.categorizer import Categorizer
from src.sorters.dummy import DummySorter
from src.post_processors.blur_removal import BlurRemoval
from src.post_processors.dummy import DummyPostProcessor

sorter_classes = {
    'categorize': Categorizer,
    'find_people': None,
    'time': None,
    'None': DummySorter
}

post_processing_classes = {
    'duplicates': None,
    'blur': BlurRemoval,
    'None': DummyPostProcessor
}


def main(args: dict):
    sorter_class = sorter_classes[args['mode']]
    post_processing_class = post_processing_classes[args['post_process']]
    sorter = sorter_class(args['input_path'], args['output_path'])
    post_processor = post_processing_class(args['output_path'], args['output_path'])

    sorter.process()
    post_processor.process()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', required=True, choices=['categorize', 'find_people', 'time', 'None'],
                        help='indicates which of the main methods should be used')
    parser.add_argument('--post_process', required=False, default='None', choices=['duplicates', 'blur', 'None'])

    args = parser.parse_args()
    args = vars(args)

    main(args)