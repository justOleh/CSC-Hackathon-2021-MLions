import argparse

classes = {}


def main(args: dict):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', required=True, choices=['categorize', 'find_people', 'time'],
                        help='indicates which of the main methods should be used')
    parser.add_argument('--post_process', required=False, default=None, choices=['duplicates', 'blur'])

    args = parser.parse_args()
    args = vars(args)

    main(args)