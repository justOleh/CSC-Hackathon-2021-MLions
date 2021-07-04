#!/bin/bash

# /conda/envs/some_env/bin/python

python3 main.py --input_path data/blur --output_path data/blur --mode None --post_process blur
python3 main.py --input_path data/find_people --output_path data/find_people --mode find_people --post_process None
python3 main.py --input_path data/image_samples/test --output_path data_samples/segment_results --mode segment --post_process None
python3 main.py --input_path ./data/image_samples/test_duplicates --output_path ./data/image_samples/res_duplicates --mode duplicates --post_process None
python3 main.py --input_path ./data/image_samples/test_duplicates --output_path ./data/image_samples/res_time --mode time --post_process None
