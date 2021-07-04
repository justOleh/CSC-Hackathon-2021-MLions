
#!/bin/bash

# /conda/envs/some_env/bin/python

python3 main.py --input_path data/blur --output_path data/blur --mode None --post_process blur
python3 main.py --input_path data/find_people --output_path data/find_people --mode find_people --post_process None