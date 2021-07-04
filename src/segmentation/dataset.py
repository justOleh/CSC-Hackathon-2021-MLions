import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
from PIL import Image

from .photo_objects import PhotoObjects
from .model import SegmentationModel
import config
import csv
from tqdm import tqdm
import re



def process_image(image_path, model):
	with tf.io.gfile.GFile(image_path, 'rb') as f:
		image = Image.open(f)
	seg_map = model.segment_image_noresize(image)
	photo = PhotoObjects(seg_map, model.label_names)

	return photo.vector_features


def create_dataset(dataset_path:str):
	model = SegmentationModel(config.MODEL_PATH, config.LABELS_PATH)
	# with open('dataset.csv', 'w', encoding='UTF8') as f:
	# 	pass

	for _, dirs, _ in os.walk(dataset_path):
		print(dirs)
		for directory in dirs:
			class_id = int(re.search(r'\d+', directory).group()) - 1
			print("Possessing ", directory, "(class", class_id, ")")
			for _, _, files in os.walk(os.path.join(dataset_path, directory)):
				with open('dataset.csv', 'a', encoding='UTF8') as csv_dataset:
					writer = csv.writer(csv_dataset)
					for image_path in tqdm(files):
						try: 
							vector_features = process_image(os.path.join(dataset_path, directory, image_path), model)
							row = vector_features.tolist()
							row.insert(0, class_id)
							writer.writerow(row)
						except:
							print("\ncouldn't open image", image_path, "skipping")
