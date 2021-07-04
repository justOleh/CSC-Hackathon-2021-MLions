import os.path
import cv2
import numpy as np
import joblib
from six import class_types
import tensorflow as tf

from src.sorters.abstract import AbstractSorter

from .  import segmentation as seg


class Segmentator(AbstractSorter):
	def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/categorizer.yml'):
		super().__init__(input_path, output_path, config_path)
	
	def process(self):
		model = seg.SegmentationModel(self.config['segmentator_path'], self.config['labels_path'])
		log_reg_model = seg.LogRegression()
		log_reg_model.load(self.config['logreg_path'])
			
		data_loader = tf.data.Dataset.list_files(os.path.join(self.input_path, '*.jpg'))
		imgs, feature_vectors, img_types = [], [], []
		for image, _ in data_loader.map(prepare_example_wrap):
			image, feature_vectors = image.numpy(), model.segment_image(image)
			img_type = log_reg_model.predict(feature_vectors)
			imgs.append(image)
			features.append(feature_vectors[0])
			img_types.append(img_type)

		img_types = np.array(img_types, dtype="int8")

		img_paths = np.array([fp.numpy().decode() for fp in data_loader])
		img_class_sorted = {}
		for type_id in range(log_reg_model.classes_num):
			class_image_paths = img_paths[type_id == img_types]
			if class_image_paths.shape[0] == 0:
				continue
			img_class_sorted[log_reg_model.get_name(type_id)] = class_image_paths.tolist()

		return img_class_sorted
