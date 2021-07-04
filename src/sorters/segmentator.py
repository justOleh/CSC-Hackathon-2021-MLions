import os.path
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from collections import Counter

from src.sorters.abstract import AbstractSorter
from src.sorters.categorizer import Categorizer
import src.sorters.segmentation as seg


class Segmentator(Categorizer):
	def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/categorizer.yml'):
		super().__init__(input_path, output_path, config_path)
	
	def process(self):
		model = seg.SegmentationModel(self.config['segmentator_path'], self.config['labels_path'])
		log_reg_model = seg.LogRegression()
		log_reg_model.load(self.config['logreg_path'])
		clusters = super().process()
		random_num = 3

		img_cluster_named = {}
		for img_cluster in clusters:
			cluster_ids = np.randint(0, len(img_cluster), random_num)
			cluster_ids = np.unique(cluster_ids)
			feature_vectors, img_types = [], []
			for cluster_id in cluster_ids:
				with tf.io.gfile.GFile(img_cluster[cluster_id], 'rb') as f:
					image = Image.open(f)
				feature, img_type = model.segment_image(image), log_reg_model.predict(feature)
				feature_vectors.append(feature)
				img_types.append(img_type)
			determined_classes = [count for item, count in Counter(img_types).items()]
			determined_id = np.argmax(np.array(determined_classes))
			cluster_name = log_reg_model.get_name(img_types[determined_id])
			img_cluster_named[cluster_name] = img_cluster
			# TODO: store data into folders
			# save the cluster to the directory
			dst_path = os.path.join(self.output_path, cluster_name)
			for img_file in cluster_id:
				shutil.copy(img_file, dst_path)
