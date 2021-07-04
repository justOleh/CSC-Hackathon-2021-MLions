import os.path
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from collections import Counter

from src.sorters.abstract import AbstractSorter
from src.sorters.categorizer import Categorizer
from utils.plotters import visualize_segmentation
import src.sorters.segmentation as seg


class Segmentator(Categorizer):
	def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/categorizer.yml'):
		super().__init__(input_path, output_path, config_path)
	
	def process(self):
		self.model = seg.SegmentationModel(self.config['segmentator_path'], self.config['labels_path'])
		self.log_reg_model = seg.LogRegression()
		self.log_reg_model.load(self.config['logreg_path'])

		# checks the value from config and
		# choses which processing alg to use
		if self.config['process'] == 'all':
			self._process_all_()
		else:
			self._process_clusters_()

	def _process_all_(self):
		imgList = []
		try:
			for files in os.listdir(self.input_path):
				isImg = (files.find('.jpg') > 0) or (files.find('.png') > 0) or (files.find('.tiff') > 0) or (files.find('.JPG') > 0)
				if isImg:
					imgList.append(files)
		except:
			pass

		for img_file in imgList:
			feature, img_type, name = self.__process_photofile__(os.path.join(self.input_path, img_file))
			# save the cluster to the directory
			cluster_name = self.log_reg_model.get_class(img_type[0])
			dst_path = os.path.join(self.output_path, cluster_name)
			if not os.path.exists(dst_path):
				os.makedirs(dst_path)
			shutil.copy(os.path.join(self.input_path, img_file), dst_path)

	def _process_clusters_(self):
		clusters = super().process()
		num_cluster_im = 5
		img_cluster_named = {}
		for img_cluster in clusters:
			cluster_ids = [r_id for r_id in range(min(num_cluster_im, len(img_cluster)))]

			img_types, names = [], [], []
			for cluster_id in cluster_ids:
				feature, img_type, name = self.__process_photofile__(img_cluster[cluster_id])
				img_types.append(img_type)
				names.append(name)

			classes = self.__get_commot_names__(names)
			cluster_name_combined = ""
			for s in classes:
				cluster_name_combined += (s.split(';')[0] + "_")

			determined_classes, repeats = np.unique(np.array(img_types), return_counts=True)
			class_index = repeats.argmax()
			determined_id = determined_classes[class_index]
			cluster_name = self.log_reg_model.get_class(determined_id)
			img_cluster_named[cluster_name] = classes

			# save the cluster to the directory
			dst_path = os.path.join(self.output_path, cluster_name)
			if not os.path.exists(dst_path):
				os.makedirs(dst_path)
			for img_file in img_cluster:
				shutil.copy(img_file, dst_path)

	def __get_commot_names__(self, names_list):
		"""
			A list of tuples is expected - the common ones will be returned
		"""
		result = set(names_list[0])
		for s in names_list[1:]:
			result.intersection_update(s)
		return result

	def __process_photofile__(self, filename):
		with tf.io.gfile.GFile(filename, 'rb') as f:
			image = Image.open(f)
		seg_map = self.model.segment_image(image)
		photo_ojs = seg.PhotoObjects(seg_map, self.model.label_names)
		feature = np.reshape(photo_ojs.vector_features, (1, -1))
		img_type = self.log_reg_model.predict(feature)
		return feature, img_type, photo_ojs.obj_dict.keys()
