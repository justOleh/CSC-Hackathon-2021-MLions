import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf


class SegmentationModel:

	def __init__(self, model_path:str, labels_path:str):
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()
		self._input_details = self.interpreter.get_input_details()
		self._output_details = self.interpreter.get_output_details()

		self.input_size=self._input_details[0]['shape'][2], self._input_details[0]['shape'][1]
		self.output_size=self._output_details[0]['shape'][2], self._output_details[0]['shape'][1]

		labels_info = pd.read_csv(labels_path)
		labels_list = list(labels_info['Name'])
		labels_list.insert(0, 'other')
		self.label_names = np.asarray(labels_list)
		self._interpreter_invoked_=False


	def segment_image(self, img:np.ndarray) -> np.ndarray:
		"""
			Performs image segmentation on one image and returns a mask of segmented objects
		"""
		img_width, img_height = img.size
		input_img = self._preprocess_image_(img)
		self.interpreter.set_tensor(self._input_details[0]['index'], input_img)
		# if not self._interpreter_invoked_:
		self.interpreter.invoke()
		# self._interpreter_invoked_ = True

		predictions = self.interpreter.tensor(
			self.interpreter.get_output_details()[0]['index']
		)()

		# resize the image to originals size and find argmax for every pixel
		seg_map = tf.argmax(tf.image.resize(predictions, (img_height, img_width)), axis=3)
		seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
		return seg_map

	def segment_image_noresize(self, img:np.ndarray) -> np.ndarray:
		"""
			Performs image withought resize at the end
		"""
		img_width, img_height = img.size
		input_img = self._preprocess_image_(img)
		self.interpreter.set_tensor(self._input_details[0]['index'], input_img)
		self.interpreter.invoke()

		predictions = self.interpreter.tensor(
			self.interpreter.get_output_details()[0]['index']
		)()

		# resize the image to originals size and find argmax for every pixel
		seg_map = tf.argmax(tf.image.resize(predictions, (self.input_size[0], self.input_size[1])), axis=3)
		seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
		return seg_map

	def _preprocess_image_(self, img:np.ndarray) -> np.ndarray:
		# Prepare input image data.
		resized_img = img.convert('RGB').resize(self.input_size, Image.ANTIALIAS)
		resized_img = np.asarray(resized_img).astype(np.float32)
		resized_img = np.expand_dims(resized_img, 0)
		resized_img = resized_img / 127.5 - 1

		return resized_img
