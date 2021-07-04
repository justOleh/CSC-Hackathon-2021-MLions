from .model import SegmentationModel
from .photo_objects import PhotoObjects
from .regression import LogRegression
from .dataset import create_dataset
from utils import visualize_segmentation
from PIL import Image
import tensorflow as tf
import config
import os


def test_segmentation(image_path:str):
	"""
		Test out the segmentation model on an image
	"""
	with tf.io.gfile.GFile(image_path, 'rb') as f:
		image = Image.open(f)
	image = image.rotate(270, expand=True)

	model = SegmentationModel(config.MODEL_PATH, config.LABELS_PATH)
	seg_map = model.segment_image(image)
	visualize_segmentation(image, seg_map, model.label_names)

	photo = PhotoObjects(seg_map, model.label_names)
	visualize_segmentation(image, photo.seg_map, model.label_names)


def train_classifier(dataset_path, make_dataset=False):
	"""
		If 'make_dataset' is false the 'dataset_path' variable should
			be a path to csv file which contains calculated feature vectors.
		If 'make_dataset' is true the function creates a csv dataset
			and fills it with feature vectors.
	"""
	if make_dataset:
		scv_dataset = os.path.join(dataset_path, "dataset.csv")
		# create_dataset(dataset_path, scv_dataset)
		dataset_path = scv_dataset

	model = LogRegression(dataset_path)
	model.fit()
	model.save("model/regression_model.sav")


if __name__ == "__main__":
	image_path = "../data_samples/test/IMG_4749.JPG"
	test_segmentation(image_path)

	train_classifier("../data_samples/dataset.csv")
