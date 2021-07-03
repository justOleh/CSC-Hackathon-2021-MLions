from segmentation import SegmentationModel
from utils import visualize_segmentation
from containers import PhotoObjects
from PIL import Image
import tensorflow as tf
import config
import cv2

if __name__ == "__main__":
	image_path = "dataset/test/IMG_4749.JPG"
	with tf.io.gfile.GFile(image_path, 'rb') as f:
		image = Image.open(f)

	# TODO: add processing for image batches

	model = SegmentationModel(config.MODEL_PATH, config.LABELS_PATH)
	seg_map = model.segment_image(image)
	visualize_segmentation(image, seg_map, model.label_names)

	photo = PhotoObjects(seg_map, model.label_names)
	visualize_segmentation(image, photo.seg_map, model.label_names)
