import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
import cv2


def create_ade20k_label_colormap():
	"""Creates a label colormap used in ADE20K segmentation benchmark.
	Returns:
		A colormap for visualizing segmentation results.
	NOTE: it's better to define colormap this way because
	if simple colormap generator is used the labels aren't
	shown correctly for 150 classes
	"""
	return np.asarray([
		[0, 0, 0],
		[120, 120, 120],
		[180, 120, 120],
		[6, 230, 230],
		[80, 50, 50],
		[4, 200, 3],
		[120, 120, 80],
		[140, 140, 140],
		[204, 5, 255],
		[230, 230, 230],
		[4, 250, 7],
		[224, 5, 255],
		[235, 255, 7],
		[150, 5, 61],
		[120, 120, 70],
		[8, 255, 51],
		[255, 6, 82],
		[143, 255, 140],
		[204, 255, 4],
		[255, 51, 7],
		[204, 70, 3],
		[0, 102, 200],
		[61, 230, 250],
		[255, 6, 51],
		[11, 102, 255],
		[255, 7, 71],
		[255, 9, 224],
		[9, 7, 230],
		[220, 220, 220],
		[255, 9, 92],
		[112, 9, 255],
		[8, 255, 214],
		[7, 255, 224],
		[255, 184, 6],
		[10, 255, 71],
		[255, 41, 10],
		[7, 255, 255],
		[224, 255, 8],
		[102, 8, 255],
		[255, 61, 6],
		[255, 194, 7],
		[255, 122, 8],
		[0, 255, 20],
		[255, 8, 41],
		[255, 5, 153],
		[6, 51, 255],
		[235, 12, 255],
		[160, 150, 20],
		[0, 163, 255],
		[140, 140, 140],
		[250, 10, 15],
		[20, 255, 0],
		[31, 255, 0],
		[255, 31, 0],
		[255, 224, 0],
		[153, 255, 0],
		[0, 0, 255],
		[255, 71, 0],
		[0, 235, 255],
		[0, 173, 255],
		[31, 0, 255],
		[11, 200, 200],
		[255, 82, 0],
		[0, 255, 245],
		[0, 61, 255],
		[0, 255, 112],
		[0, 255, 133],
		[255, 0, 0],
		[255, 163, 0],
		[255, 102, 0],
		[194, 255, 0],
		[0, 143, 255],
		[51, 255, 0],
		[0, 82, 255],
		[0, 255, 41],
		[0, 255, 173],
		[10, 0, 255],
		[173, 255, 0],
		[0, 255, 153],
		[255, 92, 0],
		[255, 0, 255],
		[255, 0, 245],
		[255, 0, 102],
		[255, 173, 0],
		[255, 0, 20],
		[255, 184, 184],
		[0, 31, 255],
		[0, 255, 61],
		[0, 71, 255],
		[255, 0, 204],
		[0, 255, 194],
		[0, 255, 82],
		[0, 10, 255],
		[0, 112, 255],
		[51, 0, 255],
		[0, 194, 255],
		[0, 122, 255],
		[0, 255, 163],
		[255, 153, 0],
		[0, 255, 10],
		[255, 112, 0],
		[143, 255, 0],
		[82, 0, 255],
		[163, 255, 0],
		[255, 235, 0],
		[8, 184, 170],
		[133, 0, 255],
		[0, 255, 92],
		[184, 0, 255],
		[255, 0, 31],
		[0, 184, 255],
		[0, 214, 255],
		[255, 0, 112],
		[92, 255, 0],
		[0, 224, 255],
		[112, 224, 255],
		[70, 184, 160],
		[163, 0, 255],
		[153, 0, 255],
		[71, 255, 0],
		[255, 0, 163],
		[255, 204, 0],
		[255, 0, 143],
		[0, 255, 235],
		[133, 255, 0],
		[255, 0, 235],
		[245, 0, 255],
		[255, 0, 122],
		[255, 245, 0],
		[10, 190, 212],
		[214, 255, 0],
		[0, 204, 255],
		[20, 0, 255],
		[255, 255, 0],
		[0, 153, 255],
		[0, 41, 255],
		[0, 255, 204],
		[41, 0, 255],
		[41, 255, 0],
		[173, 0, 255],
		[0, 245, 255],
		[71, 0, 255],
		[122, 0, 255],
		[0, 255, 184],
		[0, 92, 255],
		[184, 255, 0],
		[0, 133, 255],
		[255, 214, 0],
		[25, 194, 194],
		[102, 255, 0],
		[92, 0, 255],
	])

def create_label_colormap():
	"""
		A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
			ind >>= 3

	return colormap

def label_to_color_image(label):
	"""
		Adds color defined by the dataset colormap to the label.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_ade20k_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]

def visualize_segmentation(image, seg_map, label_names):
	"""Visualizes input image, segmentation map and overlay view."""
	label_map = np.arange(len(label_names)).reshape(len(label_names), 1)
	full_colormap = label_to_color_image(label_map)

	plt.figure(figsize=(15, 5))
	grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

	plt.subplot(grid_spec[0])
	plt.imshow(image)
	plt.axis('off')
	plt.title('input image')

	plt.subplot(grid_spec[1])
	seg_image = label_to_color_image(seg_map).astype(np.uint8)
	plt.imshow(seg_image)
	plt.axis('off')
	plt.title('segmentation map')

	plt.subplot(grid_spec[2])
	plt.imshow(image)
	plt.imshow(seg_image, alpha=0.7)
	plt.axis('off')
	plt.title('segmentation overlay')

	unique_labels = np.unique(seg_map)
	ax = plt.subplot(grid_spec[3])
	plt.imshow(
		full_colormap[unique_labels].astype(np.uint8), interpolation='nearest')
	ax.yaxis.tick_right()
	plt.yticks(range(len(unique_labels)), label_names[unique_labels])
	plt.xticks([], [])
	ax.tick_params(width=0.0)
	plt.grid('off')
	plt.show()
