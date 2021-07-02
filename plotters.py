import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt


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

	colormap = create_label_colormap()

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
