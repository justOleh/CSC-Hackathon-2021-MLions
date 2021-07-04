import numpy as np
from itertools import groupby

class PhotoObjects:
	"""
		A class which creates a dict of objects in the
		segmented image and calculates their area in the image
		Needed for determening the focus of the photo
	"""

	# the object is accounted for if it's area is larger than threshold
	AREA_THRESHOLD = 0.05

	def __init__(self, seg_map:np.ndarray, label_map:np.ndarray):
		self.seg_map = seg_map.astype("uint8")
		self.label_map = label_map
		self.obj_dict = {}
		self.vector_features = np.zeros(len(label_map))
		self._find_objects_()

	def _find_objects_(self):
		seg_map_flattened = np.ravel(self.seg_map)
		seg_map_flattened = np.sort(seg_map_flattened)
		photo_objects = np.array([list(elm) for _,elm in groupby(seg_map_flattened)])

		img_area = self.seg_map.shape[0] * self.seg_map.shape[1]

		# we have the array of objects - new we can determine the area of every objects and it's label
		# note: if the obj is too small it's ignored
		for obj in photo_objects:
			obj_area = len(obj) / img_area
			if obj_area < self.AREA_THRESHOLD:
				# the object is too small skipping and updating the segmap
				self.seg_map = np.where(self.seg_map == obj[0], 0, self.seg_map)
				continue
			self.obj_dict[self.label_map[obj[0]]] = obj_area
			self.vector_features[obj[0]] = obj_area
