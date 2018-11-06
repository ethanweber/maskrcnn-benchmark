import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from skimage import io


class MaskEnhancer:
	def __init__(self, image_string, mask_string=None, resolution = 250):
		raw_img = io.imread(image_string)
		self.image = img_as_float(raw_img)
		self.mask = io.imread(mask_string)
		self.segments_slic = []
		self.n_segments = resolution

	def get_SLIC_segments(self):
		print("Getting Segments...")
		segments_slic = slic(self.image, self.n_segments, compactness=10, sigma=1)

		if self.n_segments < 5000:
			print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
			fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)

			ax.imshow(mark_boundaries(self.image, segments_slic))
			ax.set_title('SLIC')
			plt.show()
		self.segments_slic = segments_slic
		return segments_slic

	def tune_mask_expand(self):
		""" Expands mask according to superpixel segmentation"""
		mask_pixels = list(zip(*np.where(self.mask == 100)))
		imgplot = plt.imshow(self.mask)
		plt.title("Before Enhancement")
		plt.show()

		#find segments with parts in mask
		print("Finding Relevant Segments...")
		present_segments = set()
		for r,c in mask_pixels:
			if self.segments_slic[r][c] not in present_segments:
				present_segments.add(self.segments_slic[r][c])

		#fill in the mask with those segments
		print("Updating Mask...")
		for r in range(len(self.segments_slic)):
			for c in range(len(self.segments_slic[0])):
				if self.segments_slic[r][c] in present_segments:
					self.mask[r][c] = 100

		imgplot = plt.imshow(self.mask)
		plt.title("After Enhancement")
		plt.show()
		return self.mask

	def tune_mask_trim(self):
		""" Trims mask according to superpixel segmentation"""
		mask_pixels = set(list(zip(*np.where(self.mask == 100))))
		imgplot = plt.imshow(self.mask)
		plt.title("Before Enhancement")
		plt.show()

		#find partial segments in mask
		print("Finding Segments not Fully Contained in Mask...")
		uncontained_segments = set()
		for r in range(len(self.segments_slic)):
			for c in range(len(self.segments_slic[0])):
				if (r,c) not in mask_pixels:
					uncontained_segments.add(self.segments_slic[r][c])

		#trim mask, removing partial segments
		print("Updating Mask...")
		for r,c in mask_pixels:
			if self.segments_slic[r][c] in uncontained_segments:
				self.mask[r][c] = 0

		
		imgplot = plt.imshow(self.mask)
		plt.title("After Enhancement")
		plt.show()
		return self.mask

enhancer = MaskEnhancer('demo_images/caterpillar.png', 'demo_images/mask.png', resolution=5e4)
enhancer.get_SLIC_segments()
enhancer.tune_mask_trim()
