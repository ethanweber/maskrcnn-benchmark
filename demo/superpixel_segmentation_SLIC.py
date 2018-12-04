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
	def __init__(self, image, mask=None, resolution = 250, string=True):
		if string:
			raw_img = io.imread(image)
			self.original_image = img_as_float(raw_img)
			self.image = img_as_float(raw_img)[:,:,0]
			self.mask = io.imread(mask)[:,:,0]
			self.segments_slic = []
			self.n_segments = resolution
		else:
			raw_img = image
			self.original_image = img_as_float(raw_img)
			self.image = img_as_float(raw_img)[:,:,0]
			self.mask = mask[:,:,0]
			self.segments_slic = []
			self.n_segments = resolution

	def get_SLIC_segments(self, show_segments=True):

		segments_slic = slic(self.image, self.n_segments, compactness=10, sigma=1)

		if self.n_segments < 5000 and show_segments:
			fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
			ax.imshow(mark_boundaries(self.image, segments_slic))
			ax.set_title('SLIC')
			plt.show()
		self.segments_slic = segments_slic
		return segments_slic

	def tune_mask_expand(self, show_masks=True):
		""" Expands mask according to superpixel segmentation"""
		mask_pixels = list(zip(*np.where(self.mask > 0)))
		if show_masks:
			imgplot = plt.imshow(self.original_image)
			plt.title("Image")
			plt.show()
			imgplot = plt.imshow(self.mask)
			plt.title("Before Enhancement")
			plt.show()
		
		#find segments with parts in mask
		present_segments = set()
		for r,c in mask_pixels:
			if self.segments_slic[r][c] not in present_segments:
				present_segments.add(self.segments_slic[r][c])

		#fill in the mask with those segments
		for r in range(len(self.segments_slic)):
			for c in range(len(self.segments_slic[0])):
				if self.segments_slic[r][c] in present_segments:
					self.mask[r][c] = 100
					
		if show_masks:
			imgplot = plt.imshow(self.mask)
			plt.title("After Enhancement")
			plt.show()
		return self.mask, mask_pixels

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

# enhancer = MaskEnhancer(io.imread('demo_images/caterpillar.png'), io.imread('demo_images/mask.png'), resolution=5e2, string=False)
# enhancer.get_SLIC_segments()
# enhancer.tune_mask_expand()
