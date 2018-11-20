from random import sample
import numpy as np

class OptimizeSLIC:

	def __init__(self, truths, masks, images, num_training_points):
		self.truths = truths
		self.masks = masks
		self.images = images
		self.n = num_training_points

	def optimize(self):
		#returns best resolution
		best_resolution = 0
		best_loss = float('inf')
		for resolution in range(1,1001):
			average_loss = self.evaluate(resolution)
			if average_loss < best_loss:
				best_loss = average_loss
				best_resolution = resolution
		return resolution

	def evaluate(self, resolution, batch_size=10):
		points = sample(range(0,self.n-1), batch_size)
		loss = np.mean([self.calculate_loss(truths[i], masks[i], images[i], resolution) for i in points])
		return loss

	def calculate_loss(truth, mask, image, resolution):
		total_pixels = numpy.prod(image.size())/3.0
		enhancer = MaskEnhancer(image, mask, resolution)
		enhancer.get_SLIC_segments(show_segments=False)
		enhanced_mask = enhancer.tune_mask_expand()

		missed_pixels = 0
		extra_pixels = 0
		for r in range(len(mask)):
			for c in range(len(mask[0])):
				if truth[r][c] > 0 and mask[r][c] == 0:
					missed_pixels += 1
				if truth[r][c] == 0 and mask[r][c] > 0:
					extra_pixels += 1

		return total_pixels*missed_pixels + extra_pixels
	










