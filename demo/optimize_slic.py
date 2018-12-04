from random import sample
from random import randint
import numpy as np
from scipy import stats
from skimage import io
import matplotlib.pyplot as plt

from superpixel_segmentation_SLIC import MaskEnhancer

class OptimizeSLIC:

	def __init__(self, truths=None, masks=None, images=None, num_training_points=1):
		self.truths = truths
		self.masks = masks
		self.images = images
		self.n = num_training_points

	def optimize(self, max_points = 500):

		#Similar to Peak Search Algorithm
		best_resolution = 0
		best_loss = float('inf')
		data = []

		resolution_max = 1000
		max_learning_rate = 100
		resA = int(resolution_max*0.3)
		previous_resolution = resA
		previous_loss = self.evaluate(resA)

		resB = int(resolution_max*0.7)
		second_previous_resolution = resB
		second_previous_loss = self.evaluate(resB)
		resolutions_sampled = 0
		while resolutions_sampled < max_points:

			case_increase = second_previous_resolution < previous_resolution and second_previous_loss > previous_loss
			case_increase = case_increase or (second_previous_resolution > previous_resolution and second_previous_loss < previous_loss)

			case_decrease = second_previous_resolution < previous_resolution and second_previous_loss < previous_loss
			case_decrease = case_decrease or (second_previous_resolution > previous_resolution and second_previous_loss > previous_loss)

			new_resolution = max(min(previous_resolution + randint(-20,20), resolution_max), 5)
			if case_increase:
				new_resolution = min(previous_resolution + randint(1,max_learning_rate), resolution_max)
			if case_decrease:
				new_resolution = max(previous_resolution - randint(1,max_learning_rate), 20)

			new_avg_loss = self.evaluate(new_resolution)
			data.append((new_resolution, new_avg_loss))
			if new_avg_loss < best_loss:
				best_loss = new_avg_loss
				best_resolution = new_resolution

			second_previous_resolution = previous_resolution
			second_previous_loss = previous_loss
			previous_resolution = new_resolution
			previous_loss = new_avg_loss
			print("Resolution: " + str(new_resolution) + " Loss: " + str(new_avg_loss) +  "  Count: " + str(resolutions_sampled))
			resolutions_sampled += 1
			if resolutions_sampled % 50 == 0:
				print("Best Resolution: " + str(best_resolution))
				print("Best Loss: " + str(best_loss))
		return data, best_resolution, best_loss

	def evaluate(self, resolution, batch_size=100):
		points = sample(range(0,self.n-1), batch_size)
		losses = [self.calculate_loss(truths[i], masks[i], images[i], resolution) for i in points]
		losses = [loss for loss in losses if loss >= 0]
		return np.mean(losses)

	def calculate_loss(self, truth_string, mask_string, image_string, resolution):
		truth = io.imread(truth_string)[:,:,0]
		total_pixels = truth.size


		enhancer = MaskEnhancer(io.imread(image_string), io.imread(mask_string), resolution, string=False)
		enhancer.get_SLIC_segments(show_segments=False)
		enhanced_mask, original_mask_pixels = enhancer.tune_mask_expand(show_masks=False)
		if len(original_mask_pixels) == 0:
			return -1 #ERROR

		#get ground truth mask pixel value to look for
		mask_value = self.find_ground_truth_mask_value(truth, original_mask_pixels)

		#Calculate missing and extra pixels
		missed_pixels = 0
		extra_pixels = 0
		for r in range(len(enhanced_mask)):
			for c in range(len(enhanced_mask[0])):
				if truth[r][c] == mask_value and enhanced_mask[r][c] == 0:
					missed_pixels += 1
				if truth[r][c] != mask_value and enhanced_mask[r][c] > 0:
					extra_pixels += 1
		#calculate loss
		loss = missed_pixels + extra_pixels*1.0/total_pixels
		return loss

	def find_ground_truth_mask_value(self,truth, mask_pixels):
		locations = sample(mask_pixels, min(100, len(mask_pixels)-1))
		values = [truth[r][c] for r,c in locations]
		value = stats.mode(values)[0][0]
		return value

	def plot(self, data):
		data2 = sorted(data, key = lambda x: x[0])
		resolutions, losses = zip(*data2)

		plt.figure()
		plt.plot(resolutions, losses)
		plt.show()

import glob
images = sorted(glob.glob("../2000_validation_cropped_dataset_time551/rgb_crop/*.jpg"))
truths = sorted(glob.glob("../2000_validation_cropped_dataset_time551/gt_mask/*.jpg"))
masks = sorted(glob.glob("../2000_validation_cropped_dataset_time551/predicted_mask/*.jpg"))


optimizer = OptimizeSLIC(truths = truths, masks=masks, images = images, num_training_points=len(images))
data, best_resolution, best_loss = optimizer.optimize()
optimizer.plot(data)
print("Best Resolution: " + str(best_resolution))
print("Best Loss: " + str(best_loss))


# mask = "crop_mask.jpg"
# image = "crop.jpg"
# truth = "actual_mask.jpg"
# loss = optimizer.calculate_loss(truth, mask, image)
# print("LOSS")
# print(loss)









