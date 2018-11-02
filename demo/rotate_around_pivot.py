import math
import numpy as np

def rotate_around_pivot(pixels, pivot, theta):
	"""pixels: dictionary: (x,y): pixel value
   pivot: pivot (a,b) to rotate around"""
	moved_pixels = dict()
	for point in pixels:
		moved_pixels[rotate_point_around_pivot(point, pivot, theta)] = pixels[point]
	return moved_pixels
		


def rotate_point_around_pivot(point, pivot, theta):
	"""rotates point clockwise around pivot by angle theta"""
	theta = math.radians(theta)

	# 1) subtract pivot from point
	(a,b) = point[0] - pivot[0], point[1] - pivot[1]
	#2) rotate temp_point about origin by theta
	(a,b) = (a*np.cos(theta)-b*np.sin(theta), a*np.sin(theta)+b*np.cos(theta))
	# print((a,b))

	#3) add back the pivot for final point
	final_point = int(a+pivot[0]+0.5), int(b+pivot[1]+0.5)

	return final_point
