import cv2
import numpy as np
from scipy.spatial import Delaunay
import sys

def image_morph(src, target, src_feature, target_feature, morph_rates):
	point_num = len(src_feature)
	assert len(src_feature) == len(target_feature)
	
	n, m = src.shape[0], src.shape[1]
	assert src.shape == target.shape
	edges = [[0, 0], [0, m / 2], [0, m - 1], [n / 2, 0], [n / 2, m - 1], [n - 1, 0], [n - 1, m / 2], [n - 1, m - 1]]
	for point in edges:
		src_feature.append(point)
		target_feature.append(point)
	src_feature = np.array(src_feature)
	target_feature = np.array(target_feature)
	delaunay = Delaunay(src_feature)
	tri_num = delaunay.simplices.shape[0]
	
	morph_num = len(morph_rates)
	output = []
	for i in range(morph_num):
		output.append(np.zeros((n, m, 3)))
	for i in range(n):
		for j in range(m):
			tri = delaunay.find_simplex((i, j))
			tri = delaunay.simplices[tri, :]
			A = np.zeros((3, 3))
			b = np.zeros(3)
			for k in range(3):
				A[0, k] = src_feature[tri[k], 0]
				A[1, k] = src_feature[tri[k], 1]
				A[2, k] = 1
			b[0] = i
			b[1] = j
			b[2] = 1
			x = np.linalg.solve(A, b)
			end_x = x[0] * target_feature[tri[0], 0] + x[1] * target_feature[tri[1], 0] + x[2] * target_feature[tri[2], 0]
			end_y = x[0] * target_feature[tri[0], 1] + x[1] * target_feature[tri[1], 1] + x[2] * target_feature[tri[2], 1]
			end_x = int(end_x)
			end_y = int(end_y)
			if (end_x >= n):
				end_x = n - 1
			if (end_y >= m):
				end_y = m - 1
			color1 = src[i, j, :]
			color2 = target[end_x, end_y, :]
			for k in range(morph_num):
				rate = 1 - morph_rates[k]
				x, y = i * rate + end_x * (1 - rate), j * rate + end_y * (1 - rate)
				dx, dy = x - int(x), y - int(y)
				x, y = int(x), int(y)
				output[k][x, y, :] = color1 * rate + color2 * (1 - rate)
				if x < n - 1:
					output[k][x + 1, y, :] = color1 * rate + color2 * (1 - rate)
				if y < m - 1:
					output[k][x, y + 1, :] = color1 * rate + color2 * (1 - rate)
				if x < n - 1 and y < m - 1:
					output[k][x + 1, y + 1, :] = color1 * rate + color2 * (1 - rate)
	
	cnt = 0
	for image in output:
		for i in range(1, n):
			for j in range(1, m):
				if image[i, j, 0] == 0 and image[i, j, 1] == 0 and image[i, j, 2] == 0:
					image[i, j, :] = (image[i - 1, j, :] + image[i, j - 1, :] + image[i - 1, j - 1, :]) / 3.
		cnt += 1
	return output
