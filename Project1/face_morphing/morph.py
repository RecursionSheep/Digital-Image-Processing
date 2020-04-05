import cv2
import numpy as np
from scipy.spatial import Delaunay
import sys

'''src = "source1.png"
target = "target1.png"
src_feature = [[240, 71], [228, 124], [224, 168], [222, 212], [214, 142], [288, 119], [287, 182], [334, 107], [325, 200], [302, 149], [235, 10], [329, 49], [210, 284], [310, 268]]
target_feature = [[229, 73], [229, 119], [224, 171], [226, 215], [216, 135], [301, 115], [295, 169], [339, 101], [334, 190], [308, 137], [230, 29], [297, 53], [206, 290], [288, 268]]'''
src = "source2.png"
target = "target2.png"
src_feature = [[170, 57], [168, 123], [170, 187], [170, 242], [168, 154], [262, 121], [252, 191], [295, 105], [292, 207], [374, 156], [40, 149]]
target_feature = [[156, 42], [155, 81], [146, 194], [139, 228], [209, 144], [277, 94], [261, 200], [370, 122], [365, 189], [409, 161], [125, 139]]
point_num = len(src_feature)
assert len(src_feature) == len(target_feature)

src = cv2.imread(src)
src = src / 255.
target = cv2.imread(target)
target = target / 255.

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

morph_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
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
				image[i, j, :] = (image[i - 1, j, :] + image[i, j - 1, :]) / 2.
	cnt += 1
	cv2.imwrite("output%d.png" % cnt, (np.clip(image, 0., 1.) * 255).astype(np.uint8))
