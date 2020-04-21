import cv2
import numpy as np
from morph import image_morph

def rotation_matrix(u, theta):
	c = np.cos(theta)
	s = np.sin(theta)
	t = 1 - np.cos(theta)
	x = u[0]
	y = u[1]
	return np.array([[t * x * x + c, t * x * y, s * y],
		[t * x * y, t * y * y + c, - s * x],
		[- s * y, s * x, c]])

def prewarp(F):
	eigvalue0, eigvector0 = np.linalg.eig(F)
	eigvalue1, eigvector1 = np.linalg.eig(F.T)
	epipole0 = eigvector0[:, np.argmin(eigvalue0)]
	epipole1 = eigvector1[:, np.argmin(eigvalue1)]
	d0 = np.array([- epipole0[1], epipole0[0], 0])

	Fd0 = F.dot(d0)
	d1 = np.array([- Fd0[1], Fd0[0], 0])
	theta0 = np.arctan(epipole0[2] / (d0[1] * epipole0[0] - d0[0] * epipole0[1]))
	theta1 = np.arctan(epipole1[2] / (d1[1] * epipole1[0] - d1[0] * epipole1[1]))
	R_d0_theta0 = rotation_matrix(d0, theta0)
	R_d1_theta1 = rotation_matrix(d1, theta1)

	new_e0 = R_d0_theta0.dot(epipole0)
	new_e1 = R_d1_theta1.dot(epipole1)
	phi0 = - np.arctan(new_e0[1] / new_e0[0])
	phi1 = - np.arctan(new_e1[1] / new_e1[0])
	R_phi0 = np.array([[np.cos(phi0), - np.sin(phi0), 0],
		[np.sin(phi0), np.cos(phi0), 0],
		[0, 0, 1]])
	R_phi1 = np.array([[np.cos(phi1), - np.sin(phi1), 0],
		[np.sin(phi1), np.cos(phi1), 0],
		[0, 0, 1]])
	H0 = R_phi0.dot(R_d0_theta0)
	H1 = R_phi1.dot(R_d1_theta1)
	
	return H0, H1

'''src = "source_1.png"
target = "target_1.png"
src_points = np.array([[ 63, 187],
 [ 65, 210],
 [ 71, 232],
 [ 80, 254],
 [ 87, 275],
 [ 96, 296],
 [106, 314],
 [119, 329],
 [139, 333],
 [164, 330],
 [188, 318],
 [210, 302],
 [229, 283],
 [243, 259],
 [249, 232],
 [250, 203],
 [249, 174],
 [ 54, 158],
 [ 62, 148],
 [ 75, 146],
 [ 89, 148],
 [103, 154],
 [132, 153],
 [150, 145],
 [170, 142],
 [190, 143],
 [209, 151],
 [117, 172],
 [116, 188],
 [114, 205],
 [112, 223],
 [105, 235],
 [111, 240],
 [119, 243],
 [129, 239],
 [139, 236],
 [ 71, 175],
 [ 79, 166],
 [ 91, 167],
 [103, 177],
 [ 91, 181],
 [ 79, 181],
 [151, 175],
 [160, 164],
 [174, 163],
 [187, 171],
 [176, 177],
 [163, 177],
 [102, 274],
 [108, 267],
 [116, 263],
 [123, 266],
 [132, 262],
 [148, 266],
 [168, 271],
 [151, 281],
 [136, 286],
 [126, 287],
 [118, 286],
 [110, 282],
 [107, 274],
 [117, 271],
 [124, 273],
 [134, 271],
 [161, 272],
 [134, 272],
 [125, 273],
 [117, 272]]
)
target_points = np.array(
[[ 89, 163],
 [ 91, 193],
 [ 94, 223],
 [102, 250],
 [118, 274],
 [142, 293],
 [170, 306],
 [198, 315],
 [224, 319],
 [243, 314],
 [256, 299],
 [264, 280],
 [269, 259],
 [274, 239],
 [279, 219],
 [282, 200],
 [280, 180],
 [140, 138],
 [158, 128],
 [180, 125],
 [201, 128],
 [220, 136],
 [245, 141],
 [257, 138],
 [270, 139],
 [280, 142],
 [284, 151],
 [234, 158],
 [237, 173],
 [241, 188],
 [246, 203],
 [217, 220],
 [227, 224],
 [236, 227],
 [243, 225],
 [249, 222],
 [164, 158],
 [177, 153],
 [190, 153],
 [199, 162],
 [188, 164],
 [176, 163],
 [245, 168],
 [256, 162],
 [267, 163],
 [272, 172],
 [266, 174],
 [255, 172],
 [192, 254],
 [211, 246],
 [227, 243],
 [235, 247],
 [243, 245],
 [251, 251],
 [254, 261],
 [248, 267],
 [240, 270],
 [231, 270],
 [222, 269],
 [208, 263],
 [200, 254],
 [225, 252],
 [234, 254],
 [242, 254],
 [250, 259],
 [241, 255],
 [233, 255],
 [224, 253]])'''
src = "mona.png"
target = "mona_mirror.png"
src_points = np.array([[ 22,  92],
 [ 21, 103],
 [ 23, 115],
 [ 25, 127],
 [ 29, 138],
 [ 35, 148],
 [ 42, 155],
 [ 49, 163],
 [ 59, 165],
 [ 71, 165],
 [ 84, 161],
 [ 97, 154],
 [107, 144],
 [115, 131],
 [119, 117],
 [120, 102],
 [121,  87],
 [ 23,  81],
 [ 27,  77],
 [ 34,  76],
 [ 41,  77],
 [ 48,  81],
 [ 61,  79],
 [ 70,  74],
 [ 81,  74],
 [ 92,  76],
 [103,  80],
 [ 54,  90],
 [ 53, 100],
 [ 52, 110],
 [ 51, 120],
 [ 46, 123],
 [ 49, 126],
 [ 54, 128],
 [ 59, 126],
 [ 64, 124],
 [ 30,  89],
 [ 35,  87],
 [ 41,  87],
 [ 46,  91],
 [ 40,  92],
 [ 34,  92],
 [ 70,  91],
 [ 76,  87],
 [ 83,  86],
 [ 89,  88],
 [ 84,  92],
 [ 77,  92],
 [ 42, 136],
 [ 46, 135],
 [ 50, 134],
 [ 54, 136],
 [ 58, 135],
 [ 65, 136],
 [ 74, 136],
 [ 65, 142],
 [ 59, 144],
 [ 54, 144],
 [ 51, 143],
 [ 47, 140],
 [ 44, 137],
 [ 50, 138],
 [ 54, 139],
 [ 58, 138],
 [ 71, 137],
 [ 58, 138],
 [ 54, 138],
 [ 50, 138]])
target_points = np.array(
[[ 36,  85],
 [ 37, 101],
 [ 39, 116],
 [ 42, 132],
 [ 50, 145],
 [ 61, 155],
 [ 75, 162],
 [ 89, 165],
 [101, 165],
 [110, 162],
 [118, 154],
 [125, 146],
 [130, 136],
 [134, 125],
 [136, 114],
 [137, 102],
 [137,  90],
 [ 56,  78],
 [ 66,  74],
 [ 78,  73],
 [ 89,  74],
 [ 98,  80],
 [111,  80],
 [118,  76],
 [125,  75],
 [132,  76],
 [136,  79],
 [105,  90],
 [106, 100],
 [107, 110],
 [108, 120],
 [ 95, 124],
 [100, 126],
 [105, 128],
 [109, 126],
 [113, 123],
 [ 69,  88],
 [ 76,  86],
 [ 83,  87],
 [ 89,  91],
 [ 82,  91],
 [ 75,  91],
 [112,  91],
 [117,  87],
 [123,  87],
 [128,  88],
 [123,  91],
 [118,  91],
 [ 85, 136],
 [ 94, 135],
 [101, 134],
 [105, 136],
 [109, 134],
 [113, 135],
 [117, 136],
 [113, 140],
 [109, 143],
 [105, 144],
 [100, 143],
 [ 93, 141],
 [ 88, 137],
 [101, 138],
 [105, 138],
 [109, 138],
 [115, 136],
 [110, 138],
 [105, 139],
 [101, 138]])
point_num = len(src_points)
assert len(src_points) == len(target_points)

src = cv2.imread(src)
src = src / 255.
target = cv2.imread(target)
target = target / 255.

n, m = src.shape[0], src.shape[1]
assert src.shape == target.shape

F, mask = cv2.findFundamentalMat(src_points, target_points)
H0, H1 = prewarp(F)
#print(H0)
#print(H1)

new_size = int(np.sqrt(np.power(src.shape[0], 2) + np.power(target.shape[1], 2))) * 4
prewarp_1 = cv2.warpPerspective(src, H0, (new_size, new_size), borderMode = cv2.BORDER_REPLICATE)
prewarp_2 = cv2.warpPerspective(target, H1, (new_size, new_size), borderMode = cv2.BORDER_REPLICATE)

src_features = []
target_features = []
for i in range(point_num):
	point = np.array([src_points[i, 0], src_points[i, 1], 1])
	point = np.matmul(H0, point)
	src_features.append([point[1] / point[2], point[0] / point[2]])
	'''x = int(point[1] / point[2]); y = int(point[0] / point[2]);
	for dx in range(-1, 2):
		for dy in range(-1, 2):
			prewarp_1[x + dx, y + dy, 0] = 1; prewarp_1[x + dx, y + dy, 1] = 0; prewarp_1[x + dx, y + dy, 2] = 0;'''
	point = np.array([target_points[i, 0], target_points[i, 1], 1])
	point = np.matmul(H1, point)
	target_features.append([point[1] / point[2], point[0] / point[2]])
	'''x = int(point[1] / point[2]); y = int(point[0] / point[2]);
	for dx in range(-1, 2):
		for dy in range(-1, 2):
			prewarp_2[x + dx, y + dy, 0] = 1; prewarp_2[x + dx, y + dy, 1] = 0; prewarp_2[x + dx, y + dy, 2] = 0;'''

cv2.imwrite('prewarp1.png', (np.clip(prewarp_1, 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('prewarp2.png', (np.clip(prewarp_2, 0., 1.) * 255).astype(np.uint8))

morph_rates = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
morph_rates = [.5]
output = image_morph(prewarp_1, prewarp_2, src_features, target_features, morph_rates)

point0 = np.array([0, 0, 1])
point0 = np.matmul(H0, point0)
point1 = np.array([n - 1, 0, 1])
point1 = np.matmul(H0, point1)
point2 = np.array([0, m - 1, 1])
point2 = np.matmul(H0, point2)
point3 = np.array([n - 1, m - 1, 1])
point3 = np.matmul(H0, point3)
src_corners = np.array([[point0[1] / point0[2], point0[0] / point0[2]], [point1[1] / point1[2], point1[0] / point1[2]], [point2[1] / point2[2], point2[0] / point2[2]], [point3[1] / point3[2], point3[0] / point3[2]]])
point0 = np.array([0, 0, 1])
point0 = np.matmul(H1, point0)
point1 = np.array([n - 1, 0, 1])
point1 = np.matmul(H1, point1)
point2 = np.array([0, m - 1, 1])
point2 = np.matmul(H1, point2)
point3 = np.array([n - 1, m - 1, 1])
point3 = np.matmul(H1, point3)
target_corners = np.array([[point0[1] / point0[2], point0[0] / point0[2]], [point1[1] / point1[2], point1[0] / point1[2]], [point2[1] / point2[2], point2[0] / point2[2]], [point3[1] / point3[2], point3[0] / point3[2]]])
orig_corners = np.array([[0, 0], [0, n - 1], [m - 1, 0], [m - 1, n - 1]])

cnt = 0
for rate in morph_rates:
	corners = src_corners * (1 - rate) + target_corners * rate
	Hs, s = cv2.findHomography(orig_corners, corners)
	#Hs = np.linalg.inv(H0) * (1 - rate) + np.linalg.inv(H1) * rate
	postwarp = cv2.warpPerspective(output[cnt], Hs, (m, n), borderMode = cv2.BORDER_REPLICATE)
	cnt += 1
	cv2.imwrite('morphed%d.png' % cnt, (np.clip(postwarp, 0., 1.) * 255).astype(np.uint8))
