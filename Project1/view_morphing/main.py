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

def normalize(p):
	x = p[:, 0]
	y = p[:, 1]
	x = x.reshape((-1, 1))
	y = y.reshape((-1, 1))
	num = len(x)
	mean_x, mean_y = np.mean(x), np.mean(y)
	shifted_x, shifted_y = x - mean_x, y - mean_y
	scale = np.sqrt(2) / np.mean(np.sqrt(shifted_x ** 2 + shifted_y ** 2))
	t = np.array([[scale, 0, - scale * mean_x], [0, scale, - scale * mean_y], [0, 0, 1]])
	
	ones = np.ones((num, 1))
	p = np.concatenate((x, y, ones), axis = 1)
	p = np.dot(t, p.T)
	return p.T, t

def fundamental_matrix(image1, image2, p1, p2):
	p1, t1 = normalize(p1)
	p2, t2 = normalize(p2)
	x1, y1 = p2[:, 0], p2[:, 1]
	x2, y2 = p1[:, 0], p1[:, 1]
	num = len(x1)
	o = np.ones((num, 1))
	a = np.concatenate((x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2,
		o[:, 0])).reshape((num, -1), order = 'F')
	
	U, D, V = np.linalg.svd(a)
	smallest = V[8, :].T
	F = smallest.reshape(3, 3)
	
	U, D, V = np.linalg.svd(F)
	r = D[0]
	s = D[1]
	
	F = np.dot(U, np.diag([r, s, 0])).dot(V)
	F = t2.T.dot(F).dot(t1)
	return F

def prewarp(F):
	eigvalue0, eigvector0 = np.linalg.eig(F)
	eigvalue1, eigvector1 = np.linalg.eig(np.transpose(F))
	e0 = eigvector0[:, np.argmin(eigvalue0)]
	e1 = eigvector1[:, np.argmin(eigvalue1)]
	d0 = np.array([-e0[1], e0[0], 0])

	Fd0 = F.dot(d0)
	d1 = np.array([-Fd0[1], Fd0[0], 0])
	theta0 = np.arctan(e0[2]/(d0[1]*e0[0] - d0[0]*e0[1]))
	theta1 = np.arctan(e1[2]/(d1[1]*e1[0] - d1[0]*e1[1]))
	R_d0_theta0 = rotation_matrix(d0, theta0)
	R_d1_theta1 = rotation_matrix(d1, theta1)

	new_e0 = R_d0_theta0.dot(e0)
	new_e1 = R_d1_theta1.dot(e1)
	phi0 = -np.arctan(new_e0[1]/new_e0[0])
	phi1 = -np.arctan(new_e1[1]/new_e1[0])
	R_phi0 = np.array([[np.cos(phi0), -np.sin(phi0), 0],
		[np.sin(phi0), np.cos(phi0), 0],
		[0, 0, 1]])
	R_phi1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
		[np.sin(phi1), np.cos(phi1), 0],
		[0, 0, 1]])
	H0 = R_phi0.dot(R_d0_theta0)
	H1 = R_phi1.dot(R_d1_theta1)
	
	return H0, H1

src = "source_1.png"
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
 [224, 253]])
point_num = len(src_points)
assert len(src_points) == len(target_points)

src = cv2.imread(src)
src = src / 255.
target = cv2.imread(target)
target = target / 255.

n, m = src.shape[0], src.shape[1]
assert src.shape == target.shape

F = fundamental_matrix(src, target, src_points, target_points)
F, mask = cv2.findFundamentalMat(src_points, target_points)
H0, H1 = prewarp(F)

new_size = int(np.sqrt(np.power(src.shape[0], 2) + np.power(target.shape[1], 2)))
prewarp_1 = cv2.warpPerspective(src, H0, (m, n), borderMode = cv2.BORDER_REPLICATE)
prewarp_2 = cv2.warpPerspective(target, H1, (m, n), borderMode = cv2.BORDER_REPLICATE)
'''for i in range(n):
	for j in range(m):
		if (prewarp_1[i, j, 0] == 0) and (prewarp_1[i, j, 1] <= 5e-3) and (prewarp_1[i, j, 2] <= 5e-3):
			prewarp_1[i, j, :] = prewarp_1[0, 0, :]
		if (prewarp_2[i, j, 0] <= 5e-3) and (prewarp_2[i, j, 1] <= 5e-3) and (prewarp_2[i, j, 2] <= 5e-3):
			prewarp_2[i, j, :] = prewarp_2[0, 0, :]'''
cv2.imwrite('prewarp1.png', (np.clip(prewarp_1, 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('prewarp2.png', (np.clip(prewarp_2, 0., 1.) * 255).astype(np.uint8))

src_features = []
target_features = []
for i in range(point_num):
	point = np.array([src_points[i, 0], src_points[i, 1], 1])
	point = np.matmul(H0, point)
	src_features.append([point[1] / point[2], point[0] / point[2]])
	point = np.array([target_points[i, 0], target_points[i, 1], 1])
	point = np.matmul(H1, point)
	target_features.append([point[1] / point[2], point[0] / point[2]])
output = image_morph(prewarp_1, prewarp_2, src_features, target_features)
#cv2.imwrite('prewarp1.png', (np.clip(prewarp_1, 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('middle0.png', (np.clip(output[0], 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('middle1.png', (np.clip(output[1], 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('middle2.png', (np.clip(output[2], 0., 1.) * 255).astype(np.uint8))
cv2.imwrite('middle3.png', (np.clip(output[3], 0., 1.) * 255).astype(np.uint8))
