import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import sys

try:
	src_path = sys.argv[1]
	mask_path = sys.argv[2]
	target_path = sys.argv[3]
	pos_x = int(sys.argv[4])
	pos_y = int(sys.argv[5])
except:
	print("Usage: fusion <src> <mask> <target> <position_x> <position_y>")
	exit(0)

keep_texture = True

src = cv2.imread(src_path)
src = src / 255.
mask = cv2.imread(mask_path)
mask = mask / 255.
target = cv2.imread(target_path)
target = target / 255.

assert src.shape == mask.shape
n, m = src.shape[0], src.shape[1]

id = {}
edge = {}
points = []
cnt = 0
edge_cnt = 0
for i in range(n):
	for j in range(m):
		if mask[i, j, 0] > .5:
			id[(i, j)] = cnt
			points.append((i, j))
			cnt += 1
			if i == 0 or j == 0 or i == n - 1 or j == m - 1:
				edge[(i, j)] = 1
				edge_cnt += 1
			elif mask[i - 1, j, 0] < .5 or mask[i, j - 1, 0] < .5 or mask[i + 1, j, 0] < .5 or mask[i, j + 1, 0] < .5:
				edge[(i, j)] = 1
				edge_cnt += 1
			else:
				edge[(i, j)] = 0

A = sparse.lil_matrix((cnt, cnt), dtype = float)
b = [np.zeros(shape = (cnt)), np.zeros(shape = (cnt)), np.zeros(shape = (cnt))]
X = [np.zeros(shape = (cnt)), np.zeros(shape = (cnt)), np.zeros(shape = (cnt))]
for (x, y) in points:
	k = id[(x, y)]
	if edge[(x, y)] == 1:
		A[k, k] = 1
		for channel in range(3):
			b[channel][k] = target[pos_x + x, pos_y + y, channel]
	else:
		A[k, k] = 4
		A[k, id[(x - 1, y)]] = -1
		A[k, id[(x, y - 1)]] = -1
		A[k, id[(x + 1, y)]] = -1
		A[k, id[(x, y + 1)]] = -1
		grad_src = 0
		grad_target = 0
		for channel in range(3):
			grad_src += (src[x + 1, y, channel] - src[x, y, channel]) ** 2
			grad_src += (src[x, y + 1, channel] - src[x, y, channel]) ** 2
			grad_target += abs(target[pos_x + x + 1, pos_y + y, channel] - target[pos_x + x, pos_y + y, channel])
			grad_target += abs(target[pos_x + x, pos_y + y + 1, channel] - target[pos_x + x, pos_y + y, channel])
		if not keep_texture:
			grad_target = 0
		delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
		for (dx, dy) in delta:
			dsrc = src[x, y] - src[x + dx, y + dy]
			grad_src = np.dot(dsrc, dsrc)
			dtar = target[pos_x + x, pos_y + y] - target[pos_x + x + dx, pos_y + y + dy]
			grad_target = np.dot(dtar, dtar)
			if not keep_texture or grad_src >= grad_target:
				for channel in range(3):
					b[channel][k] += dsrc[channel]
			else:
				for channel in range(3):
					b[channel][k] += dtar[channel]
				
A = A.tocsc()
for channel in range(3):
	X[channel] = spsolve(A, b[channel])
#print(cnt, edge_cnt)
for (x, y) in points:
	k = id[(x, y)]
	for channel in range(3):
		target[pos_x + x, pos_y + y, channel] = X[channel][k]
target = (np.clip(target, 0., 1.) * 255).astype(np.uint8)
cv2.imwrite('output.png', target)
