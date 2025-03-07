import numpy as np
import cv2
import math

import MapGen

class vec2:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, o):
		return vec2(self.x + o.x, self.y + o.y)

	def __sub__(self, o):
		return vec2(self.x - o.x, self.y - o.y)

	def len2(self):
		return self.x * self.x + self.y * self.y

	def len(self):
		return math.sqrt(self.len2())

	def __str__(self):
		return f'({self.x}, {self.y})'

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def __hash__(self):
		return hash(self.x + self.y)

def getHigherNeighbors(H, v):
	output = []

	height = H[v.y, v.x]
	neighbors = getAllNeighbors(H, v)

	for neighbor in neighbors:
		other = H[neighbor.y, neighbor.x]
		if height < other:
			output += [neighbor]

	return output

def getLowerNeighbors(H, v):
	output = []

	height = H[v.y, v.x]
	neighbors = getAllNeighbors(H, v)

	for neighbor in neighbors:
		other = H[neighbor.y, neighbor.x]
		if height > other:
			output += [neighbor]

	return output

def getAllNeighbors(H, v):
	output = []

	s = H.shape
	ys = range(max(0, v.y - 1), min(v.y + 2, s[0]))
	xs = range(max(0, v.x - 1), min(v.x + 2, s[1]))

	for y in ys:
		for x in xs:
			if (x == v.x and y == v.y): continue

			output += [vec2(x, y)]

	return output

def getSteepestSlope(H, v):
	neighbors = getHigherNeighbors(H, v)

	h = H[v.y, v.x]

	maxSlope = 0
	for neighbor in neighbors:
		maxSlope = max(maxSlope, (H[neighbor.y, neighbor.x] - h) / (neighbor - v).len())

	return maxSlope

def setWeights(M, H, w, v):
	neighbors = getLowerNeighbors(H, v)
	s_vk = dict()

	h = H[v.y, v.x]

	s_sum = 0
	for neighbor in neighbors:
		s_val = (h - H[neighbor.y, neighbor.x]) / (v - neighbor).len()
		s_val = pow(s_val, 4)
		s_vk[neighbor] = s_val
		s_sum += s_val

	if s_sum == 0: return

	for neighbor in neighbors:
		w[(neighbor, v)] = s_vk[neighbor] / s_sum

def setDrainage(M, H, a, w, v):
	if (a[v.y, v.x] != -1): return

	neighbors = getHigherNeighbors(H, v)

	for neighbor in neighbors:
		setWeights(M, H, w, neighbor)

	drainage = 1
	for neighbor in neighbors:
		setDrainage(M, H, a, w, neighbor)
		drainage += w[(v, neighbor)] * a[neighbor.y, neighbor.x]

	a[v.y, v.x] = drainage

def setDrainageMap(M, H, a, w):
	for y in range(M.shape[0]):
		for x in range(M.shape[1]):
			setDrainage(M, H, a, w, vec2(x, y))

DIM = 32

#uplift = MapGen.genMap(DIM, 5)
uplift = MapGen.genSimple(DIM, 10)
cv2.imwrite("sample_uplift.png", uplift * 255)
uplift += 1

heights = np.zeros((DIM, DIM))
drainage = np.full((DIM, DIM), -1)
weights = dict()

TIME_STEP = 0.1
ITERATIONS = 20

for i in range(ITERATIONS):
	print(f'ITERATION: {i + 1}')

	setDrainageMap(uplift, heights, drainage, weights)
	newHeights = np.zeros((DIM, DIM))
	for y in range(heights.shape[0]):
		for x in range(heights.shape[1]):
			v = vec2(x, y)
			newHeights[y, x] = heights[y, x] + TIME_STEP * (uplift[y, x] - pow(getSteepestSlope(heights, v), 8) * pow(drainage[y, x], 4))

	heights = newHeights
	heights[heights < 0] = 0

	print(np.min(heights), np.max(heights))
	print(np.min(drainage), np.max(drainage))

	drainage = np.full((DIM, DIM), -1)
	weights = dict()


h_min = np.min(heights)
h_max = np.max(heights)

heights -= h_min

if (h_max - h_min) != 0:
	heights /= (h_max - h_min)

cv2.imwrite("sample_output.png", heights * 255)