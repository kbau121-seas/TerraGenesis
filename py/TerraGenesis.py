import numpy as np
import cv2
import math

import MapGen

import time

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

def getSteepestSlope_OLD(H, v):
	neighbors = getLowerNeighbors(H, v)

	h = H[v.y, v.x]

	maxSlope = 0
	for neighbor in neighbors:
		maxSlope = max(maxSlope, (h - H[neighbor.y, neighbor.x]) / (neighbor - v).len())

	return maxSlope

offsets = [
	vec2(-1, -1),
	vec2( 0, -1),
	vec2( 1, -1),
	vec2(-1,  0),
	vec2( 1,  0),
	vec2(-1,  1),
	vec2( 0,  1),
	vec2( 1,  1),
]

offset_dist = [offset.len() for offset in offsets]

def getSteepestSlopeMap(H):
	padded_H = np.pad(H, (1, 1), 'edge')
	slopes = np.stack([padded_H[1 + offset.y:offset.y - 1 or None, 1 + offset.x:offset.x - 1 or None] for offset in offsets])

	higher_neighbors = H < slopes
	slopes = (H - slopes) / np.reshape(offset_dist, (8, 1, 1))
	slopes[higher_neighbors] = -1

	max_slopes = np.max(slopes, axis=0, initial=0)

	return max_slopes

def getSteepestSlopeMap_OLD(H):
	steepestSlopeMap = np.zeros_like(H, dtype=np.double)

	for y in range(steepestSlopeMap.shape[0]):
		for x in range(steepestSlopeMap.shape[1]):
			steepestSlopeMap[y, x] = getSteepestSlope_OLD(H, vec2(x, y))

	return steepestSlopeMap

def getDrainageAreaMap(H):
	xx, yy = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]))
	positions = np.stack([xx.ravel(), yy.ravel()], axis=-1)
	
	heights = H.ravel()
	sortedHeightInd = np.argsort(-heights)

	sortedPositions = positions[sortedHeightInd]

	drainageAreaMap = np.ones_like(H, dtype=np.double)

	tmp = []
	for position in sortedPositions:
		pos = vec2(position[0], position[1])
		height = H[pos.y, pos.x]

		sVals = dict()
		sValSum = 0

		neighbors = getLowerNeighbors(H, pos)
		for neighbor in neighbors:
			nHeight = H[neighbor.y, neighbor.x]

			sVal = (height - nHeight) / (pos - neighbor).len()
			sVal = pow(sVal, 4)

			sVals[neighbor] = sVal
			sValSum += sVal

		if sValSum == 0: continue

		for neighbor in neighbors:
			drainageAreaMap[neighbor.y, neighbor.x] += sVals[neighbor] * drainageAreaMap[pos.y, pos.x] / sValSum

	return drainageAreaMap

DIM = 32 * 4

#uplift = MapGen.genMap(DIM, 5 * 2)
uplift = MapGen.genSimple(DIM, 10)
#uplift = cv2.imread('custom_uplift.png', cv2.IMREAD_GRAYSCALE) / 255
cv2.imwrite("sample_uplift.png", uplift * 255)

heights = np.zeros((DIM, DIM))
drainage = np.full((DIM, DIM), -1)
weights = dict()

ITERATIONS = 1000

#heights = np.zeros((DIM, DIM))
heights = uplift
for i in range(ITERATIONS):
	print(f'ITERATION: {i + 1}', f'({np.min(heights)}, {np.max(heights)})')

	steepestSlopeMap = getSteepestSlopeMap(heights)
	drainageAreaMap = getDrainageAreaMap(heights)
	nextHeights = np.copy(heights)

	erosion = np.pow(steepestSlopeMap, 2) * np.pow(drainageAreaMap, 1)

	nextHeights += uplift * 0.01
	nextHeights -= erosion * 0.1
	nextHeights = np.clip(nextHeights, 0, 1)

	heights = np.copy(nextHeights)

	print(f'EROSION: ({np.min(erosion) * 0.1}, {np.max(erosion) * 0.1})')
	maxErosionPos = vec2(np.argmax(erosion) // erosion.shape[0], np.argmax(erosion) % erosion.shape[0])

	#col = np.stack((heights, heights, heights), axis=2)
	#col[maxErosionPos.y, maxErosionPos.x] = (255, 0, 255)
	#cv2.imwrite("sample_output.png", col * 255)

	#cv2.imwrite("sample_output.png", np.hstack((heights, erosion, steepestSlopeMap, drainageAreaMap / np.max(drainageAreaMap))) * 255)

	cv2.imwrite("sample_output.png", heights * 255)

h_min = np.min(heights)
h_max = np.max(heights)

heights -= h_min

if (h_max - h_min) != 0:
	heights /= (h_max - h_min)

#cv2.imwrite("sample_output.png", heights * 255)