import numpy as np
from scipy import signal
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

def getWeightMap(H):
	# Create an 8xMxN matrix [offset, y, x]->weight
	padded_H = np.pad(H, (1, 1), mode='constant', constant_values=(2))
	weights = np.stack([padded_H[1 + offset.y:offset.y - 1 or None, 1 + offset.x:offset.x - 1 or None] for offset in offsets], axis=2)

	higher_neighbors = H[..., np.newaxis] < weights
	weights = np.pow((weights - H[..., np.newaxis]) / np.reshape(offset_dist, (1, 1, 8)), 4)
	weights[higher_neighbors] = 0

	weight_sums = np.sum(weights, axis=2)
	weights = np.divide(weights, weight_sums[..., np.newaxis], where=weight_sums[..., np.newaxis] != 0)

	weights = np.insert(weights, 4, np.zeros_like(H), axis=2)
	weights = np.reshape(weights, (weights.shape[0], weights.shape[1], 3, 3))

	return weights

def getDrainageAreaMap(H):
	xx, yy = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]))
	positions = np.stack([xx.ravel(), yy.ravel()], axis=-1)
	
	heights = H.ravel()
	sortedHeightInd = np.argsort(-heights)

	sortedPositions = positions[sortedHeightInd]

	drainageAreaMap = np.pad(np.ones_like(H, dtype=np.double), (1, 1))

	weights = getWeightMap(H)

	tmp = []
	for position in sortedPositions:
		pos = vec2(position[0], position[1])
		height = H[pos.y, pos.x]

		drainageAreaMap[pos.y:pos.y+3, pos.x:pos.x+3] += weights[pos.y, pos.x] * drainageAreaMap[pos.y + 1, pos.x + 1]

	return drainageAreaMap[1:-1,1:-1]

DIM = 32 * 4

uplift = MapGen.genSimple(DIM, 10)
cv2.imwrite("sample_uplift.png", uplift * 255)

heights = np.zeros((DIM, DIM))
drainage = np.full((DIM, DIM), -1)
weights = dict()

ITERATIONS = 1000

heights = uplift
for i in range(ITERATIONS):
	print(f'ITERATION: {i + 1}')

	start = time.time()

	steepestSlopeMap = getSteepestSlopeMap(heights)
	drainageAreaMap = getDrainageAreaMap(heights)
	nextHeights = np.copy(heights)

	erosion = np.pow(steepestSlopeMap, 2) * np.pow(drainageAreaMap, 1)

	nextHeights += uplift * 0.01
	nextHeights -= erosion * 0.1
	nextHeights = np.clip(nextHeights, 0, 1)

	heights = nextHeights

	cv2.imwrite("sample_output.png", cv2.resize(heights * 255, (720, 720), interpolation=cv2.INTER_NEAREST))

	end = time.time()
	print(f'TIME: {end - start}')
