import numpy as np
from scipy import signal
import cv2
import math

import MapGen

import time

offsets = np.array([
	[-1, -1],
	[-1,  0],
	[-1,  1],
	[ 0, -1],
	[ 0,  1],
	[ 1, -1],
	[ 1,  0],
	[ 1,  1],
	])

offset_dist = np.sqrt(np.sum(offsets*offsets, axis=1))

def getSteepestSlopeMap(H):
	padded_H = np.pad(H, (1, 1), 'edge')
	slopes = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in offsets])

	higher_neighbors = H < slopes
	slopes = (H - slopes) / np.reshape(offset_dist, (8, 1, 1))
	slopes[higher_neighbors] = -1

	max_slopes = np.max(slopes, axis=0, initial=0)

	return max_slopes

def getWeightMap(H):
	# Create an MxNx3x3 matrix [y, x]->weight_matrix
	padded_H = np.pad(H, (1, 1), mode='constant', constant_values=(2))
	weights = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in offsets], axis=2)

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
	positions = np.stack([yy.ravel(), xx.ravel()], axis=-1)
	
	heights = H.ravel()
	sortedHeightInd = np.argsort(-heights)

	sortedPositions = positions[sortedHeightInd]

	drainageAreaMap = np.pad(np.ones_like(H, dtype=np.double), (1, 1))

	weights = getWeightMap(H)

	tmp = []
	for pos in sortedPositions:
		height = H[pos[0], pos[1]]
		drainageAreaMap[pos[0]:pos[0]+3, pos[1]:pos[1]+3] += weights[pos[0], pos[1]] * drainageAreaMap[pos[0] + 1, pos[1] + 1]

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
