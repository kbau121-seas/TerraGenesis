import numpy as np

import MapGen

class Simulator:
	OFFSETS = np.array([
		[-1, -1],
		[-1,  0],
		[-1,  1],
		[ 0, -1],
		[ 0,  1],
		[ 1, -1],
		[ 1,  0],
		[ 1,  1],
		])
	OFFSET_DIST = np.sqrt(np.sum(OFFSETS*OFFSETS, axis=1))

	def __init__(self):
		self.dim = 128
		self.timeScale = 1

		self.upliftScale = 0.01
		self.erosionScale = 0.1

		self.steepestSlopeDegree = 2
		self.drainageDegree = 1

		self.upliftMap = MapGen.genSimple(self.dim, 10)
		self.heightMap = self.upliftMap

	# Gets the steepest slope downard slope at each position in the height map
	# The slope is only calculated when a neighbor is lower than the current position
	def getSteepestSlopeMap(self):
		# Creates an 8xMxN matrix where each MxN slice is a different neighboring height
		# for the MxN height map heights, according to the following:
		# [0, ...] [1, ...] [2, ...]
		# [3, ...]          [4, ...]
		# [5, ...] [6, ...] [7, ...]
		padded_H = np.pad(self.heightMap, (1, 1), 'edge')
		slopes = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in self.OFFSETS])

		# Calculates the slope to each valid neighbor slice
		higher_neighbors = self.heightMap < slopes
		slopes = (self.heightMap - slopes) / np.reshape(self.OFFSET_DIST, (8, 1, 1))
		slopes[higher_neighbors] = -1

		max_slopes = np.max(slopes, axis=0, initial=0)
		return max_slopes

	# Create an MxNx3x3 matrix position->weight_matrix
	def getWeightMap(self):
		# Creates an MxNx8 matrix where each MxN slice is a different neighboring height
		# for the MxN height map heights, according to the following:
		# [0, ...] [1, ...] [2, ...]
		# [3, ...]          [4, ...]
		# [5, ...] [6, ...] [7, ...]
		padded_H = np.pad(self.heightMap, (1, 1), mode='constant', constant_values=(2))
		weights = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in self.OFFSETS], axis=2)

		# Calculate the unnormalized weights
		higher_neighbors = self.heightMap[..., np.newaxis] < weights
		weights = np.pow((weights - self.heightMap[..., np.newaxis]) / np.reshape(self.OFFSET_DIST, (1, 1, 8)), 4)
		weights[higher_neighbors] = 0

		# Normalize the weights
		weight_sums = np.sum(weights, axis=2)
		weights = np.divide(weights, weight_sums[..., np.newaxis], where=weight_sums[..., np.newaxis] != 0)

		# Convert to MxNx3x3
		weights = np.insert(weights, 4, np.zeros_like(self.heightMap), axis=2)
		weights = np.reshape(weights, (weights.shape[0], weights.shape[1], 3, 3))

		return weights

	# Create an MxN matrix position->drainage_area
	def getDrainageAreaMap(self):
		# Sort all positions based on height (ascending -> descending)
		xx, yy = np.meshgrid(np.arange(self.dim), np.arange(self.dim))
		positions = np.stack([yy.ravel(), xx.ravel()], axis=-1)
		
		heights = self.heightMap.ravel()
		sortedHeightInd = np.argsort(-heights)
		sortedPositions = positions[sortedHeightInd]

		drainageAreaMap = np.pad(np.ones_like(self.heightMap, dtype=np.double), (1, 1))

		weights = self.getWeightMap()

		# Add the weighted drainage area to the neighboring areas
		for pos in sortedPositions:
			height = self.heightMap[pos[0], pos[1]]
			drainageAreaMap[pos[0]:pos[0]+3, pos[1]:pos[1]+3] += weights[pos[0], pos[1]] * drainageAreaMap[pos[0] + 1, pos[1] + 1]

		return drainageAreaMap[1:-1,1:-1]

	# Iterate the simulation and update the height map
	def run(self, iterations=1):
		for i in range(iterations):
			steepestSlopeMap = self.getSteepestSlopeMap()
			drainageAreaMap = self.getDrainageAreaMap()

			erosion = np.pow(steepestSlopeMap, self.steepestSlopeDegree) * np.pow(drainageAreaMap, self.drainageDegree)

			deltaHeight  = self.upliftMap * self.upliftScale
			deltaHeight -= erosion * self.erosionScale
			deltaHeight *= self.timeScale

			self.heightMap = np.clip(self.heightMap + deltaHeight, 0, 1)
