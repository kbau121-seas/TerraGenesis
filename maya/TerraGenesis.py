import numpy as np
from PIL import Image

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

	def __init__(self, upliftMap=None):
		if not isinstance(upliftMap, (np.ndarray)):
			upliftMap = np.zeros((128, 128))

		self.dim = upliftMap.shape
		self.timeScale = 1

		self.upliftScale = 0.01
		self.erosionScale = np.full(upliftMap.shape, 0.1)

		self.steepestSlopeDegree = np.full(upliftMap.shape, 2)
		self.drainageDegree = np.full(upliftMap.shape, 1)

		self.erosion = np.zeros_like(upliftMap)
		self.upliftMap = upliftMap
		self.heightMap = upliftMap

	def setSimulationResolution(self, resolution):
		if self.upliftMap.shape[0] == resolution:
			return

		newDim = (resolution, resolution)

		upliftImage = Image.fromarray((self.upliftMap * 255).astype(np.uint8))
		upliftImage = upliftImage.resize(newDim, Image.BILINEAR)
		self.upliftMap = np.array(upliftImage, dtype=np.float32) / 255.0

		erosionScaleImage = Image.fromarray((self.erosionScale * 255).astype(np.uint8))
		erosionScaleImage = erosionScaleImage.resize(newDim, Image.BILINEAR)
		self.erosionScale = np.array(erosionScaleImage, dtype=np.float32) / 255.0

		steepestSlopeImage = Image.fromarray((self.steepestSlopeDegree * 255).astype(np.uint8))
		steepestSlopeImage = steepestSlopeImage.resize(newDim, Image.BILINEAR)
		self.steepestSlopeDegree = np.array(steepestSlopeImage, dtype=np.float32) / 255.0

		drainageImage = Image.fromarray((self.drainageDegree * 255).astype(np.uint8))
		drainageImage = drainageImage.resize(newDim, Image.BILINEAR)
		self.drainageDegree = np.array(drainageImage, dtype=np.float32) / 255.0

		heightImage = Image.fromarray((self.heightMap * 255).astype(np.uint8))
		heightImage = heightImage.resize(newDim, Image.BILINEAR)
		self.heightMap = np.array(heightImage, dtype=np.float32) / 255.0

		erosionImage = Image.fromarray((self.erosion * 255).astype(np.uint8))
		erosionImage = erosionImage.resize(newDim, Image.BILINEAR)
		self.erosion = np.array(erosionImage, dtype=np.float32) / 255.0

		self.dim = newDim

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
		weights = np.power((weights - self.heightMap[..., np.newaxis]) / np.reshape(self.OFFSET_DIST, (1, 1, 8)), 4)
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
		xx, yy = np.meshgrid(np.arange(self.dim[0]), np.arange(self.dim[1]))
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

			erosion = np.power(steepestSlopeMap, self.steepestSlopeDegree) * np.power(drainageAreaMap, self.drainageDegree)
			self.erosion=erosion

			deltaHeight  = self.upliftMap * self.upliftScale
			deltaHeight -= erosion * self.erosionScale
			deltaHeight *= self.timeScale

			self.heightMap = np.clip(self.heightMap + deltaHeight, 0, 1)