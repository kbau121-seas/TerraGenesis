import numpy as np
import cv2
import math

import MapGen

import time

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
		self.iterations = 1000
		self.timeScale = 1

		self.upliftScale = 0.01
		self.erosionScale = 0.1

		self.steepestSlopeDegree = 2
		self.drainageDegree = 1

		self.uplift = MapGen.genSimple(self.dim, 10)
		self.heights = self.uplift

	def getSteepestSlopeMap(self, H):
		padded_H = np.pad(H, (1, 1), 'edge')
		slopes = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in self.OFFSETS])

		higher_neighbors = H < slopes
		slopes = (H - slopes) / np.reshape(self.OFFSET_DIST, (8, 1, 1))
		slopes[higher_neighbors] = -1

		max_slopes = np.max(slopes, axis=0, initial=0)

		return max_slopes

	def getWeightMap(self, H):
		# Create an MxNx3x3 matrix [y, x]->weight_matrix
		padded_H = np.pad(H, (1, 1), mode='constant', constant_values=(2))
		weights = np.stack([padded_H[1 + offset[0]:offset[0] - 1 or None, 1 + offset[1]:offset[1] - 1 or None] for offset in self.OFFSETS], axis=2)

		higher_neighbors = H[..., np.newaxis] < weights
		weights = np.pow((weights - H[..., np.newaxis]) / np.reshape(self.OFFSET_DIST, (1, 1, 8)), 4)
		weights[higher_neighbors] = 0

		weight_sums = np.sum(weights, axis=2)
		weights = np.divide(weights, weight_sums[..., np.newaxis], where=weight_sums[..., np.newaxis] != 0)

		weights = np.insert(weights, 4, np.zeros_like(H), axis=2)
		weights = np.reshape(weights, (weights.shape[0], weights.shape[1], 3, 3))

		return weights

	def getDrainageAreaMap(self, H):
		xx, yy = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]))
		positions = np.stack([yy.ravel(), xx.ravel()], axis=-1)
		
		heights = H.ravel()
		sortedHeightInd = np.argsort(-heights)
		sortedPositions = positions[sortedHeightInd]

		drainageAreaMap = np.pad(np.ones_like(H, dtype=np.double), (1, 1))

		weights = self.getWeightMap(H)

		for pos in sortedPositions:
			height = H[pos[0], pos[1]]
			drainageAreaMap[pos[0]:pos[0]+3, pos[1]:pos[1]+3] += weights[pos[0], pos[1]] * drainageAreaMap[pos[0] + 1, pos[1] + 1]

		return drainageAreaMap[1:-1,1:-1]

	def run(self):
		cv2.imwrite("sample_uplift.png", self.uplift * 255)

		self.heights = self.uplift
		for i in range(self.iterations):
			print(f'ITERATION: {i + 1}')

			start = time.time()

			steepestSlopeMap = self.getSteepestSlopeMap(self.heights)
			drainageAreaMap = self.getDrainageAreaMap(self.heights)

			erosion = np.pow(steepestSlopeMap, self.steepestSlopeDegree) * np.pow(drainageAreaMap, self.drainageDegree)

			deltaHeight  = self.uplift * self.upliftScale
			deltaHeight -= erosion * self.erosionScale
			deltaHeight *= self.timeScale

			self.heights = np.clip(self.heights + deltaHeight, 0, 1)

			cv2.imwrite("sample_output.png", cv2.resize(self.heights * 255, (720, 720), interpolation=cv2.INTER_NEAREST))

			end = time.time()
			print(f'TIME: {end - start}')

if __name__ == "__main__":
	Simulator().run()
