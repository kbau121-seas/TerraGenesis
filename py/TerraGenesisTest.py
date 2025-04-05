from TerraGenesis import Simulator
import cv2
import time

if __name__ == "__main__":
	ITERATIONS=1000
	sim = Simulator()

	cv2.imwrite("sample_uplift.png", cv2.resize(sim.upliftMap * 255, (720, 720), interpolation=cv2.INTER_NEAREST))

	for i in range(ITERATIONS):
		start = time.time()
		sim.run(1)
		end = time.time()

		print(f'ITERATION: {i + 1}')
		print(f'TIME: {end - start}')

		cv2.imwrite("sample_output.png", cv2.resize(sim.heightMap * 255, (720, 720), interpolation=cv2.INTER_NEAREST))