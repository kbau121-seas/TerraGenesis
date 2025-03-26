import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

height = cv2.imread('sample_output.png', cv2.IMREAD_GRAYSCALE) / 255
xx, yy = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dz = height.ravel()
cmap = plt.get_cmap('plasma')
norm = Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))

sc = cm.ScalarMappable(cmap=cmap,norm=norm)
sc.set_array([])

ax.set_zlim(0, 2)
ax.bar3d(xx.ravel(), yy.ravel(), np.zeros(height.shape).ravel(), np.ones(height.shape).ravel(), np.ones(height.shape).ravel(), dz, color=colors)

plt.show()