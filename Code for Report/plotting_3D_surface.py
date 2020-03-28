import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image

# Open Image into array
im = Image.open("/home/tom/Desktop/thesis/Report/images/chessboard.jpg").convert('L')  # Converts to grayscale.
im = np.array(im)
# plt.imshow(im)

x = np.arange(im.shape[0])
y = np.arange(im.shape[1])
X, Y = np.meshgrid(y, x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, im)

plt.show()
