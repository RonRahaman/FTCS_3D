import numpy as np
import matplotlib.pyplot as plt

# Read binary file and reshape it into a square matrix
A = np.fromfile('build/Debug/cart_demo.out')
n = np.sqrt(A.size)
A = A.reshape((n, n))

# Plot the square matrix
plt.matshow(A)
plt.show()
