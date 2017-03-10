"""
This reads and plots a binary file ('cart_demo.out') representing an n x n matrix of doubles.

"""

import numpy as np
import matplotlib.pyplot as plt

# Read binary file and reshape it into a square matrix
A = np.fromfile('cart_demo.out')
n = np.sqrt(A.size)
A = A.reshape((n, n))

# Plot the square matrix
plt.matshow(A)
plt.show()
