"""Draw the points produced by make_data."""

import matplotlib.pyplot as plt

from perceptron import make_data

X, y = make_data(200, seed=0)

class0_x1 = [point[0] for point, label in zip(X, y) if label == 0]
class0_x2 = [point[1] for point, label in zip(X, y) if label == 0]
class1_x1 = [point[0] for point, label in zip(X, y) if label == 1]
class1_x2 = [point[1] for point, label in zip(X, y) if label == 1]

plt.scatter(class0_x1, class0_x2, label="class 0")
plt.scatter(class1_x1, class1_x2, label="class 1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("points.png")
plt.show()
