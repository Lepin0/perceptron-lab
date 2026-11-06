"""Plots for the perceptron: decision boundary and errors per epoch."""

import matplotlib.pyplot as plt


def plot_errors(errors_per_epoch: list[int], path: str) -> None:
    """Save a plot of the number of mistakes in every epoch."""
    plt.figure()
    epochs = range(1, len(errors_per_epoch) + 1)
    plt.plot(epochs, errors_per_epoch, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("number of errors")
    plt.title("Errors per epoch")
    plt.savefig(path)
    plt.close()


def plot_decision_boundary(
    X: list[list[float]], y: list[int], w: list[float], bias: float, path: str
) -> None:
    """Save a plot of 2D points of both classes and the line w.x + bias = 0."""
    plt.figure()
    for label in (0, 1):
        x1 = [point[0] for point, cls in zip(X, y) if cls == label]
        x2 = [point[1] for point, cls in zip(X, y) if cls == label]
        plt.scatter(x1, x2, label=f"class {label}")

    x1_min = min(point[0] for point in X) - 1
    x1_max = max(point[0] for point in X) + 1
    x2_min = min(point[1] for point in X) - 1
    x2_max = max(point[1] for point in X) + 1

    w1, w2 = w
    if w2 != 0:
        # w1*x1 + w2*x2 + bias = 0  =>  x2 = -(w1*x1 + bias) / w2
        line_x2 = [-(w1 * x1 + bias) / w2 for x1 in (x1_min, x1_max)]
        plt.plot([x1_min, x1_max], line_x2, color="black", label="boundary")
    elif w1 != 0:
        # the line is vertical: x1 = -bias / w1
        plt.axvline(-bias / w1, color="black", label="boundary")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig(path)
    plt.close()
