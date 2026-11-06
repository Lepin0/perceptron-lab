"""Perceptron: a single-layer neural network for binary classification.

Example:
    python perceptron.py --samples 200 --lr 0.1 --epochs 50
"""

import argparse

from datasets import load_iris_binary, make_data
from plots import plot_decision_boundary, plot_errors


class Perceptron:
    """Single-layer perceptron with a step activation function."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Create a perceptron with zero weights and zero bias."""
        self.w: list[float] = [0.0] * n_features
        self.bias: float = 0.0
        self.lr: float = learning_rate
        self.epochs: int = epochs
        self.errors_per_epoch: list[int] = []

    def activate(self, z: float) -> int:
        """Return 1 if z >= 0, otherwise 0."""
        return 1 if z >= 0 else 0

    def predict(self, x: list[float]) -> int:
        """Return the predicted class (0 or 1) for one point."""
        z = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.bias
        return self.activate(z)

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Fit weights: after each mistake shift them towards the right answer.

        The number of mistakes made in every epoch is saved in errors_per_epoch.
        """
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for x, target in zip(X, y):
                error = target - self.predict(x)
                if error != 0:
                    errors += 1
                for i in range(len(self.w)):
                    self.w[i] += self.lr * error * x[i]
                self.bias += self.lr * error
            self.errors_per_epoch.append(errors)

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Return the share of correctly classified points."""
        correct = sum(1 for x, t in zip(X, y) if self.predict(x) == t)
        return correct / len(y)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on random 2D points or on the iris flowers."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "iris"],
        default="synthetic",
        help="data to learn from (default: synthetic)",
    )
    parser.add_argument("--samples", type=int, default=200, help="number of points")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--plot", action="store_true", help="save errors.png and boundary.png"
    )
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def load_dataset(
    name: str, samples: int, seed: int
) -> tuple[list[list[float]], list[int]]:
    """Return points and their classes for the chosen dataset."""
    if name == "iris":
        return load_iris_binary()
    return make_data(samples, seed)


def save_plots(model: Perceptron, X: list[list[float]], y: list[int]) -> None:
    """Save the error plot, and the decision boundary for 2D data."""
    plot_errors(model.errors_per_epoch, "errors.png")
    print("Saved errors.png")
    if len(model.w) == 2:
        plot_decision_boundary(X, y, model.w, model.bias, "boundary.png")
        print("Saved boundary.png")
    else:
        print("Decision boundary is drawn only for 2D data, skipped.")


def main(argv: list[str] | None = None) -> None:
    """Load data, train the perceptron, print the accuracy, optionally plot."""
    args = parse_args(argv)
    X, y = load_dataset(args.dataset, args.samples, args.seed)
    model = Perceptron(n_features=len(X[0]), learning_rate=args.lr, epochs=args.epochs)
    model.train(X, y)
    print(f"Accuracy: {model.score(X, y):.3f}")
    if args.plot:
        save_plots(model, X, y)


if __name__ == "__main__":
    main()
