"""Perceptron: a single-layer neural network for binary classification."""

import argparse
import random


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

    def activate(self, z: float) -> int:
        """Return 1 if z >= 0, otherwise 0."""
        return 1 if z >= 0 else 0

    def predict(self, x: list[float]) -> int:
        """Return the predicted class (0 or 1) for one point."""
        z = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.bias
        return self.activate(z)

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Fit weights: after each mistake shift them towards the right answer."""
        for _ in range(self.epochs):
            for x, target in zip(X, y):
                error = target - self.predict(x)
                for i in range(len(self.w)):
                    self.w[i] += self.lr * error * x[i]
                self.bias += self.lr * error

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Return the share of correctly classified points."""
        correct = sum(1 for x, t in zip(X, y) if self.predict(x) == t)
        return correct / len(y)


def make_data(n_samples: int, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Generate two 2D classes around the centres (-2, -2) and (2, 2)."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(n_samples):
        label = i % 2
        centre = 2.0 if label == 1 else -2.0
        X.append([rng.gauss(centre, 0.5), rng.gauss(centre, 0.5)])
        y.append(label)
    return X, y


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on synthetic 2D data."
    )
    parser.add_argument("--samples", type=int, default=200, help="number of points")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Generate data, train the perceptron and print the accuracy."""
    args = parse_args(argv)
    X, y = make_data(args.samples, args.seed)
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    model.train(X, y)
    print(f"Accuracy: {model.score(X, y):.3f}")


if __name__ == "__main__":
    main()
