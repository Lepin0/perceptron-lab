"""Datasets for the perceptron: random 2D points and the iris flowers."""

import csv
import random
from pathlib import Path

IRIS_PATH = Path(__file__).with_name("iris.csv")


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


def load_iris_binary(
    path: Path = IRIS_PATH,
) -> tuple[list[list[float]], list[int]]:
    """Load iris and keep species 0 and 1: they can be split by a straight line."""
    X: list[list[float]] = []
    y: list[int] = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip the header row
        for row in reader:
            label = int(row[4])
            if label < 2:
                X.append([float(value) for value in row[:4]])
                y.append(label)
    return X, y
