"""Tests for the perceptron."""

from perceptron import Perceptron, make_data


def test_activate_threshold() -> None:
    p = Perceptron(n_features=2)
    assert p.activate(0) == 1
    assert p.activate(-1) == 0


def test_weights_length() -> None:
    p = Perceptron(n_features=3)
    assert len(p.w) == 3


def test_predict_with_known_weights() -> None:
    p = Perceptron(n_features=2)
    p.w = [1.0, -1.0]
    p.bias = 0.5
    assert p.predict([2.0, 3.0]) == 0  # z = 2 - 3 + 0.5 = -0.5
    assert p.predict([3.0, 1.0]) == 1  # z = 3 - 1 + 0.5 = 2.5


def test_hand_example() -> None:
    p = Perceptron(n_features=2, learning_rate=1.0, epochs=5)
    p.train([[1.0, 2.0], [1.0, 0.0]], [0, 1])
    assert p.w == [0.0, -2.0]
    assert p.bias == 0.0


def test_training_separates_classes() -> None:
    X, y = make_data(200, seed=42)
    p = Perceptron(n_features=2, learning_rate=0.1, epochs=50)
    p.train(X, y)
    assert p.score(X, y) >= 0.99


def test_bias_is_learned() -> None:
    # Граница между классами около x = 3 не проходит через 0, нужен bias.
    X = [[1.0], [2.0], [4.0], [5.0]]
    y = [0, 0, 1, 1]
    p = Perceptron(n_features=1, learning_rate=0.1, epochs=100)
    p.train(X, y)
    assert p.score(X, y) == 1.0