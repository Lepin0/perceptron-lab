"""Tests for the data generators and loaders."""

from datasets import load_iris_binary, make_data


def test_make_data_sizes_and_labels() -> None:
    X, y = make_data(10, seed=1)
    assert len(X) == 10
    assert len(y) == 10
    assert all(len(point) == 2 for point in X)
    assert set(y) == {0, 1}


def test_make_data_is_reproducible() -> None:
    assert make_data(20, seed=5) == make_data(20, seed=5)


def test_make_data_seed_changes_points() -> None:
    assert make_data(20, seed=5) != make_data(20, seed=6)


def test_load_iris_binary() -> None:
    X, y = load_iris_binary()
    assert len(X) == 100
    assert all(len(flower) == 4 for flower in X)
    assert set(y) == {0, 1}
    assert X[0] == [5.1, 3.5, 1.4, 0.2]
