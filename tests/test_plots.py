"""Tests for the plotting functions: a PNG file must appear."""

from pathlib import Path

from plots import plot_decision_boundary, plot_errors

POINTS = [[-2.0, -2.0], [-1.5, -2.5], [2.0, 2.0], [2.5, 1.5]]
CLASSES = [0, 0, 1, 1]


def test_plot_errors_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "errors.png"
    plot_errors([3, 1, 0], str(path))
    assert path.exists()


def test_plot_boundary_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "boundary.png"
    plot_decision_boundary(POINTS, CLASSES, [0.5, 0.5], 0.0, str(path))
    assert path.exists()


def test_plot_boundary_vertical_line(tmp_path: Path) -> None:
    path = tmp_path / "vertical.png"
    plot_decision_boundary(POINTS, CLASSES, [1.0, 0.0], 0.0, str(path))
    assert path.exists()


def test_plot_boundary_zero_weights(tmp_path: Path) -> None:
    path = tmp_path / "zero.png"
    plot_decision_boundary(POINTS, CLASSES, [0.0, 0.0], 0.0, str(path))
    assert path.exists()
