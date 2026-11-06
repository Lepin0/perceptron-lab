"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from perceptron import main


def test_main_synthetic(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--samples", "100", "--epochs", "20"])
    assert "Accuracy: 1.000" in capsys.readouterr().out


def test_main_iris(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--dataset", "iris", "--epochs", "10"])
    assert "Accuracy: 1.000" in capsys.readouterr().out


def test_main_rejects_negative_lr() -> None:
    with pytest.raises(SystemExit):
        main(["--lr", "-1"])


def test_main_rejects_zero_epochs() -> None:
    with pytest.raises(SystemExit):
        main(["--epochs", "0"])


def test_main_plot_creates_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    main(["--plot"])
    assert (tmp_path / "errors.png").exists()
    assert (tmp_path / "boundary.png").exists()


def test_main_plot_iris_skips_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    main(["--dataset", "iris", "--plot"])
    assert (tmp_path / "errors.png").exists()
    assert not (tmp_path / "boundary.png").exists()
