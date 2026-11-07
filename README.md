# Perceptron

Лабораторная работа 1, тема А1. Однослойный перцептрон. 

Перцептрон принимает признаки, умножает их на веса, суммирует, добавляет сдвиг (bias) и пропускает через пороговую функцию, выдавая класс 0 или 1. Обучается по правилу: если ошибся, веса подправляются: `w += lr · (правильный − предсказанный) · x`.

После запуска программа обучает перцептрон и печатает точность (`Accuracy`). С флагом `--plot` дополнительно сохраняет графики: границу решения и число ошибок по эпохам.

## Структура проекта

* `perceptron.py` — класс `Perceptron` (обучение, предсказание, пороговая функция) и CLI.
* `datasets.py` — случайные 2D-точки двух классов и загрузка датасета iris.
* `plots.py` — графики через `matplotlib`.
* `iris.csv` — локальная копия датасета iris.
* `tests/` — тесты (`pytest`).
* `.github/workflows/ci.yml` — CI: `ruff`, `mypy` и `pytest` при каждом push.
* `pyproject.toml`, `requirements.txt` — зависимости и настройки инструментов.

## Как запустить

Нужен [uv](https://docs.astral.sh/uv/). Клонируем репозиторий и запускаем одной командой:

```
git clone https://github.com/Lepin0/perceptron-lab.git
cd perceptron-lab
uv run python perceptron.py --samples 200 --lr 0.1 --epochs 50
```

Результат: `Accuracy: 1.000`.

Для реальных данных укажи `--dataset iris` (параметр `--samples` тогда не нужен), для графиков добавь `--plot`:

```
uv run python perceptron.py --dataset iris --plot
```

Значения по умолчанию: `--dataset synthetic`, `--samples 200`, `--lr 0.1`, `--epochs 50`, `--seed 0`.

## Тесты и проверки

```
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```