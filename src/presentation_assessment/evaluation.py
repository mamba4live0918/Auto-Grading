from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationResult:
    n: int
    pearson: float | None
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "pearson": self.pearson,
            "mae": self.mae,
            "rmse": self.rmse,
        }


def evaluate_pairs(actual: list[float], predicted: list[float]) -> EvaluationResult:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("at least one labeled prediction is required")

    errors = [p - a for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    pearson = _pearson(actual, predicted) if len(actual) >= 2 else None
    return EvaluationResult(
        n=len(actual),
        pearson=None if pearson is None else round(pearson, 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
    )


def evaluate_prediction_file(
    path: str | Path,
    label_column: str = "human_score",
    prediction_column: str = "final_score",
) -> EvaluationResult:
    actual: list[float] = []
    predicted: list[float] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get(label_column)
            prediction = row.get(prediction_column)
            if label in (None, "") or prediction in (None, ""):
                continue
            actual.append(float(label))
            predicted.append(float(prediction))
    return evaluate_pairs(actual, predicted)


def _pearson(x: list[float], y: list[float]) -> float | None:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0 or denom_y == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)
