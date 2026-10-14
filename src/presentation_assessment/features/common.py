from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModalityScore:
    score: float
    explanation: str
    features: dict[str, float | int | str | None] = field(default_factory=dict)


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))
