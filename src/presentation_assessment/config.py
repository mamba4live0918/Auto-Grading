from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DIMENSION_NAMES = (
    "content",
    "speech_delivery",
    "visual_engagement",
    "gesture_posture",
)


@dataclass(frozen=True)
class ActiveDimensions:
    content: bool = True
    speech_delivery: bool = True
    visual_engagement: bool = True
    gesture_posture: bool = True


@dataclass(frozen=True)
class Weights:
    content: float = 0.35
    speech_delivery: float = 0.25
    visual_engagement: float = 0.20
    gesture_posture: float = 0.20

    def normalized_for(self, active: ActiveDimensions) -> "Weights":
        content = self.content if active.content else 0.0
        speech_delivery = self.speech_delivery if active.speech_delivery else 0.0
        visual_engagement = self.visual_engagement if active.visual_engagement else 0.0
        gesture_posture = self.gesture_posture if active.gesture_posture else 0.0
        total = content + speech_delivery + visual_engagement + gesture_posture
        if total <= 0:
            raise ValueError("At least one active dimension must have a positive weight.")
        return Weights(
            content=content / total,
            speech_delivery=speech_delivery / total,
            visual_engagement=visual_engagement / total,
            gesture_posture=gesture_posture / total,
        )


@dataclass(frozen=True)
class ScoringConfig:
    min_score: float = 0.0
    max_score: float = 100.0
    neutral_score: float = 60.0
    missing_modality_policy: str = "neutral"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    active_dimensions: ActiveDimensions
    weights: Weights
    scoring: ScoringConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        scoring_raw = raw.get("scoring", {})
        active = _active_dimensions_from_raw(raw)
        weights = _weights_from_raw(raw).normalized_for(active)
        scoring = ScoringConfig(**scoring_raw)
        return cls(
            experiment_name=raw.get("experiment_name", "experiment"),
            active_dimensions=active,
            weights=weights,
            scoring=scoring,
        )


def _active_dimensions_from_raw(raw: dict[str, Any]) -> ActiveDimensions:
    if "active_dimensions" in raw:
        return ActiveDimensions(**raw.get("active_dimensions", {}))

    legacy = raw.get("active_modalities", {})
    if legacy:
        speech_enabled = bool(legacy.get("speech", True))
        return ActiveDimensions(
            content=speech_enabled,
            speech_delivery=speech_enabled,
            visual_engagement=bool(legacy.get("visual", True)),
            gesture_posture=bool(legacy.get("gesture", True)),
        )

    return ActiveDimensions()


def _weights_from_raw(raw: dict[str, Any]) -> Weights:
    weights_raw = raw.get("weights", {})
    if any(name in weights_raw for name in DIMENSION_NAMES):
        return Weights(**weights_raw)

    legacy_speech = float(weights_raw.get("speech", 0.60))
    return Weights(
        content=legacy_speech * 0.60,
        speech_delivery=legacy_speech * 0.40,
        visual_engagement=float(weights_raw.get("visual", 0.20)),
        gesture_posture=float(weights_raw.get("gesture", 0.20)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig.from_dict(raw)
