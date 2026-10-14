from __future__ import annotations

from presentation_assessment.config import DIMENSION_NAMES, ExperimentConfig
from presentation_assessment.features.common import ModalityScore, clamp


def fuse_scores(
    dimension_scores: dict[str, ModalityScore],
    config: ExperimentConfig,
) -> float:
    score = sum(
        getattr(config.weights, dimension_name) * dimension_scores[dimension_name].score
        for dimension_name in DIMENSION_NAMES
    )
    return round(clamp(score, config.scoring.min_score, config.scoring.max_score), 2)
