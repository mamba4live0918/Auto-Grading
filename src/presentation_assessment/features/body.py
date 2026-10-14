from __future__ import annotations

from presentation_assessment.config import ScoringConfig
from presentation_assessment.features.common import ModalityScore


class BodyScorer:
    """Reads gesture/posture scores until a real pose/gesture model is attached."""

    def __init__(self, scoring: ScoringConfig):
        self.scoring = scoring

    def score(self, gesture_posture_score: float | None) -> ModalityScore:
        if gesture_posture_score is None:
            return ModalityScore(
                score=self.scoring.neutral_score,
                explanation="gesture/posture score missing; neutral fallback used",
                features={"source": "neutral_fallback"},
            )
        return ModalityScore(
            score=round(gesture_posture_score, 2),
            explanation="gesture/posture score loaded from precomputed features or manifest",
            features={"source": "manifest"},
        )
