from __future__ import annotations

from presentation_assessment.config import ScoringConfig
from presentation_assessment.features.common import ModalityScore


class VisionScorer:
    """Reads visual engagement scores until a face/attention model is attached."""

    def __init__(self, scoring: ScoringConfig):
        self.scoring = scoring

    def score(self, visual_engagement_score: float | None) -> ModalityScore:
        if visual_engagement_score is None:
            return ModalityScore(
                score=self.scoring.neutral_score,
                explanation="visual engagement score missing; neutral fallback used",
                features={"source": "neutral_fallback"},
            )
        return ModalityScore(
            score=round(visual_engagement_score, 2),
            explanation="visual engagement score loaded from precomputed features or manifest",
            features={"source": "manifest"},
        )
