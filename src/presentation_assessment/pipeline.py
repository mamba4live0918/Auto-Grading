from __future__ import annotations

import csv
import json
from pathlib import Path

from presentation_assessment.config import ExperimentConfig
from presentation_assessment.data import PresentationSample
from presentation_assessment.features.body import BodyScorer
from presentation_assessment.features.common import ModalityScore
from presentation_assessment.features.speech import SpeechScorer
from presentation_assessment.features.vision import VisionScorer
from presentation_assessment.fusion import fuse_scores


class AssessmentPipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.speech_scorer = SpeechScorer()
        self.vision_scorer = VisionScorer(config.scoring)
        self.body_scorer = BodyScorer(config.scoring)

    def run_sample(self, sample: PresentationSample) -> dict[str, str | float | None]:
        content = self._content(sample)
        speech_delivery = self._speech_delivery(sample)
        visual_engagement = self._visual_engagement(sample)
        gesture_posture = self._gesture_posture(sample)
        dimension_scores = {
            "content": content,
            "speech_delivery": speech_delivery,
            "visual_engagement": visual_engagement,
            "gesture_posture": gesture_posture,
        }
        final_score = fuse_scores(dimension_scores, self.config)
        explanation = {
            "content": content.explanation,
            "speech_delivery": speech_delivery.explanation,
            "visual_engagement": visual_engagement.explanation,
            "gesture_posture": gesture_posture.explanation,
            "weights": self.config.weights.__dict__,
        }
        return {
            "sample_id": sample.sample_id,
            "experiment_name": self.config.experiment_name,
            "human_score": sample.human_score,
            "final_score": final_score,
            "content_score": content.score if self.config.active_dimensions.content else None,
            "speech_delivery_score": (
                speech_delivery.score if self.config.active_dimensions.speech_delivery else None
            ),
            "visual_engagement_score": (
                visual_engagement.score
                if self.config.active_dimensions.visual_engagement
                else None
            ),
            "gesture_posture_score": (
                gesture_posture.score if self.config.active_dimensions.gesture_posture else None
            ),
            "explanation": json.dumps(explanation, ensure_ascii=False),
        }

    def run(self, samples: list[PresentationSample]) -> list[dict[str, str | float | None]]:
        return [self.run_sample(sample) for sample in samples]

    def _content(self, sample: PresentationSample) -> ModalityScore:
        if not self.config.active_dimensions.content:
            return ModalityScore(self.config.scoring.neutral_score, "content disabled")
        return self.speech_scorer.score_content(sample.transcript)

    def _speech_delivery(self, sample: PresentationSample) -> ModalityScore:
        if not self.config.active_dimensions.speech_delivery:
            return ModalityScore(self.config.scoring.neutral_score, "speech delivery disabled")
        return self.speech_scorer.score_delivery(
            sample.transcript, sample.speech_delivery_score
        )

    def _visual_engagement(self, sample: PresentationSample) -> ModalityScore:
        if not self.config.active_dimensions.visual_engagement:
            return ModalityScore(self.config.scoring.neutral_score, "visual engagement disabled")
        return self.vision_scorer.score(sample.visual_engagement_score)

    def _gesture_posture(self, sample: PresentationSample) -> ModalityScore:
        if not self.config.active_dimensions.gesture_posture:
            return ModalityScore(self.config.scoring.neutral_score, "gesture/posture disabled")
        return self.body_scorer.score(sample.gesture_posture_score)


def write_predictions(rows: list[dict[str, str | float | None]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "experiment_name",
        "human_score",
        "final_score",
        "content_score",
        "speech_delivery_score",
        "visual_engagement_score",
        "gesture_posture_score",
        "explanation",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
