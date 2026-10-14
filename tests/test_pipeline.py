from presentation_assessment.config import load_config
from presentation_assessment.data import load_manifest
from presentation_assessment.pipeline import AssessmentPipeline


def test_pipeline_runs_sample_manifest() -> None:
    config = load_config("configs/full_multimodal.yaml")
    samples = load_manifest("data/sample_manifest.csv")
    rows = AssessmentPipeline(config).run(samples)

    assert len(rows) == 5
    assert all(0 <= float(row["final_score"]) <= 100 for row in rows)
    assert rows[0]["content_score"] is not None
    assert rows[0]["speech_delivery_score"] is not None
    assert rows[0]["visual_engagement_score"] is not None
    assert rows[0]["gesture_posture_score"] is not None
