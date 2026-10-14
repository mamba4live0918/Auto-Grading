import csv
import io

import pytest

from apps.audio_review.server import (
    AnnotationError,
    build_boundaries_csv,
    build_status_csv,
    ordered_clips,
    validate_clip_update,
)


def test_review_order_keeps_development_before_audit():
    rows = [
        {"clip_id": "A02_1", "phase": "audit", "duration_seconds": "30"},
        {"clip_id": "D02_2", "phase": "development", "duration_seconds": "30"},
        {"clip_id": "D01_1", "phase": "development", "duration_seconds": "30"},
        {"clip_id": "A01_2", "phase": "audit", "duration_seconds": "30"},
    ]
    assert [r["clip_id"] for r in ordered_clips(rows)] == [
        "D01_1", "D02_2", "A01_2", "A02_1"
    ]


def test_clip_update_rejects_overlapping_or_out_of_bounds_segments():
    with pytest.raises(AnnotationError, match="overlap"):
        validate_clip_update(
            {
                "segments": [
                    {"start_seconds": 1, "end_seconds": 3, "label": "speech"},
                    {"start_seconds": 2, "end_seconds": 4, "label": "speech"},
                ]
            },
            duration=30,
        )
    with pytest.raises(AnnotationError, match="duration"):
        validate_clip_update(
            {"segments": [{"start_seconds": 28, "end_seconds": 31, "label": "speech"}]},
            duration=30,
        )


def test_empty_reviewed_clip_is_distinct_from_unreviewed_and_exports_protocol_columns():
    clips = [
        {"clip_id": "D01_1", "phase": "development", "duration_seconds": "30"},
        {"clip_id": "D01_2", "phase": "development", "duration_seconds": "30"},
    ]
    state = {
        "annotator": "student_1",
        "clips": {
            "D01_1": {"reviewed": True, "difficult_to_judge": False, "no_speech_confirmed": True, "notes": "No speech", "segments": []},
            "D01_2": {
                "reviewed": False,
                "difficult_to_judge": True,
                "notes": "draft",
                "segments": [{"start_seconds": 1, "end_seconds": 2, "label": "uncertain", "notes": "quiet"}],
            },
        },
    }
    statuses = list(csv.DictReader(io.StringIO(build_status_csv(clips, state))))
    assert statuses[0] == {
        "clip_id": "D01_1", "annotator": "student_1", "reviewed": "yes",
        "difficult_to_judge": "no", "notes": "No speech"
    }
    assert statuses[1]["reviewed"] == "no"
    assert list(csv.DictReader(io.StringIO(build_boundaries_csv(clips, state)))) == []
