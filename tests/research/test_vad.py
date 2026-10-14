import numpy as np
import pytest

from presentation_assessment.research.vad import run_webrtc_vad


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 480


def _indexed_frames(frame_count: int) -> np.ndarray:
    """Make each 30 ms PCM frame identifiable to an injected classifier."""
    samples = np.zeros(frame_count * FRAME_SAMPLES, dtype=np.int16)
    for index in range(frame_count):
        samples[index * FRAME_SAMPLES] = index
    return samples


def test_vad_removes_short_runs_merges_a_180ms_gap_and_pads_the_result() -> None:
    """Break caught: cleanup applied in the wrong order changes retained speech boundaries."""
    samples = _indexed_frames(50)
    speech_frames = set(range(4, 14)) | set(range(20, 30)) | set(range(35, 42))

    result = run_webrtc_vad(
        samples,
        frame_classifier=lambda frame, _rate: int.from_bytes(frame[:2], "little", signed=True)
        in speech_frames,
    )

    assert result.frame_decisions == tuple(index in speech_frames for index in range(50))
    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (pytest.approx(0.02), pytest.approx(1.0))
    ]
    assert result.speech_duration_seconds == pytest.approx(0.98)


def test_vad_keeps_a_210ms_gap_separate_and_pads_at_file_edges() -> None:
    """Break caught: the 30 ms-representable non-short gap merges, or padding exceeds bounds."""
    samples = _indexed_frames(50)
    speech_frames = set(range(0, 10)) | set(range(17, 27)) | set(range(40, 50))

    result = run_webrtc_vad(
        samples,
        frame_classifier=lambda frame, _rate: int.from_bytes(frame[:2], "little", signed=True)
        in speech_frames,
    )

    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (pytest.approx(0.0), pytest.approx(0.4)),
        (pytest.approx(0.41), pytest.approx(0.91)),
        (pytest.approx(1.1), pytest.approx(1.5)),
    ]
    assert result.duration_seconds == pytest.approx(1.5)
