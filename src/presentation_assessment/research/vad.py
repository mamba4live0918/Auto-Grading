"""Deterministic WebRTC VAD segmentation for canonical PCM audio."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


FRAME_DURATION_MS = 30
AGGRESSIVENESS = 2
MIN_SPEECH_DURATION_MS = 300
MERGE_GAP_MS = 200
PADDING_MS = 100
_SUPPORTED_SAMPLE_RATES = frozenset((8_000, 16_000, 32_000, 48_000))

FrameClassifier = Callable[[bytes, int], bool]


@dataclass(frozen=True)
class SpeechSegment:
    """One post-processed speech interval, expressed in seconds."""

    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class VadResult:
    """Frame decisions and the post-processed speech intervals they produce."""

    sample_rate: int
    sample_count: int
    frame_duration_ms: int
    aggressiveness: int
    frame_decisions: tuple[bool, ...]
    segments: tuple[SpeechSegment, ...]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def speech_duration_seconds(self) -> float:
        return sum(segment.duration_seconds for segment in self.segments)


def run_webrtc_vad(
    samples_int16: np.ndarray,
    sample_rate: int = 16_000,
    *,
    frame_classifier: FrameClassifier | None = None,
) -> VadResult:
    """Classify 30 ms PCM frames, then apply the frozen cleanup configuration.

    ``frame_classifier`` is intentionally injectable so segmentation behavior can
    be tested without relying on WebRTC's signal classifier.  In production it
    defaults to a WebRTC VAD instance configured at aggressiveness 2.
    """
    samples = _validated_samples(samples_int16)
    if sample_rate not in _SUPPORTED_SAMPLE_RATES:
        raise ValueError("WebRTC VAD supports 8000, 16000, 32000, or 48000 Hz")
    frame_samples = sample_rate * FRAME_DURATION_MS // 1_000
    if frame_samples * 1_000 != sample_rate * FRAME_DURATION_MS:
        raise ValueError("sample rate does not produce integral 30 ms PCM frames")
    classifier = frame_classifier or _webrtc_classifier()
    complete_frame_count = samples.size // frame_samples
    decisions = tuple(
        bool(classifier(samples[index * frame_samples : (index + 1) * frame_samples].tobytes(), sample_rate))
        for index in range(complete_frame_count)
    )
    segments = _clean_segments(
        decisions,
        frame_seconds=FRAME_DURATION_MS / 1_000,
        recording_duration_seconds=samples.size / sample_rate,
    )
    return VadResult(
        sample_rate=sample_rate,
        sample_count=int(samples.size),
        frame_duration_ms=FRAME_DURATION_MS,
        aggressiveness=AGGRESSIVENESS,
        frame_decisions=decisions,
        segments=segments,
    )


def _validated_samples(samples_int16: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples_int16)
    if samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError("samples_int16 must be a one-dimensional int16 array")
    return np.ascontiguousarray(samples)


def _webrtc_classifier() -> FrameClassifier:
    try:
        import webrtcvad
    except ImportError as error:  # pragma: no cover - dependency is installed in research environments.
        raise RuntimeError("webrtcvad is required for VAD frame classification") from error
    vad = webrtcvad.Vad(AGGRESSIVENESS)
    return vad.is_speech


def _clean_segments(
    decisions: tuple[bool, ...],
    *,
    frame_seconds: float,
    recording_duration_seconds: float,
) -> tuple[SpeechSegment, ...]:
    """Apply duration filtering, short-gap merging, then boundary-safe padding."""
    raw = _contiguous_speech_segments(decisions, frame_seconds)
    minimum_seconds = MIN_SPEECH_DURATION_MS / 1_000
    retained = [
        segment
        for segment in raw
        if segment.duration_seconds + 1e-12 >= minimum_seconds
    ]
    merged: list[SpeechSegment] = []
    maximum_gap_seconds = MERGE_GAP_MS / 1_000
    for segment in retained:
        if merged and segment.start_seconds - merged[-1].end_seconds < maximum_gap_seconds:
            previous = merged[-1]
            merged[-1] = SpeechSegment(previous.start_seconds, segment.end_seconds)
        else:
            merged.append(segment)
    padding_seconds = PADDING_MS / 1_000
    return tuple(
        SpeechSegment(
            max(0.0, segment.start_seconds - padding_seconds),
            min(recording_duration_seconds, segment.end_seconds + padding_seconds),
        )
        for segment in merged
    )


def _contiguous_speech_segments(
    decisions: tuple[bool, ...], frame_seconds: float
) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, is_speech in enumerate(decisions):
        if is_speech and start_index is None:
            start_index = index
        elif not is_speech and start_index is not None:
            segments.append(SpeechSegment(start_index * frame_seconds, index * frame_seconds))
            start_index = None
    if start_index is not None:
        segments.append(SpeechSegment(start_index * frame_seconds, len(decisions) * frame_seconds))
    return segments
