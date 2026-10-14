"""Audit-only recording-quality measurements for decoded PCM WAV files."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_pipeline import LoudnessMeasurement, measure_loudness_true_peak
from .vad import VadResult, run_webrtc_vad


MIN_DURATION_SECONDS = 3.0
SEVERE_CLIPPING_RATIO = 0.01
LOW_SNR_DB = 10.0
_FRAME_DURATION_MS = 30


@dataclass(frozen=True)
class DecodedWav:
    """One validated PCM decode that downstream stages can safely reuse."""

    source_path: Path
    samples: np.ndarray
    sample_rate_hz: int
    channel_count: int
    frame_count: int


@dataclass(frozen=True)
class RecordingQuality:
    """Quality measurements and review-only warnings for one recording.

    This object deliberately contains no clustering eligibility field: quality
    warnings remain audit information and do not exclude a sample from later
    analysis.
    """

    duration_seconds: float
    sample_count: int
    sample_rate_hz: int
    channel_count: int
    sample_peak_amplitude: float
    rms_amplitude: float
    dc_offset_amplitude: float
    clipped_sample_ratio: float
    integrated_loudness_lufs: float | None
    loudness_range_lu: float | None
    true_peak_dbtp: float | None
    loudness_reason: str | None
    noise_floor_dbfs: float | None
    noise_floor_reason: str | None
    robust_snr_db: float | None
    robust_snr_reason: str | None
    is_silent: bool
    is_too_short: bool
    is_severely_clipped: bool
    is_low_snr: bool
    warnings: tuple[str, ...]


def measure_recording_quality(
    path: Path,
    *,
    vad_result: VadResult | None = None,
    decoded_wav: DecodedWav | None = None,
    ffmpeg: str = "ffmpeg",
) -> RecordingQuality:
    """Measure audit-only PCM, loudness, and VAD-conditioned signal quality.

    Unavailable values are represented by ``None`` with a reason code.  Raw PCM
    measures retain genuine zero values; only measurements that cannot be made
    are null.  Loudness and true-peak parsing is intentionally delegated to the
    shared Task 3 FFmpeg parser.
    """
    decoded = decoded_wav or read_pcm16_wav(path)
    if decoded.source_path.resolve() != Path(path).resolve():
        raise ValueError("decoded_wav must belong to the requested recording")
    samples = decoded.samples
    sample_rate = decoded.sample_rate_hz
    channel_count = decoded.channel_count
    frame_count = decoded.frame_count
    duration_seconds = frame_count / sample_rate
    normalized = samples.astype(np.float64) / 32_768.0
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
    dc_offset = float(np.mean(normalized)) if normalized.size else 0.0
    clipped = float(np.mean(np.abs(samples.astype(np.int32)) >= 32_767)) if samples.size else 0.0
    is_silent = peak == 0.0
    is_too_short = duration_seconds < MIN_DURATION_SECONDS

    loudness, loudness_reason = _measure_loudness(
        path,
        is_silent=is_silent,
        is_too_short=is_too_short,
        ffmpeg=ffmpeg,
    )
    mono_samples = _to_mono_int16(samples, channel_count)
    if vad_result is not None:
        _validate_injected_vad(vad_result, mono_samples, sample_rate)
    noise_floor, noise_reason, snr, snr_reason = _measure_noise_and_snr(
        mono_samples,
        sample_rate,
        is_silent=is_silent,
        vad_result=vad_result,
    )
    severely_clipped = clipped > SEVERE_CLIPPING_RATIO
    low_snr = snr is not None and snr < LOW_SNR_DB
    warnings = tuple(
        warning
        for warning, enabled in (
            ("silent_audio", is_silent),
            ("too_short", is_too_short),
            ("severely_clipped", severely_clipped),
            ("low_snr", low_snr),
        )
        if enabled
    )
    return RecordingQuality(
        duration_seconds=duration_seconds,
        sample_count=int(samples.size),
        sample_rate_hz=sample_rate,
        channel_count=channel_count,
        sample_peak_amplitude=peak,
        rms_amplitude=rms,
        dc_offset_amplitude=dc_offset,
        clipped_sample_ratio=clipped,
        integrated_loudness_lufs=None if loudness is None else loudness.integrated_lufs,
        loudness_range_lu=None if loudness is None else loudness.loudness_range_lu,
        true_peak_dbtp=None if loudness is None else loudness.true_peak_dbtp,
        loudness_reason=loudness_reason,
        noise_floor_dbfs=noise_floor,
        noise_floor_reason=noise_reason,
        robust_snr_db=snr,
        robust_snr_reason=snr_reason,
        is_silent=is_silent,
        is_too_short=is_too_short,
        is_severely_clipped=severely_clipped,
        is_low_snr=low_snr,
        warnings=warnings,
    )


def read_pcm16_wav(path: Path) -> DecodedWav:
    """Decode canonical PCM once so quality and feature stages share the samples."""
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            frame_count = source.getnframes()
            payload = source.readframes(frame_count)
    except (OSError, wave.Error) as error:
        raise ValueError("recording could not be decoded as a WAV file") from error
    if channels < 1 or sample_rate < 1 or sample_width != 2:
        raise ValueError("recording must contain 16-bit PCM WAV samples")
    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size != frame_count * channels:
        raise ValueError("recording PCM payload was incomplete")
    return DecodedWav(
        source_path=Path(path),
        samples=samples.copy(),
        sample_rate_hz=sample_rate,
        channel_count=channels,
        frame_count=frame_count,
    )


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Backward-compatible private tuple adapter for pre-Task 6 callers."""
    decoded = read_pcm16_wav(path)
    return (
        decoded.samples,
        decoded.sample_rate_hz,
        decoded.channel_count,
        decoded.frame_count,
    )


def _measure_loudness(
    path: Path, *, is_silent: bool, is_too_short: bool, ffmpeg: str
) -> tuple[LoudnessMeasurement | None, str | None]:
    if is_silent:
        return None, "silent_audio"
    if is_too_short:
        return None, "insufficient_duration"
    try:
        return measure_loudness_true_peak(path, ffmpeg=ffmpeg), None
    except (OSError, RuntimeError, ValueError):
        return None, "measurement_failed"


def _to_mono_int16(samples: np.ndarray, channel_count: int) -> np.ndarray:
    if channel_count == 1:
        return samples
    interleaved = samples.reshape(-1, channel_count).astype(np.float64)
    return np.rint(np.mean(interleaved, axis=1)).astype(np.int16)


def _validate_injected_vad(vad: VadResult, samples: np.ndarray, sample_rate: int) -> None:
    """Reject injected VAD output that was not made with this frozen contract."""
    frame_samples = sample_rate * _FRAME_DURATION_MS // 1_000
    expected_frame_count = samples.size // frame_samples
    if vad.sample_rate != sample_rate:
        raise ValueError("injected VadResult.sample_rate must match the recording")
    if vad.sample_count != samples.size:
        raise ValueError("injected VadResult.sample_count must match the recording")
    if vad.frame_duration_ms != _FRAME_DURATION_MS:
        raise ValueError("injected VadResult.frame_duration_ms must be 30")
    if vad.aggressiveness != 2:
        raise ValueError("injected VadResult.aggressiveness must be 2")
    if len(vad.frame_decisions) != expected_frame_count:
        raise ValueError("injected VadResult.frame_decisions must match 30 ms frames")


def _measure_noise_and_snr(
    samples: np.ndarray,
    sample_rate: int,
    *,
    is_silent: bool,
    vad_result: VadResult | None,
) -> tuple[float | None, str | None, float | None, str | None]:
    if is_silent:
        return None, "silent_audio", None, "silent_audio"
    try:
        vad = vad_result or run_webrtc_vad(samples, sample_rate)
    except (RuntimeError, ValueError):
        return None, "measurement_failed", None, "measurement_failed"
    frame_samples = sample_rate * _FRAME_DURATION_MS // 1_000
    complete_frame_count = samples.size // frame_samples
    if complete_frame_count < 1:
        return None, "insufficient_frames", None, "insufficient_frames"
    if vad.sample_rate != sample_rate or vad.sample_count != samples.size:
        return None, "measurement_failed", None, "measurement_failed"
    if len(vad.frame_decisions) != complete_frame_count:
        return None, "measurement_failed", None, "measurement_failed"
    frames = samples[: complete_frame_count * frame_samples].astype(np.float64).reshape(
        complete_frame_count, frame_samples
    ) / 32_768.0
    levels = np.sqrt(np.mean(np.square(frames), axis=1))
    decisions = np.asarray(vad.frame_decisions, dtype=bool)
    noise_levels = levels[~decisions]
    speech_levels = levels[decisions]
    if noise_levels.size == 0:
        return None, "no_non_speech_frames", None, "no_non_speech_frames"
    noise_level = float(np.median(noise_levels))
    if noise_level <= 0.0 or not math.isfinite(noise_level):
        return None, "measurement_failed", None, "measurement_failed"
    noise_floor = _amplitude_to_dbfs(noise_level)
    if speech_levels.size == 0:
        return noise_floor, None, None, "no_speech_frames"
    speech_level = float(np.median(speech_levels))
    if speech_level <= 0.0 or not math.isfinite(speech_level):
        return noise_floor, None, None, "measurement_failed"
    return noise_floor, None, _amplitude_to_dbfs(speech_level) - noise_floor, None


def _amplitude_to_dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(amplitude)
