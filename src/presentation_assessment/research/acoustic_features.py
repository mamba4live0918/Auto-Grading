"""Frozen, registry-bound acoustic measurements and stability QC."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from presentation_assessment.transcription import TranscriptionResult

from .feature_registry import FeatureRegistry
from .quality import DecodedWav, RecordingQuality, measure_recording_quality, read_pcm16_wav
from .vad import SpeechSegment, VadResult


FeatureValue = float | int | bool | str | None

_FRAME_SECONDS = 0.025
_HOP_SECONDS = 0.010
_PITCH_FRAME_SECONDS = 0.050
_PITCH_HOP_SECONDS = 0.020
_PITCH_MIN_HZ = 60.0
_PITCH_MAX_HZ = 500.0
_MIN_VOICED_FRAMES = 3
_TRANSIENT_FRAME_SECONDS = 0.020
_TRANSIENT_HOP_SECONDS = 0.010
_ENERGY_Z_THRESHOLD = 4.0
_AUXILIARY_Z_THRESHOLD = 2.5
_TRANSIENT_SUPPRESSION_SECONDS = 0.080
_CONTINUOUS_WRITING_LANGUAGES = frozenset(("ja", "zh", "yue"))
_WHITESPACE_LANGUAGES = frozenset(
    (
        "af",
        "ar",
        "be",
        "bg",
        "bn",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gl",
        "gu",
        "he",
        "hi",
        "hr",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "kk",
        "kn",
        "ko",
        "lt",
        "lv",
        "mk",
        "ml",
        "mr",
        "ms",
        "ne",
        "nl",
        "no",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sq",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "tr",
        "uk",
        "ur",
        "vi",
    )
)

_STABILITY_SOURCES = (
    ("speech_ratio", "stability_speech_ratio", "absolute", 0.02),
    ("pause_ratio", "stability_pause_ratio", "absolute", 0.02),
    (
        "acoustic_envelope_peaks_per_speech_minute",
        "stability_acoustic_envelope_peak_rate",
        "relative",
        0.10,
    ),
    (
        "pitch_p10_p90_range_semitones",
        "stability_pitch_p10_p90_range_semitones",
        "absolute",
        0.5,
    ),
    (
        "candidate_transients_per_speech_minute",
        "stability_candidate_transient_rate",
        "candidate",
        0.20,
    ),
)


@dataclass(frozen=True)
class RobustSummary:
    """Median, interquartile range, and median absolute deviation."""

    median: float
    iqr: float
    mad: float


@dataclass(frozen=True)
class CandidateTransient:
    """One retained heuristic transient event, never a confirmed phoneme."""

    time_seconds: float
    energy_derivative_robust_z: float
    low_frequency_ratio_robust_z: float
    spectral_flux_robust_z: float
    confidence_level: str

    def __post_init__(self) -> None:
        measurements = (
            self.time_seconds,
            self.energy_derivative_robust_z,
            self.low_frequency_ratio_robust_z,
            self.spectral_flux_robust_z,
        )
        if not all(math.isfinite(value) for value in measurements):
            raise ValueError("candidate transient measurements must be finite")
        if self.time_seconds < 0.0:
            raise ValueError("candidate transient time must be non-negative")
        if self.confidence_level not in {"medium", "high"}:
            raise ValueError("candidate transient confidence must be medium or high")


@dataclass(frozen=True)
class FeatureBundle:
    """Values, explicit missingness, events, and measured QC warnings."""

    values: Mapping[str, FeatureValue]
    missing_reasons: Mapping[str, str] = field(default_factory=dict)
    candidate_transients: tuple[CandidateTransient, ...] = ()
    qc_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.values)
        reasons = dict(self.missing_reasons)
        for name, value in values.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError(f"feature {name!r} must be finite or explicitly missing")
            if value is None and name not in reasons:
                raise ValueError(f"feature {name!r} is missing without a reason")
            if value is not None and name in reasons:
                raise ValueError(f"feature {name!r} has both a value and a missing reason")
        if set(reasons) - set(values):
            raise ValueError("missing reasons must correspond to feature values")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "missing_reasons", MappingProxyType(reasons))
        object.__setattr__(self, "candidate_transients", tuple(self.candidate_transients))
        object.__setattr__(self, "qc_warnings", tuple(self.qc_warnings))


def frozen_qc_warnings(
    values: Mapping[str, FeatureValue],
    transcript: TranscriptionResult | None,
    *,
    existing: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the stable code-only QC warnings frozen for downstream review."""
    warnings = list(existing)
    loudness = values.get("integrated_loudness_lufs")
    if isinstance(loudness, (int, float, np.number)) and float(loudness) < -40.0:
        warnings.append("low_integrated_loudness")
    speech_ratio = values.get("speech_ratio")
    if isinstance(speech_ratio, (int, float, np.number)) and float(speech_ratio) < 0.25:
        warnings.append("low_speech_ratio")
    if transcript is None or not transcript.text.strip():
        warnings.append("empty_asr")
        return tuple(dict.fromkeys(warnings))
    if (
        transcript.language_probability is not None
        and math.isfinite(float(transcript.language_probability))
        and float(transcript.language_probability) < 0.60
    ):
        warnings.append("low_language_probability")
    usable_probabilities = [
        float(word.probability)
        for segment in transcript.segments
        for word in segment.words
        if word.word.strip()
        and word.probability is not None
        and math.isfinite(float(word.probability))
        and 0.0 <= float(word.probability) <= 1.0
    ]
    if usable_probabilities and float(np.mean(usable_probabilities)) < 0.50:
        warnings.append("low_mean_usable_word_probability")
    vad_duration = values.get("speech_duration_seconds")
    asr_duration = sum(
        max(0.0, float(segment.end) - float(segment.start))
        for segment in transcript.segments
        if math.isfinite(float(segment.start)) and math.isfinite(float(segment.end))
    )
    if (
        isinstance(vad_duration, (int, float, np.number))
        and float(vad_duration) > 0.0
        and abs(asr_duration - float(vad_duration)) / float(vad_duration) > 0.15 + 1e-12
    ):
        warnings.append("asr_vad_speech_duration_disagreement")
    return tuple(dict.fromkeys(warnings))


def extract_acoustic_features(
    analysis_path: Path,
    transcript: TranscriptionResult | None,
    vad: VadResult,
    registry: FeatureRegistry,
) -> FeatureBundle:
    """Return all 73 registry keys with either a finite value or declared reason.

    The PCM WAV is decoded once here.  Task 5 quality measurement receives that
    decode and the caller's VAD result, so neither WAV parsing nor VAD is
    duplicated.
    """
    path = Path(analysis_path)
    decoded = read_pcm16_wav(path)
    if decoded.frame_count <= 0:
        raise ValueError("analysis WAV must have positive duration")
    quality = measure_recording_quality(path, vad_result=vad, decoded_wav=decoded)
    signal = _mono_normalized(decoded)
    duration = quality.duration_seconds
    segments = _validated_segments(vad, duration)
    speech_duration = float(sum(segment.duration_seconds for segment in segments))
    speech_reason = _speech_denominator_reason(segments, speech_duration)

    values: dict[str, FeatureValue] = {name: None for name in registry.by_name}
    reasons: dict[str, str] = {}
    _add_quality_features(values, reasons, quality)
    _add_vad_features(values, reasons, segments, duration, speech_duration, speech_reason)
    _add_language_features(values, reasons, transcript, duration, speech_duration, speech_reason)

    frames = _frame_signal(signal, decoded.sample_rate_hz, _FRAME_SECONDS, _HOP_SECONDS)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    _add_envelope_features(
        values,
        reasons,
        _envelope_peak_times(rms, decoded.sample_rate_hz),
        duration,
        speech_duration,
        speech_reason,
    )
    pitch, voiced_ratio = _estimate_pitch(signal, decoded.sample_rate_hz)
    _add_pitch_features(values, reasons, pitch, voiced_ratio)
    _add_energy_features(values, reasons, rms)
    centroids, flux = _spectral_metrics(frames, decoded.sample_rate_hz)
    if centroids.size == 0:
        for name in (
            "spectral_centroid_median_hz",
            "spectral_flux_median",
            "spectral_flux_dispersion",
        ):
            _set_missing(values, reasons, name, "no_valid_spectral_frames")
    else:
        values["spectral_centroid_median_hz"] = float(np.median(centroids))
        if flux.size == 0:
            for name in ("spectral_flux_median", "spectral_flux_dispersion"):
                _set_missing(values, reasons, name, "insufficient_frames")
        else:
            values["spectral_flux_median"] = float(np.median(flux))
            values["spectral_flux_dispersion"] = _robust_summary(flux).iqr

    candidates = detect_candidate_transients(signal, decoded.sample_rate_hz)
    _add_transient_features(
        values,
        reasons,
        candidates,
        duration,
        speech_duration,
        speech_reason,
    )
    for definition in registry.features:
        if definition.family == "normalization_stability":
            _set_missing(values, reasons, definition.name, "normalized_derivative_unavailable")

    _validate_registry_binding(values, reasons, registry)
    return FeatureBundle(
        values=values,
        missing_reasons=reasons,
        candidate_transients=candidates,
        qc_warnings=frozen_qc_warnings(values, transcript, existing=quality.warnings),
    )


def detect_candidate_transients(
    samples: np.ndarray, sample_rate: int
) -> tuple[CandidateTransient, ...]:
    """Detect energy attacks supported by low-frequency ratio or spectral flux."""
    if sample_rate != 16_000:
        raise ValueError("candidate transient detection requires exactly 16000 Hz")
    signal = _normalized_mono(samples)
    if signal.size == 0:
        return ()
    frames = _frame_signal(
        signal,
        sample_rate,
        _TRANSIENT_FRAME_SECONDS,
        _TRANSIENT_HOP_SECONDS,
    )
    if frames.shape[0] < 2:
        return ()
    window = np.hanning(frames.shape[1])
    rms = np.sqrt(np.sum(np.square(frames) * window, axis=1) / np.sum(window))
    energy_derivative = np.diff(rms)
    windowed = frames * window
    magnitudes = np.abs(np.fft.rfft(windowed, axis=1))
    frequencies = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    power = np.square(magnitudes)
    total_power = np.sum(power, axis=1)
    low_power = np.sum(power[:, frequencies <= 300.0], axis=1)
    low_ratio = np.divide(
        low_power,
        total_power,
        out=np.zeros_like(low_power),
        where=total_power > 0.0,
    )
    totals = np.sum(magnitudes, axis=1, keepdims=True)
    normalized = np.divide(
        magnitudes,
        totals,
        out=np.zeros_like(magnitudes),
        where=totals > 0.0,
    )
    positive_change = np.maximum(np.diff(normalized, axis=0), 0.0)
    flux = np.sqrt(np.sum(np.square(positive_change), axis=1))
    times = _frame_center_times(
        frames.shape[0],
        sample_rate,
        _TRANSIENT_FRAME_SECONDS,
        _TRANSIENT_HOP_SECONDS,
    )[1:]
    return _select_candidate_transients(
        times=times,
        energy_z=_robust_z_scores(energy_derivative),
        low_frequency_z=_robust_z_scores(low_ratio[1:]),
        flux_z=_robust_z_scores(flux),
    )


def compare_normalization_stability(
    base: FeatureBundle, peak: FeatureBundle, loudness: FeatureBundle
) -> FeatureBundle:
    """Measure maximum base-to-derivative differences across all three WAVs."""
    values: dict[str, FeatureValue] = dict(base.values)
    reasons: dict[str, str] = dict(base.missing_reasons)
    warnings: list[str] = list(base.qc_warnings)
    for source, stem, tolerance_kind, threshold in _STABILITY_SOURCES:
        absolute_name = f"{stem}_absolute_difference"
        relative_name = f"{stem}_relative_difference"
        base_value = base.values.get(source)
        derivatives = (peak.values.get(source), loudness.values.get(source))
        if base_value is None:
            _set_missing(values, reasons, absolute_name, "base_feature_missing")
            _set_missing(values, reasons, relative_name, "base_feature_missing")
            continue
        if any(value is None for value in derivatives):
            _set_missing(values, reasons, absolute_name, "normalized_derivative_unavailable")
            _set_missing(values, reasons, relative_name, "normalized_derivative_unavailable")
            continue
        numeric_base = _finite_float(base_value, source)
        numeric_derivatives = tuple(_finite_float(value, source) for value in derivatives)
        absolute = max(abs(value - numeric_base) for value in numeric_derivatives)
        values[absolute_name] = absolute
        reasons.pop(absolute_name, None)
        if numeric_base == 0.0 and absolute != 0.0:
            _set_missing(values, reasons, relative_name, "comparison_failed")
            relative = None
        else:
            relative = 0.0 if numeric_base == 0.0 else absolute / abs(numeric_base)
            values[relative_name] = relative
            reasons.pop(relative_name, None)

        warning_name: str | None = None
        warning_value: float | None = None
        warning_threshold: float | None = None
        if tolerance_kind == "absolute":
            warning_name, warning_value, warning_threshold = absolute_name, absolute, threshold
        elif tolerance_kind == "relative" and relative is not None:
            warning_name, warning_value, warning_threshold = relative_name, relative, threshold
        elif tolerance_kind == "candidate":
            if numeric_base < 5.0:
                warning_name, warning_value, warning_threshold = absolute_name, absolute, 1.0
            elif relative is not None:
                warning_name, warning_value, warning_threshold = relative_name, relative, threshold
        if (
            warning_name is not None
            and warning_value is not None
            and warning_threshold is not None
            and warning_value > warning_threshold + 1e-12
        ):
            warnings.append(
                f"{warning_name}={warning_value:g} exceeds {warning_threshold:g}"
            )
    return FeatureBundle(
        values=values,
        missing_reasons=reasons,
        candidate_transients=base.candidate_transients,
        qc_warnings=tuple(warnings),
    )


def _robust_summary(values: np.ndarray) -> RobustSummary:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("robust summaries require finite non-empty values")
    median = float(np.median(array))
    q25, q75 = np.percentile(array, (25.0, 75.0))
    return RobustSummary(
        median=median,
        iqr=float(q75 - q25),
        mad=float(np.median(np.abs(array - median))),
    )


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("robust z scores require finite one-dimensional values")
    summary = _robust_summary(array)
    scale = max(
        1.482602218505602 * summary.mad,
        summary.iqr / 1.3489795003921634,
        np.finfo(np.float64).eps,
    )
    return (array - summary.median) / scale


def _select_candidate_transients(
    *,
    times: np.ndarray,
    energy_z: np.ndarray,
    low_frequency_z: np.ndarray,
    flux_z: np.ndarray,
) -> tuple[CandidateTransient, ...]:
    arrays = tuple(
        np.asarray(item, dtype=np.float64)
        for item in (times, energy_z, low_frequency_z, flux_z)
    )
    if any(item.ndim != 1 for item in arrays):
        raise ValueError("candidate metrics must be one-dimensional")
    if len({item.size for item in arrays}) != 1 or any(
        not np.all(np.isfinite(item)) for item in arrays
    ):
        raise ValueError("candidate metrics must have equal finite lengths")
    candidate_indices = [
        index
        for index in range(times.size)
        if energy_z[index] >= _ENERGY_Z_THRESHOLD
        and (
            low_frequency_z[index] >= _AUXILIARY_Z_THRESHOLD
            or flux_z[index] >= _AUXILIARY_Z_THRESHOLD
        )
    ]
    if not candidate_indices:
        return ()
    groups: list[list[int]] = [[candidate_indices[0]]]
    for index in candidate_indices[1:]:
        if times[index] - times[groups[-1][-1]] <= _TRANSIENT_SUPPRESSION_SECONDS + 1e-12:
            groups[-1].append(index)
        else:
            groups.append([index])
    retained = [
        max(group, key=lambda index: (energy_z[index], low_frequency_z[index] + flux_z[index]))
        for group in groups
    ]
    return tuple(
        CandidateTransient(
            time_seconds=float(times[index]),
            energy_derivative_robust_z=float(energy_z[index]),
            low_frequency_ratio_robust_z=float(low_frequency_z[index]),
            spectral_flux_robust_z=float(flux_z[index]),
            confidence_level=(
                "high"
                if low_frequency_z[index] >= _AUXILIARY_Z_THRESHOLD
                and flux_z[index] >= _AUXILIARY_Z_THRESHOLD
                else "medium"
            ),
        )
        for index in retained
    )


def _mono_normalized(decoded: DecodedWav) -> np.ndarray:
    samples = decoded.samples
    if decoded.channel_count > 1:
        samples = np.rint(
            np.mean(
                samples.reshape(-1, decoded.channel_count).astype(np.float64), axis=1
            )
        ).astype(np.int16)
    return samples.astype(np.float64) / 32_768.0


def _normalized_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("samples must be a one-dimensional numeric array")
    signal = array.astype(np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError("samples must be finite")
    if np.issubdtype(array.dtype, np.integer):
        signal /= 32_768.0
    return signal


def _frame_signal(
    samples: np.ndarray, sample_rate: int, frame_seconds: float, hop_seconds: float
) -> np.ndarray:
    frame_length = max(1, round(sample_rate * frame_seconds))
    hop_length = max(1, round(sample_rate * hop_seconds))
    if samples.size < frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        return padded.reshape(1, -1)
    frame_count = 1 + (samples.size - frame_length) // hop_length
    starts = np.arange(frame_count) * hop_length
    indices = starts[:, None] + np.arange(frame_length)[None, :]
    return np.asarray(samples[indices], dtype=np.float64)


def _frame_center_times(
    frame_count: int,
    sample_rate: int,
    frame_seconds: float,
    hop_seconds: float,
) -> np.ndarray:
    """Return centres of half-open analysis frames from integer sample indices."""
    frame_length = max(1, round(sample_rate * frame_seconds))
    hop_length = max(1, round(sample_rate * hop_seconds))
    starts = np.arange(frame_count, dtype=np.float64) * hop_length
    return (starts + frame_length / 2.0) / sample_rate


def _validated_segments(vad: VadResult, duration: float) -> tuple[SpeechSegment, ...]:
    segments = tuple(vad.segments)
    previous_end = 0.0
    for segment in segments:
        if (
            not math.isfinite(segment.start_seconds)
            or not math.isfinite(segment.end_seconds)
            or segment.start_seconds < previous_end - 1e-12
            or segment.start_seconds < 0.0
            or segment.end_seconds <= segment.start_seconds
            or segment.end_seconds > duration + 1e-9
        ):
            raise ValueError(
                "VAD segments must be finite, ordered, non-overlapping, and in bounds"
            )
        previous_end = segment.end_seconds
    return segments


def _speech_denominator_reason(
    segments: tuple[SpeechSegment, ...], speech_duration: float
) -> str | None:
    if not segments:
        return "no_speech_detected"
    if speech_duration <= 0.0:
        return "zero_speech_duration"
    return None


def _add_quality_features(
    values: dict[str, FeatureValue], reasons: dict[str, str], quality: RecordingQuality
) -> None:
    values.update(
        {
            "duration_seconds": quality.duration_seconds,
            "valid_sample_count": quality.sample_count,
            "sample_rate_hz": quality.sample_rate_hz,
            "channel_count": quality.channel_count,
            "sample_peak_amplitude": quality.sample_peak_amplitude,
            "rms_amplitude": quality.rms_amplitude,
            "dc_offset_amplitude": quality.dc_offset_amplitude,
            "clipped_sample_ratio": quality.clipped_sample_ratio,
            "quality_is_silent": quality.is_silent,
            "quality_is_too_short": quality.is_too_short,
            "quality_is_severely_clipped": quality.is_severely_clipped,
        }
    )
    for name, value in (
        ("integrated_loudness_lufs", quality.integrated_loudness_lufs),
        ("loudness_range_lu", quality.loudness_range_lu),
        ("true_peak_dbtp", quality.true_peak_dbtp),
    ):
        if value is None:
            _set_missing(
                values, reasons, name, quality.loudness_reason or "measurement_failed"
            )
        else:
            values[name] = value
    if quality.noise_floor_dbfs is None:
        _set_missing(
            values,
            reasons,
            "noise_floor_dbfs",
            _registry_quality_reason(quality.noise_floor_reason),
        )
    else:
        values["noise_floor_dbfs"] = quality.noise_floor_dbfs
    if quality.robust_snr_db is None:
        reason = _registry_quality_reason(quality.robust_snr_reason)
        _set_missing(values, reasons, "robust_snr_db", reason)
        _set_missing(values, reasons, "quality_is_low_snr", reason)
    else:
        values["robust_snr_db"] = quality.robust_snr_db
        values["quality_is_low_snr"] = quality.is_low_snr


def _registry_quality_reason(reason: str | None) -> str:
    if reason in {"no_non_speech_frames", "insufficient_frames", "measurement_failed"}:
        return reason
    return "measurement_failed"


def _add_vad_features(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    segments: tuple[SpeechSegment, ...],
    duration: float,
    speech_duration: float,
    speech_reason: str | None,
) -> None:
    non_speech = max(0.0, duration - speech_duration)
    segment_durations = np.asarray(
        [segment.duration_seconds for segment in segments], dtype=np.float64
    )
    pauses = np.asarray(
        [
            segments[index + 1].start_seconds - segments[index].end_seconds
            for index in range(len(segments) - 1)
        ],
        dtype=np.float64,
    )
    values.update(
        {
            "speech_duration_seconds": speech_duration,
            "non_speech_duration_seconds": non_speech,
            "speech_ratio": speech_duration / duration,
            "pause_ratio": non_speech / duration,
            "speech_segment_count": len(segments),
            "speech_segments_per_recording_minute": len(segments) * 60.0 / duration,
            "internal_pause_count": int(pauses.size),
            "internal_pauses_per_recording_minute": pauses.size * 60.0 / duration,
            "leading_non_speech_duration_seconds": (
                segments[0].start_seconds if segments else duration
            ),
            "trailing_non_speech_duration_seconds": (
                duration - segments[-1].end_seconds if segments else duration
            ),
        }
    )
    for name, count in (
        ("speech_segments_per_speech_minute", len(segments)),
        ("internal_pauses_per_speech_minute", int(pauses.size)),
    ):
        if speech_reason:
            _set_missing(values, reasons, name, speech_reason)
        else:
            values[name] = count * 60.0 / speech_duration
    _add_distribution(
        values,
        reasons,
        segment_durations,
        prefix="speech_segment_duration",
        empty_reason="no_speech_segments",
    )
    _add_distribution(
        values,
        reasons,
        pauses,
        prefix="internal_pause_duration",
        empty_reason="no_internal_pauses",
    )
    if len(segments) < 2:
        reason = "no_speech_segments" if not segments else "insufficient_segments"
        _set_missing(
            values, reasons, "speech_segment_onset_interval_median_seconds", reason
        )
    else:
        values["speech_segment_onset_interval_median_seconds"] = float(
            np.median(np.diff([segment.start_seconds for segment in segments]))
        )
    if not segments:
        _set_missing(
            values,
            reasons,
            "continuous_speech_run_duration_median_seconds",
            "no_speech_segments",
        )
    else:
        values["continuous_speech_run_duration_median_seconds"] = float(
            np.median(segment_durations)
        )


def _add_distribution(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    observations: np.ndarray,
    *,
    prefix: str,
    empty_reason: str,
) -> None:
    names = (
        f"{prefix}_median_seconds",
        f"{prefix}_mean_seconds",
        f"{prefix}_p90_seconds",
        f"{prefix}_max_seconds",
    )
    if observations.size == 0:
        for name in names:
            _set_missing(values, reasons, name, empty_reason)
        return
    values[names[0]] = float(np.median(observations))
    values[names[1]] = float(np.mean(observations))
    values[names[2]] = float(np.percentile(observations, 90.0))
    values[names[3]] = float(np.max(observations))


def _add_language_features(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    transcript: TranscriptionResult | None,
    duration: float,
    speech_duration: float,
    speech_reason: str | None,
) -> None:
    count, reason = _language_unit_count(transcript, duration)
    names = (
        "asr_language_unit_count",
        "asr_units_per_recording_minute",
        "asr_units_per_speech_minute",
    )
    if reason:
        for name in names:
            _set_missing(values, reasons, name, reason)
        return
    assert count is not None
    values[names[0]] = count
    values[names[1]] = count * 60.0 / duration
    if speech_reason:
        _set_missing(values, reasons, names[2], speech_reason)
    else:
        values[names[2]] = count * 60.0 / speech_duration


def _language_unit_count(
    transcript: TranscriptionResult | None,
    recording_duration: float,
) -> tuple[int | None, str | None]:
    if transcript is None:
        return None, "empty_asr"
    words = [word for segment in transcript.segments for word in segment.words]
    units = [word for word in words if _contains_language_character(word.word)]
    if not units:
        return None, "empty_asr"
    language = (transcript.language or "").lower().split("-", maxsplit=1)[0]
    if language not in _CONTINUOUS_WRITING_LANGUAGES | _WHITESPACE_LANGUAGES:
        return None, "unsupported_language_unitization"
    if not _valid_word_timestamps(units, recording_duration):
        return None, "missing_word_timestamps"
    if language in _CONTINUOUS_WRITING_LANGUAGES:
        return (
            sum(
                1
                for word in units
                for character in word.word
                if _is_language_character(character)
            ),
            None,
        )
    return len(units), None


def _valid_word_timestamps(words: list[object], recording_duration: float) -> bool:
    previous_start = -math.inf
    previous_end = -math.inf
    for word in words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0.0
            or float(end) <= float(start)
            or float(end) > recording_duration + 1e-12
            or float(start) < previous_start
            or float(end) < previous_end
            or float(start) < previous_end
        ):
            return False
        previous_start = float(start)
        previous_end = float(end)
    return True


def _contains_language_character(text: str) -> bool:
    return any(_is_language_character(character) for character in text)


def _is_language_character(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "N"}


def _envelope_peak_times(rms: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    if rms.size < 3 or float(np.max(rms)) <= np.finfo(np.float64).eps:
        return np.array([], dtype=np.float64)
    smoothed = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    summary = _robust_summary(smoothed)
    threshold = summary.median + max(0.25 * summary.iqr, 0.5 * summary.mad)
    local = (
        np.flatnonzero(
            (smoothed[1:-1] > smoothed[:-2])
            & (smoothed[1:-1] >= smoothed[2:])
            & (smoothed[1:-1] > threshold)
        )
        + 1
    )
    minimum_frames = max(1, round(0.12 / _HOP_SECONDS))
    retained: list[int] = []
    for index_value in local:
        index = int(index_value)
        if not retained or index - retained[-1] >= minimum_frames:
            retained.append(index)
        elif smoothed[index] > smoothed[retained[-1]]:
            retained[-1] = index
    centers = _frame_center_times(
        rms.size, sample_rate, _FRAME_SECONDS, _HOP_SECONDS
    )
    return centers[np.asarray(retained, dtype=np.int64)]


def _add_envelope_features(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    peak_times: np.ndarray,
    duration: float,
    speech_duration: float,
    speech_reason: str | None,
) -> None:
    values["acoustic_envelope_peaks_per_recording_minute"] = (
        peak_times.size * 60.0 / duration
    )
    if speech_reason:
        _set_missing(
            values,
            reasons,
            "acoustic_envelope_peaks_per_speech_minute",
            speech_reason,
        )
    else:
        values["acoustic_envelope_peaks_per_speech_minute"] = (
            peak_times.size * 60.0 / speech_duration
        )
    if peak_times.size < 2:
        for name in (
            "envelope_peak_interval_median_seconds",
            "envelope_peak_interval_cv",
        ):
            _set_missing(values, reasons, name, "fewer_than_two_envelope_peaks")
        return
    intervals = np.diff(peak_times)
    mean_interval = float(np.mean(intervals))
    values["envelope_peak_interval_median_seconds"] = float(np.median(intervals))
    values["envelope_peak_interval_cv"] = float(np.std(intervals) / mean_interval)


def _estimate_pitch(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    frames = _frame_signal(
        samples, sample_rate, _PITCH_FRAME_SECONDS, _PITCH_HOP_SECONDS
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    energy_threshold = max(1e-4, float(np.median(rms)) * 0.05)
    minimum_lag = max(1, math.floor(sample_rate / _PITCH_MAX_HZ))
    maximum_lag = min(
        frames.shape[1] - 2, math.ceil(sample_rate / _PITCH_MIN_HZ)
    )
    estimates: list[float] = []
    for frame, level in zip(frames, rms, strict=True):
        if level < energy_threshold:
            continue
        centered = frame - np.mean(frame)
        correlation = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
        if correlation[0] <= 0.0:
            continue
        lag: float = minimum_lag + int(
            np.argmax(correlation[minimum_lag : maximum_lag + 1])
        )
        lag_index = int(lag)
        if correlation[lag_index] / correlation[0] < 0.30:
            continue
        if 1 <= lag_index < correlation.size - 1:
            left, center, right = correlation[lag_index - 1 : lag_index + 2]
            denominator = left - 2.0 * center + right
            if denominator != 0.0:
                lag += float(0.5 * (left - right) / denominator)
        frequency = sample_rate / lag
        if _PITCH_MIN_HZ <= frequency <= _PITCH_MAX_HZ and math.isfinite(frequency):
            estimates.append(float(frequency))
    return np.asarray(estimates, dtype=np.float64), len(estimates) / frames.shape[0]


def _add_pitch_features(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    pitch: np.ndarray,
    voiced_ratio: float,
) -> None:
    values["voiced_frame_ratio"] = voiced_ratio
    names = (
        "pitch_median_hz",
        "pitch_p10_p90_range_hz",
        "pitch_p10_p90_range_semitones",
        "pitch_iqr_hz",
        "pitch_std_semitones",
    )
    if pitch.size < _MIN_VOICED_FRAMES:
        for name in names:
            _set_missing(values, reasons, name, "insufficient_voiced_frames")
        return
    p10, q25, median, q75, p90 = np.percentile(
        pitch, (10.0, 25.0, 50.0, 75.0, 90.0)
    )
    semitones = 12.0 * np.log2(pitch / median)
    values.update(
        {
            "pitch_median_hz": float(median),
            "pitch_p10_p90_range_hz": float(p90 - p10),
            "pitch_p10_p90_range_semitones": float(
                12.0 * math.log2(p90 / p10)
            ),
            "pitch_iqr_hz": float(q75 - q25),
            "pitch_std_semitones": float(np.std(semitones)),
        }
    )


def _add_energy_features(
    values: dict[str, FeatureValue], reasons: dict[str, str], rms: np.ndarray
) -> None:
    summary = _robust_summary(rms)
    values["rms_energy_median"] = summary.median
    values["rms_energy_iqr"] = summary.iqr
    if rms.size < 2:
        _set_missing(values, reasons, "rms_robust_cv", "insufficient_frames")
    elif summary.median <= np.finfo(np.float64).eps:
        _set_missing(values, reasons, "rms_robust_cv", "zero_median_energy")
    else:
        values["rms_robust_cv"] = summary.iqr / summary.median


def _spectral_metrics(
    frames: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    windowed = frames * np.hanning(frames.shape[1])
    magnitudes = np.abs(np.fft.rfft(windowed, axis=1))
    totals = np.sum(magnitudes, axis=1)
    frequencies = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    valid = totals > 0.0
    centroids = np.divide(
        magnitudes @ frequencies,
        totals,
        out=np.zeros(frames.shape[0], dtype=np.float64),
        where=valid,
    )
    normalized = np.divide(
        magnitudes,
        totals[:, None],
        out=np.zeros_like(magnitudes),
        where=totals[:, None] > 0.0,
    )
    changes = np.maximum(np.diff(normalized, axis=0), 0.0)
    consecutive_valid = valid[:-1] & valid[1:]
    flux = np.sqrt(np.sum(np.square(changes[consecutive_valid]), axis=1))
    return centroids[valid], flux


def _add_transient_features(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    candidates: tuple[CandidateTransient, ...],
    duration: float,
    speech_duration: float,
    speech_reason: str | None,
) -> None:
    count = len(candidates)
    values["candidate_transient_count"] = count
    values["candidate_transients_per_recording_minute"] = count * 60.0 / duration
    if speech_reason:
        _set_missing(
            values,
            reasons,
            "candidate_transients_per_speech_minute",
            speech_reason,
        )
    else:
        values["candidate_transients_per_speech_minute"] = (
            count * 60.0 / speech_duration
        )
    if not candidates:
        _set_missing(
            values,
            reasons,
            "candidate_transient_intensity_median",
            "no_candidate_transients",
        )
        _set_missing(
            values,
            reasons,
            "candidate_transient_confidence_level",
            "no_candidate_transients",
        )
        return
    values["candidate_transient_intensity_median"] = float(
        np.median([event.energy_derivative_robust_z for event in candidates])
    )
    values["candidate_transient_confidence_level"] = (
        "high" if any(event.confidence_level == "high" for event in candidates) else "medium"
    )


def _set_missing(
    values: dict[str, FeatureValue], reasons: dict[str, str], name: str, reason: str
) -> None:
    values[name] = None
    reasons[name] = reason


def _finite_float(value: FeatureValue, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"feature {name!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"feature {name!r} must be finite")
    return numeric


def _validate_registry_binding(
    values: dict[str, FeatureValue],
    reasons: dict[str, str],
    registry: FeatureRegistry,
) -> None:
    expected = set(registry.by_name)
    if set(values) != expected:
        raise ValueError(
            "feature bundle does not match registry; "
            f"missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}"
        )
    for name, definition in registry.by_name.items():
        value = values[name]
        if value is None:
            reason = reasons.get(name)
            if not definition.missing_policy.allow_missing:
                raise ValueError(f"required feature {name!r} is missing")
            if reason not in definition.missing_policy.reason_codes:
                raise ValueError(
                    f"feature {name!r} has undeclared missing reason {reason!r}"
                )
            continue
        if name in reasons:
            raise ValueError(f"feature {name!r} has both a value and a missing reason")
        if definition.dtype == "float":
            if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
                raise ValueError(f"feature {name!r} must be a float")
            numeric = float(value)
        elif definition.dtype == "integer":
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"feature {name!r} must be an integer")
            numeric = float(value)
        elif definition.dtype == "boolean":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"feature {name!r} must be boolean")
            continue
        else:
            if not isinstance(value, str) or not value:
                raise ValueError(f"feature {name!r} must be a non-empty string")
            continue
        if not math.isfinite(numeric):
            raise ValueError(f"feature {name!r} must be finite")
        if definition.minimum is not None and numeric < float(definition.minimum) - 1e-12:
            raise ValueError(f"feature {name!r} is below the registry minimum")
        if definition.maximum is not None and numeric > float(definition.maximum) + 1e-12:
            raise ValueError(f"feature {name!r} exceeds the registry maximum")
