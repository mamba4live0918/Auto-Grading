import importlib.util
import math
import warnings
import wave
from pathlib import Path

import numpy as np
import pytest

from presentation_assessment.transcription import (
    TranscriptSegment,
    TranscriptionResult,
    WordTiming,
)
from presentation_assessment.research import acoustic_features
from presentation_assessment.research.feature_registry import load_feature_registry
from presentation_assessment.research.vad import SpeechSegment, VadResult


SAMPLE_RATE = 16_000
REGISTRY_PATH = Path("configs/audio_feature_registry.v1.yaml")


def _write_wav(path: Path, samples: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return path


def _vad(sample_count: int, segments: tuple[SpeechSegment, ...]) -> VadResult:
    frame_count = sample_count // 480
    decisions = tuple(
        any(
            segment.start_seconds <= index * 0.03 < segment.end_seconds
            for segment in segments
        )
        for index in range(frame_count)
    )
    return VadResult(
        sample_rate=SAMPLE_RATE,
        sample_count=sample_count,
        frame_duration_ms=30,
        aggressiveness=2,
        frame_decisions=decisions,
        segments=segments,
    )


def _transcript(language: str, words: list[WordTiming], *, text: str = "content") -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language=language,
        language_probability=0.99,
        duration_s=1.0,
        model_size="synthetic",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=1.0,
                text=text,
                words=words,
            )
        ],
    )


def _bundle(**values: float | None) -> acoustic_features.FeatureBundle:
    return acoustic_features.FeatureBundle(
        values=values,
        missing_reasons={name: "base_feature_missing" for name, value in values.items() if value is None},
    )


def _qc_transcript(
    *,
    text: str = "synthetic speech",
    language_probability: float = 0.60,
    word_probabilities: tuple[float, ...] = (0.50,),
    speech_duration: float = 1.15,
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language="en",
        language_probability=language_probability,
        duration_s=speech_duration,
        model_size="synthetic",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=speech_duration,
                text=text,
                words=[
                    WordTiming(
                        word=f"word{index}",
                        start=index * 0.1,
                        end=index * 0.1 + 0.05,
                        probability=probability,
                    )
                    for index, probability in enumerate(word_probabilities)
                ],
            )
        ],
    )


def test_frozen_qc_thresholds_do_not_warn_at_exact_boundaries() -> None:
    """Break caught: frozen QC thresholds use inclusive comparisons."""
    warnings = acoustic_features.frozen_qc_warnings(
        {
            "integrated_loudness_lufs": -40.0,
            "speech_ratio": 0.25,
            "speech_duration_seconds": 1.0,
        },
        _qc_transcript(),
        existing=("too_short",),
    )

    assert warnings == ("too_short",)


def test_frozen_qc_surfaces_all_low_confidence_and_disagreement_conditions() -> None:
    """Break caught: requested ASR/VAD and recording warnings are silently absent."""
    warnings = acoustic_features.frozen_qc_warnings(
        {
            "integrated_loudness_lufs": -40.01,
            "speech_ratio": 0.249,
            "speech_duration_seconds": 1.0,
        },
        _qc_transcript(
            language_probability=0.599,
            word_probabilities=(0.49, 0.50),
            speech_duration=1.151,
        ),
    )

    assert set(warnings) == {
        "low_integrated_loudness",
        "low_speech_ratio",
        "low_language_probability",
        "low_mean_usable_word_probability",
        "asr_vad_speech_duration_disagreement",
    }


def test_frozen_qc_marks_empty_asr_without_fabricating_probability_warnings() -> None:
    """Break caught: a blank ASR success is treated as usable transcript evidence."""
    transcript = _qc_transcript(text="", word_probabilities=())

    warnings = acoustic_features.frozen_qc_warnings(
        {"speech_ratio": 0.5, "speech_duration_seconds": 1.0}, transcript
    )

    assert warnings == ("empty_asr",)


def test_acoustic_feature_module_is_available() -> None:
    """Break caught: the frozen feature-extraction implementation is absent."""
    assert importlib.util.find_spec(
        "presentation_assessment.research.acoustic_features"
    ) is not None


def test_acoustic_feature_public_contract_is_available() -> None:
    """Break caught: a required Task 6 interface is renamed or omitted."""
    assert acoustic_features.FeatureBundle
    assert acoustic_features.CandidateTransient
    assert callable(acoustic_features.extract_acoustic_features)
    assert callable(acoustic_features.detect_candidate_transients)
    assert callable(acoustic_features.compare_normalization_stability)


def test_robust_summary_uses_literal_median_iqr_and_mad_formulas() -> None:
    """Break caught: ordinary mean/SD replaces the frozen robust summaries."""
    summary = acoustic_features._robust_summary(np.array([1.0, 2.0, 3.0, 100.0]))

    assert summary.median == 2.5
    assert summary.iqr == 25.5
    assert summary.mad == 1.0


def test_feature_bundle_rejects_non_finite_measurements() -> None:
    """Break caught: NaN or infinity silently enters the auditable feature map."""
    with pytest.raises(ValueError, match="finite"):
        acoustic_features.FeatureBundle(values={"broken": math.nan})


def test_extracts_vad_rhythm_pitch_energy_and_spectral_families(tmp_path: Path) -> None:
    """Break caught: a complete synthetic signal loses a frozen feature family or unit."""
    duration = 4.0
    time = np.arange(round(SAMPLE_RATE * duration)) / SAMPLE_RATE
    phase = 2 * math.pi * (180.0 * time + 10.0 * np.square(time))
    envelope = 0.55 + 0.35 * np.sin(2 * math.pi * 2.0 * time - math.pi / 2)
    samples = np.round(32_767 * envelope * 0.25 * np.sin(phase)).astype(np.int16)
    source = _write_wav(tmp_path / "complete.wav", samples)
    segments = (SpeechSegment(0.5, 1.5), SpeechSegment(2.0, 3.0))

    bundle = acoustic_features.extract_acoustic_features(
        source,
        _transcript(
            "en",
            [
                WordTiming("one", 0.5, 0.8),
                WordTiming("two", 1.0, 1.3),
                WordTiming("three", 2.0, 2.3),
            ],
        ),
        _vad(samples.size, segments),
        load_feature_registry(REGISTRY_PATH),
    )
    values = bundle.values

    assert values["speech_duration_seconds"] == pytest.approx(2.0)
    assert values["non_speech_duration_seconds"] == pytest.approx(2.0)
    assert values["speech_ratio"] == pytest.approx(0.5)
    assert values["pause_ratio"] == pytest.approx(0.5)
    assert values["speech_segment_count"] == 2
    assert values["speech_segments_per_recording_minute"] == pytest.approx(30.0)
    assert values["speech_segments_per_speech_minute"] == pytest.approx(60.0)
    assert values["speech_segment_duration_median_seconds"] == pytest.approx(1.0)
    assert values["internal_pause_count"] == 1
    assert values["internal_pauses_per_recording_minute"] == pytest.approx(15.0)
    assert values["internal_pauses_per_speech_minute"] == pytest.approx(30.0)
    assert values["internal_pause_duration_median_seconds"] == pytest.approx(0.5)
    assert values["leading_non_speech_duration_seconds"] == pytest.approx(0.5)
    assert values["trailing_non_speech_duration_seconds"] == pytest.approx(1.0)
    assert values["asr_language_unit_count"] == 3
    assert values["asr_units_per_recording_minute"] == pytest.approx(45.0)
    assert values["asr_units_per_speech_minute"] == pytest.approx(90.0)
    assert values["acoustic_envelope_peaks_per_recording_minute"] == pytest.approx(120.0, abs=20.0)
    assert values["acoustic_envelope_peaks_per_speech_minute"] == pytest.approx(240.0, abs=40.0)
    assert values["envelope_peak_interval_median_seconds"] == pytest.approx(0.5, abs=0.08)
    assert values["envelope_peak_interval_cv"] is not None
    assert values["envelope_peak_interval_cv"] < 0.15
    assert values["speech_segment_onset_interval_median_seconds"] == pytest.approx(1.5)
    assert values["continuous_speech_run_duration_median_seconds"] == pytest.approx(1.0)
    assert values["voiced_frame_ratio"] > 0.8
    assert values["pitch_median_hz"] == pytest.approx(220.0, abs=12.0)
    assert values["pitch_p10_p90_range_hz"] > 45.0
    assert values["pitch_p10_p90_range_semitones"] > 3.0
    assert values["pitch_iqr_hz"] > 25.0
    assert values["pitch_std_semitones"] > 1.0
    assert values["rms_energy_median"] > 0.0
    assert values["rms_energy_iqr"] > 0.0
    assert values["rms_robust_cv"] == pytest.approx(
        values["rms_energy_iqr"] / values["rms_energy_median"]
    )
    assert values["spectral_centroid_median_hz"] == pytest.approx(220.0, abs=80.0)
    assert values["spectral_flux_median"] >= 0.0
    assert values["spectral_flux_dispersion"] >= 0.0


def test_language_units_prefer_timestamped_words_and_count_cjk_non_punctuation(
    tmp_path: Path,
) -> None:
    """Break caught: transcript text, punctuation, or CJK chunks inflate ASR unit counts."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / "language.wav", samples)
    vad = _vad(samples.size, (SpeechSegment(0.0, 1.0),))
    registry = load_feature_registry(REGISTRY_PATH)

    english = acoustic_features.extract_acoustic_features(
        source,
        _transcript(
            "en",
            [
                WordTiming("Hello", 0.0, 0.2),
                WordTiming(",", 0.2, 0.3),
                WordTiming("world", 0.3, 0.6),
            ],
            text="ignored words outside timestamp units",
        ),
        vad,
        registry,
    )
    chinese = acoustic_features.extract_acoustic_features(
        source,
        _transcript("zh", [WordTiming("你好，世界！", 0.0, 0.8)], text="ignored"),
        vad,
        registry,
    )

    assert english.values["asr_language_unit_count"] == 2
    assert chinese.values["asr_language_unit_count"] == 4


@pytest.mark.parametrize(
    "words",
    [
        [WordTiming("word", math.nan, 0.2)],
        [WordTiming("word", 0.0, math.inf)],
        [WordTiming("word", -0.1, 0.2)],
        [WordTiming("word", 0.4, 0.2)],
        [WordTiming("word", 0.5, 0.5)],
        [WordTiming("word", 0.8, 1.2)],
        [WordTiming("one", 0.5, 0.7), WordTiming("two", 0.2, 0.4)],
        [WordTiming("one", 0.0, 0.6), WordTiming("two", 0.5, 0.8)],
    ],
    ids=(
        "nan",
        "infinite",
        "negative",
        "reversed",
        "zero-duration",
        "out-of-duration",
        "unordered",
        "overlapping",
    ),
)
def test_invalid_word_timestamps_use_the_declared_missing_reason(
    tmp_path: Path, words: list[WordTiming]
) -> None:
    """Break caught: malformed timestamps count as usable ASR language units."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / "invalid-timestamps.wav", samples)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        _transcript("en", words),
        _vad(samples.size, (SpeechSegment(0.0, 1.0),)),
        load_feature_registry(REGISTRY_PATH),
    )

    for name in (
        "asr_language_unit_count",
        "asr_units_per_recording_minute",
        "asr_units_per_speech_minute",
    ):
        assert bundle.values[name] is None
        assert bundle.missing_reasons[name] == "missing_word_timestamps"


@pytest.mark.parametrize(
    ("language", "token"),
    [("th", "สวัสดี"), ("lo", "ສະບາຍດີ"), ("km", "សួស្តី")],
)
def test_languages_without_frozen_unitization_are_explicitly_unsupported(
    tmp_path: Path, language: str, token: str
) -> None:
    """Break caught: every unknown language silently falls through to word units."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / f"unsupported-{language}.wav", samples)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        _transcript(language, [WordTiming(token, 0.0, 0.5)]),
        _vad(samples.size, (SpeechSegment(0.0, 1.0),)),
        load_feature_registry(REGISTRY_PATH),
    )

    for name in (
        "asr_language_unit_count",
        "asr_units_per_recording_minute",
        "asr_units_per_speech_minute",
    ):
        assert bundle.values[name] is None
        assert bundle.missing_reasons[name] == "unsupported_language_unitization"


def test_missing_timestamps_and_unvoiced_pitch_are_null_with_declared_reasons(
    tmp_path: Path,
) -> None:
    """Break caught: unavailable ASR/pitch values are silently omitted or zero-filled."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / "silent.wav", samples)
    registry = load_feature_registry(REGISTRY_PATH)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        _transcript("en", [WordTiming("hello", None, None)]),
        _vad(samples.size, ()),
        registry,
    )

    assert set(bundle.values) == set(registry.by_name)
    assert bundle.values["asr_language_unit_count"] is None
    assert bundle.missing_reasons["asr_language_unit_count"] == "missing_word_timestamps"
    assert bundle.values["pitch_median_hz"] is None
    assert bundle.missing_reasons["pitch_median_hz"] == "insufficient_voiced_frames"
    assert bundle.values["rms_robust_cv"] is None
    assert bundle.missing_reasons["rms_robust_cv"] == "zero_median_energy"
    assert bundle.values["candidate_transient_intensity_median"] is None
    assert bundle.missing_reasons["candidate_transient_intensity_median"] == "no_candidate_transients"
    assert all(
        bundle.missing_reasons[name] in registry.by_name[name].missing_policy.reason_codes
        for name, value in bundle.values.items()
        if value is None
    )
    assert all(name in bundle.missing_reasons for name, value in bundle.values.items() if value is None)
    assert all(name not in bundle.missing_reasons for name, value in bundle.values.items() if value is not None)


def test_silence_keeps_undefined_spectral_features_missing(tmp_path: Path) -> None:
    """Break caught: zero-total spectra become fabricated zero-Hz/zero-flux values."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / "spectral-silence.wav", samples)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        None,
        _vad(samples.size, ()),
        load_feature_registry(REGISTRY_PATH),
    )

    for name in (
        "spectral_centroid_median_hz",
        "spectral_flux_median",
        "spectral_flux_dispersion",
    ):
        assert bundle.values[name] is None
        assert bundle.missing_reasons[name] == "no_valid_spectral_frames"


def test_substantial_silence_does_not_dominate_valid_spectral_centroid(
    tmp_path: Path,
) -> None:
    """Break caught: undefined silent frames pull a valid tone centroid to zero Hz."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    tone_time = np.arange(SAMPLE_RATE // 4) / SAMPLE_RATE
    samples[: tone_time.size] = np.round(
        32_767 * 0.2 * np.sin(2 * math.pi * 440.0 * tone_time)
    ).astype(np.int16)
    source = _write_wav(tmp_path / "tone-and-silence.wav", samples)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        None,
        _vad(samples.size, ()),
        load_feature_registry(REGISTRY_PATH),
    )

    assert bundle.values["spectral_centroid_median_hz"] == pytest.approx(
        440.0, abs=20.0
    )


def test_one_spectral_frame_has_centroid_but_no_fabricated_flux(tmp_path: Path) -> None:
    """Break caught: prepending a spectrum invents a zero flux observation."""
    time = np.arange(100) / SAMPLE_RATE
    samples = np.round(32_767 * 0.2 * np.sin(2 * math.pi * 997.0 * time)).astype(
        np.int16
    )
    source = _write_wav(tmp_path / "one-frame.wav", samples)

    bundle = acoustic_features.extract_acoustic_features(
        source,
        None,
        _vad(samples.size, ()),
        load_feature_registry(REGISTRY_PATH),
    )

    assert bundle.values["spectral_centroid_median_hz"] is not None
    for name in ("spectral_flux_median", "spectral_flux_dispersion"):
        assert bundle.values[name] is None
        assert bundle.missing_reasons[name] == "insufficient_frames"


def test_zero_duration_is_a_hard_failure(tmp_path: Path) -> None:
    """Break caught: rate features divide by zero or fabricate values for an empty WAV."""
    source = _write_wav(tmp_path / "empty.wav", np.array([], dtype=np.int16))

    with pytest.raises(ValueError, match="positive duration"):
        acoustic_features.extract_acoustic_features(
            source,
            None,
            _vad(0, ()),
            load_feature_registry(REGISTRY_PATH),
        )


def test_candidate_selection_uses_exact_thresholds_and_80ms_suppression() -> None:
    """Break caught: threshold equality is rejected or nearby candidate frames double-count."""
    events = acoustic_features._select_candidate_transients(
        times=np.array([0.00, 0.08, 0.20, 0.30, 0.50]),
        energy_z=np.array([4.0, 8.0, 3.99, 4.0, 5.0]),
        low_frequency_z=np.array([2.5, 0.0, 3.0, 2.49, 3.0]),
        flux_z=np.array([0.0, 3.0, 3.0, 2.5, 3.0]),
    )

    assert [event.time_seconds for event in events] == [0.08, 0.30, 0.50]
    assert [event.confidence_level for event in events] == ["medium", "medium", "high"]
    assert events[0].energy_derivative_robust_z == 8.0


def test_low_frequency_impulses_are_detected_and_nearby_frames_merge() -> None:
    """Break caught: public signal detection ignores low-frequency attacks or double-counts bursts."""
    time = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
    samples = 0.001 * np.sin(2 * math.pi * 700.0 * time)
    for start in (1.00, 1.04, 2.00):
        begin = round(start * SAMPLE_RATE)
        length = round(0.03 * SAMPLE_RATE)
        burst_time = np.arange(length) / SAMPLE_RATE
        samples[begin : begin + length] += 0.7 * np.hanning(length) * np.sin(
            2 * math.pi * 100.0 * burst_time
        )

    events = acoustic_features.detect_candidate_transients(
        np.round(32_767 * samples).astype(np.int16), SAMPLE_RATE
    )

    assert len(events) == 2
    assert events[0].time_seconds == pytest.approx(1.01, abs=1 / SAMPLE_RATE)
    assert events[1].time_seconds == pytest.approx(2.01, abs=1 / SAMPLE_RATE)
    assert all(event.confidence_level in {"medium", "high"} for event in events)


@pytest.mark.parametrize("sample_rate", [100, 8_000, 44_100])
def test_candidate_detector_rejects_non_16khz_before_signal_math(
    sample_rate: int,
) -> None:
    """Break caught: unsupported rates reach framing/Hann math or silently pass."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="exactly 16000 Hz"):
            acoustic_features.detect_candidate_transients(
                np.zeros(sample_rate, dtype=np.float64), sample_rate
            )

    assert caught == []


@pytest.mark.parametrize(
    ("name", "samples"),
    [
        (
            "stationary-997hz",
            0.25
            * np.sin(
                2
                * math.pi
                * 997.0
                * (np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE)
            ),
        ),
        (
            "smooth-half-cosine-attack",
            (
                0.5
                - 0.5
                * np.cos(
                    math.pi
                    * np.minimum(
                        np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE, 2.0
                    )
                    / 2.0
                )
            )
            * 0.25
            * np.sin(
                2
                * math.pi
                * 997.0
                * (np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE)
            ),
        ),
        ("silence", np.zeros(SAMPLE_RATE * 4)),
        (
            "steady-noise",
            np.random.default_rng(20260825).normal(0.0, 0.05, SAMPLE_RATE * 4),
        ),
    ],
)
def test_stationary_and_smooth_signals_are_not_candidate_transients(
    name: str, samples: np.ndarray
) -> None:
    """Break caught: phase ripple or collapsed robust scale becomes a transient."""
    assert acoustic_features.detect_candidate_transients(samples, SAMPLE_RATE) == (), name


def test_envelope_peaks_use_analysis_frame_centres() -> None:
    """Break caught: an envelope event is timestamped at its frame start."""
    peak_times = acoustic_features._envelope_peak_times(
        np.array([0.0, 0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0])
    )

    assert peak_times.tolist() == pytest.approx([0.0525], abs=1 / SAMPLE_RATE)


def test_stability_reports_measured_differences_and_frozen_tolerance_warnings() -> None:
    """Break caught: one derivative or the wrong absolute/relative tolerance is used."""
    base = _bundle(
        speech_ratio=0.50,
        pause_ratio=0.50,
        acoustic_envelope_peaks_per_speech_minute=120.0,
        pitch_p10_p90_range_semitones=8.0,
        candidate_transients_per_speech_minute=4.0,
    )
    peak = _bundle(
        speech_ratio=0.51,
        pause_ratio=0.49,
        acoustic_envelope_peaks_per_speech_minute=126.0,
        pitch_p10_p90_range_semitones=8.4,
        candidate_transients_per_speech_minute=5.0,
    )
    loudness = _bundle(
        speech_ratio=0.53,
        pause_ratio=0.47,
        acoustic_envelope_peaks_per_speech_minute=138.0,
        pitch_p10_p90_range_semitones=8.6,
        candidate_transients_per_speech_minute=6.0,
    )

    stability = acoustic_features.compare_normalization_stability(base, peak, loudness)

    assert len(stability.values) == 15
    assert stability.values["stability_speech_ratio_absolute_difference"] == pytest.approx(0.03)
    assert stability.values["stability_speech_ratio_relative_difference"] == pytest.approx(0.06)
    assert stability.values["stability_pause_ratio_absolute_difference"] == pytest.approx(0.03)
    assert stability.values[
        "stability_acoustic_envelope_peak_rate_relative_difference"
    ] == pytest.approx(0.15)
    assert stability.values[
        "stability_pitch_p10_p90_range_semitones_absolute_difference"
    ] == pytest.approx(0.6)
    assert stability.values[
        "stability_candidate_transient_rate_absolute_difference"
    ] == pytest.approx(2.0)
    assert set(stability.qc_warnings) == {
        "stability_speech_ratio_absolute_difference=0.03 exceeds 0.02",
        "stability_pause_ratio_absolute_difference=0.03 exceeds 0.02",
        "stability_acoustic_envelope_peak_rate_relative_difference=0.15 exceeds 0.1",
        "stability_pitch_p10_p90_range_semitones_absolute_difference=0.6 exceeds 0.5",
        "stability_candidate_transient_rate_absolute_difference=2 exceeds 1",
    }


def test_stability_tolerance_boundaries_do_not_warn() -> None:
    """Break caught: frozen tolerances warn at equality instead of only above it."""
    base = _bundle(
        speech_ratio=0.50,
        pause_ratio=0.50,
        acoustic_envelope_peaks_per_speech_minute=120.0,
        pitch_p10_p90_range_semitones=8.0,
        candidate_transients_per_speech_minute=4.0,
    )
    derivative = _bundle(
        speech_ratio=0.52,
        pause_ratio=0.48,
        acoustic_envelope_peaks_per_speech_minute=132.0,
        pitch_p10_p90_range_semitones=8.5,
        candidate_transients_per_speech_minute=5.0,
    )

    stability = acoustic_features.compare_normalization_stability(
        base, derivative, derivative
    )

    assert stability.qc_warnings == ()


def test_stability_comparison_preserves_full_registry_binding(tmp_path: Path) -> None:
    """Break caught: stability output drops the base bundle's other 63 registry keys."""
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    source = _write_wav(tmp_path / "stable.wav", samples)
    registry = load_feature_registry(REGISTRY_PATH)
    base = acoustic_features.extract_acoustic_features(
        source, None, _vad(samples.size, ()), registry
    )

    stability = acoustic_features.compare_normalization_stability(base, base, base)

    assert set(stability.values) == set(registry.by_name)
    assert all(
        stability.missing_reasons[name]
        in registry.by_name[name].missing_policy.reason_codes
        for name, value in stability.values.items()
        if value is None
    )
    assert stability.values["duration_seconds"] == 1.0
