import math
import wave
from pathlib import Path

import numpy as np
import pytest

from presentation_assessment.research import quality as quality_module
from presentation_assessment.research.quality import measure_recording_quality
from presentation_assessment.research.vad import SpeechSegment, VadResult


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 480


def test_quality_exposes_reusable_decoded_pcm_contract() -> None:
    """Break caught: downstream extraction must decode the same WAV a second time."""
    assert quality_module.DecodedWav
    assert callable(quality_module.read_pcm16_wav)


def test_quality_reuses_injected_decode_without_reopening_the_wav(tmp_path: Path) -> None:
    """Break caught: Task 6 injects PCM but quality silently parses the WAV again."""
    source = _write_wav(
        tmp_path / "decode-once.wav", np.zeros(SAMPLE_RATE, dtype=np.int16)
    )
    decoded = quality_module.read_pcm16_wav(source)
    source.unlink()

    quality = measure_recording_quality(source, decoded_wav=decoded)

    assert quality.sample_count == SAMPLE_RATE
    assert quality.is_silent is True


def _write_wav(path: Path, samples: np.ndarray, *, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return path


def _sine(*, duration_seconds: float, amplitude: float, frequency_hz: float = 440.0) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration_seconds)) / SAMPLE_RATE
    return np.round(32_767 * amplitude * np.sin(2 * math.pi * frequency_hz * time)).astype(np.int16)


def _speech_after_noise_vad(sample_count: int) -> VadResult:
    frame_count = sample_count // FRAME_SAMPLES
    speech_start = frame_count // 2
    return VadResult(
        sample_rate=SAMPLE_RATE,
        sample_count=sample_count,
        frame_duration_ms=30,
        aggressiveness=2,
        frame_decisions=tuple(index >= speech_start for index in range(frame_count)),
        segments=(SpeechSegment(speech_start * 0.03, frame_count * 0.03),),
    )


def _constant_level_vad(sample_count: int, *, speech: bool) -> VadResult:
    frame_count = sample_count // FRAME_SAMPLES
    return VadResult(
        sample_rate=SAMPLE_RATE,
        sample_count=sample_count,
        frame_duration_ms=30,
        aggressiveness=2,
        frame_decisions=(speech,) * frame_count,
        segments=(SpeechSegment(0.0, frame_count * 0.03),) if speech else (),
    )


def test_quality_measures_pcm_loudness_and_robust_vad_conditioned_snr(tmp_path: Path) -> None:
    """Break caught: a quality report omits PCM, loudness, or speech/noise measurements."""
    noise = _sine(duration_seconds=1.98, amplitude=0.01, frequency_hz=150.0)
    speech = _sine(duration_seconds=2.02, amplitude=0.1)
    source = _write_wav(tmp_path / "recording.wav", np.concatenate((noise, speech)))

    quality = measure_recording_quality(source, vad_result=_speech_after_noise_vad(64_000))

    assert quality.duration_seconds == pytest.approx(4.0)
    assert quality.sample_count == 64_000
    assert quality.sample_rate_hz == SAMPLE_RATE
    assert quality.channel_count == 1
    assert quality.sample_peak_amplitude == pytest.approx(0.1, abs=0.001)
    assert quality.rms_amplitude == pytest.approx(0.05, abs=0.002)
    assert quality.dc_offset_amplitude == pytest.approx(0.0, abs=0.001)
    assert quality.clipped_sample_ratio == 0.0
    assert quality.integrated_loudness_lufs is not None
    assert quality.loudness_range_lu is not None
    assert quality.true_peak_dbtp is not None
    assert quality.noise_floor_dbfs == pytest.approx(-43.0, abs=0.5)
    assert quality.robust_snr_db == pytest.approx(20.0, abs=0.5)
    assert quality.warnings == ()


def test_quality_flags_a_severely_clipped_recording_without_excluding_it(tmp_path: Path) -> None:
    """Break caught: clipping is either missed or turned into a clustering-eligibility decision."""
    source = _write_wav(
        tmp_path / "clipped.wav",
        np.tile(np.array([32_767, -32_768], dtype=np.int16), SAMPLE_RATE * 2),
    )

    quality = measure_recording_quality(source)

    assert quality.clipped_sample_ratio == 1.0
    assert quality.is_severely_clipped is True
    assert "severely_clipped" in quality.warnings
    assert not hasattr(quality, "clustering_eligible")


def test_silent_recording_keeps_unmeasurable_values_null_with_reasons(tmp_path: Path) -> None:
    """Break caught: silence is reported as a fabricated zero loudness, noise floor, or SNR."""
    source = _write_wav(tmp_path / "silent.wav", np.zeros(SAMPLE_RATE * 4, dtype=np.int16))

    quality = measure_recording_quality(source)

    assert quality.is_silent is True
    assert quality.integrated_loudness_lufs is None
    assert quality.loudness_range_lu is None
    assert quality.true_peak_dbtp is None
    assert quality.loudness_reason == "silent_audio"
    assert quality.noise_floor_dbfs is None
    assert quality.noise_floor_reason == "silent_audio"
    assert quality.robust_snr_db is None
    assert quality.robust_snr_reason == "silent_audio"
    assert "silent_audio" in quality.warnings


def test_short_recording_flags_the_threshold_and_does_not_invent_loudness(tmp_path: Path) -> None:
    """Break caught: sub-three-second audio is treated as a valid EBU R128 measurement."""
    source = _write_wav(tmp_path / "short.wav", _sine(duration_seconds=1.0, amplitude=0.1))

    quality = measure_recording_quality(source)

    assert quality.is_too_short is True
    assert quality.integrated_loudness_lufs is None
    assert quality.loudness_reason == "insufficient_duration"
    assert "too_short" in quality.warnings


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sample_rate", 8_000, "sample_rate"),
        ("sample_count", 1, "sample_count"),
        ("frame_duration_ms", 20, "frame_duration_ms"),
        ("aggressiveness", 3, "aggressiveness"),
        ("frame_decisions", (), "frame_decisions"),
    ],
)
def test_injected_vad_configuration_must_match_the_quality_contract(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    """Break caught: SNR uses frame decisions from a mismatched VAD configuration."""
    source = _write_wav(tmp_path / "recording.wav", _sine(duration_seconds=4.0, amplitude=0.1))
    valid = _speech_after_noise_vad(SAMPLE_RATE * 4)
    invalid = VadResult(
        sample_rate=replacement if field == "sample_rate" else valid.sample_rate,
        sample_count=replacement if field == "sample_count" else valid.sample_count,
        frame_duration_ms=replacement if field == "frame_duration_ms" else valid.frame_duration_ms,
        aggressiveness=replacement if field == "aggressiveness" else valid.aggressiveness,
        frame_decisions=replacement if field == "frame_decisions" else valid.frame_decisions,
        segments=valid.segments,
    )

    with pytest.raises(ValueError, match=message):
        measure_recording_quality(source, vad_result=invalid)


def test_quality_warns_for_clipping_only_above_the_one_percent_cutoff(tmp_path: Path) -> None:
    """Break caught: the severe-clipping warning fires at, rather than above, one percent."""
    base = _sine(duration_seconds=4.0, amplitude=0.1)
    exact_count = base.size // 100
    exact = base.copy()
    exact[:exact_count] = 32_767
    above = base.copy()
    above[: exact_count + 1] = 32_767

    exact_quality = measure_recording_quality(_write_wav(tmp_path / "exact.wav", exact))
    above_quality = measure_recording_quality(_write_wav(tmp_path / "above.wav", above))

    assert exact_quality.clipped_sample_ratio == pytest.approx(0.01)
    assert exact_quality.is_severely_clipped is False
    assert "severely_clipped" not in exact_quality.warnings
    assert above_quality.clipped_sample_ratio > 0.01
    assert above_quality.is_severely_clipped is True
    assert "severely_clipped" in above_quality.warnings


def test_quality_warns_for_snr_only_below_the_ten_db_cutoff(tmp_path: Path) -> None:
    """Break caught: the low-SNR warning fires at, rather than below, 10 dB."""
    frame_count = (SAMPLE_RATE * 4) // FRAME_SAMPLES
    noise_frame = np.full(FRAME_SAMPLES, 1_000, dtype=np.int16)
    ten_db_speech_frame = np.concatenate(
        (np.full(288, 4_000, dtype=np.int16), np.full(192, 1_000, dtype=np.int16))
    )
    below_ten_db_speech_frame = np.concatenate(
        (np.full(278, 4_000, dtype=np.int16), np.full(202, 1_000, dtype=np.int16))
    )
    half = frame_count // 2
    exact = np.concatenate(
        [noise_frame] * half + [ten_db_speech_frame] * (frame_count - half) + [noise_frame[:160]]
    )
    below = np.concatenate(
        [noise_frame] * half
        + [below_ten_db_speech_frame] * (frame_count - half)
        + [noise_frame[:160]]
    )
    decisions = VadResult(
        sample_rate=SAMPLE_RATE,
        sample_count=SAMPLE_RATE * 4,
        frame_duration_ms=30,
        aggressiveness=2,
        frame_decisions=(False,) * half + (True,) * (frame_count - half),
        segments=(SpeechSegment(half * 0.03, frame_count * 0.03),),
    )

    exact_quality = measure_recording_quality(
        _write_wav(tmp_path / "exact-snr.wav", exact), vad_result=decisions
    )
    below_quality = measure_recording_quality(
        _write_wav(tmp_path / "below-snr.wav", below), vad_result=decisions
    )

    assert exact_quality.robust_snr_db == pytest.approx(10.0, abs=1e-12)
    assert exact_quality.is_low_snr is False
    assert "low_snr" not in exact_quality.warnings
    assert below_quality.robust_snr_db is not None and below_quality.robust_snr_db < 10.0
    assert below_quality.is_low_snr is True
    assert "low_snr" in below_quality.warnings


def test_quality_uses_no_speech_frames_for_deterministic_missing_snr(tmp_path: Path) -> None:
    """Break caught: all-non-speech VAD output reports an opaque SNR failure reason."""
    source = _write_wav(tmp_path / "non-speech.wav", _sine(duration_seconds=4.0, amplitude=0.1))

    quality = measure_recording_quality(
        source, vad_result=_constant_level_vad(SAMPLE_RATE * 4, speech=False)
    )

    assert quality.noise_floor_dbfs is not None
    assert quality.robust_snr_db is None
    assert quality.robust_snr_reason == "no_speech_frames"
