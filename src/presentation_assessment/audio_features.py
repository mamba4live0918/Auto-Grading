from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from presentation_assessment.transcription import TranscriptionResult


FILLER_WORDS = {"um", "uh", "er", "ah", "like", "\u55ef", "\u554a", "\u7136\u540e"}


@dataclass(frozen=True)
class AudioFeatureResult:
    speech_delivery_score: float
    features: dict[str, float | int | str | None]


def extract_audio_features(
    audio_path: str | Path,
    transcription: TranscriptionResult | None = None,
    transcript_text: str | None = None,
) -> AudioFeatureResult:
    samples, sample_rate = _read_wav_mono(audio_path)
    duration_s = len(samples) / sample_rate if sample_rate else 0.0
    rms = _frame_rms(samples, sample_rate)
    speech_mask = _speech_mask(rms)
    hop_s = 0.01
    speech_duration_s = min(duration_s, float(np.sum(speech_mask)) * hop_s)
    pause_duration_s = max(0.0, duration_s - speech_duration_s)
    pause_ratio = pause_duration_s / duration_s if duration_s else 0.0
    energy_mean = float(np.mean(rms)) if rms.size else 0.0
    energy_std = float(np.std(rms)) if rms.size else 0.0
    energy_variation = energy_std / energy_mean if energy_mean else 0.0
    pitch_features = _pitch_features(samples, sample_rate)

    text = transcription.text if transcription else (transcript_text or "")
    word_count = _word_count_from_transcription(transcription) if transcription else 0
    if not word_count and text:
        word_count = _count_text_units(text)
    speech_rate_wpm = word_count / speech_duration_s * 60 if speech_duration_s > 0 else None
    filler_count = _count_fillers(text)

    features: dict[str, float | int | str | None] = {
        "duration_s": round(duration_s, 3),
        "speech_duration_s": round(speech_duration_s, 3),
        "pause_duration_s": round(pause_duration_s, 3),
        "pause_ratio": round(pause_ratio, 4),
        "word_count": word_count,
        "speech_rate_wpm": None if speech_rate_wpm is None else round(speech_rate_wpm, 2),
        "filler_count": filler_count,
        "energy_mean": round(energy_mean, 6),
        "energy_variation": round(energy_variation, 4),
        **pitch_features,
    }
    return AudioFeatureResult(
        speech_delivery_score=score_speech_delivery(features),
        features=features,
    )


def score_speech_delivery(features: dict[str, float | int | str | None]) -> float:
    pause_ratio = float(features.get("pause_ratio") or 0.0)
    speech_rate_wpm = features.get("speech_rate_wpm")
    energy_variation = float(features.get("energy_variation") or 0.0)
    pitch_range_hz = features.get("pitch_range_hz")
    filler_count = int(features.get("filler_count") or 0)

    pause_score = _target_score(pause_ratio, target=0.18, tolerance=0.22)
    rate_score = 65.0 if speech_rate_wpm is None else _target_score(
        float(speech_rate_wpm), target=135.0, tolerance=90.0
    )
    energy_score = _target_score(energy_variation, target=0.55, tolerance=0.65)
    pitch_score = 65.0 if pitch_range_hz is None else _target_score(
        float(pitch_range_hz), target=90.0, tolerance=120.0
    )
    filler_penalty = min(18.0, filler_count * 2.0)
    score = (
        0.35 * rate_score
        + 0.30 * pause_score
        + 0.20 * energy_score
        + 0.15 * pitch_score
        - filler_penalty
    )
    return round(max(0.0, min(100.0, score)), 2)


def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV. Use ffmpeg extraction first.")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def _frame_rms(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = max(1, int(sample_rate * 0.03))
    hop_length = max(1, int(sample_rate * 0.01))
    if samples.size < frame_length:
        return np.array([float(np.sqrt(np.mean(samples * samples)))]) if samples.size else np.array([])
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop_length]
    return np.sqrt(np.mean(frames * frames, axis=1))


def _speech_mask(rms: np.ndarray) -> np.ndarray:
    if not rms.size:
        return np.array([], dtype=bool)
    threshold = max(float(np.percentile(rms, 60)) * 0.30, float(np.max(rms)) * 0.025, 1e-5)
    return rms > threshold


def _pitch_features(samples: np.ndarray, sample_rate: int) -> dict[str, float | int | str | None]:
    try:
        import librosa
    except ImportError:
        return {"pitch_mean_hz": None, "pitch_range_hz": None, "pitch_source": "unavailable"}

    if samples.size < sample_rate // 2:
        return {"pitch_mean_hz": None, "pitch_range_hz": None, "pitch_source": "too_short"}

    y = samples.astype(np.float32)
    pitches = librosa.yin(y, fmin=65, fmax=500, sr=sample_rate)
    voiced = pitches[np.isfinite(pitches)]
    if voiced.size == 0:
        return {"pitch_mean_hz": None, "pitch_range_hz": None, "pitch_source": "unvoiced"}

    low, high = np.percentile(voiced, [10, 90])
    return {
        "pitch_mean_hz": round(float(np.mean(voiced)), 2),
        "pitch_range_hz": round(float(high - low), 2),
        "pitch_source": "librosa_yin",
    }


def _word_count_from_transcription(transcription: TranscriptionResult | None) -> int:
    if transcription is None:
        return 0
    word_timings = [
        word.word
        for segment in transcription.segments
        for word in segment.words
        if word.word.strip()
    ]
    if word_timings:
        return len(word_timings)
    return _count_text_units(transcription.text)


def _count_fillers(text: str) -> int:
    normalized = text.lower()
    return sum(normalized.count(filler) for filler in FILLER_WORDS)


def _count_text_units(text: str) -> int:
    english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(english_words) + len(cjk_chars)


def _target_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - abs(value - target) / tolerance))
