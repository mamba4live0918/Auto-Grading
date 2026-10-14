import wave
from pathlib import Path

import numpy as np

from presentation_assessment.audio_features import extract_audio_features


def test_extract_audio_features_from_synthetic_wav(tmp_path: Path) -> None:
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate) * 0.2
    silence = np.zeros(sample_rate // 2)
    samples = np.concatenate([tone, silence, tone])
    wav_path = tmp_path / "sample.wav"
    pcm = (samples * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    result = extract_audio_features(wav_path)

    assert 0 <= result.speech_delivery_score <= 100
    assert result.features["duration_s"] == 2.5
    assert float(result.features["pause_ratio"]) > 0
