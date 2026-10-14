import json
import math
import subprocess
import wave
from pathlib import Path

import pytest
import presentation_assessment.research.audio_pipeline as audio_pipeline_module

from presentation_assessment.research.audio_pipeline import (
    AudioRepresentationStatus,
    build_audio_representations,
    measure_loudness_true_peak,
    measure_peak_dbfs,
    probe_audio,
    validate_audio_artifacts,
)


SAMPLE_RATE = 16_000


def write_pcm_wav(path: Path, samples: list[int]) -> Path:
    """Write a deterministic mono 16-bit PCM fixture without audio libraries."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return path


def write_stereo_pcm_wav(path: Path, samples: list[int]) -> Path:
    """Write the same deterministic signal to two source channels."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(
            b"".join(
                sample.to_bytes(2, "little", signed=True) * 2 for sample in samples
            )
        )
    return path


@pytest.mark.parametrize(
    ("tool", "command", "input_token"),
    [
        ("ffprobe", ["ffprobe", "-v", "error", "source.mp4"], "source.mp4"),
        ("ffmpeg", ["ffmpeg", "-nostdin", "-i", "source.mp4", "out.wav"], "-i"),
    ],
)
def test_source_command_boundary_injects_file_only_protocol_policy(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    command: list[str],
    input_token: str,
) -> None:
    """Break caught: a source command can reach FFmpeg without explicit offline protocols."""
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 1, "", "synthetic rejection")

    monkeypatch.setattr(audio_pipeline_module.subprocess, "run", run)

    audio_pipeline_module._run(command, tool=tool)

    secured = calls[0]
    input_index = secured.index(input_token)
    whitelist_index = secured.index("-protocol_whitelist")
    blacklist_index = secured.index("-protocol_blacklist")
    assert whitelist_index < input_index
    assert blacklist_index < input_index
    assert secured[whitelist_index + 1] == "file"
    blacklisted = set(secured[blacklist_index + 1].split(","))
    assert {"http", "https", "tcp", "tls", "udp", "rtsp", "rtmp"} <= blacklisted


def sine_samples(*, duration_seconds: float, amplitude: float = 0.25) -> list[int]:
    count = round(SAMPLE_RATE * duration_seconds)
    return [
        round(32_767 * amplitude * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE))
        for index in range(count)
    ]


def test_builds_three_validated_representations_from_a_deterministic_sine(tmp_path: Path) -> None:
    """Break caught: omitting conversion or either normalisation derivative."""
    source = write_pcm_wav(tmp_path / "private-source.wav", sine_samples(duration_seconds=4.0))

    artifacts = build_audio_representations(source, tmp_path / "outputs")

    assert artifacts.analysis.status is AudioRepresentationStatus.CREATED
    assert artifacts.peak_normalized.status is AudioRepresentationStatus.CREATED
    assert artifacts.loudness_normalized.status is AudioRepresentationStatus.CREATED
    assert artifacts.analysis.path is not None
    analysis = probe_audio(artifacts.analysis.path)
    assert analysis.sample_rate == 16_000
    assert analysis.channels == 1
    assert analysis.codec_name == "pcm_s16le"
    assert analysis.duration_seconds == pytest.approx(4.0, abs=0.05)
    assert measure_peak_dbfs(artifacts.analysis.path) == pytest.approx(
        measure_peak_dbfs(source), abs=0.1
    )
    assert measure_peak_dbfs(artifacts.peak_normalized.path) == pytest.approx(-1.0, abs=0.1)
    loudness = measure_loudness_true_peak(artifacts.loudness_normalized.path)
    assert loudness.integrated_lufs == pytest.approx(-23.0, abs=0.5)
    assert loudness.true_peak_dbtp <= -0.9
    validation = validate_audio_artifacts(artifacts)
    assert {name: item.status for name, item in validation.items()} == {
        "analysis": AudioRepresentationStatus.CREATED,
        "peak_normalized": AudioRepresentationStatus.CREATED,
        "loudness_normalized": AudioRepresentationStatus.CREATED,
    }


def test_stereo_source_channel_count_survives_mono_analysis_conversion(
    tmp_path: Path,
) -> None:
    """Break caught: source provenance is inferred from the mono derivative."""
    source = write_stereo_pcm_wav(
        tmp_path / "stereo-source.wav", sine_samples(duration_seconds=4.0)
    )

    artifacts = build_audio_representations(source, tmp_path / "outputs")

    assert artifacts.source_channel_count == 2
    assert artifacts.analysis_channel_count == 1
    metadata = json.loads(
        (tmp_path / "outputs" / "audio_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_channel_count"] == 2
    assert metadata["analysis_channel_count"] == 1


def test_probe_audio_uses_container_duration_when_stream_duration_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: WebM/Opus with stream duration N/A rejects every converted WAV."""
    source = tmp_path / "source.webm"
    source.write_bytes(b"synthetic-container")
    payload = {
        "streams": [
            {
                "codec_name": "opus",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "N/A",
            }
        ],
        "format": {"duration": "12.345"},
    }

    monkeypatch.setattr(
        audio_pipeline_module,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    probe = probe_audio(source)

    assert probe.duration_seconds == pytest.approx(12.345)
    assert probe.sample_rate == 48_000
    assert probe.channels == 2
    assert probe.codec_name == "opus"


def test_peak_measurement_uses_astats_overall_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: stereo peak measurement returns the first channel block."""
    stderr = """
[Parsed_astats_0] Channel: 1
[Parsed_astats_0] Peak level dB: -8.0
[Parsed_astats_0] Channel: 2
[Parsed_astats_0] Peak level dB: -3.0
[Parsed_astats_0] Overall
[Parsed_astats_0] Peak level dB: -1.5
"""
    monkeypatch.setattr(
        audio_pipeline_module,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout="", stderr=stderr
        ),
    )

    assert measure_peak_dbfs(tmp_path / "synthetic.wav") == pytest.approx(-1.5)


def test_clipped_source_still_preserves_duration_and_has_a_peak_normalized_derivative(
    tmp_path: Path,
) -> None:
    """Break caught: a clipped input causes the complete sample to be discarded."""
    clipped = [32_767 if index % 2 else -32_768 for index in range(SAMPLE_RATE * 4)]
    source = write_pcm_wav(tmp_path / "clipped.wav", clipped)

    artifacts = build_audio_representations(source, tmp_path / "outputs")

    assert artifacts.analysis.status is AudioRepresentationStatus.CREATED
    assert artifacts.peak_normalized.status is AudioRepresentationStatus.CREATED
    assert artifacts.analysis.duration_seconds == pytest.approx(4.0, abs=0.05)
    assert measure_peak_dbfs(artifacts.peak_normalized.path) == pytest.approx(-1.0, abs=0.1)


def test_quiet_nonzero_lsb_signal_is_peak_normalized_not_misclassified_as_silence(
    tmp_path: Path,
) -> None:
    """Break caught: finite ±1 LSB audio is discarded by a silence threshold."""
    source = write_pcm_wav(
        tmp_path / "quiet-nonzero.wav",
        [1 if index % 2 else -1 for index in range(SAMPLE_RATE * 4)],
    )

    artifacts = build_audio_representations(source, tmp_path / "outputs")

    assert measure_peak_dbfs(source) is not None
    assert artifacts.peak_normalized.status is AudioRepresentationStatus.CREATED
    assert artifacts.peak_normalized.path is not None
    assert measure_peak_dbfs(artifacts.peak_normalized.path) == pytest.approx(-1.0, abs=0.1)


@pytest.mark.parametrize(
    ("name", "samples"),
    [
        ("silence", [0] * (SAMPLE_RATE * 4)),
        ("too-short", sine_samples(duration_seconds=0.5)),
    ],
)
def test_silent_or_too_short_loudness_analysis_is_explicit_without_aborting_analysis(
    tmp_path: Path, name: str, samples: list[int]
) -> None:
    """Break caught: invalid loudness is faked or aborts usable representations."""
    source = write_pcm_wav(tmp_path / f"{name}.wav", samples)

    artifacts = build_audio_representations(source, tmp_path / "outputs")

    assert artifacts.analysis.status is AudioRepresentationStatus.CREATED
    if name == "silence":
        assert artifacts.peak_normalized.status is AudioRepresentationStatus.FAILED
        assert artifacts.peak_normalized.error_code == "silent_input"
    else:
        assert artifacts.peak_normalized.status is AudioRepresentationStatus.CREATED
    assert artifacts.loudness_normalized.status is AudioRepresentationStatus.FAILED
    assert artifacts.loudness_normalized.error_code == "silent_or_too_short"
    assert artifacts.loudness_normalized.path is None
    assert artifacts.loudness_normalized.error_summary


def test_command_metadata_and_failures_do_not_leak_private_absolute_paths(tmp_path: Path) -> None:
    """Break caught: a subprocess error or provenance record retains the source path."""
    private_root = tmp_path / "private" / "advisor"
    private_root.mkdir(parents=True)
    source = write_pcm_wav(private_root / "source.wav", sine_samples(duration_seconds=4.0))

    artifacts = build_audio_representations(
        source, tmp_path / "outputs", ffmpeg="missing-ffmpeg-for-test"
    )

    assert artifacts.analysis.status is AudioRepresentationStatus.FAILED
    assert artifacts.peak_normalized.status is AudioRepresentationStatus.FAILED
    assert artifacts.loudness_normalized.status is AudioRepresentationStatus.FAILED
    assert artifacts.analysis.error_code == "ffmpeg_unavailable"
    metadata = json.loads((tmp_path / "outputs" / "audio_metadata.json").read_text(encoding="utf-8"))
    serialised = json.dumps(metadata)
    assert str(private_root) not in serialised
    assert str(source) not in serialised
    assert "private" not in serialised.lower()


def test_validation_rejects_a_derivative_with_a_changed_duration(tmp_path: Path) -> None:
    """Break caught: a truncated derivative is accepted after generation."""
    source = write_pcm_wav(tmp_path / "source.wav", sine_samples(duration_seconds=4.0))
    artifacts = build_audio_representations(source, tmp_path / "outputs")
    assert artifacts.peak_normalized.path is not None
    with wave.open(str(artifacts.peak_normalized.path), "rb") as original:
        frames = original.readframes(SAMPLE_RATE)
    with wave.open(str(artifacts.peak_normalized.path), "wb") as truncated:
        truncated.setnchannels(1)
        truncated.setsampwidth(2)
        truncated.setframerate(SAMPLE_RATE)
        truncated.writeframes(frames)

    validation = validate_audio_artifacts(artifacts)

    assert validation["peak_normalized"].status is AudioRepresentationStatus.FAILED
    assert validation["peak_normalized"].error_code == "duration_validation_failed"


def test_metadata_records_both_redacted_loudnorm_passes(tmp_path: Path) -> None:
    """Break caught: provenance cannot establish that loudness used a two-pass process."""
    source = write_pcm_wav(tmp_path / "source.wav", sine_samples(duration_seconds=4.0))

    build_audio_representations(source, tmp_path / "outputs")

    metadata = json.loads((tmp_path / "outputs" / "audio_metadata.json").read_text(encoding="utf-8"))
    filters = [
        command["arguments"][index + 1]
        for command in metadata["commands"]
        if "-af" in command["arguments"]
        for index, argument in enumerate(command["arguments"])
        if argument == "-af"
    ]
    loudnorm_filters = [item for item in filters if item.startswith("loudnorm=")]
    assert len(loudnorm_filters) == 2
    assert any("measured_I=" not in item for item in loudnorm_filters)
    assert any("measured_I=" in item and "linear=true" in item for item in loudnorm_filters)
