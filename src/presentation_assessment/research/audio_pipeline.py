"""Validated FFmpeg audio representations for the frozen research dataset."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_json
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file


ANALYSIS_SAMPLE_RATE = 16_000
ANALYSIS_CHANNELS = 1
ANALYSIS_CODEC = "pcm_s16le"
PEAK_TARGET_DBFS = -1.0
PEAK_TOLERANCE_DB = 0.1
LOUDNESS_TARGET_LUFS = -23.0
LOUDNESS_TOLERANCE_LU = 0.5
TRUE_PEAK_TARGET_DBTP = -1.0
TRUE_PEAK_CEILING_DBTP = -0.9
LOUDNESS_RANGE_TARGET_LU = 7.0
MIN_LOUDNESS_DURATION_SECONDS = 3.0
_ERROR_SUMMARY_LIMIT = 240
_NETWORK_PROTOCOL_BLACKLIST = ",".join(
    (
        "ftp",
        "gopher",
        "gophers",
        "hls",
        "http",
        "httpproxy",
        "https",
        "icecast",
        "mmsh",
        "mmst",
        "rist",
        "rtmp",
        "rtmpe",
        "rtmps",
        "rtmpt",
        "rtmpte",
        "rtmpts",
        "rtp",
        "rtsp",
        "sctp",
        "srt",
        "srtp",
        "tcp",
        "tls",
        "udp",
        "udplite",
    )
)


class AudioRepresentationStatus(str, Enum):
    """Terminal state of one generated audio representation."""

    CREATED = "created"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioProbe:
    """The minimum media properties required to validate a WAV representation."""

    duration_seconds: float
    sample_rate: int
    channels: int
    codec_name: str


@dataclass(frozen=True)
class LoudnessMeasurement:
    """FFmpeg loudnorm measurement, shared with recording-quality checks."""

    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float
    threshold_lufs: float
    target_offset_lu: float


@dataclass(frozen=True)
class AudioRepresentation:
    """A redacted terminal outcome for a single derivative."""

    name: str
    status: AudioRepresentationStatus
    path: Path | None
    duration_seconds: float | None
    peak_dbfs: float | None = None
    loudness: LoudnessMeasurement | None = None
    error_code: str | None = None
    error_summary: str | None = None


@dataclass(frozen=True)
class AudioArtifacts:
    """All representation outcomes for one source, including partial failures."""

    output_dir: Path
    source_duration_seconds: float | None
    analysis: AudioRepresentation
    peak_normalized: AudioRepresentation
    loudness_normalized: AudioRepresentation
    metadata_path: Path | None
    source_channel_count: int | None
    analysis_channel_count: int | None


class _PipelineError(RuntimeError):
    def __init__(self, code: str, summary: str) -> None:
        super().__init__(summary)
        self.code = code
        self.summary = summary


def build_audio_representations(
    source_path: Path,
    output_dir: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> AudioArtifacts:
    """Build analysis, peak, and EBU R128 derivatives without exposing source paths.

    A failure in one derivative is retained on that derivative and does not prevent
    independently valid output from being returned for the other representations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command_metadata: list[dict[str, object]] = []
    analysis_path = output_dir / "analysis.wav"
    peak_path = output_dir / "peak_normalized.wav"
    loudness_path = output_dir / "loudness_normalized.wav"

    source_duration: float | None
    source_channel_count: int | None
    try:
        source_probe = probe_audio(source_path, ffprobe=ffprobe)
        source_duration = source_probe.duration_seconds
        source_channel_count = source_probe.channels
    except _PipelineError:
        source_duration = None
        source_channel_count = None

    analysis = _create_analysis(
        source_path, analysis_path, ffmpeg, ffprobe, command_metadata
    )
    if analysis.status is AudioRepresentationStatus.CREATED:
        peak = _create_peak_normalized(analysis_path, peak_path, ffmpeg, ffprobe, command_metadata)
        loudness = _create_loudness_normalized(
            analysis_path,
            loudness_path,
            analysis.duration_seconds,
            ffmpeg,
            ffprobe,
            command_metadata,
        )
    else:
        peak = _dependency_failure("peak_normalized")
        loudness = _dependency_failure("loudness_normalized")

    provisional = AudioArtifacts(
        output_dir=output_dir,
        source_duration_seconds=source_duration,
        analysis=analysis,
        peak_normalized=peak,
        loudness_normalized=loudness,
        metadata_path=output_dir / "audio_metadata.json",
        source_channel_count=source_channel_count,
        analysis_channel_count=(
            ANALYSIS_CHANNELS
            if analysis.status is AudioRepresentationStatus.CREATED
            else None
        ),
    )
    validation = validate_audio_artifacts(provisional, ffmpeg=ffmpeg, ffprobe=ffprobe)
    artifacts = replace(
        provisional,
        analysis=validation["analysis"],
        peak_normalized=validation["peak_normalized"],
        loudness_normalized=validation["loudness_normalized"],
    )
    _write_metadata(artifacts, source_path, command_metadata)
    return artifacts


def validate_audio_artifacts(
    artifacts: AudioArtifacts, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"
) -> dict[str, AudioRepresentation]:
    """Re-measure each created artifact and return independently validated outcomes."""
    return {
        "analysis": _validate_analysis(artifacts.analysis, artifacts.source_duration_seconds, ffprobe),
        "peak_normalized": _validate_peak(
            artifacts.peak_normalized, artifacts.source_duration_seconds, ffmpeg, ffprobe
        ),
        "loudness_normalized": _validate_loudness(
            artifacts.loudness_normalized, artifacts.source_duration_seconds, ffmpeg, ffprobe
        ),
    }


def probe_audio(path: Path, *, ffprobe: str = "ffprobe") -> AudioProbe:
    """Read an audio stream's duration and canonical format through FFprobe."""
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        tool="ffprobe",
    )
    if completed.returncode != 0:
        raise _PipelineError("ffprobe_failed", "Audio properties could not be measured")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        duration = _finite_duration(stream.get("duration"))
        if duration is None:
            container = payload.get("format", {})
            duration = _finite_duration(
                container.get("duration") if isinstance(container, dict) else None
            )
        if duration is None:
            raise ValueError("audio duration is unavailable")
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec_name = str(stream["codec_name"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise _PipelineError("invalid_audio_probe", "Audio properties were incomplete") from error
    if not math.isfinite(duration) or duration < 0:
        raise _PipelineError("invalid_audio_probe", "Audio duration was invalid")
    return AudioProbe(duration, sample_rate, channels, codec_name)


def _finite_duration(value: object) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def measure_peak_dbfs(path: Path, *, ffmpeg: str = "ffmpeg") -> float | None:
    """Return FFmpeg's absolute sample peak in dBFS, or ``None`` for digital silence."""
    completed = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            "astats=metadata=1:reset=0",
            "-f",
            "null",
            "-",
        ],
        tool="ffmpeg",
    )
    if completed.returncode != 0:
        raise _PipelineError("peak_measurement_failed", "Absolute peak could not be measured")
    overall_start = completed.stderr.rfind("Overall")
    overall_output = (
        completed.stderr[overall_start:] if overall_start >= 0 else completed.stderr
    )
    matches = tuple(
        re.finditer(r"Peak level dB:\s*([-+]?\d+(?:\.\d+)?|-inf)", overall_output)
    )
    match = matches[-1] if matches else None
    if match is None:
        raise _PipelineError("peak_measurement_failed", "Absolute peak was unavailable")
    value = _finite_float(match.group(1))
    return value


def measure_loudness_true_peak(path: Path, *, ffmpeg: str = "ffmpeg") -> LoudnessMeasurement:
    """Parse one FFmpeg loudnorm JSON measurement for reusable LUFS/true-peak semantics."""
    completed = _run(_loudness_measurement_command(path, ffmpeg), tool="ffmpeg")
    if completed.returncode != 0:
        raise _PipelineError("loudness_measurement_failed", "Loudness could not be measured")
    return _parse_loudnorm_json(completed.stderr)


def _create_analysis(
    source_path: Path,
    analysis_path: Path,
    ffmpeg: str,
    ffprobe: str,
    command_metadata: list[dict[str, object]],
) -> AudioRepresentation:
    _remove_if_present(analysis_path)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source_path),
        "-ac",
        str(ANALYSIS_CHANNELS),
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-c:a",
        ANALYSIS_CODEC,
        str(analysis_path),
    ]
    try:
        completed = _run(command, tool="ffmpeg")
        command_metadata.append(_redacted_command(command, source_path, analysis_path))
        if completed.returncode != 0 or not analysis_path.is_file():
            _remove_if_present(analysis_path)
            return _failed("analysis", "ffmpeg_conversion_failed", "Analysis WAV could not be created")
        probe = probe_audio(analysis_path, ffprobe=ffprobe)
        return AudioRepresentation(
            "analysis", AudioRepresentationStatus.CREATED, analysis_path, probe.duration_seconds
        )
    except _PipelineError as error:
        command_metadata.append(_redacted_command(command, source_path, analysis_path))
        _remove_if_present(analysis_path)
        return _failed("analysis", error.code, error.summary)


def _create_peak_normalized(
    analysis_path: Path,
    peak_path: Path,
    ffmpeg: str,
    ffprobe: str,
    command_metadata: list[dict[str, object]],
) -> AudioRepresentation:
    try:
        peak = measure_peak_dbfs(analysis_path, ffmpeg=ffmpeg)
        if peak is None:
            return _failed("peak_normalized", "silent_input", "Peak normalization requires non-silent audio")
        gain = PEAK_TARGET_DBFS - peak
        _remove_if_present(peak_path)
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(analysis_path),
            "-af",
            f"volume={gain:.12f}dB",
            "-ac",
            str(ANALYSIS_CHANNELS),
            "-ar",
            str(ANALYSIS_SAMPLE_RATE),
            "-c:a",
            ANALYSIS_CODEC,
            str(peak_path),
        ]
        completed = _run(command, tool="ffmpeg")
        command_metadata.append(_redacted_command(command, analysis_path, peak_path))
        if completed.returncode != 0 or not peak_path.is_file():
            _remove_if_present(peak_path)
            return _failed(
                "peak_normalized", "ffmpeg_peak_normalization_failed", "Peak derivative could not be created"
            )
        duration = probe_audio(peak_path, ffprobe=ffprobe).duration_seconds
        return AudioRepresentation(
            "peak_normalized", AudioRepresentationStatus.CREATED, peak_path, duration
        )
    except _PipelineError as error:
        _remove_if_present(peak_path)
        return _failed("peak_normalized", error.code, error.summary)


def _create_loudness_normalized(
    analysis_path: Path,
    loudness_path: Path,
    duration: float | None,
    ffmpeg: str,
    ffprobe: str,
    command_metadata: list[dict[str, object]],
) -> AudioRepresentation:
    if duration is None or duration < MIN_LOUDNESS_DURATION_SECONDS:
        return _failed(
            "loudness_normalized", "silent_or_too_short", "Loudness normalization requires at least 3 seconds"
        )
    try:
        measurement_command = _loudness_measurement_command(analysis_path, ffmpeg)
        command_metadata.append(_redacted_command(measurement_command, analysis_path, Path("-")))
        measurement = measure_loudness_true_peak(analysis_path, ffmpeg=ffmpeg)
        if not _valid_loudness_measurement(measurement):
            return _failed(
                "loudness_normalized", "silent_or_too_short", "Loudness measurement was not finite"
            )
        _remove_if_present(loudness_path)
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(analysis_path),
            "-af",
            _loudnorm_filter(measurement),
            "-ac",
            str(ANALYSIS_CHANNELS),
            "-ar",
            str(ANALYSIS_SAMPLE_RATE),
            "-c:a",
            ANALYSIS_CODEC,
            str(loudness_path),
        ]
        completed = _run(command, tool="ffmpeg")
        command_metadata.append(_redacted_command(command, analysis_path, loudness_path))
        if completed.returncode != 0 or not loudness_path.is_file():
            _remove_if_present(loudness_path)
            return _failed(
                "loudness_normalized",
                "ffmpeg_loudness_normalization_failed",
                "Loudness derivative could not be created",
            )
        output_measurement = measure_loudness_true_peak(loudness_path, ffmpeg=ffmpeg)
        duration_seconds = probe_audio(loudness_path, ffprobe=ffprobe).duration_seconds
        return AudioRepresentation(
            "loudness_normalized",
            AudioRepresentationStatus.CREATED,
            loudness_path,
            duration_seconds,
            loudness=output_measurement,
        )
    except _PipelineError as error:
        _remove_if_present(loudness_path)
        if error.code == "invalid_loudness_measurement":
            return _failed(
                "loudness_normalized", "silent_or_too_short", "Loudness measurement was not finite"
            )
        return _failed("loudness_normalized", error.code, error.summary)


def _validate_analysis(
    artifact: AudioRepresentation, source_duration: float | None, ffprobe: str
) -> AudioRepresentation:
    if artifact.status is AudioRepresentationStatus.FAILED or artifact.path is None:
        return artifact
    try:
        probe = probe_audio(artifact.path, ffprobe=ffprobe)
        valid_format = (
            probe.sample_rate == ANALYSIS_SAMPLE_RATE
            and probe.channels == ANALYSIS_CHANNELS
            and probe.codec_name == ANALYSIS_CODEC
        )
        valid_duration = source_duration is not None and abs(probe.duration_seconds - source_duration) <= 0.05
        if not valid_format or not valid_duration:
            return _failed("analysis", "analysis_validation_failed", "Analysis WAV did not meet its format contract")
        return replace(artifact, duration_seconds=probe.duration_seconds)
    except _PipelineError as error:
        return _failed("analysis", error.code, error.summary)


def _validate_peak(
    artifact: AudioRepresentation, source_duration: float | None, ffmpeg: str, ffprobe: str
) -> AudioRepresentation:
    if artifact.status is AudioRepresentationStatus.FAILED or artifact.path is None:
        return artifact
    try:
        if not _canonical_duration_matches(artifact.path, source_duration, ffprobe):
            return _failed(
                "peak_normalized", "duration_validation_failed", "Peak derivative duration changed"
            )
        peak = measure_peak_dbfs(artifact.path, ffmpeg=ffmpeg)
        if peak is None or abs(peak - PEAK_TARGET_DBFS) > PEAK_TOLERANCE_DB:
            return _failed("peak_normalized", "peak_validation_failed", "Peak derivative missed its target")
        return replace(artifact, peak_dbfs=peak)
    except _PipelineError as error:
        return _failed("peak_normalized", error.code, error.summary)


def _validate_loudness(
    artifact: AudioRepresentation, source_duration: float | None, ffmpeg: str, ffprobe: str
) -> AudioRepresentation:
    if artifact.status is AudioRepresentationStatus.FAILED or artifact.path is None:
        return artifact
    try:
        if not _canonical_duration_matches(artifact.path, source_duration, ffprobe):
            return _failed(
                "loudness_normalized", "duration_validation_failed", "Loudness derivative duration changed"
            )
        measurement = measure_loudness_true_peak(artifact.path, ffmpeg=ffmpeg)
        if (
            not _valid_loudness_measurement(measurement)
            or abs(measurement.integrated_lufs - LOUDNESS_TARGET_LUFS) > LOUDNESS_TOLERANCE_LU
            or measurement.true_peak_dbtp > TRUE_PEAK_CEILING_DBTP
        ):
            return _failed(
                "loudness_normalized", "loudness_validation_failed", "Loudness derivative missed its target"
            )
        return replace(artifact, loudness=measurement)
    except _PipelineError as error:
        return _failed("loudness_normalized", error.code, error.summary)


def _canonical_duration_matches(path: Path, source_duration: float | None, ffprobe: str) -> bool:
    if source_duration is None:
        return False
    probe = probe_audio(path, ffprobe=ffprobe)
    return (
        probe.sample_rate == ANALYSIS_SAMPLE_RATE
        and probe.channels == ANALYSIS_CHANNELS
        and probe.codec_name == ANALYSIS_CODEC
        and abs(probe.duration_seconds - source_duration) <= 0.05
    )


def _loudnorm_filter(measurement: LoudnessMeasurement | None = None) -> str:
    base = f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={TRUE_PEAK_TARGET_DBTP}:LRA={LOUDNESS_RANGE_TARGET_LU}"
    if measurement is None:
        return f"{base}:print_format=json"
    return (
        f"{base}:measured_I={measurement.integrated_lufs}:measured_TP={measurement.true_peak_dbtp}"
        f":measured_LRA={measurement.loudness_range_lu}:measured_thresh={measurement.threshold_lufs}"
        f":offset={measurement.target_offset_lu}:linear=true:print_format=json"
    )


def _loudness_measurement_command(path: Path, ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-af",
        _loudnorm_filter(),
        "-f",
        "null",
        "-",
    ]


def _parse_loudnorm_json(stderr: str) -> LoudnessMeasurement:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        raise _PipelineError("invalid_loudness_measurement", "FFmpeg loudness JSON was unavailable")
    try:
        payload = json.loads(stderr[start : end + 1])
        values = {
            "integrated_lufs": _finite_float(payload["input_i"]),
            "true_peak_dbtp": _finite_float(payload["input_tp"]),
            "loudness_range_lu": _finite_float(payload["input_lra"]),
            "threshold_lufs": _finite_float(payload["input_thresh"]),
            "target_offset_lu": _finite_float(payload["target_offset"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise _PipelineError("invalid_loudness_measurement", "FFmpeg loudness JSON was invalid") from error
    if any(value is None for value in values.values()):
        raise _PipelineError("invalid_loudness_measurement", "FFmpeg loudness measurement was non-finite")
    return LoudnessMeasurement(**values)  # type: ignore[arg-type]


def _finite_float(value: object) -> float | None:
    try:
        converted = float(str(value))
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _valid_loudness_measurement(measurement: LoudnessMeasurement) -> bool:
    return all(math.isfinite(value) for value in asdict(measurement).values())


def _run(command: Sequence[str], *, tool: str) -> subprocess.CompletedProcess[str]:
    secured_command = _offline_source_command(command, tool)
    try:
        return subprocess.run(
            secured_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except FileNotFoundError as error:
        raise _PipelineError(f"{tool}_unavailable", f"{tool.upper()} is unavailable") from error
    except subprocess.TimeoutExpired as error:
        raise _PipelineError(f"{tool}_timeout", f"{tool.upper()} did not finish in time") from error
    except OSError as error:
        raise _PipelineError(f"{tool}_execution_failed", f"{tool.upper()} could not be run") from error


def _offline_source_command(command: Sequence[str], tool: str) -> list[str]:
    secured = list(command)
    policy = [
        "-protocol_whitelist",
        "file",
        "-protocol_blacklist",
        _NETWORK_PROTOCOL_BLACKLIST,
    ]
    if tool == "ffmpeg":
        try:
            input_index = secured.index("-i")
        except ValueError as error:
            raise _PipelineError(
                "invalid_ffmpeg_command", "FFmpeg source command has no input boundary"
            ) from error
    elif tool == "ffprobe":
        if len(secured) < 2:
            raise _PipelineError(
                "invalid_ffprobe_command", "FFprobe source command has no input boundary"
            )
        input_index = len(secured) - 1
    else:
        raise _PipelineError("invalid_media_tool", "Unsupported local media tool")
    return [*secured[:input_index], *policy, *secured[input_index:]]


def _redacted_command(command: Sequence[str], input_path: Path, output_path: Path) -> dict[str, object]:
    safe_arguments: list[str] = []
    for argument in command[1:]:
        if argument == str(input_path):
            safe_arguments.append("<input-audio>")
        elif argument == str(output_path):
            safe_arguments.append(output_path.name)
        elif Path(argument).is_absolute():
            safe_arguments.append("<redacted-path>")
        else:
            safe_arguments.append(argument[:_ERROR_SUMMARY_LIMIT])
    return {"tool": Path(command[0]).name, "arguments": safe_arguments}


def _failed(name: str, code: str, summary: str) -> AudioRepresentation:
    return AudioRepresentation(
        name,
        AudioRepresentationStatus.FAILED,
        None,
        None,
        error_code=code,
        error_summary=_bounded_summary(summary),
    )


def _dependency_failure(name: str) -> AudioRepresentation:
    return _failed(name, "analysis_unavailable", "Analysis WAV was unavailable")


def _remove_if_present(path: Path) -> None:
    path.unlink(missing_ok=True)


def _bounded_summary(value: str) -> str:
    return " ".join(value.split())[:_ERROR_SUMMARY_LIMIT]


def _write_metadata(
    artifacts: AudioArtifacts, source_path: Path, command_metadata: list[dict[str, object]]
) -> None:
    if artifacts.metadata_path is None:
        return
    try:
        source_sha256 = sha256_file(source_path)
    except OSError:
        source_sha256 = None
    payload: Mapping[str, Any] = {
        "stage": "audio_representations",
        "input_sha256": source_sha256,
        "config_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "analysis": {
                        "sample_rate": ANALYSIS_SAMPLE_RATE,
                        "channels": ANALYSIS_CHANNELS,
                        "codec": ANALYSIS_CODEC,
                    },
                    "peak_target_dbfs": PEAK_TARGET_DBFS,
                    "loudness_target_lufs": LOUDNESS_TARGET_LUFS,
                    "true_peak_target_dbtp": TRUE_PEAK_TARGET_DBTP,
                    "loudness_range_lu": LOUDNESS_RANGE_TARGET_LU,
                }
            )
        ),
        "source_duration_seconds": artifacts.source_duration_seconds,
        "source_channel_count": artifacts.source_channel_count,
        "analysis_channel_count": artifacts.analysis_channel_count,
        "commands": command_metadata[-20:],
        "representations": {
            name: _metadata_representation(item, artifacts.output_dir)
            for name, item in {
                "analysis": artifacts.analysis,
                "peak_normalized": artifacts.peak_normalized,
                "loudness_normalized": artifacts.loudness_normalized,
            }.items()
        },
    }
    atomic_write_json(artifacts.metadata_path, payload)


def _metadata_representation(item: AudioRepresentation, output_dir: Path) -> dict[str, object]:
    relative_path: str | None = None
    if item.path is not None:
        try:
            relative_path = item.path.relative_to(output_dir).as_posix()
        except ValueError:
            relative_path = None
    payload: dict[str, object] = {
        "status": item.status.value,
        "path": relative_path,
        "duration_seconds": item.duration_seconds,
        "peak_dbfs": item.peak_dbfs,
        "error_code": item.error_code,
        "error_summary": item.error_summary,
    }
    if item.loudness is not None:
        payload["loudness"] = asdict(item.loudness)
    return payload
