from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None


def ensure_media_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing media tool(s): "
            + ", ".join(missing)
            + ". Install ffmpeg and make sure it is on PATH."
        )


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> Path:
    ensure_media_tools()
    video = Path(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    _run(cmd)
    return output


def probe_video(video_path: str | Path) -> VideoInfo:
    ensure_media_tools()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = _run(cmd)
    raw = json.loads(completed.stdout or "{}")
    streams = raw.get("streams") or [{}]
    stream = streams[0] if streams else {}
    fmt = raw.get("format") or {}
    duration_s = _optional_float(stream.get("duration")) or _optional_float(fmt.get("duration"))
    frame_count = _optional_int(stream.get("nb_frames"))
    return VideoInfo(
        duration_s=duration_s,
        fps=_parse_frame_rate(stream.get("avg_frame_rate")),
        frame_count=frame_count,
        width=_optional_int(stream.get("width")),
        height=_optional_int(stream.get("height")),
    )


def find_video_files(video_dir: str | Path) -> list[Path]:
    root = Path(video_dir)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    return float(value)


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(float(value))
