"""Resumable, redacted YouTube source acquisition."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from .artifacts import StageMetadata, atomic_write_json, cache_matches
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .source_manifest import SourceRow


class AcquisitionStatus(str, Enum):
    """Terminal outcomes for an individual source acquisition."""

    ACQUIRED = "acquired"
    CACHED = "cached"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class AcquisitionResult:
    """Redacted, terminal result for one source row."""

    sample_id: str
    status: AcquisitionStatus
    video_id: str
    source_path: str | None
    metadata_path: str | None
    source_sha256: str | None
    attempts: int
    error_code: str | None
    error_summary: str | None
    source_kind: str = "youtube"
    source_locator: str | None = None
    local_basename: str | None = None
    duration_seconds: float | None = None


class YouTubeRunner(Protocol):
    """Small seam around yt-dlp; tests supply a file-writing fake."""

    def download(self, url: str, output_template: Path) -> Path:
        """Download a validated URL and return the acquired source path."""


class _YtDlpRunner:
    """Lazy yt-dlp adapter, kept out of network-disabled execution paths."""

    def download(self, url: str, output_template: Path) -> Path:
        try:
            import yt_dlp
        except ImportError as error:  # pragma: no cover - environment specific
            raise RuntimeError("yt_dlp_unavailable") from error

        options = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": str(output_template),
            "overwrites": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if not isinstance(info, dict):
                raise RuntimeError("invalid_yt_dlp_response")
            return Path(downloader.prepare_filename(info))


_TERMINAL_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("private_video", ("private_video", "private video")),
    ("removed_video", ("removed_video", "removed video", "video unavailable")),
    ("region_blocked", ("region_blocked", "not available in your country", "geo restricted")),
    ("age_restricted", ("age_restricted", "age restricted", "confirm your age")),
    (
        "authentication_required",
        ("authentication_required", "sign in", "login required", "authentication required"),
    ),
    ("copyright_blocked", ("copyright_blocked", "copyright", "blocked by copyright")),
)
_TRANSIENT_ERROR_PATTERNS = (
    "network_error",
    "network error",
    "connection",
    "timeout",
    "timed out",
    "http 5",
    "http_5",
    "rate limit",
    "too many requests",
    "temporary",
)
_SUMMARY_LIMIT = 240
_SAFE_ERROR_CODE = re.compile(r"[a-z0-9_]{1,64}\Z")
_SAMPLE_ID = re.compile(r"(?:S[0-9]{2}|N[0-9]{3})\Z")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class YouTubeAcquirer:
    """Acquire one strictly validated source row, without leaking its URL to metadata."""

    def __init__(
        self,
        *,
        runner: YouTubeRunner | None = None,
        max_retries: int = 2,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._runner = runner
        self._max_retries = max_retries
        self._sleeper = sleeper if sleeper is not None else _sleep

    def acquire(
        self, row: SourceRow, output_root: Path, allow_network: bool
    ) -> AcquisitionResult:
        """Return a terminal acquisition result, persisting only redacted metadata."""
        input_sha256 = sha256_bytes(
            canonical_json_bytes(
                {"sample_id": row.sample_id, "video_id": row.video_id, "source_url": row.source_url}
            )
        )
        safe_identifiers = _safe_sample_id(row.sample_id) and _safe_video_id(row.video_id)
        sample_id = row.sample_id if safe_identifiers else "invalid"
        video_id = row.video_id if safe_identifiers else "invalid"
        source_relative, metadata_relative = _relative_paths(
            sample_id, video_id, input_sha256, safe_identifiers
        )
        source_path = output_root / source_relative
        metadata_path = output_root / metadata_relative
        config_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "adapter": "yt_dlp",
                    "format": "bestaudio/best",
                    "keep_original_encoding": True,
                    "max_retries": self._max_retries,
                }
            )
        )

        if row.is_acquisition_eligible and safe_identifiers:
            cached = self._cached_result(
                sample_id,
                video_id,
                output_root,
                source_path,
                metadata_path,
                source_relative,
                metadata_relative,
                input_sha256,
                config_sha256,
            )
            if cached is not None:
                return cached

        if not allow_network:
            return self._record_result(
                result=AcquisitionResult(
                    sample_id=sample_id,
                    status=AcquisitionStatus.BLOCKED,
                    video_id=video_id,
                    source_path=None,
                    metadata_path=metadata_relative.as_posix(),
                    source_sha256=None,
                    attempts=0,
                    error_code="network_disabled",
                    error_summary="Network acquisition is disabled",
                ),
                metadata_path=metadata_path,
                input_sha256=input_sha256,
                config_sha256=config_sha256,
            )

        if not row.is_acquisition_eligible or not safe_identifiers:
            return self._record_result(
                result=AcquisitionResult(
                    sample_id=sample_id,
                    status=AcquisitionStatus.FAILED,
                    video_id=video_id,
                    source_path=None,
                    metadata_path=metadata_relative.as_posix(),
                    source_sha256=None,
                    attempts=0,
                    error_code="invalid_source_row",
                    error_summary="Source row is not eligible for acquisition",
                ),
                metadata_path=metadata_path,
                input_sha256=input_sha256,
                config_sha256=config_sha256,
            )

        runner = self._runner if self._runner is not None else _YtDlpRunner()
        attempts = 0
        for retry in range(self._max_retries + 1):
            attempts += 1
            try:
                downloaded_path = runner.download(
                    row.source_url, source_path.with_name(f"{video_id}.%(ext)s")
                )
                relative_download = downloaded_path.resolve().relative_to(output_root.resolve())
                if relative_download.parent != source_relative.parent or not downloaded_path.is_file():
                    raise RuntimeError("invalid_runner_output")
                result = AcquisitionResult(
                    sample_id=sample_id,
                    status=AcquisitionStatus.ACQUIRED,
                    video_id=video_id,
                    source_path=relative_download.as_posix(),
                    metadata_path=metadata_relative.as_posix(),
                    source_sha256=sha256_file(downloaded_path),
                    attempts=attempts,
                    error_code=None,
                    error_summary=None,
                )
                return self._record_result(result, metadata_path, input_sha256, config_sha256)
            except Exception as error:  # runners translate third-party errors at this boundary
                error_code, summary, transient = _classify_error(error)
                if transient and retry < self._max_retries:
                    self._sleeper(float(2**retry))
                    continue
                result = AcquisitionResult(
                    sample_id=sample_id,
                    status=AcquisitionStatus.FAILED,
                    video_id=video_id,
                    source_path=None,
                    metadata_path=metadata_relative.as_posix(),
                    source_sha256=None,
                    attempts=attempts,
                    error_code=error_code,
                    error_summary=summary,
                )
                return self._record_result(result, metadata_path, input_sha256, config_sha256)
        raise AssertionError("retry loop must return a terminal result")

    def _cached_result(
        self,
        sample_id: str,
        video_id: str,
        output_root: Path,
        source_path: Path,
        metadata_path: Path,
        source_relative: Path,
        metadata_relative: Path,
        input_sha256: str,
        config_sha256: str,
    ) -> AcquisitionResult | None:
        if (
            not cache_matches(metadata_path, input_sha256, config_sha256)
            or not source_path.parent.is_dir()
        ):
            return None
        try:
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            relative_source = metadata.get("source_path")
            source_sha256 = metadata.get("source_sha256")
            if not isinstance(relative_source, str) or not isinstance(source_sha256, str):
                return None
            relative_source_path = Path(relative_source)
            if relative_source_path.is_absolute() or relative_source_path.parent != source_relative.parent:
                return None
            candidate = (output_root / relative_source_path).resolve()
            candidate.relative_to(output_root.resolve())
            if not candidate.is_file() or sha256_file(candidate) != source_sha256:
                return None
            return AcquisitionResult(
                sample_id=sample_id,
                status=AcquisitionStatus.CACHED,
                video_id=video_id,
                source_path=relative_source,
                metadata_path=metadata_relative.as_posix(),
                source_sha256=source_sha256,
                attempts=0,
                error_code=None,
                error_summary=None,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _record_result(
        self,
        result: AcquisitionResult,
        metadata_path: Path,
        input_sha256: str,
        config_sha256: str,
    ) -> AcquisitionResult:
        payload = {
            "stage": "youtube_acquisition",
            **StageMetadata(input_sha256, config_sha256).to_dict(),
            **asdict(result),
            "status": result.status.value,
        }
        try:
            atomic_write_json(metadata_path, payload)
        except OSError:
            source_path, source_sha256 = _safe_source_details(
                result.source_path, result.source_sha256
            )
            return AcquisitionResult(
                sample_id=result.sample_id,
                status=AcquisitionStatus.FAILED,
                video_id=result.video_id,
                source_path=source_path,
                metadata_path=None,
                source_sha256=source_sha256,
                attempts=result.attempts,
                error_code="metadata_write_failed",
                error_summary="Acquisition metadata could not be written",
            )
        return result


def _relative_paths(
    sample_id: str, video_id: str, input_sha256: str, safe_identifiers: bool
) -> tuple[Path, Path]:
    if safe_identifiers:
        base = Path("source") / sample_id / video_id
        return base / f"{video_id}.source", base / "metadata.json"
    base = Path("source") / "invalid" / input_sha256[:16]
    return base / "source.source", base / "metadata.json"


def _safe_sample_id(value: str) -> bool:
    return _SAMPLE_ID.fullmatch(value) is not None


def _safe_video_id(value: str) -> bool:
    return _VIDEO_ID.fullmatch(value) is not None


def _safe_source_details(
    source_path: str | None, source_sha256: str | None
) -> tuple[str | None, str | None]:
    if source_path is None or source_sha256 is None or _SHA256.fullmatch(source_sha256) is None:
        return None, None
    relative_path = Path(source_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None, None
    return relative_path.as_posix(), source_sha256


def _classify_error(error: Exception) -> tuple[str, str, bool]:
    supplied_code = getattr(error, "code", "")
    text = str(error)
    normalized = f"{supplied_code} {text}".lower()
    for code, patterns in _TERMINAL_ERROR_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return code, _redacted_summary(text), False
    transient = any(pattern in normalized for pattern in _TRANSIENT_ERROR_PATTERNS)
    code = str(supplied_code).lower().strip()
    if not _SAFE_ERROR_CODE.fullmatch(code):
        code = "network_error" if transient else "acquisition_error"
    return code, _redacted_summary(text), transient


def _redacted_summary(value: str) -> str:
    collapsed = " ".join(value.split())
    collapsed = re.sub(r"https?://\S+", "[redacted-url]", collapsed, flags=re.IGNORECASE)
    collapsed = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s]+", "[redacted-path]", collapsed)
    return (collapsed or "Acquisition failed")[:_SUMMARY_LIMIT]


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)
