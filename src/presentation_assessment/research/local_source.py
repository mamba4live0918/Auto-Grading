"""Strict, offline local-file source catalog and in-memory path resolution."""

from __future__ import annotations

import csv
import math
import stat
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

from .acquisition import AcquisitionResult, AcquisitionStatus
from .audio_pipeline import build_audio_representations, probe_audio
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .source_manifest import SourceRow, VerificationStatus, _normalise_youtube_video_id


LOCAL_CATALOG_FIELDS = (
    "local_row_id",
    "source_url",
    "video_id",
    "video_title",
    "total_score",
    "duration_seconds",
    "local_filename",
    "source_sha256",
    "status",
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "video_root",
        "catalog_path",
        "expected_total_rows",
        "expected_available_rows",
        "expected_unavailable_rows",
        "duration_tolerance_seconds",
    }
)
_LOCAL_ROW_ID = re.compile(r"local-r[0-9]{3}\Z")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class LocalAvailability(str, Enum):
    """Whether one catalog source is backed by a validated local file."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LocalSourceConfig:
    """Private runtime paths and non-sensitive validation expectations."""

    video_root: Path
    catalog_path: Path
    expected_total_rows: int
    expected_available_rows: int
    expected_unavailable_rows: int
    duration_tolerance_seconds: float


@dataclass(frozen=True)
class LocalCatalogRow:
    """One strict local catalog record with no absolute path value."""

    local_row_id: str
    source_url: str
    video_id: str
    video_title: str
    total_score: float | None
    duration_seconds: float | None
    local_filename: str
    source_sha256: str
    status: LocalAvailability

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", LocalAvailability(self.status))


@dataclass(frozen=True)
class LocalSourceCatalog:
    """Validated rows plus an identity digest that excludes private roots."""

    config: LocalSourceConfig
    rows: tuple[LocalCatalogRow, ...]
    identity_sha256: str

    @property
    def available_count(self) -> int:
        return sum(row.status is LocalAvailability.AVAILABLE for row in self.rows)

    @property
    def unavailable_count(self) -> int:
        return sum(row.status is LocalAvailability.UNAVAILABLE for row in self.rows)


@dataclass(frozen=True)
class LocalSourceReference:
    """Opaque local source input passed only between acquisition and audio."""

    locator: str
    source_sha256: str


@dataclass(frozen=True)
class TrustedSourceIdentity(Mapping[str, str]):
    """Immutable catalog-approved source identity carried through the pipeline."""

    source_kind: str
    source_row_id: str
    video_id: str
    local_basename: str
    source_sha256: str
    catalog_identity_sha256: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) for value in self.to_dict().values()):
            raise TypeError("trusted source identity values must be strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_kind": self.source_kind,
            "source_row_id": self.source_row_id,
            "video_id": self.video_id,
            "local_basename": self.local_basename,
            "source_sha256": self.source_sha256,
            "catalog_identity_sha256": self.catalog_identity_sha256,
        }

    def __getitem__(self, key: str) -> str:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 6


def local_source_reference_valid(
    payload: Mapping[str, object],
    *,
    expected_local_row_id: object,
    expected_basename: object,
    expected_source_sha256: object,
) -> bool:
    """Validate one canonical path-free locator against its source catalog row."""
    if (
        payload.get("source_kind") != "local_file"
        or not isinstance(expected_local_row_id, str)
        or _LOCAL_ROW_ID.fullmatch(expected_local_row_id) is None
        or not isinstance(expected_basename, str)
        or not _safe_local_basename(expected_basename)
        or not isinstance(expected_source_sha256, str)
        or _SHA256.fullmatch(expected_source_sha256) is None
    ):
        return False
    locator = payload.get("source_locator")
    basename = payload.get("local_basename")
    source_sha256 = payload.get("source_sha256")
    if (
        not isinstance(locator, str)
        or basename != expected_basename
        or source_sha256 != expected_source_sha256
    ):
        return False
    if not locator or "\\" in locator or "\x00" in locator:
        return False
    parts = locator.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        return False
    if (
        parts[0] != "local-file"
        or parts[1] != expected_local_row_id
        or parts[2] != expected_basename
    ):
        return False
    return True


def _safe_local_basename(value: str) -> bool:
    if (
        not value
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or value in {".", ".."}
    ):
        return False
    windows_name = PureWindowsPath(value)
    posix_name = PurePosixPath(value)
    return (
        not windows_name.drive
        and not windows_name.is_absolute()
        and not windows_name.is_reserved()
        and not posix_name.is_absolute()
        and Path(value).name == value
        and Path(value).suffix.lower() == ".mp4"
    )


@dataclass(frozen=True)
class _LocalFileGuard:
    entry_path: Path
    resolved_path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def load_local_source_config(path: Path) -> LocalSourceConfig:
    """Load an exact-schema private YAML config without normalizing relative paths."""
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("local source config is unavailable or invalid") from error
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        raise ValueError("local source config must use the exact schema")
    if payload.get("schema_version") != 1:
        raise ValueError("local source config schema_version must be 1")
    video_root = _absolute_config_path(payload.get("video_root"), "video_root")
    catalog_path = _absolute_config_path(payload.get("catalog_path"), "catalog_path")
    counts = {
        key: payload.get(key)
        for key in (
            "expected_total_rows",
            "expected_available_rows",
            "expected_unavailable_rows",
        )
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("local source expected counts must be non-negative integers")
    if counts["expected_available_rows"] + counts["expected_unavailable_rows"] != counts[
        "expected_total_rows"
    ]:
        raise ValueError("local source expected counts must sum to expected_total_rows")
    tolerance = payload.get("duration_tolerance_seconds")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ValueError("duration_tolerance_seconds must be a finite non-negative number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("duration_tolerance_seconds must be a finite non-negative number")
    return LocalSourceConfig(
        video_root=video_root,
        catalog_path=catalog_path,
        expected_total_rows=counts["expected_total_rows"],
        expected_available_rows=counts["expected_available_rows"],
        expected_unavailable_rows=counts["expected_unavailable_rows"],
        duration_tolerance_seconds=tolerance,
    )


def load_local_catalog(
    config: LocalSourceConfig,
    *,
    duration_probe: Callable[[Path], float] | None = None,
) -> LocalSourceCatalog:
    """Load, validate, hash, and probe a strict UTF-8 local-file catalog."""
    probe = duration_probe or _probe_duration
    try:
        root = config.video_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("video_root must be an existing directory")
        with config.catalog_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != LOCAL_CATALOG_FIELDS:
                raise ValueError("local catalog columns must exactly match the required schema")
            rows = tuple(
                _parse_catalog_row(raw, number, root, config, probe)
                for number, raw in enumerate(reader, 2)
            )
    except FileNotFoundError as error:
        raise ValueError("local source root or catalog is unavailable") from error
    except UnicodeError as error:
        raise ValueError("local catalog must be UTF-8") from error
    except OSError as error:
        raise ValueError("local source root or catalog is unavailable") from error
    _validate_catalog_set(rows, config)
    identity_payload = {
        "schema_version": 1,
        "duration_tolerance_seconds": config.duration_tolerance_seconds,
        "expected_counts": {
            "total": config.expected_total_rows,
            "available": config.expected_available_rows,
            "unavailable": config.expected_unavailable_rows,
        },
        "rows": [
            {
                **asdict(row),
                "status": row.status.value,
            }
            for row in rows
        ],
    }
    return LocalSourceCatalog(
        config=config,
        rows=rows,
        identity_sha256=sha256_bytes(canonical_json_bytes(identity_payload)),
    )


class LocalSourceResolver:
    """Resolve validated opaque locators inside one private root and recheck bytes."""

    def __init__(self, catalog: LocalSourceCatalog) -> None:
        self._root = catalog.config.video_root.resolve(strict=True)
        self._rows = MappingProxyType({row.local_row_id: row for row in catalog.rows})
        self._identity_sha256 = catalog.identity_sha256

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def row(self, local_row_id: str) -> LocalCatalogRow:
        try:
            return self._rows[local_row_id]
        except KeyError as error:
            raise ValueError("unknown local source row") from error

    def locator_for(self, local_row_id: str) -> str:
        row = self.row(local_row_id)
        if row.status is not LocalAvailability.AVAILABLE or not row.local_filename:
            raise ValueError("local source is unavailable")
        return f"local-file/{row.local_row_id}/{row.local_filename}"

    def resolve(self, locator: str, expected_sha256: str) -> Path:
        return self.resolve_guarded(locator, expected_sha256).resolved_path

    def resolve_guarded(self, locator: str, expected_sha256: str) -> _LocalFileGuard:
        parts = Path(str(locator)).parts
        if (
            len(parts) != 3
            or parts[0] != "local-file"
            or _LOCAL_ROW_ID.fullmatch(parts[1]) is None
            or Path(parts[2]).name != parts[2]
            or parts[2] in {".", ".."}
        ):
            raise ValueError("invalid local source locator")
        row = self.row(parts[1])
        if row.local_filename != parts[2] or expected_sha256 != row.source_sha256:
            raise ValueError("invalid local source locator")
        return _capture_local_file(self._root, row.local_filename, expected_sha256)

    def verify_unchanged(self, guard: _LocalFileGuard) -> None:
        _verify_local_file_guard(guard)


class LocalFileAcquirer:
    """Hash a local source in place and return a terminal, path-free result."""

    def __init__(self, resolver: LocalSourceResolver) -> None:
        self._resolver = resolver

    @property
    def catalog_identity_sha256(self) -> str:
        return self._resolver.identity_sha256

    def trusted_identity(self, row: SourceRow) -> TrustedSourceIdentity | None:
        """Return one catalog-approved identity without reading media bytes."""
        try:
            catalog_row = self._resolver.row(row.source_row_id)
        except ValueError:
            return None
        if (
            row.video_id != catalog_row.video_id
            or _normalise_youtube_video_id(row.source_url) != catalog_row.video_id
            or _normalise_youtube_video_id(catalog_row.source_url)
            != catalog_row.video_id
        ):
            return None
        if catalog_row.status is LocalAvailability.AVAILABLE:
            basename = catalog_row.local_filename
            source_sha256 = catalog_row.source_sha256
        else:
            basename = ""
            source_sha256 = ""
        return TrustedSourceIdentity(
            source_kind="local_file",
            source_row_id=catalog_row.local_row_id,
            video_id=catalog_row.video_id,
            local_basename=basename,
            source_sha256=source_sha256,
            catalog_identity_sha256=self.catalog_identity_sha256,
        )

    def acquire(
        self, row: SourceRow, output_root: Path, allow_network: bool
    ) -> AcquisitionResult:
        del output_root, allow_network
        try:
            catalog_row = self._resolver.row(row.source_row_id)
        except ValueError:
            return _local_failure(row, "local_source_unknown", "Local source row is unavailable")
        if (
            row.verification_status is not VerificationStatus.VERIFIED
            or catalog_row.status is not LocalAvailability.AVAILABLE
        ):
            return _local_failure(
                row, "local_source_unavailable", "Local source is explicitly unavailable"
            )
        if (
            row.video_id != catalog_row.video_id
            or _normalise_youtube_video_id(row.source_url) != catalog_row.video_id
            or _normalise_youtube_video_id(catalog_row.source_url)
            != catalog_row.video_id
        ):
            return _local_failure(
                row,
                "local_source_identity_mismatch",
                "Source row identity does not match the local catalog",
            )
        try:
            locator = self._resolver.locator_for(row.source_row_id)
            self._resolver.resolve(locator, catalog_row.source_sha256)
        except (OSError, ValueError):
            return _local_failure(
                row, "local_source_hash_mismatch", "Local source bytes do not match the catalog"
            )
        return AcquisitionResult(
            sample_id=row.sample_id,
            status=AcquisitionStatus.ACQUIRED,
            video_id=row.video_id,
            source_path=None,
            metadata_path=None,
            source_sha256=catalog_row.source_sha256,
            attempts=1,
            error_code=None,
            error_summary=None,
            source_kind="local_file",
            source_locator=locator,
            local_basename=catalog_row.local_filename,
            duration_seconds=catalog_row.duration_seconds,
        )


class LocalAudioAdapter:
    """Resolve one opaque local input, recheck its SHA, then invoke audio processing."""

    def __init__(
        self,
        resolver: LocalSourceResolver,
        *,
        builder: Callable[[Path, Path], object] = build_audio_representations,
    ) -> None:
        self._resolver = resolver
        self._builder = builder

    def __call__(self, source: object, output_dir: Path) -> object:
        if isinstance(source, LocalSourceReference):
            reference = source
        elif isinstance(source, Mapping):
            if source.get("source_kind") != "local_file":
                raise ValueError("local audio adapter requires source_kind=local_file")
            reference = LocalSourceReference(
                locator=str(source.get("source_locator", "")),
                source_sha256=str(source.get("source_sha256", "")),
            )
        else:
            raise TypeError("local audio adapter requires an opaque local source reference")
        guard = self._resolver.resolve_guarded(
            reference.locator, reference.source_sha256
        )
        try:
            result = self._builder(guard.resolved_path, output_dir)
        except BaseException as build_error:
            try:
                self._resolver.verify_unchanged(guard)
            except (OSError, ValueError) as drift_error:
                _discard_local_audio_outputs(output_dir)
                raise ValueError("local source changed during audio build") from drift_error
            raise build_error
        try:
            self._resolver.verify_unchanged(guard)
        except (OSError, ValueError) as error:
            _discard_local_audio_outputs(output_dir)
            raise ValueError("local source changed during audio build") from error
        return result


def _absolute_config_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be absolute")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _parse_catalog_row(
    raw: dict[str | None, str | None],
    number: int,
    root: Path,
    config: LocalSourceConfig,
    duration_probe: Callable[[Path], float],
) -> LocalCatalogRow:
    if raw.get(None):
        raise ValueError(f"row {number}: unexpected columns")
    values = {field: (raw.get(field) or "").strip() for field in LOCAL_CATALOG_FIELDS}
    local_row_id = values["local_row_id"]
    if _LOCAL_ROW_ID.fullmatch(local_row_id) is None:
        raise ValueError(f"row {number}: local_row_id must use local-r###")
    if not values["video_title"]:
        raise ValueError(f"row {number}: video_title is required")
    try:
        status = LocalAvailability(values["status"])
    except ValueError as error:
        raise ValueError(f"row {number}: invalid status") from error
    total_score = _optional_number(
        values["total_score"], number, "total_score", minimum=0, maximum=100
    )
    declared_duration = _optional_number(
        values["duration_seconds"], number, "duration_seconds", minimum=0
    )
    source_url = values["source_url"]
    video_id = values["video_id"]
    if source_url or video_id:
        normalized = _normalise_youtube_video_id(source_url)
        if not normalized:
            raise ValueError(f"row {number}: source_url must be an HTTPS YouTube URL")
        if _VIDEO_ID.fullmatch(video_id) is None:
            raise ValueError(f"row {number}: video_id must be an 11-character video ID")
        if video_id != normalized:
            raise ValueError(f"row {number}: video_id conflicts with source_url")
    if status is LocalAvailability.UNAVAILABLE:
        if not source_url or not video_id:
            raise ValueError(f"row {number}: unavailable rows require YouTube identity")
        if any(
            values[field]
            for field in (
                "duration_seconds",
                "local_filename",
                "source_sha256",
            )
        ):
            raise ValueError(f"row {number}: unavailable rows must omit media fields")
        return LocalCatalogRow(
            local_row_id,
            source_url,
            video_id,
            values["video_title"],
            total_score,
            None,
            "",
            "",
            status,
        )
    if not source_url or not video_id:
        raise ValueError(f"row {number}: available rows require YouTube identity")
    if declared_duration is not None and declared_duration <= 0:
        raise ValueError(f"row {number}: duration_seconds must be greater than zero")
    filename = values["local_filename"]
    filename_path = Path(filename)
    if (
        not filename
        or filename_path.name != filename
        or filename_path.is_absolute()
        or filename in {".", ".."}
    ):
        raise ValueError(f"row {number}: local_filename must be basename-only")
    if filename_path.suffix.lower() != ".mp4":
        raise ValueError(f"row {number}: local_filename must end in .mp4")
    expected_sha256 = values["source_sha256"]
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"row {number}: source_sha256 must be lowercase SHA-256")
    try:
        guard = _capture_local_file(root, filename, expected_sha256)
    except ValueError as error:
        message = str(error)
        if "symbolic links" in message:
            raise ValueError(f"row {number}: symbolic links are not accepted") from error
        if "SHA-256 mismatch" in message:
            raise ValueError(f"row {number}: local source SHA-256 mismatch") from error
        raise ValueError(f"row {number}: local file must remain inside video_root") from error
    path = guard.resolved_path
    try:
        measured_duration = float(duration_probe(path))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"row {number}: local source probe failed") from error
    if not math.isfinite(measured_duration) or measured_duration <= 0:
        raise ValueError(f"row {number}: local source duration must be greater than zero")
    try:
        _verify_local_file_guard(guard)
    except (OSError, ValueError) as error:
        raise ValueError(f"row {number}: local source changed during probe") from error
    if (
        declared_duration is not None
        and abs(declared_duration - measured_duration) > config.duration_tolerance_seconds
    ):
        raise ValueError(f"row {number}: local source duration mismatch")
    return LocalCatalogRow(
        local_row_id,
        source_url,
        video_id,
        values["video_title"],
        total_score,
        measured_duration,
        filename,
        expected_sha256,
        status,
    )


def _validate_catalog_set(
    rows: Sequence[LocalCatalogRow], config: LocalSourceConfig
) -> None:
    available = sum(row.status is LocalAvailability.AVAILABLE for row in rows)
    unavailable = len(rows) - available
    if (
        len(rows) != config.expected_total_rows
        or available != config.expected_available_rows
        or unavailable != config.expected_unavailable_rows
    ):
        raise ValueError("local catalog counts do not match the configured expectations")
    for label, values in (
        ("local_row_id", [row.local_row_id for row in rows]),
        ("video_id", [row.video_id for row in rows if row.video_id]),
        ("source_url", [row.source_url for row in rows if row.source_url]),
        (
            "local_filename",
            [row.local_filename for row in rows if row.local_filename],
        ),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"local catalog contains duplicate {label} values")


def _optional_number(
    value: str,
    row_number: int,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: {name} must be numeric") from error
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        if maximum is None:
            raise ValueError(f"row {row_number}: {name} must be at least {minimum}")
        raise ValueError(
            f"row {row_number}: {name} must be between {minimum:g} and {maximum:g}"
        )
    return number


def _probe_duration(path: Path) -> float:
    return probe_audio(path).duration_seconds


def _capture_local_file(
    root: Path, filename: str, expected_sha256: str
) -> _LocalFileGuard:
    entry = root / filename
    try:
        if entry.is_symlink():
            raise ValueError("local source symbolic links are not accepted")
        before = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("local source is not a regular file")
        resolved = entry.resolve(strict=True)
        resolved.relative_to(root)
        digest = sha256_file(resolved)
        after = entry.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("local source is unavailable") from error
    except OSError as error:
        raise ValueError("local source cannot be inspected") from error
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError("local source changed during validation")
    if digest != expected_sha256:
        raise ValueError("local source SHA-256 mismatch")
    return _LocalFileGuard(
        entry_path=entry,
        resolved_path=resolved,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
        sha256=digest,
    )


def _verify_local_file_guard(guard: _LocalFileGuard) -> None:
    if guard.entry_path.is_symlink():
        raise ValueError("local source symbolic links are not accepted")
    current_resolved = guard.entry_path.resolve(strict=True)
    if current_resolved != guard.resolved_path:
        raise ValueError("local source path identity changed")
    current = guard.entry_path.stat(follow_symlinks=False)
    expected = (
        guard.device,
        guard.inode,
        guard.size,
        guard.mtime_ns,
        guard.ctime_ns,
    )
    if _stat_identity(current) != expected:
        raise ValueError("local source stat identity changed")
    if sha256_file(current_resolved) != guard.sha256:
        raise ValueError("local source SHA-256 changed")
    after_hash = guard.entry_path.stat(follow_symlinks=False)
    if _stat_identity(after_hash) != expected:
        raise ValueError("local source changed during hash verification")


def _stat_identity(value: object) -> tuple[int, int, int, int, int]:
    return (
        int(getattr(value, "st_dev")),
        int(getattr(value, "st_ino")),
        int(getattr(value, "st_size")),
        int(getattr(value, "st_mtime_ns")),
        int(getattr(value, "st_ctime_ns")),
    )


def _discard_local_audio_outputs(output_dir: Path) -> None:
    for filename in (
        "analysis.wav",
        "peak_normalized.wav",
        "loudness_normalized.wav",
        "audio_metadata.json",
    ):
        candidate = output_dir / filename
        try:
            if candidate.is_file() or candidate.is_symlink():
                candidate.unlink()
        except OSError:
            continue


def _local_failure(row: SourceRow, code: str, summary: str) -> AcquisitionResult:
    return AcquisitionResult(
        sample_id=row.sample_id,
        status=AcquisitionStatus.FAILED,
        video_id=row.video_id,
        source_path=None,
        metadata_path=None,
        source_sha256=None,
        attempts=0,
        error_code=code,
        error_summary=summary,
        source_kind="local_file",
    )
