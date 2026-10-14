"""Validated private advisor-source manifests and stable sample identities."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse


class ScoreStatus(str, Enum):
    """Whether a source-table score can be used as an observed value."""

    OBSERVED = "observed"
    NOT_AVAILABLE = "not_available"
    INVALID = "invalid"


class VerificationStatus(str, Enum):
    """Confidence state from the two-pass source-table transcription."""

    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceRow:
    """One auditable row from the private advisor source manifest."""

    source_row_id: str
    source_page: str
    sample_id: str
    legacy_sample_id: str
    source_url: str
    video_id: str
    video_title: str
    total_score: float | None
    score_status: ScoreStatus
    source_course: str
    verification_status: VerificationStatus
    verification_note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_status", ScoreStatus(self.score_status))
        object.__setattr__(self, "verification_status", VerificationStatus(self.verification_status))

    @property
    def is_acquisition_eligible(self) -> bool:
        """Whether this complete, verified row may be acquired locally."""
        return (
            self.verification_status is VerificationStatus.VERIFIED
            and self.score_status is not ScoreStatus.INVALID
            and bool(self.source_row_id and self.source_url and self.video_id)
            and _normalise_youtube_video_id(self.source_url) == self.video_id
            and (
                self.score_status is not ScoreStatus.OBSERVED
                or _is_valid_observed_score(self.total_score)
            )
        )


@dataclass(frozen=True)
class ManifestAudit:
    """Non-sensitive counts and identity defects found in a manifest."""

    total_rows: int
    observed_score_rows: int
    not_available_rows: int
    unresolved_rows: int
    duplicate_urls: list[str]
    duplicate_video_ids: list[str]
    invalid_rows: list[str]
    identity_errors: list[str]

    @property
    def na_rows(self) -> int:
        """Alias for the number of rows whose score is not available."""
        return self.not_available_rows


_FIELDNAMES = (
    "source_row_id",
    "source_page",
    "sample_id",
    "legacy_sample_id",
    "source_url",
    "video_id",
    "video_title",
    "total_score",
    "score_status",
    "source_course",
    "verification_status",
    "verification_note",
)

# The crosswalk from the course's earlier sample IDs to their videos is private and not included here.
FROZEN_LEGACY_VIDEO_IDS: Mapping[str, str] = MappingProxyType({})
_YOUTUBE_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_NEW_SAMPLE_ID = re.compile(r"N(?P<number>[0-9]{3})\Z")
_YOUTUBE_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
)
_YOUTUBE_NOCOOKIE_HOSTS = frozenset(
    {"youtube-nocookie.com", "www.youtube-nocookie.com"}
)


def load_source_manifest(path: Path) -> list[SourceRow]:
    """Load a complete CSV manifest, enforcing score semantics at the boundary."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_fieldnames(reader.fieldnames)
            return [_parse_source_row(raw_row, row_number) for row_number, raw_row in enumerate(reader, 2)]
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"Unable to read source manifest: {path}") from error


def validate_source_rows(rows: Sequence[SourceRow]) -> ManifestAudit:
    """Report manifest completeness, score availability, duplicates, and invalid rows."""
    duplicate_urls = _duplicates(row.source_url for row in rows if row.source_url)
    duplicate_video_ids = _duplicates(row.video_id for row in rows if row.video_id)
    source_id_counts = _counts(row.source_row_id for row in rows if row.source_row_id)
    identity_errors, identity_error_rows = _identity_errors(rows, FROZEN_LEGACY_VIDEO_IDS)
    invalid_rows = [
        _row_label(row)
        for index, row in enumerate(rows)
        if _is_invalid_row(row, source_id_counts) or index in identity_error_rows
    ]
    return ManifestAudit(
        total_rows=len(rows),
        observed_score_rows=sum(row.score_status is ScoreStatus.OBSERVED for row in rows),
        not_available_rows=sum(row.score_status is ScoreStatus.NOT_AVAILABLE for row in rows),
        unresolved_rows=sum(
            row.verification_status is not VerificationStatus.VERIFIED for row in rows
        ),
        duplicate_urls=duplicate_urls,
        duplicate_video_ids=duplicate_video_ids,
        invalid_rows=invalid_rows,
        identity_errors=identity_errors,
    )


def assign_new_sample_ids(
    rows: Sequence[SourceRow], legacy_video_ids: Mapping[str, str] | None = None
) -> list[SourceRow]:
    """Assign legacy IDs or deterministic ``N###`` IDs by immutable row identity."""
    crosswalk = FROZEN_LEGACY_VIDEO_IDS if legacy_video_ids is None else legacy_video_ids
    _validate_assignment_rows(rows, crosswalk)
    existing_new_numbers = [
        int(match.group("number"))
        for row in rows
        if (match := _NEW_SAMPLE_ID.fullmatch(row.sample_id)) is not None
    ]
    new_rows = sorted(
        (
            (index, row)
            for index, row in enumerate(rows)
            if row.video_id not in crosswalk and not row.sample_id
        ),
        key=lambda item: item[1].source_row_id,
    )
    next_number = max(existing_new_numbers, default=0) + 1
    if new_rows and next_number + len(new_rows) - 1 > 999:
        raise ValueError("N### sample_id namespace is exhausted")
    new_ids = {
        index: f"N{number:03d}"
        for number, (index, _) in enumerate(new_rows, next_number)
    }
    assigned: list[SourceRow] = []
    for index, row in enumerate(rows):
        legacy_id = crosswalk.get(row.video_id)
        if legacy_id is not None:
            assigned.append(replace(row, sample_id=legacy_id, legacy_sample_id=legacy_id))
        elif row.sample_id:
            assigned.append(row)
        else:
            assigned.append(replace(row, sample_id=new_ids[index], legacy_sample_id=""))
    return assigned


def _require_fieldnames(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("source manifest must include a header row")
    if tuple(fieldnames) != _FIELDNAMES:
        raise ValueError("source manifest columns must exactly match the required schema")


def _parse_source_row(raw_row: dict[str, str | None], row_number: int) -> SourceRow:
    if raw_row.get(None):
        raise ValueError(f"row {row_number}: unexpected columns")
    values = {field: (raw_row[field] or "").strip() for field in _FIELDNAMES}
    score_status = _parse_enum(ScoreStatus, values["score_status"], "score_status", row_number)
    verification_status = _parse_enum(
        VerificationStatus, values["verification_status"], "verification_status", row_number
    )
    total_score = _parse_total_score(values["total_score"], score_status, row_number)
    video_id = _normalise_youtube_video_id(values["source_url"])
    if values["source_url"] and not video_id:
        raise ValueError(
            f"row {row_number}: source_url must be an HTTPS YouTube URL with a valid video ID"
        )
    if values["video_id"] and not video_id:
        raise ValueError(f"row {row_number}: video_id requires a valid YouTube source_url")
    if values["video_id"] and _YOUTUBE_VIDEO_ID.fullmatch(values["video_id"]) is None:
        raise ValueError(f"row {row_number}: video_id must be an 11-character YouTube video ID")
    if video_id and values["video_id"] and values["video_id"] != video_id:
        raise ValueError(f"row {row_number}: video_id conflicts with source_url")
    values.update(
        video_id=video_id or values["video_id"],
        total_score=total_score,
        score_status=score_status,
        verification_status=verification_status,
    )
    return SourceRow(**values)


def _parse_enum(enum_type: type[Enum], value: str, field: str, row_number: int) -> Enum:
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: invalid {field}: {value!r}") from error


def _parse_total_score(value: str, status: ScoreStatus, row_number: int) -> float | None:
    if status is ScoreStatus.OBSERVED:
        if not value:
            raise ValueError(f"row {row_number}: total_score is required when score_status=observed")
        try:
            score = float(value)
        except ValueError as error:
            raise ValueError(f"row {row_number}: total_score must be numeric when observed") from error
        if not math.isfinite(score):
            raise ValueError(f"row {row_number}: total_score must be finite when observed")
        if not 0 <= score <= 100:
            raise ValueError(f"row {row_number}: total_score must be between 0 and 100")
        return score
    if value:
        raise ValueError(f"row {row_number}: total_score must be blank when score_status={status.value}")
    return None


def _normalise_youtube_video_id(source_url: str) -> str:
    """Return a video ID for recognised YouTube URL forms without network access."""
    parsed = urlparse(source_url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    if parsed.scheme != "https" or host not in _YOUTUBE_HOSTS | _YOUTUBE_NOCOOKIE_HOSTS:
        return ""
    path_parts = [part for part in parsed.path.split("/") if part]
    if host == "youtu.be":
        candidate = path_parts[0] if len(path_parts) == 1 else ""
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif len(path_parts) == 2 and path_parts[0] in {"shorts", "embed", "live"}:
            candidate = path_parts[1]
        else:
            candidate = ""
    elif len(path_parts) == 2 and path_parts[0] == "embed":
        candidate = path_parts[1]
    else:
        candidate = ""
    return candidate if _YOUTUBE_VIDEO_ID.fullmatch(candidate) else ""


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = _counts(values)
    return [value for value, count in counts.items() if count > 1]


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _is_invalid_row(row: SourceRow, source_id_counts: Mapping[str, int]) -> bool:
    return (
        not row.source_row_id
        or not row.source_page
        or not row.source_url
        or not row.video_id
        or row.score_status is ScoreStatus.INVALID
        or (row.score_status is ScoreStatus.OBSERVED and row.total_score is None)
        or (row.score_status is ScoreStatus.OBSERVED and not _is_valid_observed_score(row.total_score))
        or (row.score_status is not ScoreStatus.OBSERVED and row.total_score is not None)
        or _normalise_youtube_video_id(row.source_url) != row.video_id
        or source_id_counts.get(row.source_row_id, 0) > 1
    )


def _is_valid_observed_score(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0 <= value <= 100


def _row_label(row: SourceRow) -> str:
    return row.source_row_id or "<missing source_row_id>"


def _validate_assignment_rows(rows: Sequence[SourceRow], crosswalk: Mapping[str, str]) -> None:
    source_row_ids = [row.source_row_id for row in rows]
    if not all(source_row_ids) or len(source_row_ids) != len(set(source_row_ids)):
        raise ValueError("source_row_id values must be nonblank and unique")

    supplied_sample_ids = [row.sample_id for row in rows if row.sample_id]
    if len(supplied_sample_ids) != len(set(supplied_sample_ids)):
        raise ValueError("sample_id values must be unique")

    for row in rows:
        legacy_id = crosswalk.get(row.video_id)
        if legacy_id is not None:
            if any(
                supplied_id and supplied_id != legacy_id
                for supplied_id in (row.sample_id, row.legacy_sample_id)
            ):
                raise ValueError(f"crosswalk mismatch for source_row_id {row.source_row_id}")
            continue
        if row.legacy_sample_id:
            raise ValueError(
                f"legacy_sample_id is stale for source_row_id {row.source_row_id}"
            )
        if row.sample_id and not _valid_new_sample_id(row.sample_id):
            raise ValueError(f"invalid sample_id for source_row_id {row.source_row_id}")


def _valid_new_sample_id(value: str) -> bool:
    match = _NEW_SAMPLE_ID.fullmatch(value)
    return match is not None and int(match.group("number")) >= 1


def _identity_errors(
    rows: Sequence[SourceRow], crosswalk: Mapping[str, str]
) -> tuple[list[str], set[int]]:
    source_counts = _counts(row.source_row_id for row in rows if row.source_row_id)
    sample_counts = _counts(row.sample_id for row in rows if row.sample_id)
    errors: set[str] = set()
    invalid_indices: set[int] = set()
    for index, row in enumerate(rows):
        row_errors: set[str] = set()
        if not row.source_row_id:
            row_errors.add("blank_source_row_id")
        elif source_counts.get(row.source_row_id, 0) > 1:
            row_errors.add("duplicate_source_row_id")
        if row.sample_id and sample_counts.get(row.sample_id, 0) > 1:
            row_errors.add("duplicate_sample_id")

        legacy_id = crosswalk.get(row.video_id)
        if legacy_id is not None:
            if row.sample_id != legacy_id or row.legacy_sample_id != legacy_id:
                row_errors.add("legacy_identity_mismatch")
        else:
            if row.legacy_sample_id:
                row_errors.add("stale_legacy_sample_id")
            if not row.sample_id:
                row_errors.add("missing_sample_id")
            elif not _valid_new_sample_id(row.sample_id):
                row_errors.add("invalid_sample_id")

        if row_errors:
            invalid_indices.add(index)
            errors.update(row_errors)
    return sorted(errors), invalid_indices
