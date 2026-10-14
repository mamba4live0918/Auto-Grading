"""Deterministic migration from the legacy private manifest to a local catalog."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .artifacts import atomic_write_text
from .local_source import LocalAvailability, LocalCatalogRow
from .source_manifest import (
    FROZEN_LEGACY_VIDEO_IDS,
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


SOURCE_MANIFEST_FIELDS = (
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
RETIRED_AUDIT_FIELDS = (
    "old_source_row_id",
    "sample_id",
    "video_id",
    "retirement_reason",
)
_NEW_SAMPLE_ID = re.compile(r"N(?P<number>[0-9]{3})\Z")


@dataclass(frozen=True)
class RetiredSourceAuditRow:
    """Minimal source-identity audit for one legacy row no longer active."""

    old_source_row_id: str
    sample_id: str
    video_id: str
    retirement_reason: str


@dataclass(frozen=True)
class LocalManifestMigration:
    """Strict active source rows and a separate retired-row audit."""

    active_rows: tuple[SourceRow, ...]
    retired_rows: tuple[RetiredSourceAuditRow, ...]


def migrate_manifest_to_local_catalog(
    old_rows: Sequence[SourceRow],
    catalog_rows: Sequence[LocalCatalogRow],
) -> LocalManifestMigration:
    """Preserve matching identities and allocate replacements above the old N maximum."""
    if len(old_rows) != 85 or len(catalog_rows) != 85:
        raise ValueError("local manifest migration requires exactly 85 old and 85 catalog rows")
    old_video_ids = [row.video_id for row in old_rows if row.video_id]
    if len(old_video_ids) != len(set(old_video_ids)):
        raise ValueError("old manifest video_id values must be unique")
    old_sample_ids = [row.sample_id for row in old_rows]
    if (
        not all(old_sample_ids)
        or len(old_sample_ids) != len(set(old_sample_ids))
    ):
        raise ValueError("old sample_id values must be unique and nonblank")
    catalog_video_ids = [row.video_id for row in catalog_rows if row.video_id]
    if len(catalog_video_ids) != len(set(catalog_video_ids)):
        raise ValueError("local catalog video_id values must be unique")
    local_row_ids = [row.local_row_id for row in catalog_rows]
    if len(local_row_ids) != len(set(local_row_ids)):
        raise ValueError("local catalog row identities must be unique")
    available_count = sum(
        row.status is LocalAvailability.AVAILABLE for row in catalog_rows
    )
    unavailable_count = len(catalog_rows) - available_count
    if (available_count, unavailable_count) != (83, 2):
        raise ValueError("local catalog must contain exactly 83 available and 2 unavailable rows")
    for row in catalog_rows:
        if row.status is not LocalAvailability.UNAVAILABLE:
            continue
        if (
            not row.source_url
            or not row.video_id
            or row.duration_seconds is not None
            or bool(row.local_filename)
            or bool(row.source_sha256)
        ):
            raise ValueError(
                "unavailable rows must retain identity and omit media fields"
            )
    frozen_catalog_ids = set(catalog_video_ids) & set(FROZEN_LEGACY_VIDEO_IDS)
    if frozen_catalog_ids != set(FROZEN_LEGACY_VIDEO_IDS):
        raise ValueError("local catalog must contain the exact frozen legacy video set")

    old_by_video = {row.video_id: row for row in old_rows if row.video_id}
    matching_count = sum(
        bool(row.video_id and row.video_id in old_by_video) for row in catalog_rows
    )
    replacement_count = len(catalog_rows) - matching_count
    catalog_video_id_set = {row.video_id for row in catalog_rows if row.video_id}
    retired_count = sum(
        bool(row.video_id and row.video_id not in catalog_video_id_set)
        for row in old_rows
    )
    if (matching_count, replacement_count, retired_count) != (70, 15, 15):
        raise ValueError(
            "local manifest migration requires 70 matching and 15 replacement rows "
            "with 15 retired old rows"
        )
    old_max = max(
        (
            int(match.group("number"))
            for row in old_rows
            if (match := _NEW_SAMPLE_ID.fullmatch(row.sample_id)) is not None
        ),
        default=0,
    )
    unmatched_count = sum(
        not row.video_id or row.video_id not in old_by_video for row in catalog_rows
    )
    if old_max + unmatched_count > 999:
        raise ValueError("N### sample_id namespace is exhausted")
    next_new_number = old_max + 1
    active: list[SourceRow] = []
    for catalog_row in catalog_rows:
        previous = old_by_video.get(catalog_row.video_id) if catalog_row.video_id else None
        frozen_id = FROZEN_LEGACY_VIDEO_IDS.get(catalog_row.video_id)
        if frozen_id is not None:
            if previous is not None and (
                previous.sample_id != frozen_id
                or previous.legacy_sample_id != frozen_id
            ):
                raise ValueError(
                    f"old frozen mapping mismatch for {catalog_row.local_row_id}"
                )
            sample_id = frozen_id
        elif previous is not None:
            sample_id = previous.sample_id
        else:
            sample_id = f"N{next_new_number:03d}"
            next_new_number += 1
        if not sample_id:
            raise ValueError("matching old manifest row has no sample_id")
        available = catalog_row.status is LocalAvailability.AVAILABLE
        active.append(
            SourceRow(
                source_row_id=catalog_row.local_row_id,
                source_page="local_catalog",
                sample_id=sample_id,
                legacy_sample_id=sample_id if sample_id.startswith("S") else "",
                source_url=catalog_row.source_url,
                video_id=catalog_row.video_id,
                video_title=catalog_row.video_title,
                total_score=catalog_row.total_score,
                score_status=(
                    ScoreStatus.OBSERVED
                    if catalog_row.total_score is not None
                    else ScoreStatus.NOT_AVAILABLE
                ),
                source_course=previous.source_course if previous is not None else "",
                verification_status=(
                    VerificationStatus.VERIFIED
                    if available
                    else VerificationStatus.FAILED
                ),
                verification_note="" if available else "local_source_unavailable",
            )
        )
    sample_ids = [row.sample_id for row in active]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("migrated active sample_id values must be unique")

    active_video_ids = catalog_video_id_set
    retired = tuple(
        RetiredSourceAuditRow(
            old_source_row_id=row.source_row_id,
            sample_id=row.sample_id,
            video_id=row.video_id,
            retirement_reason="not_in_local_catalog",
        )
        for row in old_rows
        if row.video_id and row.video_id not in active_video_ids
    )
    retired_active_collisions = set(sample_ids) & {
        row.sample_id for row in retired
    }
    if retired_active_collisions:
        raise ValueError("retired and active sample_id collision")
    return LocalManifestMigration(tuple(active), retired)


def write_local_manifest_migration(
    migration: LocalManifestMigration,
    active_path: Path,
    retired_path: Path,
) -> None:
    """Write exact-schema UTF-8 CSV outputs without adding catalog paths."""
    active_rows = [
        {
            "source_row_id": row.source_row_id,
            "source_page": row.source_page,
            "sample_id": row.sample_id,
            "legacy_sample_id": row.legacy_sample_id,
            "source_url": row.source_url,
            "video_id": row.video_id,
            "video_title": row.video_title,
            "total_score": "" if row.total_score is None else row.total_score,
            "score_status": row.score_status.value,
            "source_course": row.source_course,
            "verification_status": row.verification_status.value,
            "verification_note": row.verification_note,
        }
        for row in migration.active_rows
    ]
    retired_rows = [
        {
            "old_source_row_id": row.old_source_row_id,
            "sample_id": row.sample_id,
            "video_id": row.video_id,
            "retirement_reason": row.retirement_reason,
        }
        for row in migration.retired_rows
    ]
    atomic_write_text(active_path, _csv_text(SOURCE_MANIFEST_FIELDS, active_rows))
    atomic_write_text(retired_path, _csv_text(RETIRED_AUDIT_FIELDS, retired_rows))


def _csv_text(fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()
