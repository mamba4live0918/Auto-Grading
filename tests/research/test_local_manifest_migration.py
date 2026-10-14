from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from presentation_assessment.research.local_manifest_migration import (
    RETIRED_AUDIT_FIELDS,
    migrate_manifest_to_local_catalog,
    write_local_manifest_migration,
)
from presentation_assessment.research.local_source import LocalAvailability, LocalCatalogRow
from presentation_assessment.research.source_manifest import (
    FROZEN_LEGACY_VIDEO_IDS,
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


SOURCE_FIELDS = (
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


def _video_id(prefix: str, number: int) -> str:
    return f"{prefix}{number:010d}"


def _url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"  # privacy-scan: synthetic-fixture


def _old_row(video_id: str, sample_id: str, row_number: int) -> SourceRow:
    return SourceRow(
        source_row_id=f"old-r{row_number:03d}",
        source_page="synthetic-old",
        sample_id=sample_id,
        legacy_sample_id=sample_id if sample_id.startswith("S") else "",
        source_url=_url(video_id),
        video_id=video_id,
        video_title=f"Old title {row_number}",
        total_score=1.0,
        score_status=ScoreStatus.OBSERVED,
        source_course="Synthetic course",
        verification_status=VerificationStatus.VERIFIED,
        verification_note="",
    )


def _catalog_row(
    row_number: int,
    video_id: str,
    *,
    available: bool = True,
) -> LocalCatalogRow:
    if not available:
        return LocalCatalogRow(
            local_row_id=f"local-r{row_number:03d}",
            source_url=_url(video_id),
            video_id=video_id,
            video_title=f"Unavailable local title {row_number}",
            total_score=None,
            duration_seconds=None,
            local_filename="",
            source_sha256="",
            status=LocalAvailability.UNAVAILABLE,
        )
    return LocalCatalogRow(
        local_row_id=f"local-r{row_number:03d}",
        source_url=_url(video_id),
        video_id=video_id,
        video_title=f"Authoritative local title {row_number}",
        total_score=99.0,
        duration_seconds=30.0 + row_number,
        local_filename=f"{video_id}.mp4",
        source_sha256=f"{row_number:064x}"[-64:],
        status=LocalAvailability.AVAILABLE,
    )


def _migration_fixture() -> tuple[list[SourceRow], list[LocalCatalogRow]]:
    frozen = list(FROZEN_LEGACY_VIDEO_IDS.items())
    common_frozen = frozen[:11]
    corrected_frozen = frozen[11:]
    common_ids = [video_id for video_id, _ in common_frozen] + [
        _video_id("c", number) for number in range(1, 60)
    ]
    retired_ids = [_video_id("r", number) for number in range(1, 16)]
    old_rows: list[SourceRow] = []
    for row_number, video_id in enumerate(common_ids, 1):
        frozen_id = FROZEN_LEGACY_VIDEO_IDS.get(video_id)
        sample_id = frozen_id or f"N{row_number - len(common_frozen):03d}"
        old_rows.append(_old_row(video_id, sample_id, row_number))
    for offset, video_id in enumerate(retired_ids, 60):
        old_rows.append(_old_row(video_id, f"N{offset:03d}", len(old_rows) + 1))

    catalog_rows = [
        _catalog_row(index, video_id)
        for index, video_id in enumerate(common_ids, 1)
    ]
    catalog_rows.extend(
        _catalog_row(index, video_id)
        for index, (video_id, _) in enumerate(corrected_frozen, 71)
    )
    catalog_rows.extend(
        _catalog_row(index, _video_id("n", index - 74))
        for index in range(75, 84)
    )
    catalog_rows.extend(
        _catalog_row(index, _video_id("u", index - 83), available=False)
        for index in range(84, 86)
    )
    assert len(old_rows) == 85
    assert len(catalog_rows) == 85
    return old_rows, catalog_rows


