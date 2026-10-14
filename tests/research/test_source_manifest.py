from pathlib import Path

import pytest

from presentation_assessment.research.source_manifest import (
    FROZEN_LEGACY_VIDEO_IDS,
    ScoreStatus,
    SourceRow,
    VerificationStatus,
    assign_new_sample_ids,
    load_source_manifest,
    validate_source_rows,
)


MANIFEST_HEADER = (
    "source_row_id,source_page,sample_id,legacy_sample_id,source_url,video_id,video_title,"
    "total_score,score_status,source_course,verification_status,verification_note\n"
)


def row(**overrides: object) -> SourceRow:
    fields: dict[str, object] = {
        "source_row_id": "r1",
        "source_page": "1",
        "sample_id": "",
        "legacy_sample_id": "",
        "source_url": "https://www.youtube.com/watch?v=abcdefghijk",
        "video_id": "abcdefghijk",
        "video_title": "Redacted example",
        "total_score": 81.5,
        "score_status": ScoreStatus.OBSERVED,
        "source_course": "",
        "verification_status": VerificationStatus.VERIFIED,
        "verification_note": "",
    }
    if "video_id" in overrides and "source_url" not in overrides:
        fields["source_url"] = f"https://youtu.be/{overrides['video_id']}"
    fields.update(overrides)
    return SourceRow(**fields)


def write_manifest_row(tmp_path: Path, row_text: str) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(MANIFEST_HEADER + row_text + "\n", encoding="utf-8")
    return path


def test_loads_strict_valid_fixture_with_observed_and_na_scores(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        MANIFEST_HEADER
        + "r1,1,N001,,https://www.youtube.com/watch?v=abcdefghijk,abcdefghijk,Example one,81.5,observed,,verified,\n"
        + "r2,1,N002,,https://youtu.be/lmnopqrstuv,lmnopqrstuv,Example two,,not_available,,needs_review,score unavailable\n",
        encoding="utf-8",
    )

    rows = load_source_manifest(path)

    assert [item.total_score for item in rows] == [81.5, None]
    assert [item.score_status for item in rows] == [
        ScoreStatus.OBSERVED,
        ScoreStatus.NOT_AVAILABLE,
    ]


def test_loader_rejects_invalid_fixture_score_semantics(tmp_path: Path) -> None:
    path = write_manifest_row(
        tmp_path,
        "r1,1,N001,,https://www.youtube.com/watch?v=abcdefghijk,abcdefghijk,Example,82,not_available,,verified,score must be blank",
    )

    with pytest.raises(ValueError, match="total_score"):
        load_source_manifest(path)


def test_loader_rejects_rows_with_unexpected_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "source_row_id,source_page,sample_id,legacy_sample_id,source_url,video_id,video_title,total_score,score_status,source_course,verification_status,verification_note\n"
        "r1,1,,,https://videos.example.invalid/watch?v=example001,example001,Example,81.5,observed,,verified,,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected columns"):
        load_source_manifest(path)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://media.example.invalid/watch?v=abcdefghijk",
        "http://www.youtube.com/watch?v=abcdefghijk",
    ],
)
def test_loader_rejects_non_allowlisted_or_non_https_urls(
    tmp_path: Path, source_url: str
) -> None:
    path = write_manifest_row(
        tmp_path,
        f"r1,1,,,{source_url},abcdefghijk,Example,81.5,observed,,verified,",
    )

    with pytest.raises(ValueError, match="YouTube"):
        load_source_manifest(path)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://youtu.be/short",
        "https://www.youtube.com/watch?v=abcdefghijkX",
    ],
)
def test_loader_rejects_malformed_video_ids(tmp_path: Path, source_url: str) -> None:
    path = write_manifest_row(
        tmp_path,
        f"r1,1,,,{source_url},,Example,81.5,observed,,verified,",
    )

    with pytest.raises(ValueError, match="video ID"):
        load_source_manifest(path)


def test_loader_rejects_supplied_video_id_for_non_youtube_url(tmp_path: Path) -> None:
    path = write_manifest_row(
        tmp_path,
        "r1,1,,,https://media.example.invalid/video,abcdefghijk,Example,81.5,observed,,verified,",
    )

    with pytest.raises(ValueError, match="YouTube"):
        load_source_manifest(path)


@pytest.mark.parametrize("score", ["-0.01", "100.01"])
def test_loader_rejects_observed_scores_outside_inclusive_range(
    tmp_path: Path, score: str
) -> None:
    path = write_manifest_row(
        tmp_path,
        f"r1,1,,,https://youtu.be/abcdefghijk,,Example,{score},observed,,verified,",
    )

    with pytest.raises(ValueError, match="between 0 and 100"):
        load_source_manifest(path)


def test_loader_accepts_youtube_nocookie_embed_url(tmp_path: Path) -> None:
    path = write_manifest_row(
        tmp_path,
        "r1,1,,,https://www.youtube-nocookie.com/embed/abcdefghijk,,Example,100,observed,,verified,",
    )

    loaded = load_source_manifest(path)

    assert loaded[0].video_id == "abcdefghijk"
    assert loaded[0].total_score == 100


def test_assignment_rejects_stale_legacy_id_for_nonlegacy_row() -> None:
    with pytest.raises(ValueError, match="legacy_sample_id"):
        assign_new_sample_ids([row(legacy_sample_id="S01")], {})


def test_assignment_preserves_existing_new_ids_and_allocates_above_the_maximum() -> None:
    assigned = assign_new_sample_ids(
        [
            row(source_row_id="r20", sample_id="N007"),
            row(source_row_id="r10"),
            row(source_row_id="r15"),
        ],
        {},
    )

    assert [item.sample_id for item in assigned] == ["N007", "N008", "N009"]


def test_inserting_an_earlier_row_does_not_renumber_an_existing_sample() -> None:
    assigned = assign_new_sample_ids(
        [
            row(source_row_id="r01"),
            row(source_row_id="r20", sample_id="N001"),
        ],
        {},
    )

    assert [item.sample_id for item in assigned] == ["N002", "N001"]


@pytest.mark.parametrize("sample_id", ["N01", "N000", "S16", "sample-1"])
def test_assignment_rejects_invalid_sample_id_formats(sample_id: str) -> None:
    with pytest.raises(ValueError, match="sample_id"):
        assign_new_sample_ids([row(sample_id=sample_id)], {})


def test_assignment_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        assign_new_sample_ids(
            [row(source_row_id="r1", sample_id="N001"), row(source_row_id="r2", sample_id="N001")],
            {},
        )


@pytest.mark.parametrize("source_row_ids", [("",), ("r1", "r1")])
def test_assignment_rejects_blank_or_duplicate_source_row_ids(source_row_ids: tuple[str, ...]) -> None:
    rows = [row(source_row_id=source_row_id) for source_row_id in source_row_ids]

    with pytest.raises(ValueError, match="source_row_id values must be nonblank and unique"):
        assign_new_sample_ids(rows, {})


def test_new_ids_ignore_failed_rows_without_renumbering() -> None:
    rows = [
        row(source_row_id="r1"),
        row(source_row_id="r2", verification_status=VerificationStatus.NEEDS_REVIEW),
    ]

    assigned = assign_new_sample_ids(rows, {})

    assert [item.sample_id for item in assigned] == ["N001", "N002"]


def test_failed_row_is_ineligible_and_keeps_its_new_id_position() -> None:
    assigned = assign_new_sample_ids(
        [
            row(source_row_id="r1"),
            row(source_row_id="r2", verification_status=VerificationStatus.FAILED),
        ],
        {},
    )

    assert [item.sample_id for item in assigned] == ["N001", "N002"]
    assert assigned[1].is_acquisition_eligible is False


def test_needs_review_row_is_not_eligible_for_acquisition() -> None:
    unresolved = row(verification_status=VerificationStatus.NEEDS_REVIEW)

    assert unresolved.is_acquisition_eligible is False


def test_new_ids_follow_source_row_id_order() -> None:
    rows = [row(source_row_id="r20"), row(source_row_id="r3")]

    assigned = assign_new_sample_ids(rows, {})

    assert [item.sample_id for item in assigned] == ["N001", "N002"]


def test_duplicate_video_ids_are_reported() -> None:
    audit = validate_source_rows(
        [
            row(video_id="abcdefghijk", sample_id="N001"),
            row(source_row_id="r2", video_id="abcdefghijk", sample_id="N002"),
        ]
    )

    assert audit.duplicate_video_ids == ["abcdefghijk"]


def test_audit_surfaces_identity_errors() -> None:
    audit = validate_source_rows(
        [
            row(source_row_id="", sample_id="bad"),
            row(source_row_id="r2", sample_id="N001", legacy_sample_id="S01"),
            row(source_row_id="r3", sample_id="N001"),
        ]
    )

    assert set(audit.identity_errors) == {
        "blank_source_row_id",
        "duplicate_sample_id",
        "invalid_sample_id",
        "stale_legacy_sample_id",
    }
    assert len(audit.invalid_rows) == 3


def test_audit_counts_scores_unresolved_duplicates_and_invalid_rows() -> None:
    rows = [
        row(sample_id="N001", source_url="https://youtu.be/abcdefghijk", video_id="abcdefghijk"),
        row(
            source_row_id="r2",
            sample_id="N002",
            source_url="https://youtu.be/abcdefghijk",
            video_id="abcdefghijk",
            score_status=ScoreStatus.NOT_AVAILABLE,
            total_score=None,
            verification_status=VerificationStatus.NEEDS_REVIEW,
        ),
        row(
            source_row_id="",
            sample_id="N003",
            source_url="",
            video_id="",
            score_status=ScoreStatus.INVALID,
            total_score=None,
        ),
    ]

    audit = validate_source_rows(rows)

    assert audit.total_rows == 3
    assert audit.observed_score_rows == 1
    assert audit.not_available_rows == 1
    assert audit.unresolved_rows == 1
    assert audit.duplicate_urls == ["https://youtu.be/abcdefghijk"]
    assert audit.duplicate_video_ids == ["abcdefghijk"]
    assert audit.invalid_rows == ["<missing source_row_id>"]
