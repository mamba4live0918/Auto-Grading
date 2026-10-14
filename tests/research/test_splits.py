"""Behavioural tests for deterministic exploratory split assignment."""

from __future__ import annotations

from presentation_assessment.research.splits import create_exploratory_splits


def _row(
    sample_id: str,
    *,
    video_id: str,
    source_sha256: str,
    total_score: float | None = 70.0,
    score_status: str = "observed",
    speaker_id: str | None = None,
    talk_id: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "source_sha256": source_sha256,
        "total_score": total_score,
        "score_status": score_status,
        "speaker_id": speaker_id,
        "talk_id": talk_id,
    }


def test_duplicate_content_and_video_components_never_split() -> None:
    """Break caught: either duplicate key assigns related samples to different folds."""
    rows = [
        _row("N001", video_id="video-00001", source_sha256="a" * 64),
        _row("N002", video_id="video-00002", source_sha256="a" * 64),
        _row("N003", video_id="video-00002", source_sha256="b" * 64),
        _row("N004", video_id="video-00004", source_sha256="c" * 64),
    ]

    registry = create_exploratory_splits(rows, seed=20260813)
    assignments = {item.sample_id: item for item in registry.assignments}

    duplicate_folds = {
        (assignments[sample_id].outer_fold, assignments[sample_id].inner_fold)
        for sample_id in ("N001", "N002", "N003")
    }
    assert len(duplicate_folds) == 1
    assert len({assignments[sample_id].group_id for sample_id in ("N001", "N002", "N003")}) == 1
    assert assignments["N004"].group_id != assignments["N001"].group_id


def test_assignments_are_order_independent_and_seeded() -> None:
    """Break caught: row order or ambient randomness changes the frozen registry."""
    rows = [
        _row(f"N{index:03d}", video_id=f"video-{index:05d}", source_sha256=f"{index:064x}")
        for index in range(1, 9)
    ]

    first = create_exploratory_splits(rows, seed=20260813)
    reordered = create_exploratory_splits(list(reversed(rows)), seed=20260813)
    other_seed = create_exploratory_splits(rows, seed=20260814)

    assert first.to_dict() == reordered.to_dict()
    assert first.seed == 20260813
    assert [item.outer_fold for item in first.assignments] != [
        item.outer_fold for item in other_seed.assignments
    ]


def test_unknown_speaker_is_explicit_without_a_speaker_holdout_claim() -> None:
    """Break caught: unknown identity is fabricated or presented as speaker-disjoint."""
    registry = create_exploratory_splits(
        [_row("N001", video_id="video-00001", source_sha256="a" * 64)],
        seed=20260813,
    )

    assignment = registry.assignments[0]
    assert assignment.speaker_id == "unknown"
    assert registry.speaker_identity_status == "not_reliably_available"
    assert registry.speaker_disjoint is False
    assert registry.no_independent_holdout_claim is True
    assert registry.evaluation_claim == "exploratory_no_speaker_holdout"


def test_unlabelled_rows_are_unsupervised_but_keep_qc_and_clustering_eligibility() -> None:
    """Break caught: absent scores are zero-filled or excluded from unsupervised analysis."""
    rows = [
        _row(
            "N001",
            video_id="video-00001",
            source_sha256="a" * 64,
            total_score=None,
            score_status="not_available",
        )
    ]

    assignment = create_exploratory_splits(rows, seed=20260813).assignments[0]

    assert assignment.modelling_scope == "unsupervised_only"
    assert assignment.supervised_eligible is False
    assert assignment.qc_eligible is True
    assert assignment.clustering_eligible is True


def test_failed_labelled_row_is_not_supervised_eligible() -> None:
    """Break caught: label presence alone marks a feature-failed sample usable."""
    row = _row("N001", video_id="video-00001", source_sha256="a" * 64)
    row.update(
        supervised_eligible=False,
        qc_eligible=False,
        clustering_eligible=False,
    )

    assignment = create_exploratory_splits([row], seed=20260813).assignments[0]

    assert assignment.modelling_scope == "supervised_exploratory"
    assert assignment.supervised_eligible is False
