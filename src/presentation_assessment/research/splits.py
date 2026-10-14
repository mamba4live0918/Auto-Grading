"""Deterministic, limitation-aware exploratory split registries."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping, Sequence

from .canonical import canonical_json_bytes, sha256_bytes


ModellingScope = Literal["supervised_exploratory", "unsupervised_only"]


@dataclass(frozen=True)
class SplitAssignment:
    """One sample's immutable grouped-fold assignment and analysis scope."""

    sample_id: str
    group_id: str
    outer_fold: int
    inner_fold: int
    speaker_id: str
    talk_id: str
    video_id: str
    source_sha256: str | None
    modelling_scope: ModellingScope
    supervised_eligible: bool
    qc_eligible: bool
    clustering_eligible: bool

    def to_dict(self) -> dict[str, object]:
        """Return the canonical public representation used by ``splits.json``."""
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "speaker_id": self.speaker_id,
            "talk_id": self.talk_id,
            "video_id": self.video_id,
            "source_sha256": self.source_sha256,
            "modelling_scope": self.modelling_scope,
            "supervised_eligible": self.supervised_eligible,
            "qc_eligible": self.qc_eligible,
            "clustering_eligible": self.clustering_eligible,
        }


@dataclass(frozen=True)
class SplitRegistry:
    """Frozen split policy plus every deterministic sample assignment."""

    seed: int
    assignments: tuple[SplitAssignment, ...]
    outer_method: str
    outer_fold_count: int
    inner_method: str
    inner_fold_count: int
    deduplicate_by: tuple[str, ...] = ("source_sha256", "video_id")
    group_fields: tuple[str, ...] = ("speaker_id", "talk_id")
    speaker_identity_status: str = "not_reliably_available"
    speaker_disjoint: bool = False
    no_independent_holdout_claim: bool = True
    evaluation_claim: str = "exploratory_no_speaker_holdout"

    def to_dict(self) -> dict[str, object]:
        """Return a canonical, order-stable serialisation."""
        return {
            "schema_version": 1,
            "seed": self.seed,
            "strategy": "nested_grouped_cross_validation",
            "outer_method": self.outer_method,
            "outer_fold_count": self.outer_fold_count,
            "inner_method": self.inner_method,
            "inner_fold_count": self.inner_fold_count,
            "deduplicate_by": list(self.deduplicate_by),
            "group_fields": list(self.group_fields),
            "speaker_identity_status": self.speaker_identity_status,
            "speaker_disjoint": self.speaker_disjoint,
            "no_independent_holdout_claim": self.no_independent_holdout_claim,
            "evaluation_claim": self.evaluation_claim,
            "assignments": [item.to_dict() for item in self.assignments],
        }

    @property
    def sha256(self) -> str:
        """Return the exact identity bound into the post-freeze protocol."""
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class _RowView:
    sample_id: str
    video_id: str
    source_sha256: str | None
    speaker_id: str
    talk_id: str
    modelling_scope: ModellingScope
    supervised_eligible: bool
    qc_eligible: bool
    clustering_eligible: bool


class _Groups:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parents[item]
        if parent != item:
            self.parents[item] = self.find(parent)
        return self.parents[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[max(left_root, right_root)] = min(left_root, right_root)


def create_exploratory_splits(rows: Sequence[object], seed: int) -> SplitRegistry:
    """Assign nested exploratory folds without inventing speaker identity.

    Duplicate source hashes and video IDs form transitive components.  Reliable
    speaker/talk identifiers may merge those components, while the literal
    ``unknown`` value is retained for reporting and never used to merge otherwise
    unrelated samples.
    """
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    viewed = tuple(_row_view(row) for row in rows)
    sample_ids = [row.sample_id for row in viewed]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("sample_id values must be nonblank")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    groups = _Groups(len(viewed))
    for field in ("source_sha256", "video_id", "speaker_id", "talk_id"):
        seen: dict[str, int] = {}
        for index, row in enumerate(viewed):
            value = getattr(row, field)
            if not value or value == "unknown":
                continue
            if value in seen:
                groups.union(index, seen[value])
            else:
                seen[value] = index

    members: dict[int, list[int]] = {}
    for index in range(len(viewed)):
        members.setdefault(groups.find(index), []).append(index)
    ordered_components = sorted(
        members.values(),
        key=lambda indexes: tuple(sorted(viewed[index].sample_id for index in indexes)),
    )
    shuffled_components = list(ordered_components)
    random.Random(seed).shuffle(shuffled_components)
    group_count = len(shuffled_components)
    outer_count = min(5, group_count)
    inner_count = min(4, group_count)
    outer_method = "group_kfold" if group_count >= 5 else "leave_one_group_out"
    inner_method = "group_kfold" if group_count >= 4 else "leave_one_group_out"

    assignments: list[SplitAssignment] = []
    for position, indexes in enumerate(shuffled_components):
        component_ids = sorted(viewed[index].sample_id for index in indexes)
        group_id = "group-" + sha256_bytes(
            canonical_json_bytes({"sample_ids": component_ids})
        )[:16]
        outer_fold = position % max(outer_count, 1)
        inner_fold = position % max(inner_count, 1)
        for index in indexes:
            row = viewed[index]
            assignments.append(
                SplitAssignment(
                    sample_id=row.sample_id,
                    group_id=group_id,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    speaker_id=row.speaker_id,
                    talk_id=row.talk_id,
                    video_id=row.video_id,
                    source_sha256=row.source_sha256,
                    modelling_scope=row.modelling_scope,
                    supervised_eligible=row.supervised_eligible,
                    qc_eligible=row.qc_eligible,
                    clustering_eligible=row.clustering_eligible,
                )
            )

    return SplitRegistry(
        seed=seed,
        assignments=tuple(sorted(assignments, key=lambda item: item.sample_id)),
        outer_method=outer_method,
        outer_fold_count=outer_count,
        inner_method=inner_method,
        inner_fold_count=inner_count,
    )


def _row_view(row: object) -> _RowView:
    sample_id = str(_field(row, "sample_id", "")).strip()
    video_id = str(_field(row, "video_id", "")).strip()
    raw_source_sha256 = _field(row, "source_sha256", None)
    source_sha256 = None if raw_source_sha256 in (None, "") else str(raw_source_sha256)
    raw_speaker = _field(row, "speaker_id", None)
    speaker_id = "unknown" if raw_speaker in (None, "") else str(raw_speaker).strip()
    if not speaker_id:
        speaker_id = "unknown"
    raw_talk = _field(row, "talk_id", None)
    talk_id = str(raw_talk).strip() if raw_talk not in (None, "") else (video_id or "unknown")
    score_status = _enum_value(_field(row, "score_status", ""))
    has_label = score_status == "observed" and _field(row, "total_score", None) is not None
    raw_supervised_eligible = _field(row, "supervised_eligible", has_label)
    if type(raw_supervised_eligible) is not bool:
        raise TypeError("supervised_eligible must be boolean")
    modelling_scope: ModellingScope = (
        "supervised_exploratory" if has_label else "unsupervised_only"
    )
    return _RowView(
        sample_id=sample_id,
        video_id=video_id,
        source_sha256=source_sha256,
        speaker_id=speaker_id,
        talk_id=talk_id,
        modelling_scope=modelling_scope,
        supervised_eligible=has_label and raw_supervised_eligible,
        qc_eligible=bool(_field(row, "qc_eligible", True)),
        clustering_eligible=bool(_field(row, "clustering_eligible", True)),
    )


def _field(row: object, name: str, default: object) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _enum_value(value: object) -> str:
    return str(value.value if isinstance(value, Enum) else value)
