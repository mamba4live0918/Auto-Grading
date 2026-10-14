"""Authority-bound, fail-closed gates for optional WSL2 QLoRA training."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import yaml

from . import dataset_pipeline
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .feature_registry import FeatureRegistry
from .protocol import BoundProtocol


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_QLORA_AUTHORITY_REGISTRY = _PROJECT_ROOT / "configs/qlora_experiment_registry.yaml"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY = object()
_CAPABILITY_KEY = secrets.token_bytes(32)
_REQUIRED_ARTIFACTS = (
    "experiment_manifest",
    "rubric",
    "rubric_labels",
    "evidence_annotations",
    "base_checkpoint",
    "runtime",
    "training_manifest",
    "hardware_manifest",
)


@dataclass(frozen=True, slots=True)
class _AuthorizedArtifact:
    name: str
    path: str
    sha256: str
    valid_at_load: bool


@dataclass(frozen=True, slots=True, init=False)
class VerifiedQloraContext:
    dataset_id: str
    protocol_id: str
    split_sha256: str
    sample_count: int
    qlora_sample_ids: tuple[str, ...]
    qlora_sample_subset_valid: bool
    registry_path: str
    registry_sha256: str
    authority_entry: bool
    protocol_enabled: bool
    frozen_protocol: bool
    transcript_only_task: bool
    frozen_target: bool
    primary_endpoint: bool
    minimum_meaningful_improvement: bool
    failure_criteria: bool
    advisor_total_score_only: bool
    confirmatory_sample_size_protocol: bool
    quality_checked_transcripts: bool
    construct_aligned_rubric_labels: bool
    human_evidence_annotations: bool
    frozen_construct_aligned_rubric: bool
    reliable_independent_speaker_groups: bool
    split_overlap_zero: bool
    locked_test: bool
    training_configuration_frozen: bool
    verified_base_checkpoint_artifact: bool
    verified_runtime_artifact: bool
    adapter_only_model_difference: bool
    wsl2: bool
    available_vram_gb: float | None
    frozen_setup_preflight_passed: bool
    training_command: tuple[str, ...] | None
    checkpoint_path: str | None
    artifacts: tuple[_AuthorizedArtifact, ...]
    _capability: object
    _seal: str

    def __init__(self, *, capability: object, **facts: object) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("VerifiedQloraContext requires the production authority loader")
        for name in _CONTEXT_FIELDS:
            object.__setattr__(self, name, facts[name])
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _context_seal(self))


@dataclass(frozen=True, slots=True, init=False)
class SyntheticQloraFixture:
    sample_count: int
    seed: int
    fixture_sha256: str
    _capability: object
    _seal: str

    def __init__(self, *, sample_count: int, seed: int, capability: object) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("SyntheticQloraFixture requires the generator")
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "sample_ids": [
                        f"synthetic-{seed}-{index}" for index in range(sample_count)
                    ]
                }
            )
        )
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fixture_sha256", digest)
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _seal((str(sample_count), str(seed), digest)))


@dataclass(frozen=True, slots=True)
class TrainingGateResult:
    status: Literal["ready", "not_run", "method_feasibility_only"]
    reasons: tuple[str, ...]
    interpretation: Literal[
        "formal_training_not_started",
        "not_effectiveness_evidence",
        "method_feasibility_only",
    ]
    training_command: tuple[str, ...] | None
    checkpoint_path: str | None
    metrics: Mapping[str, float] | None


_CONTEXT_FIELDS = (
    "dataset_id",
    "protocol_id",
    "split_sha256",
    "sample_count",
    "qlora_sample_ids",
    "qlora_sample_subset_valid",
    "registry_path",
    "registry_sha256",
    "authority_entry",
    "protocol_enabled",
    "frozen_protocol",
    "transcript_only_task",
    "frozen_target",
    "primary_endpoint",
    "minimum_meaningful_improvement",
    "failure_criteria",
    "advisor_total_score_only",
    "confirmatory_sample_size_protocol",
    "quality_checked_transcripts",
    "construct_aligned_rubric_labels",
    "human_evidence_annotations",
    "frozen_construct_aligned_rubric",
    "reliable_independent_speaker_groups",
    "split_overlap_zero",
    "locked_test",
    "training_configuration_frozen",
    "verified_base_checkpoint_artifact",
    "verified_runtime_artifact",
    "adapter_only_model_difference",
    "wsl2",
    "available_vram_gb",
    "frozen_setup_preflight_passed",
    "training_command",
    "checkpoint_path",
    "artifacts",
)


def load_verified_qlora_context(
    dataset_dir: str | Path, registry: FeatureRegistry
) -> VerifiedQloraContext:
    """Derive all gates from strict frozen data and the committed authority registry."""
    loaded = dataset_pipeline.load_strict_frozen_dataset(Path(dataset_dir), registry)
    if not isinstance(loaded.protocol, BoundProtocol):
        raise ValueError("strict loader did not return a bound protocol")
    root = Path(loaded.path)
    splits = _json_mapping(root / "splits.json")
    manifest_rows = _csv_rows(root / "manifest.csv")
    transcripts = _json_lines(root / "transcripts.jsonl")
    sample_ids = set(loaded.sample_ids)
    assignments = splits.get("assignments")
    if (
        {row.get("sample_id") for row in manifest_rows} != sample_ids
        or {row.get("sample_id") for row in transcripts} != sample_ids
        or not isinstance(assignments, list)
        or {item.get("sample_id") for item in assignments if isinstance(item, Mapping)}
        != sample_ids
    ):
        raise ValueError("strict dataset views disagree on sample identities")
    if sha256_bytes(canonical_json_bytes(splits)) != loaded.protocol.split_sha256:
        raise ValueError("strict split registry does not match bound protocol")

    registry_path = Path(_QLORA_AUTHORITY_REGISTRY).resolve(strict=True)
    authority_root = _load_authority_registry(registry_path)
    entries = authority_root["entries"]
    matching = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("dataset_id") == loaded.dataset_id
        and entry.get("protocol_id") == loaded.protocol.protocol_id
    ]
    if len(matching) > 1:
        raise ValueError("QLoRA authority contains duplicate dataset/protocol entry")
    entry = matching[0] if matching else None
    qlora_sample_ids, qlora_sample_subset_valid = _authority_subset(
        entry, sample_ids, assignments if isinstance(assignments, list) else []
    )
    qlora_sample_id_set = set(qlora_sample_ids) if entry is not None else sample_ids
    subset_assignments = [
        item
        for item in assignments
        if isinstance(item, Mapping) and item.get("sample_id") in qlora_sample_id_set
    ]
    subset_transcripts = [
        row for row in transcripts if row.get("sample_id") in qlora_sample_id_set
    ]
    artifacts = _authority_artifacts(entry, registry_path.parent)
    by_name = {artifact.name: artifact for artifact in artifacts}

    experiment = _artifact_json(by_name.get("experiment_manifest"))
    rubric_payload = _artifact_json(by_name.get("rubric"))
    labels_payload = _artifact_json(by_name.get("rubric_labels"))
    evidence_payload = _artifact_json(by_name.get("evidence_annotations"))
    training = _artifact_json(by_name.get("training_manifest"))
    hardware = _artifact_json(by_name.get("hardware_manifest"))
    template = loaded.protocol.template

    frozen_protocol = _experiment_binding_matches(experiment, loaded.protocol)
    target = experiment.get("target") if frozen_protocol else None
    frozen_target = (
        isinstance(target, Mapping)
        and target.get("name") == template.target.name
        and _same_number(target.get("scale_min"), template.target.scale_min)
        and _same_number(target.get("scale_max"), template.target.scale_max)
    )
    transcript_only = frozen_protocol and experiment.get("task_type") == "transcript_only"
    primary_endpoint = frozen_protocol and isinstance(
        experiment.get("primary_endpoint"), str
    ) and bool(str(experiment["primary_endpoint"]).strip())
    improvement = experiment.get("minimum_meaningful_improvement") if frozen_protocol else None
    improvement_ok = _positive_finite(improvement)
    failure_raw = experiment.get("failure_criteria") if frozen_protocol else None
    failure_ok = (
        isinstance(failure_raw, list)
        and bool(failure_raw)
        and all(isinstance(item, str) and item.strip() for item in failure_raw)
    )
    sample_size_ok = _valid_sample_size_justification(
        experiment.get("sample_size_justification") if frozen_protocol else None
    )
    rubric_ok, labels_ok = _valid_rubric_and_labels(
        rubric_payload, labels_payload, qlora_sample_id_set, template.target.name
    )
    evidence_ok = _valid_evidence(
        evidence_payload, qlora_sample_id_set, template.target.name
    )
    partitions = _partitions(subset_assignments)
    subset_gate = qlora_sample_subset_valid if entry is not None else True
    speakers_ok = subset_gate and _speaker_independence(
        subset_assignments, partitions
    )
    overlap_zero = subset_gate and _cross_partition_overlap_zero(
        subset_assignments, partitions
    )
    locked_test = (
        frozen_protocol
        and experiment.get("test_partition") == "test"
        and "train" in set(partitions.values())
        and "test" in set(partitions.values())
        and splits.get("no_independent_holdout_claim") is False
    )
    training_ok, command, checkpoint_path = _valid_training(
        training, loaded.dataset_id, loaded.protocol.protocol_id
    )
    adapter_only = frozen_protocol and experiment.get("adapter_only_model_difference") is True
    hardware_ok, wsl2, vram, preflight = _valid_hardware(hardware)
    quality_transcripts = (
        subset_gate
        and len(subset_transcripts) == len(qlora_sample_id_set)
        and {row.get("sample_id") for row in subset_transcripts}
        == qlora_sample_id_set
        and all(_usable_transcript(row) for row in subset_transcripts)
    )
    base_ok = _artifact_current(by_name.get("base_checkpoint"))
    runtime_ok = _artifact_current(by_name.get("runtime"))

    facts: dict[str, object] = {
        "dataset_id": loaded.dataset_id,
        "protocol_id": loaded.protocol.protocol_id,
        "split_sha256": loaded.protocol.split_sha256,
        "sample_count": loaded.row_count,
        "qlora_sample_ids": qlora_sample_ids,
        "qlora_sample_subset_valid": qlora_sample_subset_valid,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "authority_entry": entry is not None,
        "protocol_enabled": template.qlora.enabled is True,
        "frozen_protocol": frozen_protocol,
        "transcript_only_task": transcript_only,
        "frozen_target": frozen_target,
        "primary_endpoint": primary_endpoint,
        "minimum_meaningful_improvement": improvement_ok,
        "failure_criteria": failure_ok,
        "advisor_total_score_only": (
            template.target.label_source == "advisor_total_score"
            and not labels_ok
            and not evidence_ok
        ),
        "confirmatory_sample_size_protocol": sample_size_ok,
        "quality_checked_transcripts": quality_transcripts,
        "construct_aligned_rubric_labels": rubric_ok and labels_ok,
        "human_evidence_annotations": evidence_ok,
        "frozen_construct_aligned_rubric": (
            rubric_ok and template.target.rubric_level_enabled is True
        ),
        "reliable_independent_speaker_groups": speakers_ok,
        "split_overlap_zero": overlap_zero,
        "locked_test": locked_test,
        "training_configuration_frozen": training_ok,
        "verified_base_checkpoint_artifact": base_ok,
        "verified_runtime_artifact": runtime_ok,
        "adapter_only_model_difference": adapter_only,
        "wsl2": wsl2,
        "available_vram_gb": vram,
        "frozen_setup_preflight_passed": preflight,
        "training_command": command,
        "checkpoint_path": checkpoint_path,
        "artifacts": artifacts,
    }
    return VerifiedQloraContext(capability=_CAPABILITY, **facts)


def create_synthetic_qlora_fixture(
    *, sample_count: int, seed: int
) -> SyntheticQloraFixture:
    if type(sample_count) is not int or sample_count < 2 or type(seed) is not int:
        raise ValueError("synthetic fixture requires sample_count >= 2 and integer seed")
    return SyntheticQloraFixture(
        sample_count=sample_count, seed=seed, capability=_CAPABILITY
    )


def evaluate_qlora_gates(
    context: VerifiedQloraContext | SyntheticQloraFixture,
) -> TrainingGateResult:
    if _valid_synthetic(context):
        return TrainingGateResult(
            "method_feasibility_only",
            ("generated_synthetic_fixture_only",),
            "method_feasibility_only",
            None,
            None,
            None,
        )
    if not _valid_context(context):
        return _not_run(("context_not_verified",))

    registry_current = _file_matches(context.registry_path, context.registry_sha256)
    current = {
        artifact.name: registry_current and _artifact_current(artifact)
        for artifact in context.artifacts
    }
    experiment_current = current.get("experiment_manifest", False)
    rubric_current = current.get("rubric", False)
    labels_current = current.get("rubric_labels", False)
    evidence_current = current.get("evidence_annotations", False)
    training_current = current.get("training_manifest", False)
    hardware_current = current.get("hardware_manifest", False)
    effective_vram = context.available_vram_gb if hardware_current else None
    checks = (
        (context.authority_entry and registry_current, "authority_entry_missing"),
        (
            not context.authority_entry or context.qlora_sample_subset_valid,
            "qlora_sample_subset_invalid",
        ),
        (context.protocol_enabled, "protocol_disables_qlora"),
        (context.frozen_protocol and experiment_current, "frozen_protocol_missing"),
        (
            context.transcript_only_task and experiment_current,
            "transcript_only_task_missing",
        ),
        (context.frozen_target and experiment_current, "frozen_target_missing"),
        (context.primary_endpoint and experiment_current, "primary_endpoint_missing"),
        (
            context.minimum_meaningful_improvement and experiment_current,
            "minimum_meaningful_improvement_missing",
        ),
        (context.failure_criteria and experiment_current, "failure_criteria_missing"),
        (not context.advisor_total_score_only, "advisor_total_score_only"),
        (
            context.confirmatory_sample_size_protocol and experiment_current,
            "confirmatory_sample_size_protocol_missing",
        ),
        (
            context.quality_checked_transcripts,
            "quality_checked_transcripts_missing",
        ),
        (
            context.construct_aligned_rubric_labels
            and rubric_current
            and labels_current,
            "construct_aligned_rubric_labels_missing",
        ),
        (
            context.human_evidence_annotations and evidence_current,
            "human_evidence_annotations_missing",
        ),
        (
            context.frozen_construct_aligned_rubric and rubric_current,
            "frozen_construct_aligned_rubric_missing",
        ),
        (
            context.reliable_independent_speaker_groups,
            "reliable_independent_speaker_groups_missing",
        ),
        (context.split_overlap_zero, "split_overlap_zero_missing"),
        (context.locked_test and experiment_current, "locked_test_missing"),
        (
            context.training_configuration_frozen and training_current,
            "training_configuration_not_frozen",
        ),
        (
            context.verified_base_checkpoint_artifact
            and current.get("base_checkpoint", False),
            "verified_base_checkpoint_artifact_missing",
        ),
        (
            context.verified_runtime_artifact and current.get("runtime", False),
            "verified_runtime_artifact_missing",
        ),
        (
            context.adapter_only_model_difference and experiment_current,
            "adapter_only_difference_not_guaranteed",
        ),
        (context.wsl2 and hardware_current, "wsl2_unavailable"),
        (effective_vram is not None, "vram_unavailable"),
        (
            effective_vram is None or float(effective_vram) >= 12.0,
            "vram_below_12_gb",
        ),
        (
            context.frozen_setup_preflight_passed and hardware_current,
            "wsl2_vram_preflight_failed",
        ),
        (
            context.training_command is not None and training_current,
            "training_command_missing",
        ),
        (
            context.checkpoint_path is not None and training_current,
            "checkpoint_path_missing",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    if reasons:
        return _not_run(reasons)
    return TrainingGateResult(
        "ready",
        (),
        "formal_training_not_started",
        context.training_command,
        context.checkpoint_path,
        None,
    )


def _load_authority_registry(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("invalid QLoRA authority registry") from error
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "registry_name", "entries"}
        or raw.get("schema_version") != 1
        or raw.get("registry_name") != "qlora_experiment_registry"
        or not isinstance(raw.get("entries"), list)
    ):
        raise ValueError("invalid QLoRA authority registry")
    return raw


def _authority_artifacts(
    entry: object, registry_root: Path
) -> tuple[_AuthorizedArtifact, ...]:
    if not isinstance(entry, Mapping) or set(entry) != {
        "dataset_id",
        "protocol_id",
        "qlora_sample_ids",
        "artifacts",
    }:
        return ()
    raw = entry.get("artifacts")
    if not isinstance(raw, Mapping) or set(raw) != set(_REQUIRED_ARTIFACTS):
        return ()
    artifacts: list[_AuthorizedArtifact] = []
    for name in _REQUIRED_ARTIFACTS:
        item = raw[name]
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            return ()
        path_value, digest = item.get("path"), item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            return ()
        path = Path(path_value)
        if not path.is_absolute():
            path = registry_root / path
        try:
            path = path.resolve(strict=True)
        except OSError:
            path = path.resolve()
        artifacts.append(
            _AuthorizedArtifact(
                name=name,
                path=str(path),
                sha256=digest,
                valid_at_load=_file_matches(str(path), digest),
            )
        )
    return tuple(artifacts)


def _authority_subset(
    entry: object,
    dataset_sample_ids: set[str],
    assignments: list[object],
) -> tuple[tuple[str, ...], bool]:
    if not isinstance(entry, Mapping):
        return (), False
    raw = entry.get("qlora_sample_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item.strip() for item in raw)
    ):
        return (), False
    sample_ids = tuple(str(item) for item in raw)
    if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= dataset_sample_ids:
        return sample_ids, False
    selected = [
        item
        for item in assignments
        if isinstance(item, Mapping) and item.get("sample_id") in set(sample_ids)
    ]
    partitions = _partitions(selected)
    split_safe = (
        len(selected) == len(sample_ids)
        and set(partitions) == set(sample_ids)
        and _speaker_independence(selected, partitions)
        and _cross_partition_overlap_zero(selected, partitions)
    )
    return sample_ids, split_safe


def _experiment_binding_matches(
    payload: Mapping[str, object], protocol: BoundProtocol
) -> bool:
    return (
        payload.get("schema_version") == 1
        and payload.get("dataset_id") == protocol.dataset_id
        and payload.get("protocol_id") == protocol.protocol_id
        and payload.get("split_sha256") == protocol.split_sha256
    )


def _valid_sample_size_justification(raw: object) -> bool:
    return (
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "pilot_variance",
            "human_score_reliability",
            "target_coverage",
            "desired_ci_precision",
        }
        and _positive_finite(raw.get("pilot_variance"))
        and _positive_finite(raw.get("human_score_reliability"))
        and float(raw["human_score_reliability"]) <= 1.0
        and isinstance(raw.get("target_coverage"), str)
        and bool(str(raw["target_coverage"]).strip())
        and _positive_finite(raw.get("desired_ci_precision"))
    )


def _valid_rubric_and_labels(
    rubric: Mapping[str, object],
    labels: Mapping[str, object],
    sample_ids: set[str],
    target_name: str,
) -> tuple[bool, bool]:
    dimensions = rubric.get("dimensions")
    rubric_ok = (
        rubric.get("schema_version") == 1
        and isinstance(rubric.get("rubric_id"), str)
        and rubric.get("target_name") == target_name
        and isinstance(dimensions, list)
        and bool(dimensions)
        and all(isinstance(item, str) and item.strip() for item in dimensions)
    )
    raw_labels = labels.get("labels")
    labels_ok = (
        rubric_ok
        and labels.get("schema_version") == 1
        and labels.get("rubric_id") == rubric.get("rubric_id")
        and labels.get("target_name") == target_name
        and isinstance(raw_labels, list)
        and {item.get("sample_id") for item in raw_labels if isinstance(item, Mapping)}
        == sample_ids
        and all(
            isinstance(item, Mapping)
            and all(dimension in item for dimension in dimensions)
            for item in raw_labels
        )
    )
    return rubric_ok, labels_ok


def _valid_evidence(
    payload: Mapping[str, object], sample_ids: set[str], target_name: str
) -> bool:
    annotations = payload.get("annotations")
    return (
        payload.get("schema_version") == 1
        and payload.get("target_name") == target_name
        and isinstance(annotations, list)
        and {item.get("sample_id") for item in annotations if isinstance(item, Mapping)}
        == sample_ids
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("quote"), str)
            and bool(str(item["quote"]).strip())
            for item in annotations
        )
    )


def _valid_training(
    payload: Mapping[str, object], dataset_id: str, protocol_id: str
) -> tuple[bool, tuple[str, ...] | None, str | None]:
    command = payload.get("training_command")
    checkpoint = payload.get("checkpoint_path")
    valid = (
        payload.get("schema_version") == 1
        and payload.get("dataset_id") == dataset_id
        and payload.get("protocol_id") == protocol_id
        and payload.get("training_configuration_frozen") is True
        and isinstance(command, list)
        and bool(command)
        and all(isinstance(item, str) and item.strip() for item in command)
        and isinstance(checkpoint, str)
        and bool(checkpoint.strip())
    )
    return (
        valid,
        tuple(command) if valid else None,
        checkpoint if valid and isinstance(checkpoint, str) else None,
    )


def _valid_hardware(
    payload: Mapping[str, object],
) -> tuple[bool, bool, float | None, bool]:
    vram = payload.get("available_vram_gb")
    valid = (
        payload.get("schema_version") == 1
        and type(payload.get("wsl2")) is bool
        and type(payload.get("frozen_setup_preflight_passed")) is bool
        and not isinstance(vram, bool)
        and isinstance(vram, (int, float))
        and math.isfinite(float(vram))
        and float(vram) >= 0.0
    )
    return (
        valid,
        bool(payload.get("wsl2")) if valid else False,
        float(vram) if valid else None,
        bool(payload.get("frozen_setup_preflight_passed")) if valid else False,
    )


def _partitions(assignments: list[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            return {}
        sample_id, partition = item.get("sample_id"), item.get("partition")
        if not isinstance(sample_id, str) or partition not in {
            "train",
            "validation",
            "test",
        }:
            return {}
        result[sample_id] = str(partition)
    return result


def _speaker_independence(
    assignments: list[object], partitions: Mapping[str, str]
) -> bool:
    if not partitions:
        return False
    memberships: dict[str, set[str]] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            return False
        speaker = item.get("speaker_id")
        sample_id = item.get("sample_id")
        if not isinstance(speaker, str) or speaker in {"", "unknown"}:
            return False
        memberships.setdefault(speaker, set()).add(partitions[str(sample_id)])
    return all(len(values) == 1 for values in memberships.values())


def _cross_partition_overlap_zero(
    assignments: list[object], partitions: Mapping[str, str]
) -> bool:
    if not partitions or "train" not in set(partitions.values()) or "test" not in set(
        partitions.values()
    ):
        return False
    for field in ("group_id", "source_sha256", "video_id"):
        memberships: dict[str, set[str]] = {}
        for item in assignments:
            if not isinstance(item, Mapping):
                return False
            value, sample_id = item.get(field), item.get("sample_id")
            if not isinstance(value, str) or not value:
                return False
            memberships.setdefault(value, set()).add(partitions[str(sample_id)])
        if any(len(values) > 1 for values in memberships.values()):
            return False
    return True


def _valid_context(value: object) -> bool:
    if not isinstance(value, VerifiedQloraContext):
        return False
    try:
        return value._capability is _CAPABILITY and hmac.compare_digest(
            value._seal, _context_seal(value)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _valid_synthetic(value: object) -> bool:
    if not isinstance(value, SyntheticQloraFixture):
        return False
    try:
        return value._capability is _CAPABILITY and hmac.compare_digest(
            value._seal,
            _seal((str(value.sample_count), str(value.seed), value.fixture_sha256)),
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _context_seal(value: VerifiedQloraContext) -> str:
    values: list[str] = []
    for name in _CONTEXT_FIELDS:
        field = getattr(value, name)
        if name == "artifacts":
            for artifact in field:
                values.extend(
                    (
                        artifact.name,
                        artifact.path,
                        artifact.sha256,
                        str(artifact.valid_at_load),
                    )
                )
        elif isinstance(field, tuple):
            values.extend((name, *field))
        else:
            values.extend((name, repr(field)))
    return _seal(values)


def _seal(values: Sequence[str]) -> str:
    return hmac.new(
        _CAPABILITY_KEY, "\0".join(values).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _artifact_json(artifact: _AuthorizedArtifact | None) -> Mapping[str, object]:
    if not _artifact_current(artifact):
        return {}
    return _json_mapping(Path(artifact.path))


def _artifact_current(artifact: _AuthorizedArtifact | None) -> bool:
    return (
        artifact is not None
        and artifact.valid_at_load
        and _file_matches(artifact.path, artifact.sha256)
    )


def _file_matches(path_value: str, digest: str) -> bool:
    if _SHA256.fullmatch(digest) is None:
        return False
    try:
        path = Path(path_value).resolve(strict=True)
        return path.is_file() and hmac.compare_digest(sha256_file(path), digest)
    except OSError:
        return False


def _json_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"invalid JSON artifact: {path.name}")
    return raw


def _json_lines(path: Path) -> list[Mapping[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid transcript rows") from error
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("invalid transcript rows")
    return rows


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _usable_transcript(row: Mapping[str, object]) -> bool:
    transcription = row.get("transcription")
    return (
        row.get("status") == "succeeded"
        and isinstance(transcription, Mapping)
        and isinstance(transcription.get("text"), str)
        and bool(str(transcription["text"]).strip())
    )


def _same_number(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == float(expected)
    )


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _not_run(reasons: tuple[str, ...]) -> TrainingGateResult:
    return TrainingGateResult(
        "not_run", reasons, "not_effectiveness_evidence", None, None, None
    )
