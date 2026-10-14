from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import presentation_assessment.research.qlora as qlora_module
from presentation_assessment.research.canonical import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from presentation_assessment.research.protocol import bind_protocol, load_protocol_template
from presentation_assessment.research.qlora import (
    VerifiedQloraContext,
    create_synthetic_qlora_fixture,
    evaluate_qlora_gates,
    load_verified_qlora_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPTY_REGISTRY = PROJECT_ROOT / "configs/qlora_experiment_registry.yaml"


def _dataset_shell(root: Path, *, ready: bool, overlap: bool = False):
    root.mkdir()
    sample_ids = tuple(f"S{index:03d}" for index in range(6 if ready else 2))
    partitions = ("train", "train", "validation", "validation", "test", "test")
    assignments: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        partition = partitions[index] if ready else "exploratory"
        assignments.append(
            {
                "sample_id": sample_id,
                "speaker_id": f"speaker-{index}" if ready else "unknown",
                "group_id": "duplicate-across-test" if overlap and index in (0, 4) else f"group-{index}",
                "source_sha256": "a" * 64 if overlap and index in (0, 4) else f"{index + 1:064x}",
                "video_id": "same-video" if overlap and index in (0, 4) else f"video-{index}",
                "partition": partition,
                "outer_fold": index % 3,
            }
        )
    split_payload: dict[str, object] = {
        "schema_version": 1,
        "speaker_identity_status": "reliably_available" if ready else "not_reliably_available",
        "speaker_disjoint": ready,
        "no_independent_holdout_claim": not ready,
        "assignments": assignments,
    }
    (root / "splits.json").write_bytes(canonical_json_bytes(split_payload))
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "total_score", "score_status", "speaker_id"),
        )
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "total_score": str(60 + index),
                    "score_status": "observed",
                    "speaker_id": assignments[index]["speaker_id"],
                }
            )
    (root / "transcripts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "status": "succeeded",
                    "transcription": {"text": f"synthetic transcript {sample_id}"},
                }
            )
            + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    template = load_protocol_template(
        PROJECT_ROOT / "configs/experiment_protocol.template.yaml"
    )
    if ready:
        template = template.model_copy(
            update={
                "target": template.target.model_copy(update={"rubric_level_enabled": True}),
                "split": template.split.model_copy(update={"no_independent_holdout_claim": False}),
                "qlora": template.qlora.model_copy(update={"enabled": True}),
            }
        )
    protocol = bind_protocol(
        template,
        dataset_id=("e" if ready else "d") * 64,
        split_sha256=sha256_bytes(canonical_json_bytes(split_payload)),
    )
    loaded = SimpleNamespace(
        path=root,
        dataset_id=protocol.dataset_id,
        protocol=protocol,
        row_count=len(sample_ids),
        sample_ids=frozenset(sample_ids),
    )
    return loaded, split_payload, sample_ids


def _install_loader(monkeypatch: pytest.MonkeyPatch, loaded: object) -> None:
    monkeypatch.setattr(
        qlora_module.dataset_pipeline,
        "load_strict_frozen_dataset",
        lambda path, registry: loaded,
    )


def _install_authority(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(qlora_module, "_QLORA_AUTHORITY_REGISTRY", path)


def _write_ready_authority(
    tmp_path: Path,
    loaded: object,
    sample_ids: tuple[str, ...],
    *,
    qlora_sample_ids: tuple[str, ...] | None = None,
) -> tuple[Path, dict[str, Path]]:
    protocol = loaded.protocol
    authority_sample_ids = sample_ids if qlora_sample_ids is None else qlora_sample_ids
    artifacts: dict[str, Path] = {}

    def write_json(name: str, payload: object) -> Path:
        path = tmp_path / name
        path.write_bytes(canonical_json_bytes(payload))
        artifacts[name] = path
        return path

    experiment = write_json(
        "experiment.json",
        {
            "schema_version": 1,
            "dataset_id": loaded.dataset_id,
            "protocol_id": protocol.protocol_id,
            "split_sha256": protocol.split_sha256,
            "task_type": "transcript_only",
            "target": {"name": "total_score", "scale_min": 0.0, "scale_max": 100.0},
            "primary_endpoint": "mae",
            "minimum_meaningful_improvement": 1.0,
            "failure_criteria": ["mae_not_improved", "interval_too_wide"],
            "sample_size_justification": {
                "pilot_variance": 12.0,
                "human_score_reliability": 0.8,
                "target_coverage": "0-100",
                "desired_ci_precision": 2.0,
            },
            "test_partition": "test",
            "adapter_only_model_difference": True,
        },
    )
    rubric = write_json(
        "rubric.json",
        {
            "schema_version": 1,
            "rubric_id": "synthetic-rubric-v1",
            "target_name": "total_score",
            "dimensions": ["clarity", "structure"],
        },
    )
    rubric_labels = write_json(
        "rubric-labels.json",
        {
            "schema_version": 1,
            "rubric_id": "synthetic-rubric-v1",
            "target_name": "total_score",
            "labels": [
                {"sample_id": sample_id, "clarity": 3, "structure": 4}
                for sample_id in authority_sample_ids
            ],
        },
    )
    evidence = write_json(
        "evidence.json",
        {
            "schema_version": 1,
            "target_name": "total_score",
            "annotations": [
                {"sample_id": sample_id, "quote": "synthetic"}
                for sample_id in authority_sample_ids
            ],
        },
    )
    checkpoint = tmp_path / "base-checkpoint.bin"
    runtime = tmp_path / "runtime.lock"
    checkpoint.write_bytes(b"verified synthetic qwen checkpoint")
    runtime.write_bytes(b"verified synthetic transformers peft runtime")
    artifacts["base-checkpoint.bin"] = checkpoint
    artifacts["runtime.lock"] = runtime
    training = write_json(
        "training.json",
        {
            "schema_version": 1,
            "dataset_id": loaded.dataset_id,
            "protocol_id": protocol.protocol_id,
            "training_configuration_frozen": True,
            "training_command": ["python", "-m", "train_qlora", "--locked"],
            "checkpoint_path": str(tmp_path / "planned-adapter"),
        },
    )
    hardware = write_json(
        "hardware.json",
        {
            "schema_version": 1,
            "wsl2": True,
            "available_vram_gb": 12.0,
            "frozen_setup_preflight_passed": True,
        },
    )
    named = {
        "experiment_manifest": experiment,
        "rubric": rubric,
        "rubric_labels": rubric_labels,
        "evidence_annotations": evidence,
        "base_checkpoint": checkpoint,
        "runtime": runtime,
        "training_manifest": training,
        "hardware_manifest": hardware,
    }
    authority = tmp_path / "qlora-registry.yaml"
    authority.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "registry_name": "qlora_experiment_registry",
                "entries": [
                    {
                        "dataset_id": loaded.dataset_id,
                        "protocol_id": protocol.protocol_id,
                        "qlora_sample_ids": list(authority_sample_ids),
                        "artifacts": {
                            key: {"path": str(path), "sha256": sha256_file(path)}
                            for key, path in named.items()
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return authority, named


def test_committed_authority_is_versioned_and_empty_for_current_dataset() -> None:
    """Break caught: local manifests become authorized without a committed entry."""
    payload = yaml.safe_load(EMPTY_REGISTRY.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "registry_name": "qlora_experiment_registry",
        "entries": [],
    }


def test_current_strict_dataset_enumerates_every_missing_mandatory_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: absent authority omits protocol, semantic, or execution gates."""
    loaded, _, _ = _dataset_shell(tmp_path / "current", ready=False)
    _install_loader(monkeypatch, loaded)
    context = load_verified_qlora_context(loaded.path, object())

    result = evaluate_qlora_gates(context)

    assert result.status == "not_run"
    assert result.reasons == (
        "authority_entry_missing",
        "protocol_disables_qlora",
        "frozen_protocol_missing",
        "transcript_only_task_missing",
        "frozen_target_missing",
        "primary_endpoint_missing",
        "minimum_meaningful_improvement_missing",
        "failure_criteria_missing",
        "advisor_total_score_only",
        "confirmatory_sample_size_protocol_missing",
        "construct_aligned_rubric_labels_missing",
        "human_evidence_annotations_missing",
        "frozen_construct_aligned_rubric_missing",
        "reliable_independent_speaker_groups_missing",
        "split_overlap_zero_missing",
        "locked_test_missing",
        "training_configuration_not_frozen",
        "verified_base_checkpoint_artifact_missing",
        "verified_runtime_artifact_missing",
        "adapter_only_difference_not_guaranteed",
        "wsl2_unavailable",
        "vram_unavailable",
        "wsl2_vram_preflight_failed",
        "training_command_missing",
        "checkpoint_path_missing",
    )
    assert result.training_command is None
    assert result.checkpoint_path is None
    assert result.metrics is None


def test_fully_authorized_synthetic_context_has_legitimate_ready_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: mandatory gates can never pass through verified production context."""
    loaded, _, sample_ids = _dataset_shell(tmp_path / "ready", ready=True)
    authority, _ = _write_ready_authority(tmp_path, loaded, sample_ids)
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)

    context = load_verified_qlora_context(loaded.path, object())
    result = evaluate_qlora_gates(context)

    assert result.status == "ready"
    assert result.reasons == ()
    assert result.interpretation == "formal_training_not_started"
    assert result.training_command == ("python", "-m", "train_qlora", "--locked")
    assert result.checkpoint_path == str(tmp_path / "planned-adapter")
    assert result.metrics is None


@pytest.mark.parametrize(
    "broken_transcript",
    [
        {"sample_id": "S000", "status": "failed", "transcription": None},
        {
            "sample_id": "S000",
            "status": "succeeded",
            "transcription": {"text": "   "},
        },
    ],
    ids=("failed", "blank"),
)
def test_ready_authority_is_blocked_by_failed_or_blank_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken_transcript: dict[str, object],
) -> None:
    """Break caught: complete authority masks an unusable training transcript."""
    loaded, _, sample_ids = _dataset_shell(tmp_path / "ready", ready=True)
    authority, _ = _write_ready_authority(tmp_path, loaded, sample_ids)
    transcript_path = Path(loaded.path) / "transcripts.jsonl"
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows[0] = broken_transcript
    transcript_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)

    result = evaluate_qlora_gates(
        load_verified_qlora_context(loaded.path, object())
    )

    assert result.status == "not_run"
    assert result.reasons == ("quality_checked_transcripts_missing",)
    assert result.training_command is None
    assert result.checkpoint_path is None
    assert result.metrics is None


def test_authorized_qlora_subset_ignores_unrelated_transcript_and_split_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: unrelated frozen rows incorrectly veto a verified QLoRA subset."""
    loaded, _, sample_ids = _dataset_shell(
        tmp_path / "subset-ready", ready=True, overlap=True
    )
    subset = ("S001", "S002", "S005")
    authority, _ = _write_ready_authority(
        tmp_path, loaded, sample_ids, qlora_sample_ids=subset
    )
    transcript_path = Path(loaded.path) / "transcripts.jsonl"
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows[0] = {"sample_id": "S000", "status": "failed", "transcription": None}
    transcript_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)

    context = load_verified_qlora_context(loaded.path, object())
    result = evaluate_qlora_gates(context)

    assert context.qlora_sample_ids == subset
    assert result.status == "ready"
    assert result.reasons == ()


@pytest.mark.parametrize(
    "subset",
    [
        ("S001", "S001", "S002", "S005"),
        ("S001", "S002", "S999"),
        (),
    ],
    ids=("duplicate", "unknown", "empty"),
)
def test_invalid_authority_qlora_subset_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subset: tuple[str, ...],
) -> None:
    """Break caught: a duplicate or unknown subset can acquire training authority."""
    loaded, _, sample_ids = _dataset_shell(tmp_path / "invalid-subset", ready=True)
    authority, _ = _write_ready_authority(
        tmp_path, loaded, sample_ids, qlora_sample_ids=subset
    )
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)

    result = evaluate_qlora_gates(
        load_verified_qlora_context(loaded.path, object())
    )

    assert result.status == "not_run"
    assert "qlora_sample_subset_invalid" in result.reasons


def test_authorized_qlora_subset_rejects_cross_partition_duplicate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the authorized subset spans one source identity across test."""
    loaded, _, sample_ids = _dataset_shell(
        tmp_path / "unsafe-subset", ready=True, overlap=True
    )
    authority, _ = _write_ready_authority(
        tmp_path,
        loaded,
        sample_ids,
        qlora_sample_ids=("S000", "S002", "S004"),
    )
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)

    result = evaluate_qlora_gates(
        load_verified_qlora_context(loaded.path, object())
    )

    assert result.status == "not_run"
    assert "qlora_sample_subset_invalid" in result.reasons


def test_authorized_artifact_drift_revokes_ready_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: authority-bound runtime can change after context construction."""
    loaded, _, sample_ids = _dataset_shell(tmp_path / "ready", ready=True)
    authority, artifacts = _write_ready_authority(tmp_path, loaded, sample_ids)
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)
    context = load_verified_qlora_context(loaded.path, object())
    artifacts["runtime"].write_bytes(b"drift")

    result = evaluate_qlora_gates(context)

    assert result.status == "not_run"
    assert result.reasons == ("verified_runtime_artifact_missing",)
    assert result.training_command is None
    assert result.checkpoint_path is None


def test_split_overlap_is_computed_from_partition_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a self-asserted overlap-free boolean hides cross-test duplicates."""
    loaded, _, sample_ids = _dataset_shell(
        tmp_path / "overlap", ready=True, overlap=True
    )
    authority, _ = _write_ready_authority(tmp_path, loaded, sample_ids)
    _install_loader(monkeypatch, loaded)
    _install_authority(monkeypatch, authority)
    context = load_verified_qlora_context(loaded.path, object())

    result = evaluate_qlora_gates(context)

    assert "split_overlap_zero_missing" in result.reasons


def test_arbitrary_metadata_and_forged_context_fail_closed() -> None:
    """Break caught: caller facts or object lookalikes bypass authority derivation."""
    result = evaluate_qlora_gates(
        {"frozen_protocol": True, "runtime_sha256": "a" * 64}  # type: ignore[arg-type]
    )
    assert result.reasons == ("context_not_verified",)
    forged = object.__new__(VerifiedQloraContext)
    assert evaluate_qlora_gates(forged).reasons == ("context_not_verified",)


def test_only_generated_synthetic_capability_gets_feasibility_status() -> None:
    """Break caught: arbitrary metadata can claim a synthetic feasibility run."""
    result = evaluate_qlora_gates(
        create_synthetic_qlora_fixture(sample_count=8, seed=20260826)
    )
    assert result.status == "method_feasibility_only"
    assert result.reasons == ("generated_synthetic_fixture_only",)
    assert result.training_command is None
    assert result.checkpoint_path is None
    assert result.metrics is None
