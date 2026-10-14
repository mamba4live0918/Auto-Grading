"""Deterministic, privacy-safe orchestration helpers for Notebook 1."""

from __future__ import annotations

import ast
import base64
import binascii
import csv
import contextlib
import io
import json
import math
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import yaml

from .artifacts import atomic_write_json
from .canonical import sha256_file
from .dataset_pipeline import (
    DatasetPipelineConfig,
    DatasetRunResult,
    DatasetStageAdapters,
    DatasetStageRecord,
    DatasetStageResult,
    FrozenDatasetResult,
    freeze_dataset,
    load_strict_frozen_dataset,
    run_dataset_pipeline,
    stage_implementation_digests,
)
from .feature_registry import FeatureDefinition, load_feature_registry
from .local_manifest_migration import migrate_manifest_to_local_catalog
from .local_source import (
    LocalFileAcquirer,
    LocalSourceResolver,
    TrustedSourceIdentity,
    load_local_catalog,
    load_local_source_config,
    local_source_reference_valid,
)
from .preflight import (
    PreflightReport,
    SystemPreflightProbes,
    run_preflight,
)
from .protocol import load_protocol_template
from .source_manifest import (
    ScoreStatus,
    SourceRow,
    VerificationStatus,
    load_source_manifest,
    validate_source_rows,
)
from .whisper_local import WhisperRuntimeConfig


_MODES = frozenset({"SMOKE_MODE", "RESUME", "FORCE_STAGE", "FULL_RUN"})
_FULL_DATA_MODES = frozenset({"RESUME", "FORCE_STAGE", "FULL_RUN"})
_STAGES = ("acquisition", "audio", "asr", "features")
_FROZEN_FILENAMES = frozenset(
    {
        "manifest.csv",
        "transcripts.jsonl",
        "acoustic_features.parquet",
        "quality_report.csv",
        "splits.json",
        "experiment_protocol.json",
        "provenance.json",
        "source_snapshot.json",
        "git_diff.patch",
        "checksums.sha256",
    }
)
_PUBLIC_NOTEBOOK = Path("notebooks/01_dataset_pipeline.ipynb")
_EXECUTED_NOTEBOOK = Path("artifacts/research/executed/01_dataset_pipeline.full.ipynb")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC_ABSOLUTE_PATH = re.compile(  # privacy-scan: synthetic-fixture
    r"(?<!\\)\\\\[^\\/\s]+\\"
)
_EMAIL_ADDRESS = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
_PRIVATE_OUTPUT_KEYS = re.compile(
    r'"(?:source_url|video_title|total_score|transcript|transcription)"\s*:',
    re.IGNORECASE,
)
_PRIVATE_ROW_FIELDS = re.compile(
    r"\b(?:source[ _]?url|video[ _]?title|total[ _]?score|"
    r"transcript(?:ion)?|source[ _]?row[ _]?id|video[ _]?id|"
    r"speaker[ _]?id|talk[ _]?id)\b",
    re.IGNORECASE,
)
_URI_SCHEME = re.compile(
    r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s'\"{}])"
)
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)*)"
)
_RASTER_IMAGE_MIMES = frozenset({"image/png", "image/jpeg"})
_NOTEBOOK2_PROVENANCE_KEYS = frozenset(
    {
        "dataset_id",
        "protocol_id",
        "included_count",
        "excluded_count",
        "exclusion_basis",
        "method",
        "uncertainty_method",
        "model_identifier",
    }
)
_AGGREGATE_SCHEMAS = {
    "preflight_summary": frozenset(
        {
            "artifact_type",
            "mode",
            "source_mode",
            "ready",
            "network_enabled",
            "check_status_counts",
            "warning_check_count",
        }
    ),
    "source_audit_summary": frozenset(
        {
            "artifact_type",
            "total_rows",
            "observed_label_rows",
            "not_available_label_rows",
            "unresolved_rows",
            "duplicate_source_rows",
            "duplicate_video_rows",
            "invalid_rows",
            "identity_error_rows",
        }
    ),
    "stage_summary": frozenset(
        {
            "artifact_type",
            "stage",
            "succeeded",
            "failed",
            "skipped",
            "resumed",
            "executed",
        }
    ),
    "feature_preview_summary": frozenset(
        {
            "artifact_type",
            "previewed_row_count",
            "previewed_feature_count",
            "available_value_count",
            "missing_value_count",
        }
    ),
    "quality_summary": frozenset(
        {
            "artifact_type",
            "total_rows",
            "feature_succeeded_rows",
            "feature_non_success_rows",
            "rows_with_qc_warnings",
            "recorded_missing_values",
            "terminal_counts",
            "pipeline_complete_rows",
            "analysis_usable_counts",
        }
    ),
    "freeze_summary": frozenset(
        {
            "artifact_type",
            "reused",
            "row_count",
            "terminal_counts",
            "pipeline_complete_rows",
            "analysis_usable_counts",
        }
    ),
    "validation_summary": frozenset(
        {
            "artifact_type",
            "row_count",
            "expected_row_count",
            "terminal_records_valid",
            "frozen_file_set_valid",
            "checksum_count",
            "checksums_valid",
            "frozen_row_counts_valid",
            "public_source_safe",
            "executed_copy_ignored",
            "valid",
        }
    ),
    "handoff_summary": frozenset(
        {
            "artifact_type",
            "row_count",
            "terminal_counts",
            "pipeline_complete_rows",
            "analysis_usable_counts",
            "valid",
        }
    ),
    "execution_summary": frozenset(
        {"artifact_type", "row_count", "valid"}
    ),
    "modelling_dataset_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "dataset_id",
            "row_count",
            "qc_count",
            "qc_excluded_count",
            "clustering_count",
            "clustering_excluded_count",
            "supervised_count",
            "supervised_excluded_count",
            "transcript_count",
            "transcript_excluded_count",
        }
    ),
    "clustering_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "status",
            "sample_count",
            "feature_count",
            "selected_k",
            "silhouette",
            "stability_min",
            "stability_median",
            "similarity_scale",
            "conditional_feature_included",
            "conditional_feature_status",
        }
    ),
    "cluster_score_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "status",
            "selected_k",
            "sample_count",
            "excluded_sample_count",
            "group_count",
            "comparison_count",
            "minimum_adjusted_p",
            "significance_threshold",
            "statistically_significant",
        }
    ),
    "composite_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "sample_count",
            "feature_count",
            "spearman",
            "pearson",
            "learned_weights",
            "calibrated_to_target",
        }
    ),
    "model_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "model_name",
            "sample_count",
            "oof_prediction_count",
            "mae",
            "rmse",
            "spearman",
            "pearson",
            "mae_ci_status",
            "mae_ci_lower",
            "mae_ci_upper",
            "fold_count",
            "fold_local_preprocessing",
        }
    ),
    "model_comparison_summary": _NOTEBOOK2_PROVENANCE_KEYS
    | frozenset(
        {
            "artifact_type",
            "ridge_minus_training_mean_estimate",
            "ridge_minus_training_mean_status",
            "ridge_minus_training_mean_ci_lower",
            "ridge_minus_training_mean_ci_upper",
            "ridge_minus_training_mean_valid_replicates",
            "ridge_minus_training_mean_required_replicates",
            "ridge_minus_training_mean_includes_zero",
            "elastic_net_minus_training_mean_estimate",
            "elastic_net_minus_training_mean_status",
            "elastic_net_minus_training_mean_ci_lower",
            "elastic_net_minus_training_mean_ci_upper",
            "elastic_net_minus_training_mean_valid_replicates",
            "elastic_net_minus_training_mean_required_replicates",
            "elastic_net_minus_training_mean_includes_zero",
            "ridge_minus_elastic_net_estimate",
            "ridge_minus_elastic_net_status",
            "ridge_minus_elastic_net_ci_lower",
            "ridge_minus_elastic_net_ci_upper",
            "ridge_minus_elastic_net_valid_replicates",
            "ridge_minus_elastic_net_required_replicates",
            "ridge_minus_elastic_net_includes_zero",
        }
    ),
    "qwen_gate_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "status",
            "reason_code",
            "warning",
            "attempt_count",
            "calibration_fit",
        }
    ),
    "qlora_gate_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "status",
            "reason_count",
            "ready",
            "training_started",
            "metrics_available",
            "qlora_reasons",
        }
    ),
    "modelling_validation_summary": _NOTEBOOK2_PROVENANCE_KEYS | frozenset(
        {
            "artifact_type",
            "dataset_id",
            "valid",
            "row_count",
            "selected_k",
            "stability_checked",
            "stability_status",
            "cohort_counts_valid",
            "one_oof_per_sample",
            "fold_local_preprocessing",
            "bootstrap_replicates",
            "bootstrap_valid",
            "paired_model_differences_valid",
            "cluster_scores_posthoc",
            "mae_intervals_overlap",
            "mae_overlap_lower",
            "mae_overlap_upper",
            "qwen_not_run",
            "qlora_not_run",
        }
    ),
}
_AGGREGATE_NESTED_KEYS = {
    "check_status_counts": frozenset(
        {"pass", "warning", "blocked", "skipped"}
    ),
    "terminal_counts": frozenset(
        {
            "succeeded",
            "acquisition_failed",
            "audio_failed",
            "asr_failed",
            "feature_failed",
        }
    ),
    "analysis_usable_counts": frozenset(
        {
            "quality_control",
            "clustering",
            "supervised_exploratory",
            "transcript_analysis",
        }
    ),
    "status_counts": frozenset({"succeeded", "failed", "skipped"}),
    "qlora_reasons": frozenset(
        {
            "authority_entry_missing",
            "qlora_sample_subset_invalid",
            "protocol_disables_qlora",
            "frozen_protocol_missing",
            "transcript_only_task_missing",
            "frozen_target_missing",
            "primary_endpoint_missing",
            "minimum_meaningful_improvement_missing",
            "failure_criteria_missing",
            "advisor_total_score_only",
            "confirmatory_sample_size_protocol_missing",
            "quality_checked_transcripts_missing",
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
            "vram_below_12_gb",
            "wsl2_vram_preflight_failed",
            "training_command_missing",
            "checkpoint_path_missing",
        }
    ),
}
_AGGREGATE_ENUMS = {
    "mode": _MODES,
    "source_mode": frozenset({"YOUTUBE", "LOCAL"}),
    "stage": frozenset(_STAGES),
    "status": frozenset(
        {"ok", "indistinguishable", "not_estimable", "not_run", "ready"}
    ),
    "model_name": frozenset({"training_mean", "ridge", "elastic_net"}),
    "mae_ci_status": frozenset({"ok", "not_estimable"}),
    "ridge_minus_training_mean_status": frozenset({"ok", "not_estimable"}),
    "elastic_net_minus_training_mean_status": frozenset({"ok", "not_estimable"}),
    "ridge_minus_elastic_net_status": frozenset({"ok", "not_estimable"}),
    "reason_code": frozenset({"expected_local_digest_missing"}),
    "exclusion_basis": frozenset(
        {
            "strict_frozen_dataset",
            "clustering_cohort_policy",
            "observed_scores_after_cluster_freeze",
            "supervised_total_score_cohort_policy",
            "transcript_model_cohort_policy",
        }
    ),
    "method": frozenset(
        {
            "strict_frozen_dataset_validation",
            "ward_euclidean_robust_scaled_pca",
            "posthoc_group_permutation_holm",
            "cross_fitted_equal_available_dimensions",
            "nested_grouped_cross_validation",
            "paired_model_mae_differences",
            "local_qwen_identity_gate",
            "authority_bound_qlora_gate",
            "fail_closed_modelling_validation",
        }
    ),
    "uncertainty_method": frozenset(
        {
            "not_applicable",
            "leave_one_feature_out_adjusted_rand_index",
            "group_permutation_10000_seed_20260813_holm",
            "descriptive_correlations_no_interval",
            "paired_group_bootstrap_percentile_5000_seed_20260813_95pct",
        }
    ),
    "model_identifier": frozenset(
        {
            "not_applicable",
            "training_fold_mean_v1",
            "sklearn_linear_model_ridge",
            "sklearn_linear_model_elastic_net",
            "training_mean_ridge_elastic_net",
            "qwen3.5:9b-q4_K_M",
        }
    ),
    "conditional_feature_status": frozenset(
        {
            "excluded_no_successful_asr_rows",
            "excluded_missing_whisper_language",
            "excluded_mixed_whisper_languages",
            "included_single_whisper_language",
        }
    ),
    "stability_status": frozenset({"passed", "failed", "not_applicable"}),
}
_AGGREGATE_PATTERN_KEYS = {
    "dataset_id": re.compile(r"[0-9a-f]{64}\Z"),
    "protocol_id": re.compile(r"[0-9a-f]{64}\Z"),
}
_NOTEBOOK2_AGGREGATE_TYPES = frozenset(
    {
        "modelling_dataset_summary",
        "clustering_summary",
        "cluster_score_summary",
        "composite_summary",
        "model_summary",
        "model_comparison_summary",
        "qwen_gate_summary",
        "qlora_gate_summary",
        "modelling_validation_summary",
    }
)
_NOTEBOOK1_AGGREGATE_TYPES = frozenset(
    {
        "preflight_summary",
        "source_audit_summary",
        "stage_summary",
        "feature_preview_summary",
        "quality_summary",
        "freeze_summary",
        "validation_summary",
        "handoff_summary",
    }
)
_EXACT_AGGREGATE_TYPES = (
    _NOTEBOOK1_AGGREGATE_TYPES
    | _NOTEBOOK2_AGGREGATE_TYPES
    | frozenset({"execution_summary"})
)
_EXACT_NESTED_AGGREGATE_KEYS = frozenset(
    {"check_status_counts", "terminal_counts", "analysis_usable_counts"}
)
_AGGREGATE_BOOLEAN_KEYS = frozenset(
    {
        "conditional_feature_included",
        "statistically_significant",
        "learned_weights",
        "calibrated_to_target",
        "fold_local_preprocessing",
        "warning",
        "calibration_fit",
        "ready",
        "training_started",
        "metrics_available",
        "valid",
        "stability_checked",
        "cohort_counts_valid",
        "one_oof_per_sample",
        "bootstrap_valid",
        "paired_model_differences_valid",
        "cluster_scores_posthoc",
        "mae_intervals_overlap",
        "qwen_not_run",
        "qlora_not_run",
    }
)
_AGGREGATE_OPTIONAL_BOOLEAN_KEYS = frozenset(
    {
        "ridge_minus_training_mean_includes_zero",
        "elastic_net_minus_training_mean_includes_zero",
        "ridge_minus_elastic_net_includes_zero",
    }
)
_AGGREGATE_LINE_KEYS = frozenset(
    {
        "completed_rows",
        "row_count",
        "succeeded",
        "failed",
        "skipped",
        "warning_count",
        "missing_value_count",
        "valid",
        "ready",
        "mode",
        "stage",
    }
)
_AGGREGATE_KEY_VALUE_LINE = re.compile(
    r"(?P<key>[a-z][a-z0-9_]*)=(?P<value>[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"|true|false|null|[A-Za-z_][A-Za-z0-9_]*)\Z"
)
_MATPLOTLIB_FIGURE_REPR = re.compile(
    r"<Figure size \d{1,5}x\d{1,5} with \d{1,4} Axes?>\Z"
)
_PLAIN_TEXT_LIMIT = 1_000
_STRUCTURED_TEXT_LIMIT = 8_000
_AGGREGATE_LINE_LIMIT = 12
_MAX_ENCODED_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_DECODED_IMAGE_BYTES = 6 * 1024 * 1024


PreflightRunner = Callable[..., PreflightReport]


@dataclass(frozen=True)
class NotebookRunContext:
    """Resolved but non-displayed state for one notebook execution."""

    repository_root: Path
    config_path: Path
    mode: str
    source_mode: str
    allow_network: bool
    raw_config: Mapping[str, Any]
    preflight: PreflightReport
    rows: tuple[SourceRow, ...]
    pipeline_config: DatasetPipelineConfig
    expected_row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_config", MappingProxyType(dict(self.raw_config)))


class _SmokeStages:
    """Small deterministic file-writing adapters with no external dependency."""

    def acquire(
        self, row: SourceRow, output_root: Path, allow_network: bool
    ) -> DatasetStageResult:
        if allow_network:
            raise RuntimeError("smoke mode attempted network acquisition")
        source_path = output_root / f"{row.sample_id}.source"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"synthetic-source-{row.sample_id}\n".encode("utf-8"))
        return DatasetStageResult(
            status="succeeded",
            sample_id=row.sample_id,
            payload={
                "source_path": source_path,
                "source_sha256": sha256_file(source_path),
            },
        )

    def audio(self, source_path: Path, output_root: Path) -> DatasetStageResult:
        sample_id = output_root.parent.name
        payload: dict[str, object] = {
            "source_channel_count": 1,
            "analysis_channel_count": 1,
        }
        source_bytes = source_path.read_bytes()
        for representation in ("analysis", "peak_normalized", "loudness_normalized"):
            audio_path = output_root / f"{representation}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(source_bytes + representation.encode("utf-8"))
            payload[f"{representation}_path"] = audio_path
            payload[f"{representation}_sha256"] = sha256_file(audio_path)
        return DatasetStageResult(
            status="succeeded", sample_id=sample_id, payload=payload
        )

    def transcribe_many(self, items: Sequence[Any]) -> list[DatasetStageResult]:
        return [
            DatasetStageResult(
                status="succeeded",
                sample_id=item.sample_id,
                payload={
                    "transcription": {
                        "text": "synthetic local speech",
                        "language": "en",
                        "language_probability": 1.0,
                        "duration_s": 1.0,
                        "model_size": "synthetic-local",
                        "segments": [],
                    },
                    "provenance": {
                        "model_name": "synthetic-local",
                        "local_files_only": True,
                    },
                },
            )
            for item in items
        ]

    def features(
        self,
        sample_id: str,
        audio_paths: dict[str, Path],
        transcript: object,
        registry: object,
    ) -> DatasetStageResult:
        del audio_paths, transcript
        feature_values = {
            feature.name: _synthetic_feature_value(feature)
            for feature in registry.features
        }
        return DatasetStageResult(
            status="succeeded",
            sample_id=sample_id,
            payload={
                "features": feature_values,
                "missing_reasons": {},
                "qc_warnings": [],
            },
        )

    def adapters(self) -> DatasetStageAdapters:
        return DatasetStageAdapters(
            acquire=self.acquire,
            audio=self.audio,
            transcribe_many=self.transcribe_many,
            features=self.features,
        )


def locate_repository_root(start: Path) -> Path:
    """Find the nearest repository root from a notebook or shell directory."""
    resolved = Path(start).resolve(strict=True)
    candidates = (resolved, *resolved.parents) if resolved.is_dir() else resolved.parents
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs" / "notebook_pipeline.yaml"
        ).is_file():
            return candidate
    raise FileNotFoundError("repository root markers were not found")


def prepare_notebook_run(
    repository_root: Path,
    *,
    mode: str,
    network_opt_in: str,
    config_path: str | Path,
    source_mode: str = "YOUTUBE",
    local_source_config: str | Path | None = None,
    preflight_runner: PreflightRunner = run_preflight,
) -> NotebookRunContext:
    """Validate notebook parameters and construct one Task 7 pipeline config."""
    root = locate_repository_root(repository_root)
    normalized_mode = str(mode).strip().upper()
    if normalized_mode not in _MODES:
        raise ValueError(f"unsupported notebook mode: {normalized_mode}")
    normalized_source_mode = str(source_mode).strip().upper()
    if normalized_source_mode not in {"YOUTUBE", "LOCAL"}:
        raise ValueError(f"unsupported source mode: {normalized_source_mode}")
    if (
        normalized_mode == "FULL_RUN"
        and normalized_source_mode == "YOUTUBE"
        and network_opt_in != "1"
    ):
        raise PermissionError("FULL_RUN requires explicit network opt-in")

    relative_config = _safe_relative_path(config_path, "dataset config")
    resolved_config = (root / relative_config).resolve(strict=True)
    if root not in resolved_config.parents:
        raise ValueError("dataset config must remain inside the repository")
    raw_config = _load_mapping(resolved_config)
    resolved_local_source_config: Path | None = None
    if normalized_mode != "SMOKE_MODE" and normalized_source_mode == "LOCAL":
        resolved_local_source_config = _absolute_private_path(
            local_source_config, "local source config"
        )
        preflight = preflight_runner(
            root,
            relative_config,
            normalized_mode,
            source_mode="LOCAL",
            local_source_config=resolved_local_source_config,
        )
    else:
        preflight = preflight_runner(root, relative_config, normalized_mode)
    if not isinstance(preflight, PreflightReport):
        raise TypeError("preflight runner must return PreflightReport")
    if not preflight.ready:
        blocked = sorted(
            check.name for check in preflight.checks if check.required and check.status.value != "pass"
        )
        raise RuntimeError(f"preflight blocked: {', '.join(blocked)}")

    artifact_root = _configured_repository_path(
        root, raw_config, "research_artifact_root", Path("artifacts/research")
    )
    protocol_template = load_protocol_template(
        root / "configs" / "experiment_protocol.template.yaml"
    )
    feature_registry = load_feature_registry(
        root / "configs" / "audio_feature_registry.v1.yaml"
    )
    allow_network = (
        normalized_source_mode == "YOUTUBE"
        and normalized_mode == "FULL_RUN"
        and network_opt_in == "1"
    )

    if normalized_mode == "SMOKE_MODE":
        rows = _smoke_rows()
        adapters = _SmokeStages().adapters()
        whisper_fingerprint = {
            "model_name": "synthetic-local",
            "revision": "smoke-v1",
            "snapshot_sha256": "0" * 64,
            "local_files_only": True,
        }
        staging_name = "notebook-01-smoke-v1"
    else:
        manifest_path = _configured_repository_path(
            root, raw_config, "source_manifest", Path("data/private/source_manifest.csv")
        )
        legacy_rows = tuple(load_source_manifest(manifest_path))
        whisper_config = _whisper_runtime_config(raw_config)
        local_identity: str | None = None
        if normalized_source_mode == "LOCAL":
            assert resolved_local_source_config is not None
            local_catalog = load_local_catalog(
                load_local_source_config(resolved_local_source_config)
            )
            rows = migrate_manifest_to_local_catalog(
                legacy_rows, local_catalog.rows
            ).active_rows
            local_identity = local_catalog.identity_sha256
            adapters = DatasetStageAdapters.production(
                whisper_config=whisper_config,
                local_source_resolver=LocalSourceResolver(local_catalog),
            )
        else:
            rows = legacy_rows
            adapters = DatasetStageAdapters.production(whisper_config=whisper_config)
        whisper_fingerprint = {
            "model_name": whisper_config.model_name,
            "revision": whisper_config.revision,
            "snapshot_sha256": whisper_config.snapshot_sha256,
            "local_files_only": whisper_config.local_files_only,
            "device": whisper_config.device,
            "compute_type": whisper_config.compute_type,
            "beam_size": whisper_config.beam_size,
            "vad_filter": whisper_config.vad_filter,
            "word_timestamps": whisper_config.word_timestamps,
            "language": whisper_config.language,
        }
        staging_name = (
            "notebook-01-local-v1"
            if normalized_source_mode == "LOCAL"
            else "notebook-01-full-v1"
        )

    expected_row_count = _expected_row_count(raw_config, normalized_mode, len(rows))
    pipeline_parameters = {
        "schema_version": 1,
        "notebook": "01_dataset_pipeline",
        "mode": normalized_mode,
        "source_mode": normalized_source_mode,
        "random_seed": protocol_template.random_seed,
        "expected_row_count": expected_row_count,
    }
    acquisition_parameters: dict[str, object] = {
        "adapter": (
            "synthetic-local"
            if normalized_mode == "SMOKE_MODE"
            else ("local_file" if normalized_source_mode == "LOCAL" else "yt-dlp")
        )
    }
    if normalized_mode != "SMOKE_MODE" and normalized_source_mode == "LOCAL":
        acquisition_parameters["local_source_identity_sha256"] = local_identity
    stage_parameters = {
        "acquisition": acquisition_parameters,
        "audio": {
            "duration_source": "stream_then_container",
            "analysis_rate_hz": 16000,
            "channels": 1,
            "pcm_bits": 16,
        },
        "asr": {
            "batch_model_instances": 1,
            "local_files_only": True,
        },
        "features": {"registry_schema_version": feature_registry.schema_version},
    }
    force_stages = _force_stages_from_environment(normalized_mode, os.environ)
    pipeline_config = DatasetPipelineConfig(
        repository_root=root,
        staging_dir=artifact_root / "staging" / staging_name,
        artifact_root=artifact_root,
        rows=rows,
        protocol_template=protocol_template,
        feature_registry=feature_registry,
        pipeline_parameters=pipeline_parameters,
        stage_parameters=stage_parameters,
        whisper_fingerprint=whisper_fingerprint,
        implementation_digests=stage_implementation_digests(root),
        adapters=adapters,
        allow_network=allow_network,
        allow_legacy_stage_cache=_legacy_cache_opt_in(os.environ),
        force_stages=force_stages,
        progress_callback=None,
    )
    return NotebookRunContext(
        repository_root=root,
        config_path=resolved_config,
        mode=normalized_mode,
        source_mode=normalized_source_mode,
        allow_network=allow_network,
        raw_config=raw_config,
        preflight=preflight,
        rows=rows,
        pipeline_config=pipeline_config,
        expected_row_count=expected_row_count,
    )


def summarize_preflight(context: NotebookRunContext) -> dict[str, object]:
    """Return aggregate preflight status without paths or command details."""
    status_counts = Counter(check.status.value for check in context.preflight.checks)
    return {
        "artifact_type": "preflight_summary",
        "mode": context.mode,
        "source_mode": context.source_mode,
        "ready": context.preflight.ready,
        "network_enabled": context.allow_network,
        "check_status_counts": {
            status: status_counts.get(status, 0)
            for status in ("pass", "warning", "blocked", "skipped")
        },
        "warning_check_count": status_counts.get("warning", 0),
    }


def summarize_source_audit(context: NotebookRunContext) -> dict[str, object]:
    """Return count-only source audit information."""
    if context.mode == "SMOKE_MODE":
        return {
            "artifact_type": "source_audit_summary",
            "total_rows": len(context.rows),
            "observed_label_rows": 0,
            "not_available_label_rows": len(context.rows),
            "unresolved_rows": 0,
            "duplicate_source_rows": 0,
            "duplicate_video_rows": 0,
            "invalid_rows": 0,
            "identity_error_rows": 0,
        }
    audit = validate_source_rows(context.rows)
    return {
        "artifact_type": "source_audit_summary",
        "total_rows": audit.total_rows,
        "observed_label_rows": audit.observed_score_rows,
        "not_available_label_rows": audit.not_available_rows,
        "unresolved_rows": audit.unresolved_rows,
        "duplicate_source_rows": len(audit.duplicate_urls),
        "duplicate_video_rows": len(audit.duplicate_video_ids),
        "invalid_rows": len(audit.invalid_rows),
        "identity_error_rows": len(audit.identity_errors),
    }


def execute_notebook_pipeline(context: NotebookRunContext) -> DatasetRunResult:
    """Execute the reviewed resumable Task 7 pipeline."""
    pipeline_config = replace(
        context.pipeline_config, progress_callback=None
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return run_dataset_pipeline(pipeline_config)


def stage_summary(result: DatasetRunResult, stage: str) -> dict[str, object]:
    """Return deterministic terminal counts for one stage."""
    if stage not in _STAGES:
        raise ValueError(f"unknown dataset stage: {stage}")
    counts = Counter(getattr(sample, stage).status for sample in result.samples)
    resumed = sum(getattr(sample, stage).resumed for sample in result.samples)
    return {
        "artifact_type": "stage_summary",
        "stage": stage,
        **{
            status: counts.get(status, 0)
            for status in ("succeeded", "failed", "skipped")
        },
        "resumed": resumed,
        "executed": len(result.samples) - resumed,
    }


def display_aggregate_payload(payload: Mapping[str, object]) -> None:
    """Display one reviewed aggregate as JSON without a text representation."""
    from IPython.display import display

    display({"application/json": dict(payload)}, raw=True)


def bounded_feature_preview(
    result: DatasetRunResult, *, limit: int = 5, feature_limit: int = 6
) -> dict[str, object]:
    """Return bounded aggregate feature availability without row-level values."""
    if limit < 0 or feature_limit < 0:
        raise ValueError("preview limits must be non-negative")
    feature_names = [
        feature.name for feature in result.feature_registry.features[:feature_limit]
    ]
    available_value_count = 0
    missing_value_count = 0
    for sample in result.samples[:limit]:
        values = sample.features.payload.get("features", {})
        for name in feature_names:
            value = values.get(name) if isinstance(values, Mapping) else None
            if value is None:
                missing_value_count += 1
            else:
                available_value_count += 1
    return {
        "artifact_type": "feature_preview_summary",
        "previewed_row_count": min(limit, len(result.samples)),
        "previewed_feature_count": len(feature_names),
        "available_value_count": available_value_count,
        "missing_value_count": missing_value_count,
    }


def summarize_quality(result: DatasetRunResult) -> dict[str, object]:
    """Return aggregate feature missingness and QC-warning counts."""
    succeeded = [sample for sample in result.samples if sample.features.status == "succeeded"]
    warning_rows = 0
    missing_values = 0
    for sample in succeeded:
        warnings = sample.features.payload.get("qc_warnings", [])
        reasons = sample.features.payload.get("missing_reasons", {})
        warning_rows += int(isinstance(warnings, list) and bool(warnings))
        missing_values += len(reasons) if isinstance(reasons, Mapping) else 0
    return {
        "artifact_type": "quality_summary",
        "total_rows": len(result.samples),
        "feature_succeeded_rows": len(succeeded),
        "feature_non_success_rows": len(result.samples) - len(succeeded),
        "rows_with_qc_warnings": warning_rows,
        "recorded_missing_values": missing_values,
        "terminal_counts": _terminal_counts(result),
        "pipeline_complete_rows": sum(
            sample.terminal_status == "succeeded" for sample in result.samples
        ),
        "analysis_usable_counts": _analysis_usable_counts(result),
    }


def freeze_notebook_dataset(result: DatasetRunResult) -> FrozenDatasetResult:
    """Create or reuse the immutable Task 7 dataset."""
    return freeze_dataset(result)


def redacted_freeze_summary(
    frozen: FrozenDatasetResult, result: DatasetRunResult
) -> dict[str, object]:
    """Return an aggregate frozen-dataset summary without identity or paths."""
    return {
        "artifact_type": "freeze_summary",
        "reused": frozen.reused,
        "row_count": len(result.samples),
        "terminal_counts": _terminal_counts(result),
        "pipeline_complete_rows": sum(
            sample.terminal_status == "succeeded" for sample in result.samples
        ),
        "analysis_usable_counts": _analysis_usable_counts(result),
    }


def validate_notebook_outputs(
    context: NotebookRunContext,
    result: DatasetRunResult,
    frozen: FrozenDatasetResult,
) -> dict[str, object]:
    """Validate row terminality, frozen formats/hashes, and privacy boundaries."""
    row_count_valid = len(result.samples) == context.expected_row_count
    expected_source_identities = _expected_local_source_identities(
        context.pipeline_config
    )
    terminal_records_valid = expected_source_identities is not None and all(
        _sample_records_valid(
            sample,
            expected_identity=expected_source_identities.get(sample.sample_id),
        )
        for sample in result.samples
    )
    if expected_source_identities is None:
        loaded = None
    else:
        try:
            loaded = load_strict_frozen_dataset(
                frozen.path,
                result.feature_registry,
                expected_dataset_id=frozen.dataset_id,
                expected_split_sha256=frozen.protocol.split_sha256,
                expected_source_identities=expected_source_identities,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError):
            loaded = None
    frozen_file_set_valid = loaded is not None
    checksum_count = 0 if loaded is None else loaded.checksum_count
    checksums_valid = loaded is not None
    frozen_row_counts_valid = (
        loaded is not None and loaded.row_count == context.expected_row_count
    )
    public_source_safe = _public_notebook_is_safe(context.repository_root / _PUBLIC_NOTEBOOK)
    executed_copy_ignored = _git_ignores(context.repository_root, _EXECUTED_NOTEBOOK)
    checks = {
        "artifact_type": "validation_summary",
        "row_count": len(result.samples),
        "expected_row_count": context.expected_row_count,
        "terminal_records_valid": terminal_records_valid,
        "frozen_file_set_valid": frozen_file_set_valid,
        "checksum_count": checksum_count,
        "checksums_valid": checksums_valid,
        "frozen_row_counts_valid": frozen_row_counts_valid,
        "public_source_safe": public_source_safe,
        "executed_copy_ignored": executed_copy_ignored,
    }
    checks["valid"] = row_count_valid and all(
        bool(checks[name])
        for name in (
            "terminal_records_valid",
            "frozen_file_set_valid",
            "checksums_valid",
            "frozen_row_counts_valid",
            "public_source_safe",
            "executed_copy_ignored",
        )
    )
    return checks


def write_redacted_handoff(
    context: NotebookRunContext,
    result: DatasetRunResult,
    frozen: FrozenDatasetResult,
    validation: Mapping[str, object],
) -> dict[str, object]:
    """Persist and display only the aggregate handoff needed by Notebook 2."""
    if validation.get("valid") is not True:
        raise ValueError("handoff requires strict validation with valid exactly true")
    handoff_path = (
        context.repository_root
        / "artifacts"
        / "research"
        / "handoffs"
        / f"01_dataset_pipeline.{context.mode.lower()}.json"
    )
    payload = {
        "schema_version": 1,
        "mode": context.mode,
        "dataset_id": frozen.dataset_id,
        "artifact_path": _repository_relative(frozen.path, context.repository_root),
        "row_count": len(result.samples),
        "terminal_counts": _terminal_counts(result),
        "pipeline_complete_rows": sum(
            sample.terminal_status == "succeeded" for sample in result.samples
        ),
        "analysis_usable_counts": _analysis_usable_counts(result),
        "validation": dict(validation),
        "next_notebook": "02_modelling_and_results.ipynb",
    }
    atomic_write_json(handoff_path, payload)
    return {
        "artifact_type": "handoff_summary",
        "row_count": len(result.samples),
        "terminal_counts": _terminal_counts(result),
        "pipeline_complete_rows": sum(
            sample.terminal_status == "succeeded" for sample in result.samples
        ),
        "analysis_usable_counts": _analysis_usable_counts(result),
        "valid": True,
    }


def validate_executed_notebook_outputs(path: Path) -> bool:
    """Reject private-shaped values from every saved executed-notebook output."""
    try:
        notebook = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        if not isinstance(cells, list):
            raise ValueError("invalid notebook cells")
        for cell in cells:
            if not isinstance(cell, dict):
                raise ValueError("invalid notebook cell")
            outputs = cell.get("outputs", [])
            if not isinstance(outputs, list):
                raise ValueError("invalid notebook outputs")
            for output in outputs:
                _validate_executed_output(output)
    except _PrivateNotebookOutputError as error:
        raise ValueError("executed notebook contains private output") from error
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("executed notebook could not be privacy-scanned") from error
    return True


class _PrivateNotebookOutputError(ValueError):
    """Internal fail-closed signal normalized by the public scanner."""


def _validate_executed_output(output: object) -> None:
    if not isinstance(output, dict):
        raise _PrivateNotebookOutputError("output is not a mapping")
    output_type = output.get("output_type")
    if output_type != "display_data":
        raise _PrivateNotebookOutputError("unsupported output type")
    data = output.get("data")
    if not isinstance(data, dict) or not data:
        raise _PrivateNotebookOutputError("output data is missing")
    if "application/json" in data and set(data) != {"application/json"}:
        raise _PrivateNotebookOutputError("aggregate JSON must be the only MIME type")
    has_raster_image = any(mime_type in _RASTER_IMAGE_MIMES for mime_type in data)
    for mime_type, payload in data.items():
        _validate_output_mime(
            str(mime_type), payload, allow_figure_repr=has_raster_image
        )
    metadata = output.get("metadata", {})
    if metadata:
        raise _PrivateNotebookOutputError("output metadata must be empty")


def _validate_output_mime(
    mime_type: str, payload: object, *, allow_figure_repr: bool = False
) -> None:
    if mime_type in _RASTER_IMAGE_MIMES:
        _validate_image_payload(mime_type, payload)
        return
    if mime_type == "text/plain" and allow_figure_repr:
        if _MATPLOTLIB_FIGURE_REPR.fullmatch(_output_text(payload).strip()) is None:
            raise _PrivateNotebookOutputError(
                "raster companion text is not a figure representation"
            )
        return
    if mime_type == "application/json":
        _validate_aggregate_structure(payload, require_mapping=True)
        return
    raise _PrivateNotebookOutputError("unsupported output MIME type")


def _validate_image_payload(mime_type: str, payload: object) -> None:
    encoded = _output_text(payload)
    compact = "".join(encoded.split())
    if not compact or len(compact) > _MAX_ENCODED_IMAGE_BYTES:
        raise _PrivateNotebookOutputError("image payload is empty or unbounded")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as error:
        raise _PrivateNotebookOutputError("image payload is not valid base64") from error
    if not decoded or len(decoded) > _MAX_DECODED_IMAGE_BYTES:
        raise _PrivateNotebookOutputError("decoded image is empty or unbounded")
    expected_magic = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
    }[mime_type]
    if not decoded.startswith(expected_magic):
        raise _PrivateNotebookOutputError("image magic does not match its MIME type")


def _output_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list) and all(isinstance(part, str) for part in payload):
        return "".join(payload)
    raise _PrivateNotebookOutputError("text output is not textual")


def _validate_plain_output(text: str, *, allow_figure_repr: bool = False) -> None:
    if _contains_private_text(text):
        raise _PrivateNotebookOutputError("private-shaped text")
    stripped = text.strip()
    if not stripped:
        return
    if allow_figure_repr and _MATPLOTLIB_FIGURE_REPR.fullmatch(stripped):
        return
    if len(stripped) <= _STRUCTURED_TEXT_LIMIT and _is_safe_structured_plain_text(stripped):
        return
    if len(stripped) > _PLAIN_TEXT_LIMIT:
        raise _PrivateNotebookOutputError("unbounded text")
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > _AGGREGATE_LINE_LIMIT or not all(
        _is_approved_aggregate_line(line) for line in lines
    ):
        raise _PrivateNotebookOutputError("unrecognized multiline text")


def _is_safe_structured_plain_text(text: str) -> bool:
    payload: object
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            payload = ast.literal_eval(text)
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            return False
    if not isinstance(payload, dict):
        return False
    _validate_aggregate_structure(payload, require_mapping=True)
    return True


def _validate_aggregate_structure(
    payload: object,
    *,
    require_mapping: bool = False,
) -> None:
    if require_mapping and not isinstance(payload, dict):
        raise _PrivateNotebookOutputError("aggregate JSON must be an object")
    if not isinstance(payload, dict):
        raise _PrivateNotebookOutputError("aggregate output must be a mapping")
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or artifact_type not in _AGGREGATE_SCHEMAS:
        raise _PrivateNotebookOutputError("aggregate output has no approved discriminator")
    allowed_keys = _AGGREGATE_SCHEMAS[artifact_type]
    if artifact_type in _EXACT_AGGREGATE_TYPES:
        if payload.keys() != allowed_keys:
            raise _PrivateNotebookOutputError(
                "aggregate output does not match its exact schema"
            )
    elif not payload.keys() <= allowed_keys:
        raise _PrivateNotebookOutputError("aggregate output contains an unapproved key")
    for key, value in payload.items():
        if key == "artifact_type":
            continue
        if key in _AGGREGATE_NESTED_KEYS:
            _validate_nested_aggregate(key, value)
        elif key in _AGGREGATE_ENUMS:
            if not isinstance(value, str) or value not in _AGGREGATE_ENUMS[key]:
                raise _PrivateNotebookOutputError("aggregate enum is not approved")
        elif key in _AGGREGATE_PATTERN_KEYS:
            if (
                not isinstance(value, str)
                or _AGGREGATE_PATTERN_KEYS[key].fullmatch(value) is None
            ):
                raise _PrivateNotebookOutputError("aggregate identifier is not approved")
        elif key in _AGGREGATE_BOOLEAN_KEYS:
            if type(value) is not bool:
                raise _PrivateNotebookOutputError("aggregate boolean is not approved")
        elif key in _AGGREGATE_OPTIONAL_BOOLEAN_KEYS:
            if value is not None and type(value) is not bool:
                raise _PrivateNotebookOutputError(
                    "aggregate optional boolean is not approved"
                )
        else:
            _validate_aggregate_scalar(value)


def _validate_nested_aggregate(key: str, value: object) -> None:
    if not isinstance(value, dict):
        raise _PrivateNotebookOutputError("nested aggregate is not a mapping")
    allowed_keys = _AGGREGATE_NESTED_KEYS[key]
    if key in _EXACT_NESTED_AGGREGATE_KEYS:
        keys_valid = value.keys() == allowed_keys
    else:
        keys_valid = value.keys() <= allowed_keys
    if not keys_valid:
        raise _PrivateNotebookOutputError("nested aggregate keys are not approved")
    for nested_value in value.values():
        if key == "qlora_reasons":
            if type(nested_value) is not bool:
                raise _PrivateNotebookOutputError("gate reason value must be boolean")
        else:
            _validate_aggregate_scalar(nested_value)


def _validate_aggregate_scalar(value: object) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return
    raise _PrivateNotebookOutputError("aggregate value is not numeric, boolean, or null")


def _is_approved_aggregate_line(line: str) -> bool:
    if any(delimiter in line for delimiter in (",", "\t", "|")):
        return False
    match = _AGGREGATE_KEY_VALUE_LINE.fullmatch(line)
    if match is None or match.group("key") not in _AGGREGATE_LINE_KEYS:
        return False
    key = match.group("key")
    value = match.group("value")
    if key in _AGGREGATE_ENUMS:
        return value in _AGGREGATE_ENUMS[key]
    if value in {"true", "false", "null"}:
        return True
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _contains_private_text(text: str) -> bool:
    return bool(
        _URI_SCHEME.search(text)
        or _POSIX_ABSOLUTE_PATH.search(text)
        or _WINDOWS_ABSOLUTE_PATH.search(text)
        or _UNC_ABSOLUTE_PATH.search(text)
        or _EMAIL_ADDRESS.search(text)
        or _PRIVATE_OUTPUT_KEYS.search(text)
        or _PRIVATE_ROW_FIELDS.search(text)
    )


def _force_stages_from_environment(
    mode: str, environment: Mapping[str, str]
) -> frozenset[str]:
    if mode != "FORCE_STAGE":
        return frozenset()
    requested = str(environment.get("THESIS_FORCE_STAGE", "")).strip().lower()
    if requested not in {*_STAGES, "all"}:
        raise ValueError(
            "THESIS_FORCE_STAGE must be acquisition, audio, asr, features, or all"
        )
    if requested == "all":
        return frozenset(_STAGES)
    index = _STAGES.index(requested)
    return frozenset(_STAGES[index:])


def _legacy_cache_opt_in(environment: Mapping[str, str]) -> bool:
    value = str(environment.get("THESIS_ALLOW_LEGACY_STAGE_CACHE", "0")).strip()
    if value not in {"0", "1"}:
        raise ValueError("THESIS_ALLOW_LEGACY_STAGE_CACHE must be 0 or 1")
    return value == "1"


def _analysis_usable_counts(result: DatasetRunResult) -> dict[str, int]:
    return {
        "quality_control": sum(sample.qc_eligible for sample in result.samples),
        "clustering": sum(sample.clustering_eligible for sample in result.samples),
        "supervised_exploratory": sum(
            sample.features.status == "succeeded"
            and sample.source_row.score_status.value == "observed"
            and sample.source_row.total_score is not None
            for sample in result.samples
        ),
        "transcript_analysis": sum(
            sample.asr.status == "succeeded"
            and bool(
                isinstance(sample.asr.payload.get("transcription"), Mapping)
                and str(sample.asr.payload["transcription"].get("text", "")).strip()
            )
            for sample in result.samples
        ),
    }


def _terminal_counts(result: DatasetRunResult) -> dict[str, int]:
    counts = result.terminal_counts
    return {
        status: counts.get(status, 0)
        for status in (
            "succeeded",
            "acquisition_failed",
            "audio_failed",
            "asr_failed",
            "feature_failed",
        )
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("dataset configuration is unavailable or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset configuration must be a mapping")
    return payload


def _safe_relative_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return path


def _absolute_private_path(value: str | Path | None, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} must be an absolute existing file")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute existing file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} must be an absolute existing file") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must be an absolute existing file")
    return resolved


def _configured_repository_path(
    root: Path, config: Mapping[str, Any], key: str, default: Path
) -> Path:
    path = _safe_relative_path(str(config.get(key, default.as_posix())), key)
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ValueError(f"{key} must remain inside the repository")
    return resolved


def _expected_row_count(
    config: Mapping[str, Any], mode: str, actual_count: int
) -> int:
    if mode == "SMOKE_MODE":
        return actual_count
    counts = config.get("source_manifest_expected_counts")
    if not isinstance(counts, Mapping) or type(counts.get("total_rows")) is not int:
        raise ValueError("full-data mode requires an expected source row count")
    return int(counts["total_rows"])


def _whisper_runtime_config(config: Mapping[str, Any]) -> WhisperRuntimeConfig:
    probes = SystemPreflightProbes()
    model_name = str(config.get("whisper_model", ""))
    revision = str(config.get("whisper_revision", ""))
    snapshot_sha256 = str(config.get("whisper_snapshot_sha256", ""))
    snapshot_path = probes.local_whisper_snapshot(
        model_name, revision, snapshot_sha256
    )
    if snapshot_path is None:
        raise RuntimeError("preflight-approved local Whisper snapshot is unavailable")
    return WhisperRuntimeConfig.from_mapping(
        config, validated_snapshot_path=snapshot_path
    )


def _smoke_rows() -> tuple[SourceRow, ...]:
    return tuple(
        SourceRow(
            source_row_id=f"smoke-row-{index}",
            source_page="synthetic-local",
            sample_id=f"N90{index}",
            legacy_sample_id="",
            source_url=f"local-smoke-{index}",
            video_id=f"smoke00000{index}",
            video_title="",
            total_score=None,
            score_status=ScoreStatus.NOT_AVAILABLE,
            source_course="synthetic-local",
            verification_status=VerificationStatus.VERIFIED,
            verification_note="synthetic-local",
        )
        for index in range(1, 4)
    )


def _synthetic_feature_value(feature: FeatureDefinition) -> object:
    if feature.dtype == "boolean":
        return False
    if feature.dtype == "string":
        return "synthetic"
    lower = feature.minimum if feature.minimum is not None else 0
    upper = feature.maximum
    value = lower if upper is None else (float(lower) + float(upper)) / 2.0
    return int(round(float(value))) if feature.dtype == "integer" else float(value)


def _repository_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError("artifact path must remain inside the repository") from error


def _expected_local_source_identities(
    config: DatasetPipelineConfig,
) -> dict[str, TrustedSourceIdentity] | None:
    """Derive local authorities from the configured resolver, never from run records."""
    acquire_owner = getattr(config.adapters.acquire, "__self__", None)
    if not isinstance(acquire_owner, LocalFileAcquirer):
        return {}
    expected: dict[str, TrustedSourceIdentity] = {}
    for row in config.rows:
        identity = acquire_owner.trusted_identity(row)
        if identity is None:
            return None
        expected[row.sample_id] = identity
    return expected


def _sample_records_valid(
    sample: object,
    *,
    expected_identity: TrustedSourceIdentity | Mapping[str, str] | None = None,
) -> bool:
    records: list[DatasetStageRecord] = [
        getattr(sample, stage) for stage in _STAGES
    ]
    if any(record.status not in {"succeeded", "failed", "skipped"} for record in records):
        return False
    for record in records:
        if record.status in {"failed", "skipped"} and not (
            record.error_code and record.error_summary
        ):
            return False
    trusted = getattr(sample, "trusted_source_identity", None)
    if not isinstance(trusted, TrustedSourceIdentity):
        try:
            trusted = TrustedSourceIdentity(**dict(trusted))
        except (TypeError, ValueError):
            return False
    if expected_identity is not None:
        if not isinstance(expected_identity, TrustedSourceIdentity):
            try:
                expected_identity = TrustedSourceIdentity(**dict(expected_identity))
            except (TypeError, ValueError):
                return False
        if trusted.to_dict() != expected_identity.to_dict():
            return False
    if sample.terminal_status != "succeeded":
        return True
    acquisition, audio, asr, features = records
    local_identity = (
        expected_identity
        if expected_identity is not None
        else trusted
    )
    if local_identity.source_kind == "local_file":
        acquisition_source_valid = (
            acquisition.payload.get("source_kind") == "local_file"
            and not acquisition.payload.get("source_path")
            and trusted.source_kind == "local_file"
            and trusted.source_row_id == sample.source_row.source_row_id
            and trusted.video_id == sample.source_row.video_id
            and re.fullmatch(
                r"[0-9a-f]{64}", trusted.catalog_identity_sha256
            )
            is not None
            and local_source_reference_valid(
                acquisition.payload,
                expected_local_row_id=local_identity.source_row_id,
                expected_basename=local_identity.local_basename,
                expected_source_sha256=local_identity.source_sha256,
            )
        )
    else:
        acquisition_source_valid = (
            bool(acquisition.payload.get("source_path"))
            and trusted.source_kind == "youtube"
            and trusted.source_row_id == sample.source_row.source_row_id
            and trusted.video_id == sample.source_row.video_id
            and trusted.local_basename == ""
            and trusted.catalog_identity_sha256 == ""
            and trusted.source_sha256 == acquisition.payload.get("source_sha256")
        )
    return (
        acquisition.status == "succeeded"
        and acquisition_source_valid
        and bool(acquisition.payload.get("source_sha256"))
        and audio.status == "succeeded"
        and bool(audio.payload.get("analysis_path"))
        and bool(audio.payload.get("analysis_sha256"))
        and asr.status == "succeeded"
        and isinstance(asr.payload.get("transcription"), Mapping)
        and isinstance(asr.payload.get("provenance"), Mapping)
        and features.status == "succeeded"
        and isinstance(features.payload.get("features"), Mapping)
        and isinstance(features.payload.get("missing_reasons"), Mapping)
        and isinstance(features.payload.get("qc_warnings"), list)
    )


def _verify_checksums(root: Path) -> tuple[int, bool]:
    try:
        lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        declarations = [line.split("  ", 1) for line in lines if line]
        declaration_count = len(declarations)
        if any(len(parts) != 2 for parts in declarations):
            return declaration_count, False
        expected_names = _FROZEN_FILENAMES - {"checksums.sha256"}
        names = [parts[1] for parts in declarations]
        if len(names) != len(expected_names) or set(names) != expected_names:
            return declaration_count, False
        declared_hashes = {parts[1]: parts[0] for parts in declarations}
        valid = all(
            (root / filename).is_file()
            and sha256_file(root / filename) == declared_hashes[filename]
            for filename in expected_names
        )
        return declaration_count, valid
    except (OSError, ValueError):
        return 0, False


def _frozen_table_counts(root: Path) -> dict[str, int]:
    try:
        with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            manifest_count = sum(1 for _ in csv.DictReader(handle))
        with (root / "quality_report.csv").open(encoding="utf-8", newline="") as handle:
            quality_count = sum(1 for _ in csv.DictReader(handle))
        transcript_count = sum(
            1
            for line in (root / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
        feature_count = len(pd.read_parquet(root / "acoustic_features.parquet"))
    except (OSError, ValueError):
        return {"manifest": -1, "quality": -1, "transcripts": -1, "features": -1}
    return {
        "manifest": manifest_count,
        "quality": quality_count,
        "transcripts": transcript_count,
        "features": feature_count,
    }


def _public_notebook_is_safe(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if any(cell.get("outputs") for cell in payload.get("cells", [])):
        return False
    if any(cell.get("execution_count") is not None for cell in payload.get("cells", [])):
        return False
    lowered = text.lower()
    return not (
        "source_url" in text
        or "video_title" in text
        or "total_score" in text
        or "http://" in lowered
        or "https://" in lowered
        or _WINDOWS_ABSOLUTE_PATH.search(text)
    )


def _git_ignores(root: Path, path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path.as_posix()],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
