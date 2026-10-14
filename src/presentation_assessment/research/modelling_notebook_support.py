"""Privacy-safe orchestration and figures for the modelling-results notebook."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from . import analysis_data
from .analysis_data import AnalysisCohort, AnalysisDataset, select_cohort
from .artifacts import atomic_write_json, atomic_write_text
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .clustering import (
    ClusteringResult,
    PosthocClusterScoreResult,
    fit_acoustic_clusters,
    posthoc_cluster_score_test,
)
from .dataset_pipeline import (
    DatasetPipelineConfig,
    DatasetStageAdapters,
    DatasetStageResult,
    LoadedFrozenDataset,
    freeze_dataset,
    load_strict_frozen_dataset,
    run_dataset_pipeline,
    stage_implementation_digests,
)
from .feature_registry import FeatureDefinition, FeatureRegistry, load_feature_registry
from .modelling import (
    AcousticCompositeResult,
    BootstrapInterval,
    ExperimentResults,
    compute_acoustic_composite,
    paired_group_bootstrap,
    run_nested_baselines,
)
from .protocol import load_protocol_template
from .provenance import SourceSnapshot
from .qlora import (
    TrainingGateResult,
    VerifiedQloraContext,
    evaluate_qlora_gates,
    load_verified_qlora_context,
)
from .qwen import QwenAvailability, create_local_qwen_session
from .source_manifest import ScoreStatus, SourceRow, VerificationStatus


RANDOM_SEED = 20260813
SMOKE_SAMPLE_COUNT = 24
_POSTHOC_SIGNIFICANCE_THRESHOLD = 0.05
_EXPECTED_NOTEBOOK2_PAYLOAD_COUNTS = {
    "modelling_dataset_summary": 1,
    "clustering_summary": 1,
    "cluster_score_summary": 1,
    "composite_summary": 1,
    "model_summary": 3,
    "model_comparison_summary": 1,
    "qwen_gate_summary": 1,
    "qlora_gate_summary": 1,
    "modelling_validation_summary": 1,
}
FIGURE_FILENAMES: Mapping[str, str] = MappingProxyType(
    {
        "pca_scatter": "01_pca_scatter.png",
        "similarity_heatmap": "02_similarity_heatmap.png",
        "ward_dendrogram": "03_ward_dendrogram.png",
        "target_histogram": "04_target_histogram.png",
        "composite_scatter": "05_composite_vs_total.png",
        "oof_scatter": "06_observed_vs_oof.png",
        "mae_comparison": "07_model_mae_ci.png",
    }
)

_BLUE = "#2F6690"
_GOLD = "#D4A72C"
_ORANGE = "#D97706"
_OLIVE = "#718355"
_PINK = "#C06C84"
_INK = "#24313D"
_MUTED = "#66737F"
_GRID = "#D7DEE4"
_LIGHT = "#EEF3F6"
_CLUSTER_COLOURS = (_BLUE, _GOLD, _ORANGE, _OLIVE, _PINK)
_CLUSTER_MARKERS = ("o", "s", "^", "D", "P")
_MODEL_COLOURS = {
    "training_mean": _BLUE,
    "ridge": _GOLD,
    "elastic_net": _ORANGE,
}
_MODEL_MARKERS = {"training_mean": "o", "ridge": "s", "elastic_net": "^"}
_MODEL_LABELS = {
    "training_mean": "Training-fold mean",
    "ridge": "Ridge",
    "elastic_net": "ElasticNet",
}


@dataclass(frozen=True, slots=True)
class ModellingNotebookResult:
    """One complete in-memory run; row-level values stay out of text outputs."""

    loaded: LoadedFrozenDataset
    analysis: AnalysisDataset
    cohorts: Mapping[str, AnalysisCohort]
    clustering: ClusteringResult
    cluster_scores: PosthocClusterScoreResult
    composite: AcousticCompositeResult
    baselines: ExperimentResults
    mae_intervals: tuple[BootstrapInterval, ...]
    mae_difference_intervals: tuple[BootstrapInterval, ...]
    qwen_availability: QwenAvailability
    feature_registry: FeatureRegistry
    qlora_context: VerifiedQloraContext
    qlora_gate: TrainingGateResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohorts", MappingProxyType(dict(self.cohorts)))


class _NoNetworkQwenTransport:
    """A sentinel transport that must remain unused when the digest is absent."""

    def probe_model(self, *args: object, **kwargs: object) -> str | None:
        raise RuntimeError("Qwen probing is disabled without a frozen digest")

    def generate(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError("Qwen inference is disabled without a frozen digest")


class _ModellingSmokeStages:
    """Deterministic local adapters for a strict, labelled smoke dataset."""

    def acquire(
        self, row: SourceRow, output_root: Path, allow_network: bool
    ) -> DatasetStageResult:
        if allow_network:
            raise RuntimeError("modelling smoke mode attempted network access")
        source = output_root / f"{row.sample_id}.source"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"modelling-smoke-{row.sample_id}\n".encode("utf-8"))
        return DatasetStageResult(
            sample_id=row.sample_id,
            status="succeeded",
            payload={"source_path": source, "source_sha256": sha256_file(source)},
        )

    def audio(self, source_path: Path, output_root: Path) -> DatasetStageResult:
        sample_id = output_root.parent.name
        payload: dict[str, object] = {
            "source_channel_count": 1,
            "analysis_channel_count": 1,
        }
        for representation in ("analysis", "peak_normalized", "loudness_normalized"):
            destination = output_root / f"{representation}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source_path.read_bytes() + representation.encode("utf-8"))
            payload[f"{representation}_path"] = destination
            payload[f"{representation}_sha256"] = sha256_file(destination)
        return DatasetStageResult(
            sample_id=sample_id, status="succeeded", payload=payload
        )

    def transcribe_many(self, items: Sequence[object]) -> list[DatasetStageResult]:
        return [
            DatasetStageResult(
                sample_id=str(item.sample_id),
                status="succeeded",
                payload={
                    "transcription": {
                        "text": "synthetic local modelling speech",
                        "language": "en",
                        "language_probability": 1.0,
                        "duration_s": 60.0,
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
        registry: FeatureRegistry,
    ) -> DatasetStageResult:
        del audio_paths, transcript
        index = int(sample_id[1:]) - 901
        values = {
            definition.name: _smoke_feature_value(definition, index)
            for definition in registry.features
        }
        return DatasetStageResult(
            sample_id=sample_id,
            status="succeeded",
            payload={"features": values, "missing_reasons": {}, "qc_warnings": []},
        )

    def adapters(self) -> DatasetStageAdapters:
        return DatasetStageAdapters(
            acquire=self.acquire,
            audio=self.audio,
            transcribe_many=self.transcribe_many,
            features=self.features,
        )


def create_smoke_modelling_dataset(repository_root: Path) -> LoadedFrozenDataset:
    """Create or reuse a strict 24-row synthetic dataset without network access."""
    root = Path(repository_root).resolve(strict=True)
    registry = load_feature_registry(root / "configs" / "audio_feature_registry.v1.yaml")
    protocol = load_protocol_template(root / "configs" / "experiment_protocol.template.yaml")
    artifact_root = root / "artifacts" / "research"
    config = DatasetPipelineConfig(
        repository_root=root,
        staging_dir=artifact_root / "staging" / "notebook-02-smoke-v1",
        artifact_root=artifact_root,
        rows=_smoke_rows(),
        protocol_template=protocol,
        feature_registry=registry,
        pipeline_parameters={
            "schema_version": 1,
            "notebook": "02_modelling_and_results",
            "mode": "SMOKE_MODE",
            "random_seed": protocol.random_seed,
            "expected_row_count": SMOKE_SAMPLE_COUNT,
        },
        stage_parameters={
            "acquisition": {"adapter": "synthetic-local", "revision": 1},
            "audio": {"adapter": "synthetic-local", "revision": 1},
            "asr": {"adapter": "synthetic-local", "revision": 1},
            "features": {"adapter": "synthetic-local", "revision": 1},
        },
        whisper_fingerprint={
            "model_name": "synthetic-local",
            "revision": "smoke-v1",
            "snapshot_sha256": "0" * 64,
            "local_files_only": True,
        },
        implementation_digests=stage_implementation_digests(root),
        adapters=_ModellingSmokeStages().adapters(),
        allow_network=False,
        force_stages=frozenset({"acquisition", "audio", "asr", "features"}),
        source_snapshotter=_smoke_snapshot,
    )
    frozen = freeze_dataset(run_dataset_pipeline(config))
    return load_strict_frozen_dataset(
        frozen.path,
        registry,
        expected_dataset_id=frozen.dataset_id,
        expected_split_sha256=frozen.protocol.split_sha256,
    )


def load_analysis_after_strict_validation(
    loaded: LoadedFrozenDataset, registry: FeatureRegistry
) -> AnalysisDataset:
    """Build the analysis view only from a previously strict-validated capability."""
    if not isinstance(loaded, LoadedFrozenDataset):
        raise TypeError("loaded must come from load_strict_frozen_dataset")
    if not isinstance(registry, FeatureRegistry):
        raise TypeError("registry must be a FeatureRegistry")
    root = Path(loaded.path).resolve(strict=True)
    if root.name != loaded.dataset_id or loaded.protocol.dataset_id != loaded.dataset_id:
        raise ValueError("strict frozen-dataset capability is inconsistent")

    manifest_rows = analysis_data._read_csv(root / "manifest.csv", "manifest")
    quality_rows = analysis_data._read_csv(root / "quality_report.csv", "quality")
    transcript_rows = analysis_data._read_json_lines(root / "transcripts.jsonl")
    feature_frame = pd.read_parquet(root / "acoustic_features.parquet")
    split_payload = analysis_data._read_json(root / "splits.json", "split registry")
    assignments = split_payload.get("assignments")
    if not isinstance(assignments, list) or not all(
        isinstance(item, dict) for item in assignments
    ):
        raise ValueError("invalid split assignments")

    feature_names = tuple(definition.name for definition in registry.features)
    if list(feature_frame.columns) != ["sample_id", *feature_names]:
        raise ValueError("analysis feature schema does not exactly match the registry")
    indexed = {
        "manifest": analysis_data._index_rows(manifest_rows, "manifest"),
        "quality": analysis_data._index_rows(quality_rows, "quality"),
        "transcripts": analysis_data._index_rows(transcript_rows, "transcripts"),
        "features": analysis_data._index_rows(feature_frame.to_dict("records"), "features"),
        "splits": analysis_data._index_rows(assignments, "splits"),
    }
    sample_ids = frozenset(indexed["manifest"])
    if any(frozenset(rows) != sample_ids for rows in indexed.values()):
        raise ValueError("analysis tables must contain exactly equal sample_id sets")
    if sample_ids != loaded.sample_ids or len(sample_ids) != loaded.row_count:
        raise ValueError("analysis sample IDs do not match the strict frozen dataset")
    rows = tuple(
        analysis_data._join_row(
            sample_id,
            manifest=indexed["manifest"][sample_id],
            quality=indexed["quality"][sample_id],
            transcript=indexed["transcripts"][sample_id],
            features=indexed["features"][sample_id],
            assignment=indexed["splits"][sample_id],
            definitions=registry.features,
        )
        for sample_id in sorted(sample_ids)
    )
    return AnalysisDataset(
        dataset_id=loaded.dataset_id,
        protocol=loaded.protocol,
        feature_names=feature_names,
        rows=rows,
    )


def run_modelling_analysis(
    loaded: LoadedFrozenDataset,
    dataset: AnalysisDataset,
    registry: FeatureRegistry,
) -> ModellingNotebookResult:
    """Run the reviewed clustering, modelling, uncertainty, and local-only gates."""
    if dataset.dataset_id != loaded.dataset_id:
        raise ValueError("analysis dataset does not match the strict dataset")
    cohorts = {
        purpose: select_cohort(dataset, purpose)  # type: ignore[arg-type]
        for purpose in (
            "quality_control",
            "clustering",
            "supervised_total_score",
            "transcript_model",
        )
    }
    clustering = fit_acoustic_clusters(
        cohorts["clustering"], loaded.protocol, registry
    )
    scores_after_cluster_freeze = {
        row.sample_id: row.total_score for row in cohorts["clustering"].rows
    }
    cluster_scores = posthoc_cluster_score_test(
        clustering, scores_after_cluster_freeze, seed=RANDOM_SEED
    )
    composite = compute_acoustic_composite(
        cohorts["supervised_total_score"], loaded.protocol
    )
    baselines = run_nested_baselines(
        cohorts["supervised_total_score"], loaded.protocol
    )
    mae_intervals = tuple(
        paired_group_bootstrap(evaluation, metric="mae", seed=RANDOM_SEED)
        for evaluation in baselines.evaluations
    )
    evaluations = baselines.evaluation_by_model
    mae_difference_intervals = (
        paired_group_bootstrap(
            evaluations["training_mean"], evaluations["ridge"], metric="mae", seed=RANDOM_SEED
        ),
        paired_group_bootstrap(
            evaluations["training_mean"], evaluations["elastic_net"], metric="mae", seed=RANDOM_SEED
        ),
        paired_group_bootstrap(
            evaluations["elastic_net"], evaluations["ridge"], metric="mae", seed=RANDOM_SEED
        ),
    )
    qwen_session = create_local_qwen_session(
        expected_digest=None,
        endpoint="http://127.0.0.1:11434/api/generate",
        transport=_NoNetworkQwenTransport(),
    )
    qlora_context = load_verified_qlora_context(loaded.path, registry)
    qlora_gate = evaluate_qlora_gates(qlora_context)
    return ModellingNotebookResult(
        loaded=loaded,
        analysis=dataset,
        cohorts=cohorts,
        clustering=clustering,
        cluster_scores=cluster_scores,
        composite=composite,
        baselines=baselines,
        mae_intervals=mae_intervals,
        mae_difference_intervals=mae_difference_intervals,
        qwen_availability=qwen_session.availability,
        feature_registry=registry,
        qlora_context=qlora_context,
        qlora_gate=qlora_gate,
    )


def validate_modelling_results(result: ModellingNotebookResult) -> dict[str, object]:
    """Reconcile cohorts, clustering, exactly-once OOF, fold locality, and gates."""
    row_count = result.analysis.row_count
    cohort_counts_valid = all(
        cohort.count + len(cohort.exclusions) == row_count
        for cohort in result.cohorts.values()
    )
    clustering_cohort = result.cohorts["clustering"]
    supervised = result.cohorts["supervised_total_score"]
    clustering_alignment = (
        result.clustering.dataset_id == result.loaded.dataset_id
        and set(result.clustering.sample_ids) == set(clustering_cohort.sample_ids)
    )
    selected_k_valid = (
        result.clustering.status == "indistinguishable"
        and result.clustering.selected_k is None
    ) or (
        result.clustering.status == "ok"
        and result.clustering.selected_k in {2, 3, 4, 5}
    )
    stability_evidence_valid = result.clustering.status == "ok" and (
        len(result.clustering.leave_one_feature_out)
        == len(result.clustering.feature_names)
        and all(
            math.isfinite(item.adjusted_rand_index)
            for item in result.clustering.leave_one_feature_out
        )
    )
    stability_checked = (
        result.clustering.status == "indistinguishable" or stability_evidence_valid
    )
    stability_status = (
        "not_applicable"
        if result.clustering.status == "indistinguishable"
        else "passed" if stability_evidence_valid else "failed"
    )
    one_oof_per_sample = all(
        evaluation.sample_ids == tuple(sorted(supervised.sample_ids))
        and evaluation.oof_prediction_count == supervised.count
        and len(set(evaluation.sample_ids)) == supervised.count
        and all(math.isfinite(value) for value in evaluation.predictions)
        for evaluation in result.baselines.evaluations
    )
    fold_local = all(
        _evaluation_has_fold_local_evidence(evaluation, supervised.sample_ids)
        for evaluation in result.baselines.evaluations
    )
    bootstrap_valid = all(
        interval.replicates == 5_000
        and interval.sample_count == supervised.count
        and (
            (
                interval.status == "ok"
                and interval.lower is not None
                and interval.upper is not None
                and math.isfinite(interval.lower)
                and math.isfinite(interval.upper)
                and interval.lower < interval.upper
            )
            or (
                interval.status == "not_estimable"
                and interval.reason is not None
                and interval.lower is None
                and interval.upper is None
            )
        )
        for interval in (*result.mae_intervals, *result.mae_difference_intervals)
    )
    paired_model_differences_valid = (
        len(result.mae_difference_intervals) == 3
        and all(
            interval.paired
            and interval.difference_direction == "comparison_minus_reference"
            and (
                interval.includes_zero
                is (
                    float(interval.lower) <= 0.0 <= float(interval.upper)
                    if interval.status == "ok"
                    and interval.lower is not None
                    and interval.upper is not None
                    else None
                )
            )
            for interval in result.mae_difference_intervals
        )
    )
    mae_intervals_overlap, mae_overlap_lower, mae_overlap_upper = (
        _mae_interval_overlap(result.mae_intervals)
    )
    cluster_scores_posthoc = (
        result.cluster_scores.repetitions == 10_000
        and result.cluster_scores.seed == RANDOM_SEED
        and result.cluster_scores.selected_k == result.clustering.selected_k
    )
    qwen_not_run = (
        result.qwen_availability.status == "not_run"
        and result.qwen_availability.reason == "expected_local_digest_missing"
    )
    qlora_not_run = (
        result.qlora_gate.status == "not_run" and bool(result.qlora_gate.reasons)
    )
    checks = (
        cohort_counts_valid,
        clustering_alignment,
        selected_k_valid,
        stability_checked,
        one_oof_per_sample,
        fold_local,
        bootstrap_valid,
        paired_model_differences_valid,
        cluster_scores_posthoc,
        qwen_not_run,
        qlora_not_run,
        result.composite.sample_ids == tuple(sorted(supervised.sample_ids)),
        result.baselines.speaker_disjoint is False,
        result.baselines.evaluation_claim == "exploratory_no_speaker_holdout",
    )
    return {
        **_aggregate_provenance(
            result,
            included_count=row_count,
            excluded_count=0,
            exclusion_basis="strict_frozen_dataset",
            method="fail_closed_modelling_validation",
            uncertainty_method="not_applicable",
        ),
        "artifact_type": "modelling_validation_summary",
        "valid": all(checks),
        "row_count": row_count,
        "selected_k": result.clustering.selected_k,
        "stability_checked": stability_checked,
        "stability_status": stability_status,
        "cohort_counts_valid": cohort_counts_valid,
        "one_oof_per_sample": one_oof_per_sample,
        "fold_local_preprocessing": fold_local,
        "bootstrap_replicates": 5_000,
        "bootstrap_valid": bootstrap_valid,
        "paired_model_differences_valid": paired_model_differences_valid,
        "cluster_scores_posthoc": cluster_scores_posthoc,
        "mae_intervals_overlap": mae_intervals_overlap,
        "mae_overlap_lower": mae_overlap_lower,
        "mae_overlap_upper": mae_overlap_upper,
        "qwen_not_run": qwen_not_run,
        "qlora_not_run": qlora_not_run,
    }


def _aggregate_provenance(
    result: ModellingNotebookResult,
    *,
    included_count: int,
    excluded_count: int,
    exclusion_basis: str,
    method: str,
    uncertainty_method: str,
    model_identifier: str = "not_applicable",
) -> dict[str, object]:
    """Return the common standalone provenance required by every aggregate."""
    return {
        "dataset_id": result.loaded.dataset_id,
        "protocol_id": result.loaded.protocol.protocol_id,
        "included_count": included_count,
        "excluded_count": excluded_count,
        "exclusion_basis": exclusion_basis,
        "method": method,
        "uncertainty_method": uncertainty_method,
        "model_identifier": model_identifier,
    }


def _cohort_provenance(
    result: ModellingNotebookResult,
    purpose: str,
    *,
    exclusion_basis: str,
    method: str,
    uncertainty_method: str,
) -> dict[str, object]:
    cohort = result.cohorts[purpose]
    return _aggregate_provenance(
        result,
        included_count=cohort.count,
        excluded_count=len(cohort.exclusions),
        exclusion_basis=exclusion_basis,
        method=method,
        uncertainty_method=uncertainty_method,
    )


def aggregate_payloads(result: ModellingNotebookResult) -> tuple[dict[str, object], ...]:
    """Return only the exact aggregate schemas approved for notebook display."""
    clustering = result.clustering
    stability = np.asarray(
        [item.adjusted_rand_index for item in clustering.leave_one_feature_out],
        dtype=float,
    )
    silhouettes = {item.k: item.silhouette for item in clustering.candidate_silhouettes}
    comparisons = result.cluster_scores.comparisons
    dataset_provenance = _aggregate_provenance(
        result,
        included_count=result.analysis.row_count,
        excluded_count=0,
        exclusion_basis="strict_frozen_dataset",
        method="strict_frozen_dataset_validation",
        uncertainty_method="not_applicable",
    )
    clustering_provenance = _cohort_provenance(
        result,
        "clustering",
        exclusion_basis="clustering_cohort_policy",
        method="ward_euclidean_robust_scaled_pca",
        uncertainty_method="leave_one_feature_out_adjusted_rand_index",
    )
    supervised_provenance = _cohort_provenance(
        result,
        "supervised_total_score",
        exclusion_basis="supervised_total_score_cohort_policy",
        method="nested_grouped_cross_validation",
        uncertainty_method="paired_group_bootstrap_percentile_5000_seed_20260813_95pct",
    )
    transcript_provenance = _cohort_provenance(
        result,
        "transcript_model",
        exclusion_basis="transcript_model_cohort_policy",
        method="local_model_authority_gate",
        uncertainty_method="not_applicable",
    )
    dataset_payload = {
        **dataset_provenance,
        "artifact_type": "modelling_dataset_summary",
        "row_count": result.analysis.row_count,
        "qc_count": result.cohorts["quality_control"].count,
        "qc_excluded_count": len(result.cohorts["quality_control"].exclusions),
        "clustering_count": result.cohorts["clustering"].count,
        "clustering_excluded_count": len(result.cohorts["clustering"].exclusions),
        "supervised_count": result.cohorts["supervised_total_score"].count,
        "supervised_excluded_count": len(
            result.cohorts["supervised_total_score"].exclusions
        ),
        "transcript_count": result.cohorts["transcript_model"].count,
        "transcript_excluded_count": len(result.cohorts["transcript_model"].exclusions),
    }
    clustering_payload = {
        **clustering_provenance,
        "artifact_type": "clustering_summary",
        "status": clustering.status,
        "sample_count": clustering.sample_count,
        "feature_count": len(clustering.feature_names),
        "selected_k": clustering.selected_k,
        "silhouette": (
            None if clustering.selected_k is None else silhouettes[clustering.selected_k]
        ),
        "stability_min": None if stability.size == 0 else float(np.min(stability)),
        "stability_median": None if stability.size == 0 else float(np.median(stability)),
        "similarity_scale": clustering.similarity_scale,
        "conditional_feature_included": (
            result.loaded.protocol.template.clustering.conditional_feature
            in clustering.feature_names
        ),
        "conditional_feature_status": clustering.conditional_feature_status,
    }
    minimum_adjusted_p = (
        None
        if not comparisons
        else min(item.adjusted_p_value for item in comparisons)
    )
    posthoc_payload = {
        **_aggregate_provenance(
            result,
            included_count=result.cluster_scores.sample_count,
            excluded_count=result.cluster_scores.excluded_sample_count,
            exclusion_basis="observed_scores_after_cluster_freeze",
            method="posthoc_group_permutation_holm",
            uncertainty_method="group_permutation_10000_seed_20260813_holm",
        ),
        "artifact_type": "cluster_score_summary",
        "status": result.cluster_scores.status,
        "selected_k": result.cluster_scores.selected_k,
        "sample_count": result.cluster_scores.sample_count,
        "excluded_sample_count": result.cluster_scores.excluded_sample_count,
        "group_count": result.cluster_scores.group_count,
        "comparison_count": len(comparisons),
        "minimum_adjusted_p": minimum_adjusted_p,
        "significance_threshold": _POSTHOC_SIGNIFICANCE_THRESHOLD,
        "statistically_significant": (
            minimum_adjusted_p is not None
            and minimum_adjusted_p < _POSTHOC_SIGNIFICANCE_THRESHOLD
        ),
    }
    composite_payload = {
        **{
            **supervised_provenance,
            "method": "cross_fitted_equal_available_dimensions",
            "uncertainty_method": "descriptive_correlations_no_interval",
        },
        "artifact_type": "composite_summary",
        "sample_count": result.composite.sample_count,
        "feature_count": len(result.composite.feature_names),
        "spearman": result.composite.spearman,
        "pearson": result.composite.pearson,
        "learned_weights": result.composite.learned_weights,
        "calibrated_to_target": result.composite.calibrated_to_target,
    }
    intervals = {item.reference_model: item for item in result.mae_intervals}
    model_payloads = tuple(
        {
            **{
                **supervised_provenance,
                "model_identifier": {
                    "training_mean": "training_fold_mean_v1",
                    "ridge": "sklearn_linear_model_ridge",
                    "elastic_net": "sklearn_linear_model_elastic_net",
                }[evaluation.model_name],
            },
            "artifact_type": "model_summary",
            "model_name": evaluation.model_name,
            "sample_count": evaluation.sample_count,
            "oof_prediction_count": evaluation.oof_prediction_count,
            "mae": evaluation.mae,
            "rmse": evaluation.rmse,
            "spearman": evaluation.spearman,
            "pearson": evaluation.pearson,
            "mae_ci_status": intervals[evaluation.model_name].status,
            "mae_ci_lower": intervals[evaluation.model_name].lower,
            "mae_ci_upper": intervals[evaluation.model_name].upper,
            "fold_count": len(evaluation.fold_audits),
            "fold_local_preprocessing": _evaluation_has_fold_local_evidence(
                evaluation, result.cohorts["supervised_total_score"].sample_ids
            ),
        }
        for evaluation in result.baselines.evaluations
    )
    difference_names = (
        "ridge_minus_training_mean",
        "elastic_net_minus_training_mean",
        "ridge_minus_elastic_net",
    )
    model_comparison_payload: dict[str, object] = {
        **{
            **supervised_provenance,
            "method": "paired_model_mae_differences",
            "model_identifier": "training_mean_ridge_elastic_net",
        },
        "artifact_type": "model_comparison_summary",
    }
    for name, interval in zip(
        difference_names, result.mae_difference_intervals, strict=True
    ):
        model_comparison_payload.update(
            {
                f"{name}_estimate": interval.estimate,
                f"{name}_status": interval.status,
                f"{name}_ci_lower": interval.lower,
                f"{name}_ci_upper": interval.upper,
                f"{name}_valid_replicates": interval.valid_replicates,
                f"{name}_required_replicates": interval.replicates,
                f"{name}_includes_zero": interval.includes_zero,
            }
        )
    qwen_payload = {
        **{
            **transcript_provenance,
            "method": "local_qwen_identity_gate",
            "model_identifier": result.qwen_availability.model_id,
        },
        "artifact_type": "qwen_gate_summary",
        "status": result.qwen_availability.status,
        "reason_code": result.qwen_availability.reason,
        "warning": result.qwen_availability.warning,
        "attempt_count": 0,
        "calibration_fit": False,
    }
    qlora_payload = {
        **{
            **transcript_provenance,
            "method": "authority_bound_qlora_gate",
            "model_identifier": "not_applicable",
        },
        "artifact_type": "qlora_gate_summary",
        "status": result.qlora_gate.status,
        "reason_count": len(result.qlora_gate.reasons),
        "ready": result.qlora_gate.status == "ready",
        "training_started": False,
        "metrics_available": result.qlora_gate.metrics is not None,
        "qlora_reasons": {reason: True for reason in result.qlora_gate.reasons},
    }
    return (
        dataset_payload,
        clustering_payload,
        posthoc_payload,
        composite_payload,
        *model_payloads,
        model_comparison_payload,
        qwen_payload,
        qlora_payload,
        validate_modelling_results(result),
    )


def display_aggregate_payload(payload: Mapping[str, object]) -> None:
    """Display one aggregate as JSON only, avoiding an unsafe text repr."""
    from IPython.display import display

    display({"application/json": dict(payload)}, raw=True)


def display_private_png(path: Path) -> None:
    """Display a saved raster without revealing its local path in notebook output."""
    from IPython.display import display

    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    display({"image/png": encoded}, raw=True)


def render_private_figures(
    result: ModellingNotebookResult, output_dir: Path
) -> Mapping[str, Path]:
    """Render all required static figures under a private artifact directory."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": _INK,
            "axes.labelcolor": _INK,
            "axes.titlecolor": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    paths = {name: destination / filename for name, filename in FIGURE_FILENAMES.items()}
    captions = _figure_captions(result)
    _plot_pca(result, paths["pca_scatter"], captions["pca_scatter"], plt)
    _plot_similarity(result, paths["similarity_heatmap"], captions["similarity_heatmap"], plt)
    _plot_dendrogram(result, paths["ward_dendrogram"], captions["ward_dendrogram"], plt)
    _plot_target_histogram(result, paths["target_histogram"], captions["target_histogram"], plt)
    _plot_composite(result, paths["composite_scatter"], captions["composite_scatter"], plt)
    _plot_oof(result, paths["oof_scatter"], captions["oof_scatter"], plt)
    _plot_mae(result, paths["mae_comparison"], captions["mae_comparison"], plt)
    atomic_write_json(
        destination / "figure_manifest.json",
        _figure_manifest_payload(result, paths),
    )
    return MappingProxyType(paths)


def write_private_results(
    result: ModellingNotebookResult,
    output_dir: Path,
    *,
    figure_paths: Mapping[str, Path],
) -> tuple[Path, ...]:
    """Persist aggregate tables and row-level audit values only under private roots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate_path = destination / "aggregate_results.json"
    cluster_values_path = destination / "cluster_assignments.jsonl"
    composite_values_path = destination / "composite_values.jsonl"
    oof_values_path = destination / "oof_predictions.jsonl"
    cluster_table_path = destination / "cluster_summary.csv"
    cohort_table_path = destination / "cohort_summary.csv"
    model_table_path = destination / "model_metrics.csv"
    manifest_path = destination / "model_run_manifest.json"
    checksum_path = destination / "result_checksums.sha256"

    atomic_write_json(
        aggregate_path,
        {
            "schema_version": 1,
            "dataset_id": result.loaded.dataset_id,
            "payloads": list(aggregate_payloads(result)),
            "evaluation_claim": result.baselines.evaluation_claim,
            "limitation": result.baselines.limitation,
        },
    )
    _write_json_lines(
        cluster_values_path,
        (
            {
                "sample_id": sample_id,
                "cluster_id": cluster_id,
                "pc1": coordinates[0],
                "pc2": coordinates[1],
            }
            for sample_id, cluster_id, coordinates in zip(
                result.clustering.sample_ids,
                result.clustering.cluster_assignments,
                result.clustering.pca_coordinates,
                strict=True,
            )
        ),
    )
    _write_json_lines(
        composite_values_path,
        (
            {
                "sample_id": sample_id,
                "observed": observed,
                "composite_index": index_value,
            }
            for sample_id, observed, index_value in zip(
                result.composite.sample_ids,
                result.composite.observed_scores,
                result.composite.index_values,
                strict=True,
            )
        ),
    )
    _write_json_lines(
        oof_values_path,
        (
            {
                "model_name": evaluation.model_name,
                "sample_id": sample_id,
                "observed": observed,
                "prediction": prediction,
            }
            for evaluation in result.baselines.evaluations
            for sample_id, observed, prediction in zip(
                evaluation.sample_ids,
                evaluation.observed,
                evaluation.predictions,
                strict=True,
            )
        ),
    )
    cluster_counts = Counter(result.clustering.cluster_assignments)
    _write_csv(
        cluster_table_path,
        ("cluster_id", "sample_count", "sample_percent"),
        (
            {
                "cluster_id": cluster_id,
                "sample_count": count,
                "sample_percent": 100.0 * count / result.clustering.sample_count,
            }
            for cluster_id, count in sorted(cluster_counts.items())
        ),
    )
    _write_csv(
        cohort_table_path,
        ("purpose", "included_count", "excluded_count"),
        (
            {
                "purpose": purpose,
                "included_count": cohort.count,
                "excluded_count": len(cohort.exclusions),
            }
            for purpose, cohort in result.cohorts.items()
        ),
    )
    intervals = {item.reference_model: item for item in result.mae_intervals}
    _write_csv(
        model_table_path,
        (
            "model_name",
            "sample_count",
            "mae",
            "mae_ci_status",
            "mae_ci_lower",
            "mae_ci_upper",
            "rmse",
            "spearman",
            "pearson",
        ),
        (
            {
                "model_name": evaluation.model_name,
                "sample_count": evaluation.sample_count,
                "mae": evaluation.mae,
                "mae_ci_status": intervals[evaluation.model_name].status,
                "mae_ci_lower": intervals[evaluation.model_name].lower,
                "mae_ci_upper": intervals[evaluation.model_name].upper,
                "rmse": evaluation.rmse,
                "spearman": evaluation.spearman,
                "pearson": evaluation.pearson,
            }
            for evaluation in result.baselines.evaluations
        ),
    )
    base_paths = (
        aggregate_path,
        cluster_values_path,
        composite_values_path,
        oof_values_path,
        cluster_table_path,
        cohort_table_path,
        model_table_path,
    )
    manifest = _model_run_manifest(result, base_paths, figure_paths)
    atomic_write_json(manifest_path, manifest)
    checksum_targets = (*base_paths, manifest_path, *_figure_audit_paths(figure_paths))
    _write_result_checksums(checksum_path, destination.parent, checksum_targets)
    validate_model_run_artifacts(result, destination, next(iter(figure_paths.values())).parent)
    return (*base_paths, manifest_path, checksum_path)


def validate_model_run_artifacts(
    result: ModellingNotebookResult,
    result_directory: Path,
    figure_directory: Path,
) -> bool:
    """Reconstruct the private run audit and verify every declared artifact hash."""
    results = Path(result_directory).resolve(strict=True)
    figures = Path(figure_directory).resolve(strict=True)
    base_names = (
        "aggregate_results.json",
        "cluster_assignments.jsonl",
        "composite_values.jsonl",
        "oof_predictions.jsonl",
        "cluster_summary.csv",
        "cohort_summary.csv",
        "model_metrics.csv",
    )
    expected_result_names = {
        *base_names,
        "model_run_manifest.json",
        "result_checksums.sha256",
    }
    if {path.name for path in results.iterdir() if path.is_file()} != expected_result_names:
        raise ValueError("private model result file inventory is invalid")
    if {path.name for path in figures.iterdir() if path.is_file()} != {
        *FIGURE_FILENAMES.values(),
        "figure_manifest.json",
    }:
        raise ValueError("private figure file inventory is invalid")
    base_paths = tuple(results / name for name in base_names)
    figure_paths = MappingProxyType(
        {name: figures / filename for name, filename in FIGURE_FILENAMES.items()}
    )
    manifest_path = results / "model_run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("model run manifest is unreadable") from error
    expected_manifest = _model_run_manifest(result, base_paths, figure_paths)
    if manifest != expected_manifest:
        raise ValueError("model run manifest does not reconstruct the in-memory run")
    checksum_targets = (*base_paths, manifest_path, *_figure_audit_paths(figure_paths))
    expected_checksums = _result_checksum_text(results.parent, checksum_targets)
    if (results / "result_checksums.sha256").read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("model result checksum registry is invalid")
    try:
        figure_manifest = json.loads(
            (figures / "figure_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("figure manifest is unreadable") from error
    if figure_manifest != _figure_manifest_payload(result, figure_paths):
        raise ValueError("figure manifest does not reconstruct figure provenance")
    expected_aggregate = {
        "schema_version": 1,
        "dataset_id": result.loaded.dataset_id,
        "payloads": list(aggregate_payloads(result)),
        "evaluation_claim": result.baselines.evaluation_claim,
        "limitation": result.baselines.limitation,
    }
    if json.loads((results / "aggregate_results.json").read_text(encoding="utf-8")) != expected_aggregate:
        raise ValueError("aggregate result output cannot be reconstructed")
    if _read_json_lines(results / "cluster_assignments.jsonl") != _cluster_value_rows(result):
        raise ValueError("cluster output cannot be reconstructed")
    if _read_json_lines(results / "composite_values.jsonl") != _composite_value_rows(result):
        raise ValueError("composite output cannot be reconstructed")
    if _read_json_lines(results / "oof_predictions.jsonl") != _oof_value_rows(result):
        raise ValueError("OOF output cannot be reconstructed")
    cluster_counts = Counter(result.clustering.cluster_assignments)
    expected_cluster_table = _csv_text(
        ("cluster_id", "sample_count", "sample_percent"),
        [
            {
                "cluster_id": cluster_id,
                "sample_count": count,
                "sample_percent": 100.0 * count / result.clustering.sample_count,
            }
            for cluster_id, count in sorted(cluster_counts.items())
        ],
    )
    if (results / "cluster_summary.csv").read_text(encoding="utf-8") != expected_cluster_table:
        raise ValueError("cluster summary output cannot be reconstructed")
    expected_cohort_table = _csv_text(
        ("purpose", "included_count", "excluded_count"),
        [
            {
                "purpose": purpose,
                "included_count": cohort.count,
                "excluded_count": len(cohort.exclusions),
            }
            for purpose, cohort in result.cohorts.items()
        ],
    )
    if (results / "cohort_summary.csv").read_text(encoding="utf-8") != expected_cohort_table:
        raise ValueError("cohort summary output cannot be reconstructed")
    intervals = {item.reference_model: item for item in result.mae_intervals}
    expected_model_table = _csv_text(
        (
            "model_name",
            "sample_count",
            "mae",
            "mae_ci_status",
            "mae_ci_lower",
            "mae_ci_upper",
            "rmse",
            "spearman",
            "pearson",
        ),
        [
            {
                "model_name": evaluation.model_name,
                "sample_count": evaluation.sample_count,
                "mae": evaluation.mae,
                "mae_ci_status": intervals[evaluation.model_name].status,
                "mae_ci_lower": intervals[evaluation.model_name].lower,
                "mae_ci_upper": intervals[evaluation.model_name].upper,
                "rmse": evaluation.rmse,
                "spearman": evaluation.spearman,
                "pearson": evaluation.pearson,
            }
            for evaluation in result.baselines.evaluations
        ],
    )
    if (results / "model_metrics.csv").read_text(encoding="utf-8") != expected_model_table:
        raise ValueError("model metrics output cannot be reconstructed")
    return True


def _model_run_manifest(
    result: ModellingNotebookResult,
    result_paths: Sequence[Path],
    figure_paths: Mapping[str, Path],
) -> dict[str, object]:
    project_root = result.loaded.path.resolve(strict=True).parents[2]
    implementation_files = (
        "src/presentation_assessment/research/modelling.py",
        "src/presentation_assessment/research/modelling_notebook_support.py",
        "src/presentation_assessment/research/notebook_support.py",
        "src/presentation_assessment/research/clustering.py",
        "src/presentation_assessment/research/qwen.py",
        "src/presentation_assessment/research/qlora.py",
        "configs/audio_feature_registry.v1.yaml",
        "notebooks/02_modelling_and_results.ipynb",
    )
    code_identities = {
        name: sha256_file(project_root / name) for name in implementation_files
    }
    cohort_bases = {
        "quality_control": "quality_control_cohort_policy",
        "clustering": "clustering_cohort_policy",
        "supervised_total_score": "supervised_total_score_cohort_policy",
        "transcript_model": "transcript_model_cohort_policy",
    }
    cohorts: dict[str, object] = {}
    for purpose, cohort in result.cohorts.items():
        cohorts[purpose] = {
            "included_count": cohort.count,
            "excluded_count": len(cohort.exclusions),
            "exclusion_basis": cohort_bases[purpose],
            "included_sample_sha256": [
                _identifier_sha256("sample", sample_id)
                for sample_id in sorted(cohort.sample_ids)
            ],
            "exclusions": [
                {
                    "sample_sha256": _identifier_sha256("sample", item.sample_id),
                    "reasons": list(item.reasons),
                }
                for item in sorted(cohort.exclusions, key=lambda value: value.sample_id)
            ],
        }
    models = {
        evaluation.model_name: {
            "model_identifier": {
                "training_mean": "training_fold_mean_v1",
                "ridge": "sklearn.linear_model.Ridge",
                "elastic_net": "sklearn.linear_model.ElasticNet",
            }[evaluation.model_name],
            "hyperparameter_grid": {
                name: list(values)
                for name, values in evaluation.hyperparameter_grid.items()
            },
            "primary_metric": evaluation.primary_metric,
            "metrics": {
                "mae": evaluation.mae,
                "rmse": evaluation.rmse,
                "spearman": evaluation.spearman,
                "pearson": evaluation.pearson,
                "spearman_reason": evaluation.spearman_reason,
                "pearson_reason": evaluation.pearson_reason,
            },
            "oof_count": evaluation.oof_prediction_count,
            "oof_sample_sha256": [
                _identifier_sha256("sample", sample_id)
                for sample_id in evaluation.sample_ids
            ],
            "outer_folds": [_outer_fold_manifest(audit) for audit in evaluation.fold_audits],
        }
        for evaluation in result.baselines.evaluations
    }
    output_paths = (*result_paths, *_figure_audit_paths(figure_paths))
    outputs = {
        _artifact_relative_path(result_paths[0].parent.parent, path): sha256_file(path)
        for path in output_paths
    }
    return {
        "schema_version": 1,
        "dataset_id": result.loaded.dataset_id,
        "protocol_id": result.loaded.protocol.protocol_id,
        "split_sha256": result.loaded.protocol.split_sha256,
        "code_identities": code_identities,
        "random_seeds": {
            "protocol": result.loaded.protocol.template.random_seed,
            "bootstrap": RANDOM_SEED,
            "cluster_permutation": result.clustering.permutation_seed,
        },
        "cohorts": cohorts,
        "feature_registry": result.feature_registry.model_dump(mode="json"),
        "composite_folds": [
            {
                "outer_fold": audit.outer_fold,
                "training_sample_sha256": [
                    _identifier_sha256("sample", value)
                    for value in audit.training_sample_ids
                ],
                "validation_sample_sha256": [
                    _identifier_sha256("sample", value)
                    for value in audit.validation_sample_ids
                ],
                "feature_names": list(audit.feature_names),
                "excluded_all_null_features": list(audit.excluded_all_null_features),
                "location": dict(audit.location),
                "scale": dict(audit.scale),
            }
            for audit in result.composite.fold_audits
        ],
        "models": models,
        "bootstrap_intervals": [
            asdict(interval)
            for interval in (*result.mae_intervals, *result.mae_difference_intervals)
        ],
        "qwen_gate": {
            "status": result.qwen_availability.status,
            "model_id": result.qwen_availability.model_id,
            "expected_digest": result.qwen_availability.expected_digest,
            "observed_digest": result.qwen_availability.observed_digest,
            "reason": result.qwen_availability.reason,
        },
        "qlora_gate": {
            "status": result.qlora_gate.status,
            "reasons": list(result.qlora_gate.reasons),
            "registry_sha256": result.qlora_context.registry_sha256,
            "authority_entry": result.qlora_context.authority_entry,
            "qlora_sample_sha256": [
                _identifier_sha256("sample", value)
                for value in result.qlora_context.qlora_sample_ids
            ],
        },
        "outputs": outputs,
    }


def _outer_fold_manifest(audit: object) -> dict[str, object]:
    return {
        "outer_fold": audit.outer_fold,
        "training_sample_sha256": [
            _identifier_sha256("sample", value) for value in audit.training_sample_ids
        ],
        "validation_sample_sha256": [
            _identifier_sha256("sample", value) for value in audit.validation_sample_ids
        ],
        "training_group_sha256": [
            _identifier_sha256("group", value) for value in audit.training_group_ids
        ],
        "validation_group_sha256": [
            _identifier_sha256("group", value) for value in audit.validation_group_ids
        ],
        "feature_names": list(audit.feature_names),
        "excluded_all_null_features": list(audit.excluded_all_null_features),
        "location": dict(audit.location),
        "scale": dict(audit.scale),
        "selected_hyperparameters": dict(audit.selected_hyperparameters),
        "inner_folds": [
            {
                "inner_fold": inner.inner_fold,
                "training_sample_sha256": [
                    _identifier_sha256("sample", value)
                    for value in inner.training_sample_ids
                ],
                "validation_sample_sha256": [
                    _identifier_sha256("sample", value)
                    for value in inner.validation_sample_ids
                ],
                "training_group_sha256": [
                    _identifier_sha256("group", value)
                    for value in inner.training_group_ids
                ],
                "validation_group_sha256": [
                    _identifier_sha256("group", value)
                    for value in inner.validation_group_ids
                ],
                "feature_names": list(inner.feature_names),
                "excluded_all_null_features": list(inner.excluded_all_null_features),
                "location": dict(inner.location),
                "scale": dict(inner.scale),
            }
            for inner in audit.inner_audits
        ],
    }


def _identifier_sha256(kind: str, value: str) -> str:
    return sha256_bytes(f"{kind}\0{value}".encode("utf-8"))


def _figure_audit_paths(figure_paths: Mapping[str, Path]) -> tuple[Path, ...]:
    if set(figure_paths) != set(FIGURE_FILENAMES):
        raise ValueError("figure paths do not match the required inventory")
    resolved = tuple(Path(figure_paths[name]).resolve(strict=True) for name in FIGURE_FILENAMES)
    parents = {path.parent for path in resolved}
    if len(parents) != 1:
        raise ValueError("figure paths must share one private directory")
    figure_manifest = next(iter(parents)) / "figure_manifest.json"
    if not figure_manifest.is_file():
        raise ValueError("figure manifest is missing")
    return (*resolved, figure_manifest)


def _artifact_relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise ValueError("model output escaped its private artifact root") from error


def _write_result_checksums(path: Path, root: Path, targets: Sequence[Path]) -> None:
    atomic_write_text(path, _result_checksum_text(root, targets))


def _result_checksum_text(root: Path, targets: Sequence[Path]) -> str:
    lines = sorted(
        f"{sha256_file(path)}  {_artifact_relative_path(root, path)}" for path in targets
    )
    return "\n".join(lines) + "\n"


def _read_json_lines(path: Path) -> list[object]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _cluster_value_rows(result: ModellingNotebookResult) -> list[object]:
    return [
        {
            "sample_id": sample_id,
            "cluster_id": cluster_id,
            "pc1": coordinates[0],
            "pc2": coordinates[1],
        }
        for sample_id, cluster_id, coordinates in zip(
            result.clustering.sample_ids,
            result.clustering.cluster_assignments,
            result.clustering.pca_coordinates,
            strict=True,
        )
    ]


def _composite_value_rows(result: ModellingNotebookResult) -> list[object]:
    return [
        {"sample_id": sample_id, "observed": observed, "composite_index": index_value}
        for sample_id, observed, index_value in zip(
            result.composite.sample_ids,
            result.composite.observed_scores,
            result.composite.index_values,
            strict=True,
        )
    ]


def _oof_value_rows(result: ModellingNotebookResult) -> list[object]:
    return [
        {
            "model_name": evaluation.model_name,
            "sample_id": sample_id,
            "observed": observed,
            "prediction": prediction,
        }
        for evaluation in result.baselines.evaluations
        for sample_id, observed, prediction in zip(
            evaluation.sample_ids,
            evaluation.observed,
            evaluation.predictions,
            strict=True,
        )
    ]


def modelling_artifact_directory(
    repository_root: Path, dataset_id: str, mode: str
) -> Path:
    """Return the ignored, dataset-scoped directory for one execution mode."""
    normalized_mode = str(mode).strip().upper()
    if normalized_mode not in {"SMOKE_MODE", "FULL_RUN"}:
        raise ValueError("THESIS_MODE must be SMOKE_MODE or FULL_RUN")
    if len(dataset_id) != 64 or any(character not in "0123456789abcdef" for character in dataset_id):
        raise ValueError("THESIS_DATASET_ID must be a lowercase SHA-256 dataset ID")
    root = Path(repository_root).resolve(strict=True)
    return root / "artifacts" / "research" / "modelling" / dataset_id / normalized_mode.lower()


def resolve_repository_relative(
    repository_root: Path, value: str | Path, label: str
) -> Path:
    """Resolve a required repository-relative configuration path."""
    root = Path(repository_root).resolve(strict=True)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be repository-relative")
    resolved = (root / relative).resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"{label} must remain inside the repository")
    return resolved


def finalize_executed_notebook_narrative(path: Path) -> Path:
    """Write tl;dr and takeaways only after executed aggregate outputs validate."""
    import nbformat

    from .notebook_support import validate_executed_notebook_outputs

    notebook_path = Path(path)
    validate_executed_notebook_outputs(notebook_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    payloads = _executed_aggregate_payloads(notebook)
    if len(payloads) != 11:
        raise ValueError("executed modelling notebook requires exactly eleven aggregate payloads")
    by_type: dict[str, list[Mapping[str, object]]] = {}
    for payload in payloads:
        by_type.setdefault(str(payload["artifact_type"]), []).append(payload)
    actual_counts = {name: len(values) for name, values in by_type.items()}
    if actual_counts != _EXPECTED_NOTEBOOK2_PAYLOAD_COUNTS:
        raise ValueError("executed modelling notebook has an unexpected payload inventory")
    dataset = _one_payload(by_type, "modelling_dataset_summary")
    clustering = _one_payload(by_type, "clustering_summary")
    cluster_scores = _one_payload(by_type, "cluster_score_summary")
    composite = _one_payload(by_type, "composite_summary")
    qwen = _one_payload(by_type, "qwen_gate_summary")
    qlora = _one_payload(by_type, "qlora_gate_summary")
    validation = _one_payload(by_type, "modelling_validation_summary")
    model_comparison = _one_payload(by_type, "model_comparison_summary")
    models = by_type.get("model_summary", [])
    model_counts = Counter(str(item.get("model_name")) for item in models)
    if model_counts != Counter(
        {"training_mean": 1, "ridge": 1, "elastic_net": 1}
    ):
        raise ValueError("executed modelling notebook requires exactly one summary for each model")
    if validation.get("valid") is not True:
        raise ValueError("executed modelling outputs are incomplete or invalid")
    posthoc_p = _optional_finite_number(
        cluster_scores.get("minimum_adjusted_p"), "post-hoc adjusted p-value"
    )
    significance_threshold = _required_finite_number(
        cluster_scores.get("significance_threshold"), "significance threshold"
    )
    if not 0.0 < significance_threshold < 1.0:
        raise ValueError("significance threshold must be between zero and one")
    computed_significance = (
        posthoc_p is not None and posthoc_p < significance_threshold
    )
    if cluster_scores.get("statistically_significant") is not computed_significance:
        raise ValueError("post-hoc significance flag does not match its p-value")
    paired_statements = _paired_difference_statements(model_comparison, models)
    best = min(models, key=lambda item: float(item["mae"]))
    stability_value = clustering.get("stability_median")
    weak_stability = stability_value is None or float(stability_value) < 0.5
    stability_clause = (
        "indicating a feature-sensitive solution"
        if weak_stability
        else "indicating moderate-to-high feature stability"
    )
    cluster_takeaway = (
        f"- Median leave-one-feature-out ARI was {_format_number(stability_value)}: "
        "the selected clusters are feature-sensitive and not a robust natural taxonomy; "
        "the post-hoc score analysis cannot alter k."
        if weak_stability
        else "- Clusters are score-blind descriptive groupings; the post-hoc permutation analysis cannot alter k."
    )
    if posthoc_p is None:
        posthoc_statement = (
            "The post-hoc cluster score comparison was not estimable; it provides no compelling evidence of cluster score differences."
        )
    elif computed_significance:
        posthoc_statement = (
            f"Holm-adjusted p={_format_number(posthoc_p)} was below the declared descriptive "
            f"alpha={_format_number(significance_threshold)} threshold; this remains exploratory post-hoc evidence."
        )
    else:
        posthoc_statement = (
            f"Holm-adjusted p={_format_number(posthoc_p)} was not statistically significant at the declared "
            f"descriptive alpha={_format_number(significance_threshold)} threshold and is not compelling evidence of cluster score differences."
        )
    paired_statement = " ".join(paired_statements)
    tl_dr = (
        "## tl;dr\n\n"
        f"- Strict dataset `{dataset['dataset_id']}` contains {dataset['row_count']} rows; "
        f"{dataset['clustering_count']} enter clustering and {dataset['supervised_count']} enter supervised OOF evaluation.\n"
        f"- Ward clustering selected k={clustering['selected_k']} across {clustering['feature_count']} acoustic features; "
        f"median leave-one-feature-out ARI was {_format_number(stability_value)}, {stability_clause}.\n"
        f"- {posthoc_statement}\n"
        f"- The fixed acoustic composite was associated with the broad advisor score at Spearman "
        f"{_format_number(composite['spearman'])} and Pearson {_format_number(composite['pearson'])}; it is an uncalibrated index, not a learned grade.\n"
        f"- Lowest observed exploratory OOF MAE was {_format_number(best['mae'])} points for {_MODEL_LABELS[str(best['model_name'])]}, "
        f"with 95% group-bootstrap CI [{_format_number(best['mae_ci_lower'])}, {_format_number(best['mae_ci_upper'])}]. "
        f"{paired_statement}\n"
        f"- Qwen and QLoRA were not run: Qwen reason `{qwen['reason_code']}`; QLoRA reported {qlora['reason_count']} failed gates."
    )
    takeaways = (
        "## Takeaways\n\n"
        f"{cluster_takeaway}\n"
        f"- {posthoc_statement}\n"
        "- Every eligible supervised observation has one OOF prediction, with preprocessing and tuning fit inside training folds.\n"
        f"- {paired_statement}\n"
        "- Model differences remain exploratory: speaker identity is not reliable, folds are not speaker-disjoint, and there is no independent held-out test.\n"
        "- The source label is an overall advisor score, so these results do not support causal, rubric-dimension, personality, or speaker-independent claims.\n"
        "- Qwen calibration and QLoRA effectiveness evidence remain unavailable until their frozen identity, label, split, artifact, and hardware gates pass."
    )
    replacements = {"narrative-tldr": tl_dr, "narrative-takeaways": takeaways}
    found: set[str] = set()
    for cell in notebook.cells:
        tags = set(cell.get("metadata", {}).get("tags", []))
        for tag, text in replacements.items():
            if tag in tags:
                cell.source = text
                found.add(tag)
    if found != set(replacements):
        raise ValueError("executed notebook is missing narrative cells")
    nbformat.write(notebook, notebook_path)
    return notebook_path


def _smoke_rows() -> tuple[SourceRow, ...]:
    rows: list[SourceRow] = []
    for index in range(SMOKE_SAMPLE_COUNT):
        sample_number = 901 + index
        video_id = f"smoke{sample_number:06d}"
        score = float(np.clip(50.0 + 1.35 * index + 2.8 * math.sin(index * 0.7), 0.0, 100.0))
        rows.append(
            SourceRow(
                source_row_id=f"smoke-row-{sample_number}",
                source_page="synthetic-local",
                sample_id=f"N{sample_number}",
                legacy_sample_id="",
                source_url="https" + "://" + "youtu.be/" + video_id,
                video_id=video_id,
                video_title="",
                total_score=score,
                score_status=ScoreStatus.OBSERVED,
                source_course="synthetic-local",
                verification_status=VerificationStatus.VERIFIED,
                verification_note="synthetic-local",
            )
        )
    return tuple(rows)


def _smoke_feature_value(definition: FeatureDefinition, index: int) -> object:
    cluster = index % 3
    trend = index / (SMOKE_SAMPLE_COUNT - 1)
    primary = {
        "speech_ratio": 0.48 + 0.28 * trend + 0.015 * cluster,
        "internal_pauses_per_speech_minute": 10.0 - 5.5 * trend + 0.35 * cluster,
        "internal_pause_duration_median_seconds": 0.52 - 0.24 * trend + 0.025 * cluster,
        "speech_segment_duration_median_seconds": 1.2 + 1.8 * trend + 0.08 * cluster,
        "acoustic_envelope_peaks_per_speech_minute": 72.0 + 38.0 * trend + 2.0 * cluster,
        "envelope_peak_interval_cv": 0.42 - 0.18 * trend + 0.015 * cluster,
        "pitch_p10_p90_range_semitones": 3.0 + 5.0 * trend + 0.3 * cluster,
        "pitch_std_semitones": 1.2 + 2.6 * trend + 0.16 * cluster,
        "rms_robust_cv": 0.18 + 0.30 * trend + 0.018 * cluster,
        "spectral_flux_dispersion": 0.04 + 0.12 * trend + 0.009 * cluster,
        "candidate_transients_per_speech_minute": 7.5 - 3.8 * trend + 0.3 * cluster,
        "asr_units_per_speech_minute": 92.0 + 42.0 * trend + 1.5 * cluster,
    }
    if definition.name in primary:
        return float(primary[definition.name])
    if definition.dtype == "boolean":
        return False
    if definition.dtype == "string":
        return "high"
    lower = None if definition.minimum is None else float(definition.minimum)
    upper = None if definition.maximum is None else float(definition.maximum)
    fraction = (index + 1) / (SMOKE_SAMPLE_COUNT + 1)
    if lower is not None and upper is not None:
        value = lower + fraction * (upper - lower)
    elif lower is not None:
        value = lower + 1.0 + 0.05 * index
    elif upper is not None:
        value = upper - 1.0 - 0.05 * index
    else:
        value = 0.1 + 0.05 * index
    return int(round(value)) if definition.dtype == "integer" else float(value)


def _smoke_snapshot(_: Path) -> SourceSnapshot:
    files = {"src/synthetic_modelling_smoke.py": "c" * 64}
    return SourceSnapshot(
        git_commit="a" * 40,
        git_dirty=False,
        tree_sha256=sha256_bytes(canonical_json_bytes(files)),
        files=files,
        git_diff_patch=b"",
        untracked_files={},
    )


def _evaluation_has_fold_local_evidence(
    evaluation: object, expected_sample_ids: Sequence[str]
) -> bool:
    audits = tuple(evaluation.fold_audits)
    validation_ids = [
        sample_id for audit in audits for sample_id in audit.validation_sample_ids
    ]
    if sorted(validation_ids) != sorted(expected_sample_ids):
        return False
    for audit in audits:
        if set(audit.training_sample_ids).intersection(audit.validation_sample_ids):
            return False
        if set(audit.training_group_ids).intersection(audit.validation_group_ids):
            return False
        if evaluation.model_name == "training_mean":
            if audit.feature_names or audit.location or audit.scale or audit.inner_audits:
                return False
            continue
        if set(audit.location) != set(audit.feature_names) or set(audit.scale) != set(
            audit.feature_names
        ):
            return False
        for inner in audit.inner_audits:
            if set(inner.training_sample_ids).intersection(inner.validation_sample_ids):
                return False
            if set(inner.training_group_ids).intersection(inner.validation_group_ids):
                return False
            if not set(inner.training_sample_ids).union(inner.validation_sample_ids) <= set(
                audit.training_sample_ids
            ):
                return False
            if set(inner.location) != set(inner.feature_names) or set(inner.scale) != set(
                inner.feature_names
            ):
                return False
    return True


def _figure_captions(result: ModellingNotebookResult) -> Mapping[str, str]:
    clustering = result.cohorts["clustering"]
    supervised = result.cohorts["supervised_total_score"]
    short_id = result.loaded.dataset_id[:12]

    def caption(
        cohort: AnalysisCohort,
        *,
        basis: str,
        units: str,
        method: str,
        uncertainty: str,
    ) -> str:
        return (
            f"dataset={short_id}; n={cohort.count} of {result.analysis.row_count}; "
            f"excluded={len(cohort.exclusions)}\n"
            f"basis={basis}; units={units}\n"
            f"method={method}\n"
            f"uncertainty={uncertainty}"
        )

    return MappingProxyType(
        {
            "pca_scatter": caption(
                clustering,
                basis="clustering_cohort_policy",
                units="robust-scaled_acoustic_features",
                method="PCA_and_score-blind_Ward_clusters",
                uncertainty="leave-one-feature-out_ARI",
            ),
            "similarity_heatmap": caption(
                clustering,
                basis="clustering_cohort_policy",
                units="RBF_similarity_percent_0-100",
                method="median-nonzero-distance_RBF",
                uncertainty="not_applicable",
            ),
            "ward_dendrogram": caption(
                clustering,
                basis="clustering_cohort_policy",
                units="robust-scaled_Euclidean_linkage_distance",
                method="Ward_hierarchical_clustering",
                uncertainty="not_applicable",
            ),
            "target_histogram": caption(
                supervised,
                basis="supervised_total_score_cohort_policy",
                units="advisor_points_0-100_and_counts",
                method="fixed_5-point_histogram_bins",
                uncertainty="not_applicable",
            ),
            "composite_scatter": caption(
                supervised,
                basis="supervised_total_score_cohort_policy",
                units="advisor_points_and_raw_standardized_index",
                method="cross-fitted_equal_available_dimensions",
                uncertainty="descriptive_correlations_no_interval",
            ),
            "oof_scatter": caption(
                supervised,
                basis="supervised_total_score_cohort_policy",
                units="advisor_score_points_0-100",
                method="nested_grouped_out-of-fold_predictions",
                uncertainty="marginal_intervals_not_shown_in_panel",
            ),
            "mae_comparison": caption(
                supervised,
                basis="supervised_total_score_cohort_policy",
                units="MAE_in_advisor_score_points",
                method="nested_grouped_out-of-fold_evaluation",
                uncertainty="95pct_group-bootstrap_percentile_5000_seed_20260813",
            ),
        }
    )


def _figure_manifest_payload(
    result: ModellingNotebookResult, figure_paths: Mapping[str, Path]
) -> dict[str, object]:
    captions = _figure_captions(result)
    return {
        "schema_version": 1,
        "dataset_id": result.loaded.dataset_id,
        "protocol_id": result.loaded.protocol.protocol_id,
        "figures": {
            name: {
                "filename": Path(path).name,
                "caption": captions[name],
                "sha256": sha256_file(Path(path)),
            }
            for name, path in figure_paths.items()
        },
    }


def _plot_header(figure: object, title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=14, color=_INK)
    figure.text(
        0.06,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.15,
        color=_MUTED,
    )


def _save_figure(
    figure: object, path: Path, title: str, caption: str, pyplot: object
) -> None:
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.80))
    figure.savefig(
        path,
        dpi=150,
        facecolor="white",
        metadata={"Title": title, "Description": caption},
    )
    pyplot.close(figure)


def _quiet_axes(axis: object, *, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis is not None:
        axis.grid(axis=grid_axis, color=_GRID, linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)


def _plot_pca(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    coordinates = np.asarray(result.clustering.pca_coordinates, dtype=float)
    assignments = np.asarray(result.clustering.cluster_assignments, dtype=int)
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    for position, cluster_id in enumerate(sorted(set(assignments.tolist()))):
        mask = assignments == cluster_id
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=54,
            marker=_CLUSTER_MARKERS[position % len(_CLUSTER_MARKERS)],
            facecolor=_CLUSTER_COLOURS[position % len(_CLUSTER_COLOURS)],
            edgecolor=_INK,
            linewidth=0.65,
            alpha=0.88,
            label=f"Cluster {cluster_id} (n={int(np.sum(mask))})",
        )
    axis.axhline(0.0, color=_GRID, linewidth=0.8)
    axis.axvline(0.0, color=_GRID, linewidth=0.8)
    axis.set_xlabel(
        f"PC1 ({100.0 * result.clustering.pca_explained_variance_ratio[0]:.1f}% variance)"
    )
    axis.set_ylabel(
        f"PC2 ({100.0 * result.clustering.pca_explained_variance_ratio[1]:.1f}% variance)"
    )
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    _quiet_axes(axis)
    title = "PCA projection of acoustic features"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_similarity(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    matrix = np.asarray(result.clustering.similarity_matrix, dtype=float)
    order = np.argsort(np.asarray(result.clustering.cluster_assignments, dtype=int), kind="stable")
    ordered = matrix[np.ix_(order, order)]
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(ordered, vmin=0.0, vmax=100.0, cmap="Blues", aspect="equal")
    axis.set_xlabel("Observations ordered by cluster")
    axis.set_ylabel("Observations ordered by cluster")
    axis.set_xticks([])
    axis.set_yticks([])
    colourbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("RBF similarity (%)")
    title = "Pairwise acoustic similarity"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_dendrogram(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    values = np.asarray(result.clustering.scaled_values, dtype=float)
    tree = linkage(values, method="ward", metric="euclidean", optimal_ordering=True)
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    dendrogram(
        tree,
        ax=axis,
        no_labels=True,
        color_threshold=0.0,
        above_threshold_color=_BLUE,
    )
    axis.set_xlabel("Observations (labels suppressed)")
    axis.set_ylabel("Ward linkage distance (robust-scaled units)")
    axis.set_ylim(bottom=0.0)
    _quiet_axes(axis, grid_axis="y")
    title = "Ward hierarchical clustering dendrogram"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_target_histogram(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    values = np.asarray(
        result.cohorts["supervised_total_score"].target_series(), dtype=float
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    bins = np.arange(0.0, 105.0, 5.0)
    axis.hist(values, bins=bins, color=_BLUE, edgecolor=_INK, linewidth=0.7)
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Advisor total score (points; source scale 0–100)")
    axis.set_ylabel("Observation count")
    _quiet_axes(axis, grid_axis="y")
    title = "Distribution of advisor total scores"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_composite(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    observed = np.asarray(result.composite.observed_scores, dtype=float)
    index_values = np.asarray(result.composite.index_values, dtype=float)
    figure, axis = plt.subplots(figsize=(7.4, 5.1))
    axis.scatter(
        observed,
        index_values,
        s=54,
        facecolor=_BLUE,
        edgecolor=_INK,
        linewidth=0.65,
        alpha=0.85,
    )
    axis.axhline(0.0, color=_GRID, linewidth=0.9)
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Advisor total score (points; source scale 0–100)")
    axis.set_ylabel("Fixed acoustic composite (raw standardized index)")
    _quiet_axes(axis, grid_axis="both")
    title = "Fixed acoustic composite and advisor total score"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_oof(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharex=True, sharey=True)
    for axis, evaluation in zip(axes, result.baselines.evaluations, strict=True):
        colour = _MODEL_COLOURS[evaluation.model_name]
        axis.scatter(
            evaluation.observed,
            evaluation.predictions,
            s=42,
            marker=_MODEL_MARKERS[evaluation.model_name],
            facecolor=colour,
            edgecolor=_INK,
            linewidth=0.6,
            alpha=0.82,
        )
        axis.plot([0.0, 100.0], [0.0, 100.0], linestyle="--", color=_MUTED, linewidth=1.0)
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 100.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(_MODEL_LABELS[evaluation.model_name], fontsize=10)
        axis.set_xlabel("Observed (points)")
        _quiet_axes(axis, grid_axis="both")
    axes[0].set_ylabel("OOF prediction (points)")
    title = "Observed and out-of-fold predicted scores"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _plot_mae(
    result: ModellingNotebookResult, path: Path, caption: str, plt: object
) -> None:
    intervals = {item.reference_model: item for item in result.mae_intervals}
    evaluations = tuple(result.baselines.evaluations)
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    upper_values: list[float] = []
    for position, evaluation in enumerate(evaluations):
        interval = intervals[evaluation.model_name]
        colour = _MODEL_COLOURS[evaluation.model_name]
        if interval.status == "ok" and interval.lower is not None and interval.upper is not None:
            lower_error = evaluation.mae - interval.lower
            upper_error = interval.upper - evaluation.mae
            axis.errorbar(
                evaluation.mae,
                position,
                xerr=np.asarray([[lower_error], [upper_error]]),
                fmt=_MODEL_MARKERS[evaluation.model_name],
                color=colour,
                markeredgecolor=_INK,
                markeredgewidth=0.6,
                markersize=8,
                capsize=4,
                linewidth=1.8,
            )
            upper_values.append(interval.upper)
        else:
            axis.scatter(
                [evaluation.mae],
                [position],
                marker=_MODEL_MARKERS[evaluation.model_name],
                color=colour,
                edgecolor=_INK,
                s=60,
            )
            upper_values.append(evaluation.mae)
    axis.set_yticks(range(len(evaluations)), [_MODEL_LABELS[item.model_name] for item in evaluations])
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(upper_values) * 1.18 if upper_values else 1.0)
    axis.set_xlabel("Mean absolute error (score points)")
    _quiet_axes(axis, grid_axis="x")
    title = "Out-of-fold model MAE with uncertainty"
    _plot_header(
        figure,
        title,
        caption,
    )
    _save_figure(figure, path, title, caption, plt)


def _write_json_lines(path: Path, rows: Sequence[object] | object) -> None:
    lines = [
        canonical_json_bytes(row).decode("utf-8")
        for row in rows  # type: ignore[union-attr]
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]] | object
) -> None:
    atomic_write_text(path, _csv_text(fieldnames, rows))


def _csv_text(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]] | object
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:  # type: ignore[union-attr]
        writer.writerow(dict(row))
    return stream.getvalue()


def _executed_aggregate_payloads(notebook: object) -> tuple[Mapping[str, object], ...]:
    payloads: list[Mapping[str, object]] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            payload = data.get("application/json") if isinstance(data, Mapping) else None
            if isinstance(payload, Mapping) and isinstance(payload.get("artifact_type"), str):
                payloads.append(dict(payload))
    return tuple(payloads)


def _one_payload(
    by_type: Mapping[str, list[Mapping[str, object]]], artifact_type: str
) -> Mapping[str, object]:
    values = by_type.get(artifact_type, [])
    if len(values) != 1:
        raise ValueError(f"executed notebook requires one {artifact_type} output")
    return values[0]


def _mae_interval_overlap(
    intervals: Sequence[BootstrapInterval],
) -> tuple[bool, float | None, float | None]:
    by_model = {item.reference_model: item for item in intervals}
    if set(by_model) != {"training_mean", "ridge", "elastic_net"} or any(
        item.status != "ok" or item.lower is None or item.upper is None
        for item in by_model.values()
    ):
        return False, None, None
    lower = max(float(item.lower) for item in by_model.values())
    upper = min(float(item.upper) for item in by_model.values())
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return False, None, None
    return True, lower, upper


def _model_payload_ci_overlap(
    models: Sequence[Mapping[str, object]],
) -> tuple[bool, float | None, float | None]:
    if any(item.get("mae_ci_status") != "ok" for item in models):
        return False, None, None
    intervals: list[tuple[float, float]] = []
    for item in models:
        mae = _required_finite_number(item.get("mae"), "model MAE")
        lower = _required_finite_number(item.get("mae_ci_lower"), "MAE CI lower")
        upper = _required_finite_number(item.get("mae_ci_upper"), "MAE CI upper")
        if lower > mae or mae > upper:
            raise ValueError("model MAE must lie within its confidence interval")
        intervals.append((lower, upper))
    lower = max(item[0] for item in intervals)
    upper = min(item[1] for item in intervals)
    if lower > upper:
        return False, None, None
    return True, lower, upper


def _paired_difference_statements(
    payload: Mapping[str, object], models: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    """Validate paired MAE-difference evidence and render claim-safe prose."""
    model_mae = {
        str(item["model_name"]): _required_finite_number(item.get("mae"), "model MAE")
        for item in models
    }
    specifications = (
        (
            "ridge_minus_training_mean",
            "Ridge",
            "the training-fold mean",
            model_mae["ridge"] - model_mae["training_mean"],
        ),
        (
            "elastic_net_minus_training_mean",
            "ElasticNet",
            "the training-fold mean",
            model_mae["elastic_net"] - model_mae["training_mean"],
        ),
        (
            "ridge_minus_elastic_net",
            "Ridge",
            "ElasticNet",
            model_mae["ridge"] - model_mae["elastic_net"],
        ),
    )
    statements: list[str] = []
    for prefix, comparison_label, reference_label, expected_estimate in specifications:
        status = payload.get(f"{prefix}_status")
        estimate = _required_finite_number(
            payload.get(f"{prefix}_estimate"), f"{prefix} estimate"
        )
        if not math.isclose(estimate, expected_estimate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("paired MAE point difference does not match model MAEs")
        required = payload.get(f"{prefix}_required_replicates")
        valid = payload.get(f"{prefix}_valid_replicates")
        if type(required) is not int or required != 5_000 or type(valid) is not int:
            raise ValueError("paired MAE replicate counts are invalid")
        if status == "not_estimable":
            if (
                valid > required
                or payload.get(f"{prefix}_ci_lower") is not None
                or payload.get(f"{prefix}_ci_upper") is not None
                or payload.get(f"{prefix}_includes_zero") is not None
            ):
                raise ValueError("paired MAE not-estimable interval is inconsistent")
            statements.append(
                f"{comparison_label} versus {reference_label} was not estimable; "
                "no pairwise model-superiority claim is supported."
            )
            continue
        if status != "ok" or valid != required:
            raise ValueError("paired MAE replicate counts are invalid")
        lower = _required_finite_number(
            payload.get(f"{prefix}_ci_lower"), f"{prefix} CI lower"
        )
        upper = _required_finite_number(
            payload.get(f"{prefix}_ci_upper"), f"{prefix} CI upper"
        )
        if lower > estimate or estimate > upper or lower >= upper:
            raise ValueError("paired MAE interval does not contain its point difference")
        includes_zero = lower <= 0.0 <= upper
        if payload.get(f"{prefix}_includes_zero") is not includes_zero:
            raise ValueError("paired MAE includes-zero flag does not match its interval")
        interval_text = (
            f"difference {_format_number(estimate)} points, 95% CI "
            f"[{_format_number(lower)}, {_format_number(upper)}]."
        )
        if includes_zero:
            statements.append(
                f"{comparison_label} versus {reference_label} includes zero: "
                f"{interval_text[:-1]}; no pairwise superiority claim is supported."
            )
        elif upper < 0.0:
            statements.append(
                f"{comparison_label} had lower paired MAE than {reference_label}: "
                f"{interval_text}"
            )
        else:
            statements.append(
                f"{reference_label.capitalize()} had lower paired MAE than "
                f"{comparison_label}: {interval_text}"
            )
    return tuple(statements)


def _required_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_finite_number(value: object, label: str) -> float | None:
    return None if value is None else _required_finite_number(value, label)


def _same_optional_number(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    try:
        actual = _required_finite_number(value, "aggregate value")
    except ValueError:
        return False
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _format_number(value: object) -> str:
    if value is None:
        return "not estimable"
    return f"{float(value):.3f}"
