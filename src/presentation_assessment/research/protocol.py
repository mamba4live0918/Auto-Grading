"""Strict experiment-protocol schema, canonical identity, and dataset binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .canonical import canonical_json_bytes, sha256_bytes


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class TargetDefinition(_StrictModel):
    name: Literal["total_score"]
    label_source: Literal["advisor_total_score"]
    scale_min: float
    scale_max: float
    semantics: str = Field(min_length=1)
    observed_score_status: Literal["observed"]
    exploratory: bool
    rubric_level_enabled: bool

    @model_validator(mode="after")
    def require_ordered_scale(self) -> "TargetDefinition":
        if self.scale_min >= self.scale_max:
            raise ValueError("target scale_min must be lower than scale_max")
        return self


class LimitationPolicy(_StrictModel):
    speaker_identity_status: Literal["not_reliably_available"]
    unknown_speaker_policy: Literal["retain_and_report_limitation"]
    prohibited_inference: tuple[str, ...] = Field(min_length=1)


class SplitPolicy(_StrictModel):
    strategy: Literal["nested_grouped_cross_validation"]
    group_fields: tuple[Literal["speaker_id", "talk_id"], ...]
    deduplicate_by: tuple[Literal["source_sha256", "video_id"], ...]
    outer_method: Literal["group_kfold"]
    outer_folds: int = Field(ge=2)
    inner_method: Literal["group_kfold"]
    inner_folds: int = Field(ge=2)
    insufficient_group_fallback: Literal["leave_one_group_out"]
    no_independent_holdout_claim: bool
    preprocessing_fit_scope: Literal["training_fold_only"]
    score_unavailable_policy: Literal["unsupervised_only"]

    @model_validator(mode="after")
    def require_unique_grouping_and_deduplication_fields(self) -> "SplitPolicy":
        if len(self.group_fields) != len(set(self.group_fields)):
            raise ValueError("split group_fields must be unique")
        if len(self.deduplicate_by) != len(set(self.deduplicate_by)):
            raise ValueError("split deduplicate_by fields must be unique")
        return self


class ClusteringProtocol(_StrictModel):
    feature_registry: Literal["audio_feature_registry.v1.yaml"]
    primary_features: tuple[str, ...] = Field(min_length=1)
    conditional_feature: Literal["asr_units_per_speech_minute"]
    conditional_feature_rule: Literal["all_successful_samples_share_whisper_language"]
    missing_value_imputation: Literal["batch_median"]
    missing_indicators: Literal["audit_only"]
    scaler: Literal["median_iqr_robust"]
    outlier_policy: Literal["flag_without_removal_or_winsorization"]
    pca_components: Literal[2]
    distance: Literal["euclidean"]
    similarity_scale: Literal["median_nonzero_distance_rbf"]
    linkage: Literal["ward"]
    candidate_cluster_counts: tuple[Literal[2, 3, 4, 5], ...]
    cluster_selection_metric: Literal["silhouette"]
    smaller_cluster_tolerance: float = Field(ge=0.0)
    sensitivity_analysis: Literal["leave_one_feature_out_adjusted_rand_index"]

    @model_validator(mode="after")
    def require_unique_clustering_inputs(self) -> "ClusteringProtocol":
        if len(self.primary_features) != len(set(self.primary_features)):
            raise ValueError("primary clustering features must be unique")
        if len(self.candidate_cluster_counts) != len(set(self.candidate_cluster_counts)):
            raise ValueError("candidate cluster counts must be unique")
        return self


class CompositeComponent(_StrictModel):
    feature: str = Field(min_length=1)
    direction: Literal[-1, 1]
    weight: float = Field(gt=0.0, le=1.0)


class CompositeProtocol(_StrictModel):
    name: Literal["exploratory_delivery_composite"]
    interpretation: Literal["fixed_expert_exploratory"]
    transform: Literal["training_fold_robust_z"]
    aggregation: Literal["signed_weighted_mean"]
    components: tuple[CompositeComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_fixed_weights(self) -> "CompositeProtocol":
        names = [item.feature for item in self.components]
        if len(names) != len(set(names)):
            raise ValueError("composite component features must be unique")
        if abs(sum(item.weight for item in self.components) - 1.0) > 1e-9:
            raise ValueError("composite component weights must sum to one")
        return self


class EvaluationProtocol(_StrictModel):
    baselines: tuple[Literal["training_mean", "ridge", "elastic_net"], ...]
    primary_metric: Literal["mae"]
    secondary_metrics: tuple[Literal["rmse", "spearman", "pearson"], ...]
    tuning_scope: Literal["grouped_training_folds_only"]
    report_sample_counts_and_exclusions: bool


class UncertaintyProtocol(_StrictModel):
    resampling_unit: Literal["speaker_or_talk_group"]
    bootstrap_method: Literal["paired_group_bootstrap"]
    bootstrap_replicates: int = Field(ge=1)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    interval_method: Literal["percentile"]
    degenerate_interval_policy: Literal["not_estimable_with_reason"]
    cluster_permutation_unit: Literal["group_level_labels"]
    permutation_replicates: int = Field(ge=1)
    multiple_comparison_correction: Literal["holm"]


class DisabledQwenProtocol(_StrictModel):
    enabled: Literal[False]
    status: Literal["not_run"]
    reason: str = Field(min_length=1)
    retry_count_if_enabled_by_future_protocol: Literal[1]


class DisabledQloraProtocol(_StrictModel):
    enabled: Literal[False]
    status: Literal["not_run"]
    formal_effectiveness_claim_allowed: Literal[False]
    smoke_test_interpretation: Literal["method_feasibility_only"]
    reasons: tuple[str, ...] = Field(min_length=1)


class ProtocolTemplate(_StrictModel):
    """Pre-freeze decisions whose canonical digest is the protocol identity."""

    schema_version: Literal[1]
    protocol_name: Literal["thesis_frozen_dataset_experiment_v1"]
    random_seed: int
    research_question: str = Field(min_length=1)
    target: TargetDefinition
    limitations: LimitationPolicy
    split: SplitPolicy
    clustering: ClusteringProtocol
    composite: CompositeProtocol
    evaluation: EvaluationProtocol
    uncertainty: UncertaintyProtocol
    qwen: DisabledQwenProtocol
    qlora: DisabledQloraProtocol


class BoundProtocol(_StrictModel):
    """A post-freeze binding that cannot affect its embedded protocol identity."""

    protocol_id: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    template: ProtocolTemplate

    @model_validator(mode="after")
    def require_protocol_id_for_embedded_template(self) -> "BoundProtocol":
        if self.protocol_id != protocol_id(self.template):
            raise ValueError("protocol_id must match the embedded template")
        return self


def load_protocol_template(path: Path) -> ProtocolTemplate:
    """Load a pre-freeze YAML protocol, rejecting unknown fields and coercion."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Unable to read experiment protocol template: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("experiment protocol template root must be a mapping")
    return ProtocolTemplate.model_validate_json(json.dumps(raw, ensure_ascii=False))


def protocol_id(template: ProtocolTemplate) -> str:
    """Return the canonical pre-freeze protocol digest, excluding later bindings."""
    return sha256_bytes(canonical_json_bytes(template.model_dump(mode="json")))


def bind_protocol(
    template: ProtocolTemplate, dataset_id: str, split_sha256: str
) -> BoundProtocol:
    """Bind immutable dataset and split identities without changing protocol identity."""
    return BoundProtocol(
        protocol_id=protocol_id(template),
        dataset_id=dataset_id,
        split_sha256=split_sha256,
        template=template,
    )
