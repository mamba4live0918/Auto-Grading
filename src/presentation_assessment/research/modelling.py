"""Leakage-safe exploratory acoustic baselines and uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import ElasticNet, Ridge

from .analysis_data import AnalysisCohort, RobustFoldPreprocessor
from .protocol import BoundProtocol, ProtocolTemplate


ModelName = Literal["training_mean", "ridge", "elastic_net"]
MetricName = Literal["mae", "rmse", "spearman", "pearson"]
BootstrapStatus = Literal["ok", "not_estimable"]

_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_ELASTIC_NET_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
_ELASTIC_NET_L1_RATIOS = (0.1, 0.5, 0.9)
_BOOTSTRAP_REPLICATES = 5_000
_BOOTSTRAP_SEED = 20260813
_BOOTSTRAP_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class InnerFoldAudit:
    """Training-only preprocessing evidence for one inner validation fold."""

    outer_fold: int
    inner_fold: int
    training_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    excluded_all_null_features: tuple[str, ...]
    location: Mapping[str, float]
    scale: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", MappingProxyType(dict(self.location)))
        object.__setattr__(self, "scale", MappingProxyType(dict(self.scale)))


@dataclass(frozen=True, slots=True)
class OuterFoldAudit:
    """One outer OOF fit, its selected parameters, and inner-fit evidence."""

    model_name: ModelName
    outer_fold: int
    training_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    excluded_all_null_features: tuple[str, ...]
    location: Mapping[str, float]
    scale: Mapping[str, float]
    selected_hyperparameters: Mapping[str, float]
    inner_audits: tuple[InnerFoldAudit, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", MappingProxyType(dict(self.location)))
        object.__setattr__(self, "scale", MappingProxyType(dict(self.scale)))
        object.__setattr__(
            self,
            "selected_hyperparameters",
            MappingProxyType(dict(self.selected_hyperparameters)),
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """One model's exactly-once OOF predictions and aggregate metrics."""

    model_name: ModelName
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    observed: tuple[float, ...]
    predictions: tuple[float, ...]
    mae: float
    rmse: float
    spearman: float | None
    pearson: float | None
    spearman_reason: str | None
    pearson_reason: str | None
    fold_audits: tuple[OuterFoldAudit, ...] = ()
    hyperparameter_grid: Mapping[str, tuple[float, ...]] = field(
        default_factory=dict
    )
    primary_metric: Literal["mae"] = "mae"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hyperparameter_grid",
            MappingProxyType(
                {
                    name: tuple(float(value) for value in values)
                    for name, values in self.hyperparameter_grid.items()
                }
            ),
        )

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)

    @property
    def oof_prediction_count(self) -> int:
        return len(self.predictions)

    @property
    def prediction_by_sample(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.sample_ids, self.predictions, strict=True)))


@dataclass(frozen=True, slots=True)
class ExperimentResults:
    """Exploratory nested-CV results with explicit grouping limitations."""

    dataset_id: str
    sample_count: int
    excluded_sample_count: int
    outer_fold_count: int
    inner_fold_counts: tuple[int, ...]
    evaluations: tuple[ModelEvaluation, ...]
    grouping_unit: Literal["speaker_or_talk_group", "source_or_sample_group"]
    speaker_disjoint: bool
    evaluation_claim: str
    limitation: str
    no_independent_holdout_claim: bool
    seed: int

    @property
    def evaluation_by_model(self) -> Mapping[str, ModelEvaluation]:
        return MappingProxyType(
            {evaluation.model_name: evaluation for evaluation in self.evaluations}
        )


@dataclass(frozen=True, slots=True)
class CompositeFoldAudit:
    """Fold-local robust transformation used for held-out index values."""

    outer_fold: int
    training_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    excluded_all_null_features: tuple[str, ...]
    location: Mapping[str, float]
    scale: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", MappingProxyType(dict(self.location)))
        object.__setattr__(self, "scale", MappingProxyType(dict(self.scale)))


@dataclass(frozen=True, slots=True)
class AcousticCompositeResult:
    """Cross-fitted fixed acoustic index and exploratory score association."""

    dataset_id: str
    name: str
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    observed_scores: tuple[float, ...]
    index_values: tuple[float, ...]
    available_dimension_counts: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_directions: Mapping[str, int]
    fold_audits: tuple[CompositeFoldAudit, ...]
    spearman: float | None
    pearson: float | None
    spearman_reason: str | None
    pearson_reason: str | None
    aggregation: Literal["equal_available_dimensions"]
    scale: Literal["raw_standardized_index"]
    learned_weights: Literal[False]
    calibrated_to_target: Literal[False]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_directions",
            MappingProxyType(dict(self.feature_directions)),
        )

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Frozen percentile interval for an OOF metric or paired model difference."""

    status: BootstrapStatus
    reason: str | None
    method: Literal["paired_group_bootstrap"]
    difference_direction: Literal["comparison_minus_reference"] | None
    metric: MetricName
    estimate: float | None
    lower: float | None
    upper: float | None
    reference_model: str
    comparison_model: str | None
    group_count: int
    sample_count: int
    analysis_unit: Literal["group"]
    paired: bool
    replicates: int
    valid_replicates: int
    seed: int
    confidence_level: float
    interval_method: Literal["percentile"]
    includes_zero: bool | None


@dataclass(frozen=True, slots=True)
class _PreparedInnerFold:
    audit: InnerFoldAudit
    training_values: np.ndarray
    validation_values: np.ndarray
    training_targets: np.ndarray
    validation_targets: np.ndarray


def compute_acoustic_composite(
    cohort: AnalysisCohort,
    protocol: ProtocolTemplate | BoundProtocol,
) -> AcousticCompositeResult:
    """Compute a label-blind, fold-standardized fixed acoustic index.

    Protocol directions are applied to held-out robust-z values. Each sample is
    averaged only over dimensions observed for that sample and retained by its
    training fold; median imputation therefore cannot dilute an index merely
    because a dimension was unavailable. Scores are read only after every index
    value has been fixed, for the separate exploratory association.
    """
    if not isinstance(cohort, AnalysisCohort):
        raise TypeError("cohort must be an AnalysisCohort")
    if cohort.purpose != "supervised_total_score":
        raise ValueError("the acoustic composite requires a supervised cohort")
    template = _protocol_template(protocol, cohort.dataset_id)
    components = template.composite.components
    _validate_fixed_composite(template)

    rows = tuple(sorted(cohort.rows, key=lambda row: row.sample_id))
    if len(rows) < 2:
        raise ValueError("the acoustic composite requires at least two samples")
    sample_ids = tuple(_identifier(row.sample_id, "sample_id") for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("composite sample IDs must be unique")
    group_ids = tuple(_identifier(row.group_id, "group_id") for row in rows)
    feature_names = tuple(component.feature for component in components)
    missing = tuple(name for name in feature_names if name not in cohort.feature_names)
    if missing:
        raise ValueError(f"composite cohort is missing protocol features: {missing!r}")
    directions = {component.feature: component.direction for component in components}
    feature_frame = _feature_frame(rows, feature_names, sample_ids)
    outer_folds = _validated_fold_values(
        rows,
        group_ids,
        field="outer_fold",
        requested_count=template.split.outer_folds,
        context="composite outer",
    )

    values_by_sample: dict[str, float] = {}
    counts_by_sample: dict[str, int] = {}
    audits: list[CompositeFoldAudit] = []
    for outer_fold in outer_folds:
        validation_indexes = tuple(
            index for index, row in enumerate(rows) if row.outer_fold == outer_fold
        )
        training_indexes = tuple(
            index for index, row in enumerate(rows) if row.outer_fold != outer_fold
        )
        _require_group_disjoint(
            group_ids,
            training_indexes,
            validation_indexes,
            context="composite outer",
        )
        preprocessor = RobustFoldPreprocessor(feature_names=feature_names).fit(
            feature_frame.iloc[list(training_indexes)]
        )
        transformed = preprocessor.transform(
            feature_frame.iloc[list(validation_indexes)]
        )
        for local_index, row_index in enumerate(validation_indexes):
            sample_id = sample_ids[row_index]
            available = tuple(
                name
                for name in preprocessor.feature_names_
                if not pd.isna(feature_frame.iloc[row_index][name])
            )
            if not available:
                raise ValueError(
                    f"sample {sample_id!r} has no available composite dimensions"
                )
            directed = tuple(
                directions[name] * float(transformed.iloc[local_index][name])
                for name in available
            )
            values_by_sample[sample_id] = float(np.mean(directed))
            counts_by_sample[sample_id] = len(available)
        audits.append(
            CompositeFoldAudit(
                outer_fold=outer_fold,
                training_sample_ids=tuple(
                    sample_ids[index] for index in training_indexes
                ),
                validation_sample_ids=tuple(
                    sample_ids[index] for index in validation_indexes
                ),
                feature_names=preprocessor.feature_names_,
                excluded_all_null_features=(
                    preprocessor.excluded_all_null_features_
                ),
                location=preprocessor.location_,
                scale=preprocessor.scale_,
            )
        )
    if set(values_by_sample) != set(sample_ids):
        raise RuntimeError("each eligible sample must receive one composite index")

    index_values = np.asarray(
        [values_by_sample[sample_id] for sample_id in sample_ids], dtype=float
    )
    observed = np.asarray([_target(row.total_score, template) for row in rows])
    spearman, spearman_reason = _correlation(observed, index_values, rank=True)
    pearson, pearson_reason = _correlation(observed, index_values, rank=False)
    return AcousticCompositeResult(
        dataset_id=cohort.dataset_id,
        name=template.composite.name,
        sample_ids=sample_ids,
        group_ids=group_ids,
        observed_scores=tuple(float(value) for value in observed),
        index_values=tuple(float(value) for value in index_values),
        available_dimension_counts=tuple(
            counts_by_sample[sample_id] for sample_id in sample_ids
        ),
        feature_names=feature_names,
        feature_directions=directions,
        fold_audits=tuple(audits),
        spearman=spearman,
        pearson=pearson,
        spearman_reason=spearman_reason,
        pearson_reason=pearson_reason,
        aggregation="equal_available_dimensions",
        scale="raw_standardized_index",
        learned_weights=False,
        calibrated_to_target=False,
    )


def run_nested_baselines(
    cohort: AnalysisCohort,
    protocol: ProtocolTemplate | BoundProtocol,
) -> ExperimentResults:
    """Run grouped nested mean, Ridge, and ElasticNet OOF baselines."""
    if not isinstance(cohort, AnalysisCohort):
        raise TypeError("cohort must be an AnalysisCohort")
    if cohort.purpose != "supervised_total_score":
        raise ValueError("baselines require a supervised_total_score cohort")
    template = _protocol_template(protocol, cohort.dataset_id)
    _validate_frozen_modelling_protocol(template)

    rows = tuple(sorted(cohort.rows, key=lambda row: row.sample_id))
    if len(rows) < 3:
        raise ValueError("nested baselines require at least three samples")
    sample_ids = tuple(_identifier(row.sample_id, "sample_id") for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("supervised sample IDs must be unique")
    group_ids = tuple(_identifier(row.group_id, "group_id") for row in rows)
    targets = np.asarray([_target(row.total_score, template) for row in rows])
    feature_names = _numeric_model_feature_names(rows, cohort.feature_names)
    if not feature_names:
        raise ValueError("at least one acoustic model feature is required")
    feature_frame = _feature_frame(rows, feature_names, sample_ids)

    outer_folds = _validated_fold_values(
        rows,
        group_ids,
        field="outer_fold",
        requested_count=template.split.outer_folds,
        context="outer",
    )
    all_predictions: dict[ModelName, dict[str, float]] = {
        "training_mean": {},
        "ridge": {},
        "elastic_net": {},
    }
    all_audits: dict[ModelName, list[OuterFoldAudit]] = {
        "training_mean": [],
        "ridge": [],
        "elastic_net": [],
    }
    inner_fold_counts: list[int] = []

    for outer_fold in outer_folds:
        validation_indexes = tuple(
            index for index, row in enumerate(rows) if row.outer_fold == outer_fold
        )
        training_indexes = tuple(
            index for index, row in enumerate(rows) if row.outer_fold != outer_fold
        )
        _require_group_disjoint(
            group_ids, training_indexes, validation_indexes, context="outer"
        )
        training_sample_ids = tuple(sample_ids[index] for index in training_indexes)
        validation_sample_ids = tuple(sample_ids[index] for index in validation_indexes)
        training_group_ids = _ordered_unique(group_ids[index] for index in training_indexes)
        validation_group_ids = _ordered_unique(
            group_ids[index] for index in validation_indexes
        )

        mean_prediction = float(np.mean(targets[list(training_indexes)]))
        _record_predictions(
            all_predictions["training_mean"],
            validation_sample_ids,
            np.full(len(validation_indexes), mean_prediction),
        )
        all_audits["training_mean"].append(
            OuterFoldAudit(
                model_name="training_mean",
                outer_fold=outer_fold,
                training_sample_ids=training_sample_ids,
                validation_sample_ids=validation_sample_ids,
                training_group_ids=training_group_ids,
                validation_group_ids=validation_group_ids,
                feature_names=(),
                excluded_all_null_features=(),
                location={},
                scale={},
                selected_hyperparameters={},
                inner_audits=(),
            )
        )

        prepared_inner = _prepare_inner_folds(
            rows=rows,
            group_ids=group_ids,
            targets=targets,
            feature_frame=feature_frame,
            feature_names=feature_names,
            outer_fold=outer_fold,
            outer_training_indexes=training_indexes,
            requested_count=template.split.inner_folds,
        )
        inner_fold_counts.append(len(prepared_inner))

        for model_name in ("ridge", "elastic_net"):
            parameters = _select_hyperparameters(model_name, prepared_inner)
            preprocessor = RobustFoldPreprocessor(feature_names=feature_names).fit(
                feature_frame.iloc[list(training_indexes)]
            )
            transformed_training = preprocessor.transform(
                feature_frame.iloc[list(training_indexes)]
            )
            transformed_validation = preprocessor.transform(
                feature_frame.iloc[list(validation_indexes)]
            )
            predictions = _fit_predict(
                model_name,
                parameters,
                transformed_training.to_numpy(dtype=float),
                targets[list(training_indexes)],
                transformed_validation.to_numpy(dtype=float),
            )
            _record_predictions(
                all_predictions[model_name], validation_sample_ids, predictions
            )
            all_audits[model_name].append(
                OuterFoldAudit(
                    model_name=model_name,
                    outer_fold=outer_fold,
                    training_sample_ids=training_sample_ids,
                    validation_sample_ids=validation_sample_ids,
                    training_group_ids=training_group_ids,
                    validation_group_ids=validation_group_ids,
                    feature_names=preprocessor.feature_names_,
                    excluded_all_null_features=(
                        preprocessor.excluded_all_null_features_
                    ),
                    location=preprocessor.location_,
                    scale=preprocessor.scale_,
                    selected_hyperparameters=parameters,
                    inner_audits=tuple(item.audit for item in prepared_inner),
                )
            )

    evaluations = tuple(
        _model_evaluation(
            model_name=model_name,
            sample_ids=sample_ids,
            group_ids=group_ids,
            targets=targets,
            predictions=all_predictions[model_name],
            audits=all_audits[model_name],
        )
        for model_name in template.evaluation.baselines
    )
    unknown_speakers = any(
        not str(row.speaker_id).strip() or str(row.speaker_id).strip() == "unknown"
        for row in rows
    )
    speaker_disjoint = False
    evaluation_claim = "exploratory_no_speaker_holdout"
    if unknown_speakers:
        grouping_unit: Literal[
            "speaker_or_talk_group", "source_or_sample_group"
        ] = "source_or_sample_group"
    else:
        grouping_unit = "speaker_or_talk_group"
    limitation = (
        "Speaker identity is not reliably available under the frozen protocol; "
        "frozen groups prevent known duplicates from crossing folds, but this is "
        "not speaker-disjoint validation or an independent held-out test."
    )
    return ExperimentResults(
        dataset_id=cohort.dataset_id,
        sample_count=len(rows),
        excluded_sample_count=len(cohort.exclusions),
        outer_fold_count=len(outer_folds),
        inner_fold_counts=tuple(inner_fold_counts),
        evaluations=evaluations,
        grouping_unit=grouping_unit,
        speaker_disjoint=speaker_disjoint,
        evaluation_claim=evaluation_claim,
        limitation=limitation,
        no_independent_holdout_claim=template.split.no_independent_holdout_claim,
        seed=template.random_seed,
    )


def paired_group_bootstrap(
    reference: ModelEvaluation,
    comparison: ModelEvaluation | None = None,
    *,
    metric: MetricName = "mae",
    seed: int = _BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Bootstrap groups, pairing models on every resampled sample.

    When ``comparison`` is supplied, the estimate and interval are for
    ``comparison - reference``. The frozen method intentionally exposes no
    repetitions override and performs no fold-level t-test.
    """
    if type(seed) is not int or seed != _BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed must remain frozen at 20260813")
    if metric not in ("mae", "rmse", "spearman", "pearson"):
        raise ValueError(f"unsupported bootstrap metric: {metric!r}")
    observed, reference_predictions, groups = _evaluation_arrays(reference)
    comparison_predictions: np.ndarray | None = None
    if comparison is not None:
        other_observed, comparison_predictions, other_groups = _evaluation_arrays(
            comparison
        )
        if comparison.sample_ids != reference.sample_ids:
            raise ValueError("paired evaluations must have identical sample alignment")
        if other_groups.tolist() != groups.tolist():
            raise ValueError("paired evaluations must have identical group alignment")
        if not np.array_equal(other_observed, observed):
            raise ValueError("paired evaluations must have identical observed targets")

    reference_estimate = _metric_value(metric, observed, reference_predictions)
    comparison_estimate = (
        None
        if comparison_predictions is None
        else _metric_value(metric, observed, comparison_predictions)
    )
    estimate = (
        reference_estimate
        if comparison_predictions is None
        else _optional_difference(comparison_estimate, reference_estimate)
    )
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    if len(unique_groups) < 2:
        return _not_estimable_bootstrap(
            reason="fewer_than_two_independent_groups",
            metric=metric,
            estimate=estimate,
            reference=reference,
            comparison=comparison,
            group_count=len(unique_groups),
            sample_count=len(observed),
            valid_replicates=0,
        )

    indexes_by_group = {
        group_id: np.flatnonzero(groups == group_id) for group_id in unique_groups
    }
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(_BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indexes = np.concatenate(
            [indexes_by_group[unique_groups[int(index)]] for index in selected]
        )
        reference_value = _metric_value(
            metric, observed[indexes], reference_predictions[indexes]
        )
        if comparison_predictions is None:
            bootstrap_value = reference_value
        else:
            comparison_value = _metric_value(
                metric, observed[indexes], comparison_predictions[indexes]
            )
            bootstrap_value = _optional_difference(
                comparison_value, reference_value
            )
        if bootstrap_value is not None and math.isfinite(bootstrap_value):
            bootstrap_values.append(float(bootstrap_value))

    if len(bootstrap_values) != _BOOTSTRAP_REPLICATES:
        return _not_estimable_bootstrap(
            reason="not_all_bootstrap_replicates_valid",
            metric=metric,
            estimate=estimate,
            reference=reference,
            comparison=comparison,
            group_count=len(unique_groups),
            sample_count=len(observed),
            valid_replicates=len(bootstrap_values),
        )
    distribution = np.asarray(bootstrap_values, dtype=float)
    if np.ptp(distribution) == 0.0:
        return _not_estimable_bootstrap(
            reason="degenerate_bootstrap_distribution",
            metric=metric,
            estimate=estimate,
            reference=reference,
            comparison=comparison,
            group_count=len(unique_groups),
            sample_count=len(observed),
            valid_replicates=len(bootstrap_values),
        )
    tail = (1.0 - _BOOTSTRAP_CONFIDENCE) / 2.0
    lower, upper = np.percentile(distribution, [100.0 * tail, 100.0 * (1.0 - tail)])
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        return _not_estimable_bootstrap(
            reason="non_finite_percentile_interval",
            metric=metric,
            estimate=estimate,
            reference=reference,
            comparison=comparison,
            group_count=len(unique_groups),
            sample_count=len(observed),
            valid_replicates=len(bootstrap_values),
        )
    if float(lower) == float(upper):
        return _not_estimable_bootstrap(
            reason="collapsed_percentile_interval",
            metric=metric,
            estimate=estimate,
            reference=reference,
            comparison=comparison,
            group_count=len(unique_groups),
            sample_count=len(observed),
            valid_replicates=len(bootstrap_values),
        )
    return BootstrapInterval(
        status="ok",
        reason=None,
        method="paired_group_bootstrap",
        difference_direction=(
            "comparison_minus_reference" if comparison is not None else None
        ),
        metric=metric,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        reference_model=reference.model_name,
        comparison_model=None if comparison is None else comparison.model_name,
        group_count=len(unique_groups),
        sample_count=len(observed),
        analysis_unit="group",
        paired=comparison is not None,
        replicates=_BOOTSTRAP_REPLICATES,
        valid_replicates=len(bootstrap_values),
        seed=_BOOTSTRAP_SEED,
        confidence_level=_BOOTSTRAP_CONFIDENCE,
        interval_method="percentile",
        includes_zero=(float(lower) <= 0.0 <= float(upper)) if comparison is not None else None,
    )


def _prepare_inner_folds(
    *,
    rows: Sequence[object],
    group_ids: tuple[str, ...],
    targets: np.ndarray,
    feature_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    outer_fold: int,
    outer_training_indexes: tuple[int, ...],
    requested_count: int,
) -> tuple[_PreparedInnerFold, ...]:
    outer_training_groups = tuple(group_ids[index] for index in outer_training_indexes)
    inner_folds = _validated_fold_values(
        tuple(rows[index] for index in outer_training_indexes),
        outer_training_groups,
        field="inner_fold",
        requested_count=requested_count,
        context=f"inner for outer fold {outer_fold}",
    )
    prepared: list[_PreparedInnerFold] = []
    validation_coverage: set[int] = set()
    for inner_fold in inner_folds:
        validation_indexes = tuple(
            index
            for index in outer_training_indexes
            if getattr(rows[index], "inner_fold") == inner_fold
        )
        training_indexes = tuple(
            index
            for index in outer_training_indexes
            if getattr(rows[index], "inner_fold") != inner_fold
        )
        _require_group_disjoint(
            group_ids,
            training_indexes,
            validation_indexes,
            context=f"inner fold {inner_fold}",
        )
        preprocessor = RobustFoldPreprocessor(feature_names=feature_names).fit(
            feature_frame.iloc[list(training_indexes)]
        )
        transformed_training = preprocessor.transform(
            feature_frame.iloc[list(training_indexes)]
        )
        transformed_validation = preprocessor.transform(
            feature_frame.iloc[list(validation_indexes)]
        )
        prepared.append(
            _PreparedInnerFold(
                audit=InnerFoldAudit(
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    training_sample_ids=tuple(
                        str(feature_frame.index[index]) for index in training_indexes
                    ),
                    validation_sample_ids=tuple(
                        str(feature_frame.index[index]) for index in validation_indexes
                    ),
                    training_group_ids=_ordered_unique(
                        group_ids[index] for index in training_indexes
                    ),
                    validation_group_ids=_ordered_unique(
                        group_ids[index] for index in validation_indexes
                    ),
                    feature_names=preprocessor.feature_names_,
                    excluded_all_null_features=(
                        preprocessor.excluded_all_null_features_
                    ),
                    location=preprocessor.location_,
                    scale=preprocessor.scale_,
                ),
                training_values=transformed_training.to_numpy(dtype=float),
                validation_values=transformed_validation.to_numpy(dtype=float),
                training_targets=targets[list(training_indexes)],
                validation_targets=targets[list(validation_indexes)],
            )
        )
        validation_coverage.update(validation_indexes)
    if validation_coverage != set(outer_training_indexes):
        raise ValueError("inner folds must validate every outer-training sample once")
    return tuple(prepared)


def _select_hyperparameters(
    model_name: Literal["ridge", "elastic_net"],
    prepared_inner: tuple[_PreparedInnerFold, ...],
) -> Mapping[str, float]:
    if model_name == "ridge":
        candidates = tuple({"alpha": alpha} for alpha in _RIDGE_ALPHAS)
    else:
        candidates = tuple(
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha in _ELASTIC_NET_ALPHAS
            for l1_ratio in _ELASTIC_NET_L1_RATIOS
        )
    scored: list[tuple[float, tuple[float, ...], Mapping[str, float]]] = []
    for parameters in candidates:
        observed: list[float] = []
        predictions: list[float] = []
        for prepared in prepared_inner:
            fold_predictions = _fit_predict(
                model_name,
                parameters,
                prepared.training_values,
                prepared.training_targets,
                prepared.validation_values,
            )
            observed.extend(prepared.validation_targets.tolist())
            predictions.extend(float(value) for value in fold_predictions)
        mae = float(np.mean(np.abs(np.asarray(observed) - np.asarray(predictions))))
        tie_break = tuple(float(parameters[name]) for name in sorted(parameters))
        scored.append((mae, tie_break, parameters))
    return dict(min(scored, key=lambda item: (item[0], item[1]))[2])


def _estimator(
    model_name: Literal["ridge", "elastic_net"], parameters: Mapping[str, float]
) -> Ridge | ElasticNet:
    if model_name == "ridge":
        return Ridge(alpha=float(parameters["alpha"]))
    return ElasticNet(
        alpha=float(parameters["alpha"]),
        l1_ratio=float(parameters["l1_ratio"]),
        max_iter=1_000_000,
        tol=1e-4,
        selection="random",
        random_state=20260813,
    )


def _fit_predict(
    model_name: Literal["ridge", "elastic_net"],
    parameters: Mapping[str, float],
    training_values: np.ndarray,
    training_targets: np.ndarray,
    validation_values: np.ndarray,
) -> np.ndarray:
    estimator = _estimator(model_name, parameters)
    estimator.fit(training_values, training_targets)
    return estimator.predict(validation_values)


def _model_evaluation(
    *,
    model_name: ModelName,
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    targets: np.ndarray,
    predictions: Mapping[str, float],
    audits: Sequence[OuterFoldAudit],
) -> ModelEvaluation:
    if set(predictions) != set(sample_ids) or len(predictions) != len(sample_ids):
        raise RuntimeError("each eligible sample must receive exactly one OOF prediction")
    ordered_predictions = np.asarray(
        [predictions[sample_id] for sample_id in sample_ids], dtype=float
    )
    if not np.all(np.isfinite(ordered_predictions)):
        raise ValueError("OOF predictions must be finite")
    errors = ordered_predictions - targets
    spearman, spearman_reason = _correlation(targets, ordered_predictions, rank=True)
    pearson, pearson_reason = _correlation(targets, ordered_predictions, rank=False)
    return ModelEvaluation(
        model_name=model_name,
        sample_ids=sample_ids,
        group_ids=group_ids,
        observed=tuple(float(value) for value in targets),
        predictions=tuple(float(value) for value in ordered_predictions),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        spearman=spearman,
        pearson=pearson,
        spearman_reason=spearman_reason,
        pearson_reason=pearson_reason,
        fold_audits=tuple(audits),
        hyperparameter_grid=_hyperparameter_grid(model_name),
    )


def _hyperparameter_grid(
    model_name: ModelName,
) -> Mapping[str, tuple[float, ...]]:
    if model_name == "ridge":
        return {"alpha": _RIDGE_ALPHAS}
    if model_name == "elastic_net":
        return {
            "alpha": _ELASTIC_NET_ALPHAS,
            "l1_ratio": _ELASTIC_NET_L1_RATIOS,
        }
    return {}


def _correlation(
    observed: np.ndarray, predicted: np.ndarray, *, rank: bool
) -> tuple[float | None, str | None]:
    if len(observed) < 2:
        return None, "fewer_than_two_samples"
    if np.ptp(observed) == 0.0:
        return None, "constant_observed_values"
    if np.ptp(predicted) == 0.0:
        return None, "constant_predictions"
    left = rankdata(observed) if rank else observed
    right = rankdata(predicted) if rank else predicted
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None, "non_finite_correlation"
    return value, None


def _evaluation_arrays(
    evaluation: ModelEvaluation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(evaluation, ModelEvaluation):
        raise TypeError("bootstrap inputs must be ModelEvaluation instances")
    size = len(evaluation.sample_ids)
    if size == 0 or not (
        len(evaluation.group_ids)
        == len(evaluation.observed)
        == len(evaluation.predictions)
        == size
    ):
        raise ValueError("model evaluation arrays must be non-empty and aligned")
    if len(set(evaluation.sample_ids)) != size:
        raise ValueError("model evaluation sample IDs must be unique")
    groups = np.asarray(
        [_identifier(value, "group_id") for value in evaluation.group_ids],
        dtype=object,
    )
    observed = np.asarray(evaluation.observed, dtype=float)
    predictions = np.asarray(evaluation.predictions, dtype=float)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predictions)):
        raise ValueError("bootstrap inputs must be finite")
    return observed, predictions, groups


def _metric_value(
    metric: MetricName, observed: np.ndarray, predicted: np.ndarray
) -> float | None:
    errors = predicted - observed
    if metric == "mae":
        return float(np.mean(np.abs(errors)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    value, _ = _correlation(observed, predicted, rank=metric == "spearman")
    return value


def _optional_difference(
    comparison: float | None, reference: float | None
) -> float | None:
    if comparison is None or reference is None:
        return None
    return float(comparison - reference)


def _not_estimable_bootstrap(
    *,
    reason: str,
    metric: MetricName,
    estimate: float | None,
    reference: ModelEvaluation,
    comparison: ModelEvaluation | None,
    group_count: int,
    sample_count: int,
    valid_replicates: int,
) -> BootstrapInterval:
    return BootstrapInterval(
        status="not_estimable",
        reason=reason,
        method="paired_group_bootstrap",
        difference_direction=(
            "comparison_minus_reference" if comparison is not None else None
        ),
        metric=metric,
        estimate=estimate,
        lower=None,
        upper=None,
        reference_model=reference.model_name,
        comparison_model=None if comparison is None else comparison.model_name,
        group_count=group_count,
        sample_count=sample_count,
        analysis_unit="group",
        paired=comparison is not None,
        replicates=_BOOTSTRAP_REPLICATES,
        valid_replicates=valid_replicates,
        seed=_BOOTSTRAP_SEED,
        confidence_level=_BOOTSTRAP_CONFIDENCE,
        interval_method="percentile",
        includes_zero=None,
    )


def _validated_fold_values(
    rows: Sequence[object],
    group_ids: Sequence[str],
    *,
    field: Literal["outer_fold", "inner_fold"],
    requested_count: int,
    context: str,
) -> tuple[int, ...]:
    group_folds: dict[str, int] = {}
    for row, group_id in zip(rows, group_ids, strict=True):
        value = getattr(row, field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{context} fold assignments must be non-negative integers")
        previous = group_folds.setdefault(group_id, value)
        if previous != value:
            raise ValueError(f"independent group {group_id!r} spans {context} folds")
    group_count = len(group_folds)
    if group_count < 2:
        raise ValueError(f"{context} cross-validation requires at least two groups")
    fold_values = tuple(sorted(set(group_folds.values())))
    expected_count = min(requested_count, group_count)
    if len(fold_values) != expected_count:
        raise ValueError(
            f"{context} fold registry has {len(fold_values)} folds; "
            f"expected {expected_count} for {group_count} groups"
        )
    return fold_values


def _feature_frame(
    rows: Sequence[object],
    feature_names: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> pd.DataFrame:
    values: list[list[object]] = []
    for row in rows:
        missing = tuple(name for name in feature_names if name not in row.features)
        if missing:
            raise ValueError(f"supervised rows are missing model features: {missing!r}")
        values.append([row.features[name] for name in feature_names])
    return pd.DataFrame(
        values,
        index=pd.Index(sample_ids, name="sample_id"),
        columns=feature_names,
    )


def _numeric_model_feature_names(
    rows: Sequence[object], feature_names: Sequence[str]
) -> tuple[str, ...]:
    """Keep registry-shaped numeric columns while excluding categorical audits."""
    selected: list[str] = []
    for name in feature_names:
        observed = [
            row.features[name]
            for row in rows
            if not pd.isna(row.features[name])
        ]
        numeric = [
            isinstance(value, (Real, np.number, np.bool_))
            and not isinstance(value, (str, bytes))
            for value in observed
        ]
        if numeric and any(numeric) and not all(numeric):
            raise ValueError(f"model feature {name!r} mixes numeric and categorical values")
        if numeric and not any(numeric):
            continue
        selected.append(str(name))
    if not selected:
        raise ValueError("at least one numeric acoustic model feature is required")
    return tuple(selected)


def _record_predictions(
    destination: dict[str, float],
    sample_ids: Sequence[str],
    values: Sequence[float] | np.ndarray,
) -> None:
    if len(sample_ids) != len(values):
        raise RuntimeError("prediction count does not match validation sample count")
    for sample_id, value in zip(sample_ids, values, strict=True):
        if sample_id in destination:
            raise RuntimeError(f"sample {sample_id!r} received multiple OOF predictions")
        prediction = float(value)
        if not math.isfinite(prediction):
            raise ValueError("OOF predictions must be finite")
        destination[sample_id] = prediction


def _require_group_disjoint(
    group_ids: Sequence[str],
    training_indexes: Sequence[int],
    validation_indexes: Sequence[int],
    *,
    context: str,
) -> None:
    if not training_indexes or not validation_indexes:
        raise ValueError(f"{context} folds require non-empty training and validation")
    training = {group_ids[index] for index in training_indexes}
    validation = {group_ids[index] for index in validation_indexes}
    overlap = tuple(sorted(training.intersection(validation)))
    if overlap:
        raise ValueError(f"independent groups cross {context} folds: {overlap!r}")


def _protocol_template(
    protocol: ProtocolTemplate | BoundProtocol, dataset_id: str
) -> ProtocolTemplate:
    if isinstance(protocol, BoundProtocol):
        if protocol.dataset_id != dataset_id:
            raise ValueError("bound protocol dataset does not match the cohort")
        return protocol.template
    if isinstance(protocol, ProtocolTemplate):
        return protocol
    raise TypeError("protocol must be a ProtocolTemplate or BoundProtocol")


def _validate_frozen_modelling_protocol(template: ProtocolTemplate) -> None:
    if template.evaluation.baselines != (
        "training_mean",
        "ridge",
        "elastic_net",
    ):
        raise ValueError("frozen baselines must be training mean, Ridge, and ElasticNet")
    if template.evaluation.primary_metric != "mae":
        raise ValueError("MAE must remain the primary metric")
    if template.evaluation.secondary_metrics != (
        "rmse",
        "spearman",
        "pearson",
    ):
        raise ValueError("frozen secondary metrics must be RMSE, Spearman, and Pearson")
    if template.evaluation.tuning_scope != "grouped_training_folds_only":
        raise ValueError("hyperparameter tuning must remain grouped-training-fold only")
    if template.split.preprocessing_fit_scope != "training_fold_only":
        raise ValueError("preprocessing must remain training-fold only")
    if not template.split.no_independent_holdout_claim:
        raise ValueError("the frozen analysis cannot claim an independent holdout")


def _validate_fixed_composite(template: ProtocolTemplate) -> None:
    composite = template.composite
    if composite.transform != "training_fold_robust_z":
        raise ValueError("composite transformation must remain training-fold robust-z")
    if composite.aggregation != "signed_weighted_mean":
        raise ValueError("composite aggregation must remain the frozen signed mean")
    expected = 1.0 / len(composite.components)
    if any(
        not math.isclose(component.weight, expected, rel_tol=0.0, abs_tol=1e-12)
        for component in composite.components
    ):
        raise ValueError("composite dimensions must retain equal fixed weights")


def _target(value: object, template: ProtocolTemplate) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("supervised total scores must be finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("supervised total scores must be finite numbers")
    if not template.target.scale_min <= result <= template.target.scale_max:
        raise ValueError("supervised total scores must remain on the frozen scale")
    return result


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} values must be nonblank strings")
    return value


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
