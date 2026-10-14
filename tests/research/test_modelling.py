"""Synthetic behavioural tests for leakage-safe exploratory baselines."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

from presentation_assessment.research.analysis_data import (
    AnalysisCohort,
    AnalysisRow,
)
from presentation_assessment.research.modelling import (
    ModelEvaluation,
    compute_acoustic_composite,
    paired_group_bootstrap,
    run_nested_baselines,
)
from presentation_assessment.research.protocol import (
    CompositeComponent,
    CompositeProtocol,
    load_protocol_template,
)


PROTOCOL_PATH = Path("configs/experiment_protocol.template.yaml")
FEATURES = ("synthetic_linear", "synthetic_stable", "synthetic_rare")
COMPOSITE_FEATURES = (
    "speech_ratio",
    "internal_pauses_per_speech_minute",
)
pytestmark = pytest.mark.filterwarnings(
    "error::sklearn.exceptions.ConvergenceWarning"
)


def _row(
    *,
    sample_id: str,
    group_id: str,
    outer_fold: int,
    inner_fold: int,
    score: float,
    features: dict[str, float | None],
) -> AnalysisRow:
    return AnalysisRow(
        sample_id=sample_id,
        total_score=score,
        score_status="observed",
        terminal_status="complete",
        acquisition_status="succeeded",
        audio_status="succeeded",
        transcript_status="failed",
        feature_status="succeeded",
        transcript_available=False,
        whisper_language_code=None,
        group_id=group_id,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        speaker_id="unknown",
        talk_id=sample_id,
        modelling_scope="supervised_exploratory",
        supervised_eligible=True,
        qc_eligible=True,
        clustering_eligible=True,
        qc_warnings=(),
        acquisition_error_code=None,
        audio_error_code=None,
        asr_error_code=None,
        feature_error_code=None,
        missing_feature_count=sum(value is None for value in features.values()),
        features=features,
        missing_reasons={
            name: "synthetic_not_observed"
            for name, value in features.items()
            if value is None
        },
    )


def _nested_cohort(*, corrupt_outer_zero: bool = False) -> AnalysisCohort:
    rows: list[AnalysisRow] = []
    for group_index in range(10):
        for duplicate_index in range(2):
            sample_id = f"S{group_index:02d}{duplicate_index}"
            linear = float(group_index) + 0.2 * duplicate_index
            stable = 2.0 * group_index + float(duplicate_index)
            rare = 100.0 if (group_index, duplicate_index) == (1, 0) else None
            score = 12.0 + 3.0 * linear + 0.25 * stable
            if corrupt_outer_zero and group_index % 5 == 0:
                linear = 100.0 + group_index + duplicate_index
                stable = 200.0 + 2.0 * group_index + duplicate_index
                rare = 500.0
                score = 0.0 if duplicate_index == 0 else 100.0
            rows.append(
                _row(
                    sample_id=sample_id,
                    group_id=f"duplicate-group-{group_index:02d}",
                    outer_fold=group_index % 5,
                    inner_fold=group_index % 4,
                    score=score,
                    features={
                        "synthetic_linear": linear,
                        "synthetic_stable": stable,
                        "synthetic_rare": rare,
                    },
                )
            )
    return AnalysisCohort(
        dataset_id="synthetic-dataset",
        purpose="supervised_total_score",
        feature_names=FEATURES,
        rows=tuple(rows),
        exclusions=(),
    )


def _composite_protocol(
    *, weights: tuple[float, float] = (0.5, 0.5)
):
    template = load_protocol_template(PROTOCOL_PATH)
    composite = CompositeProtocol(
        name="exploratory_delivery_composite",
        interpretation="fixed_expert_exploratory",
        transform="training_fold_robust_z",
        aggregation="signed_weighted_mean",
        components=(
            CompositeComponent(
                feature=COMPOSITE_FEATURES[0], direction=1, weight=weights[0]
            ),
            CompositeComponent(
                feature=COMPOSITE_FEATURES[1], direction=-1, weight=weights[1]
            ),
        ),
    )
    return template.model_copy(update={"composite": composite})


def _composite_cohort() -> AnalysisCohort:
    values = (
        (0.0, 30.0),
        (1.0, None),
        (2.0, 20.0),
        (4.0, 10.0),
    )
    scores = (10.0, 20.0, 80.0, 90.0)
    rows = tuple(
        _row(
            sample_id=f"C{index:03d}",
            group_id=f"composite-group-{0 if index < 2 else 1}",
            outer_fold=0 if index < 2 else 1,
            inner_fold=index % 2,
            score=scores[index],
            features=dict(zip(COMPOSITE_FEATURES, row, strict=True)),
        )
        for index, row in enumerate(values)
    )
    return AnalysisCohort(
        dataset_id="synthetic-composite",
        purpose="supervised_total_score",
        feature_names=COMPOSITE_FEATURES,
        rows=rows,
        exclusions=(),
    )


def _evaluation(results, model_name: str):
    return next(
        evaluation
        for evaluation in results.evaluations
        if evaluation.model_name == model_name
    )


def _outer_audit(results, model_name: str, outer_fold: int):
    return next(
        audit
        for audit in _evaluation(results, model_name).fold_audits
        if audit.outer_fold == outer_fold
    )


def _manual_evaluation(
    model_name: str,
    predictions: tuple[float, ...],
    *,
    group_ids: tuple[str, ...] = ("A", "A", "B", "B", "C", "C"),
    observed: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
) -> ModelEvaluation:
    errors = np.asarray(predictions) - np.asarray(observed)
    return ModelEvaluation(
        model_name=model_name,  # type: ignore[arg-type]
        sample_ids=tuple(f"B{index:03d}" for index in range(len(observed))),
        group_ids=group_ids,
        observed=observed,
        predictions=predictions,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        spearman=0.0,
        pearson=0.0,
        spearman_reason=None,
        pearson_reason=None,
    )


def test_outer_validation_outliers_and_labels_cannot_change_fold_zero_fit() -> None:
    """Break caught: held-out rows leak into filtering, scaling, or tuning."""
    protocol = load_protocol_template(PROTOCOL_PATH)

    clean = run_nested_baselines(_nested_cohort(), protocol)
    corrupted = run_nested_baselines(
        _nested_cohort(corrupt_outer_zero=True), protocol
    )

    for model_name in ("ridge", "elastic_net"):
        clean_audit = _outer_audit(clean, model_name, 0)
        corrupt_audit = _outer_audit(corrupted, model_name, 0)
        assert clean_audit.training_sample_ids == corrupt_audit.training_sample_ids
        assert clean_audit.validation_sample_ids == corrupt_audit.validation_sample_ids
        assert clean_audit.feature_names == corrupt_audit.feature_names
        assert clean_audit.excluded_all_null_features == (
            corrupt_audit.excluded_all_null_features
        )
        assert clean_audit.location == corrupt_audit.location
        assert clean_audit.scale == corrupt_audit.scale
        assert clean_audit.selected_hyperparameters == (
            corrupt_audit.selected_hyperparameters
        )
        assert clean_audit.inner_audits == corrupt_audit.inner_audits


def test_inner_validation_outliers_cannot_change_that_inner_training_fit() -> None:
    """Break caught: inner validation values enter imputation/scaling/filtering."""
    cohort = _nested_cohort()
    changed_rows: list[AnalysisRow] = []
    for row in cohort.rows:
        if row.outer_fold != 0 and row.inner_fold == 0:
            changed_rows.append(
                replace(
                    row,
                    features={
                        "synthetic_linear": 100.0,
                        "synthetic_stable": 200.0,
                        "synthetic_rare": 500.0,
                    },
                    missing_feature_count=0,
                    missing_reasons={},
                )
            )
        else:
            changed_rows.append(row)
    changed = replace(cohort, rows=tuple(changed_rows))
    protocol = load_protocol_template(PROTOCOL_PATH)

    clean_result = run_nested_baselines(cohort, protocol)
    changed_result = run_nested_baselines(changed, protocol)
    clean_inner = next(
        audit
        for audit in _outer_audit(clean_result, "ridge", 0).inner_audits
        if audit.inner_fold == 0
    )
    changed_inner = next(
        audit
        for audit in _outer_audit(changed_result, "ridge", 0).inner_audits
        if audit.inner_fold == 0
    )

    assert clean_inner.training_sample_ids == changed_inner.training_sample_ids
    assert clean_inner.validation_sample_ids == changed_inner.validation_sample_ids
    assert clean_inner.feature_names == changed_inner.feature_names
    assert clean_inner.excluded_all_null_features == (
        changed_inner.excluded_all_null_features
    )
    assert clean_inner.location == changed_inner.location
    assert clean_inner.scale == changed_inner.scale


def test_frozen_groups_drive_nested_folds_and_unknown_speakers_stay_exploratory() -> None:
    """Break caught: duplicates split or sample grouping becomes a speaker claim."""
    results = run_nested_baselines(
        _nested_cohort(), load_protocol_template(PROTOCOL_PATH)
    )

    assert results.outer_fold_count == 5
    assert results.inner_fold_counts == (4, 4, 4, 4, 4)
    assert results.speaker_disjoint is False
    assert results.grouping_unit == "source_or_sample_group"
    assert results.evaluation_claim == "exploratory_no_speaker_holdout"
    assert "speaker" in results.limitation.lower()
    assert tuple(item.model_name for item in results.evaluations) == (
        "training_mean",
        "ridge",
        "elastic_net",
    )

    expected_ids = frozenset(_nested_cohort().sample_ids)
    for evaluation in results.evaluations:
        assert evaluation.primary_metric == "mae"
        assert evaluation.sample_count == len(expected_ids)
        assert evaluation.oof_prediction_count == len(expected_ids)
        assert frozenset(evaluation.sample_ids) == expected_ids
        assert np.isfinite(evaluation.mae)
        assert np.isfinite(evaluation.rmse)
        validation_counts: dict[str, int] = {sample_id: 0 for sample_id in expected_ids}
        for audit in evaluation.fold_audits:
            assert set(audit.training_group_ids).isdisjoint(audit.validation_group_ids)
            for sample_id in audit.validation_sample_ids:
                validation_counts[sample_id] += 1
            for inner in audit.inner_audits:
                assert set(inner.training_group_ids).isdisjoint(
                    inner.validation_group_ids
                )
        assert set(validation_counts.values()) == {1}

    ridge = _evaluation(results, "ridge")
    elastic_net = _evaluation(results, "elastic_net")
    assert all(
        set(audit.selected_hyperparameters) == {"alpha"}
        for audit in ridge.fold_audits
    )
    assert all(
        set(audit.selected_hyperparameters) == {"alpha", "l1_ratio"}
        for audit in elastic_net.fold_audits
    )
    assert any(
        "synthetic_rare" in inner.excluded_all_null_features
        for outer in ridge.fold_audits
        for inner in outer.inner_audits
    )
    for group_index in range(10):
        pair = {f"S{group_index:02d}0", f"S{group_index:02d}1"}
        containing_folds = [
            audit.outer_fold
            for audit in ridge.fold_audits
            if pair.intersection(audit.validation_sample_ids)
        ]
        assert len(containing_folds) == 1
        held_out = next(
            audit for audit in ridge.fold_audits if audit.outer_fold == containing_folds[0]
        )
        assert pair.issubset(held_out.validation_sample_ids)


def test_small_fixture_reduces_frozen_outer_and_inner_fold_counts_explicitly() -> None:
    """Break caught: five-by-four CV is forced when too few groups are available."""
    full = _nested_cohort()
    small = replace(
        full,
        rows=tuple(
            row
            for row in full.rows
            if row.group_id in {
                "duplicate-group-00",
                "duplicate-group-01",
                "duplicate-group-02",
            }
        ),
    )

    results = run_nested_baselines(small, load_protocol_template(PROTOCOL_PATH))

    assert results.outer_fold_count == 3
    assert results.inner_fold_counts == (2, 2, 2)
    assert all(evaluation.oof_prediction_count == 6 for evaluation in results.evaluations)


def test_inconsistent_frozen_group_assignment_is_rejected() -> None:
    """Break caught: a duplicate group is silently allowed across outer folds."""
    cohort = _nested_cohort()
    broken_rows = list(cohort.rows)
    broken_rows[0] = replace(broken_rows[0], outer_fold=1)

    with pytest.raises(ValueError, match="independent group.*spans outer folds"):
        run_nested_baselines(
            replace(cohort, rows=tuple(broken_rows)),
            load_protocol_template(PROTOCOL_PATH),
        )


def test_row_labels_cannot_upgrade_the_frozen_no_speaker_holdout_claim() -> None:
    """Break caught: a convenient cohort overrides the global speaker limitation."""
    cohort = _nested_cohort()
    apparently_known = replace(
        cohort,
        rows=tuple(
            replace(row, speaker_id=f"speaker-{row.group_id}")
            for row in cohort.rows
        ),
    )

    results = run_nested_baselines(
        apparently_known, load_protocol_template(PROTOCOL_PATH)
    )

    assert results.speaker_disjoint is False
    assert results.evaluation_claim == "exploratory_no_speaker_holdout"
    assert "not reliably available" in results.limitation.lower()


def test_training_mean_predictions_use_only_each_outer_training_partition() -> None:
    """Break caught: the mean baseline uses the full cohort or held-out labels."""
    cohort = _nested_cohort()
    results = run_nested_baselines(cohort, load_protocol_template(PROTOCOL_PATH))
    evaluation = _evaluation(results, "training_mean")
    audit = _outer_audit(results, "training_mean", 0)
    expected = np.mean(
        [float(row.total_score) for row in cohort.rows if row.outer_fold != 0]
    )

    assert expected != pytest.approx(
        np.mean([float(row.total_score) for row in cohort.rows])
    )
    for sample_id in audit.validation_sample_ids:
        assert evaluation.prediction_by_sample[sample_id] == pytest.approx(expected)


def test_oof_output_order_is_sample_sorted_and_independent_of_input_order() -> None:
    """Break caught: row order changes OOF alignment or serialized predictions."""
    cohort = _nested_cohort()
    protocol = load_protocol_template(PROTOCOL_PATH)

    forward = run_nested_baselines(cohort, protocol)
    reversed_result = run_nested_baselines(
        replace(cohort, rows=tuple(reversed(cohort.rows))), protocol
    )

    expected_ids = tuple(sorted(cohort.sample_ids))
    for first, second in zip(
        forward.evaluations, reversed_result.evaluations, strict=True
    ):
        assert first.sample_ids == second.sample_ids == expected_ids
        assert first.group_ids == second.group_ids
        np.testing.assert_allclose(first.observed, second.observed)
        np.testing.assert_allclose(first.predictions, second.predictions)


def test_outer_training_all_null_feature_is_excluded_despite_held_out_values() -> None:
    """Break caught: held-out feature availability rescues an all-null train column."""
    cohort = _nested_cohort()
    changed_rows = tuple(
        replace(
            row,
            features={**dict(row.features), "synthetic_rare": 999_999.0},
            missing_feature_count=max(0, row.missing_feature_count - 1),
            missing_reasons={
                name: reason
                for name, reason in row.missing_reasons.items()
                if name != "synthetic_rare"
            },
        )
        if row.outer_fold == 1
        else row
        for row in cohort.rows
    )
    protocol = load_protocol_template(PROTOCOL_PATH)

    original = run_nested_baselines(cohort, protocol)
    changed = run_nested_baselines(replace(cohort, rows=changed_rows), protocol)

    for model_name in ("ridge", "elastic_net"):
        original_audit = _outer_audit(original, model_name, 1)
        changed_audit = _outer_audit(changed, model_name, 1)
        assert "synthetic_rare" in original_audit.excluded_all_null_features
        assert "synthetic_rare" not in original_audit.feature_names
        assert original_audit.location == changed_audit.location
        assert original_audit.scale == changed_audit.scale
        original_predictions = _evaluation(original, model_name).prediction_by_sample
        changed_predictions = _evaluation(changed, model_name).prediction_by_sample
        for sample_id in original_audit.validation_sample_ids:
            assert changed_predictions[sample_id] == pytest.approx(
                original_predictions[sample_id]
            )


def test_model_results_record_the_complete_deterministic_tuning_grids() -> None:
    """Break caught: run artifacts cannot reconstruct the alpha/l1-ratio search."""
    results = run_nested_baselines(
        _nested_cohort(), load_protocol_template(PROTOCOL_PATH)
    )

    assert _evaluation(results, "training_mean").hyperparameter_grid == {}
    assert _evaluation(results, "ridge").hyperparameter_grid == {
        "alpha": (0.01, 0.1, 1.0, 10.0, 100.0)
    }
    assert _evaluation(results, "elastic_net").hyperparameter_grid == {
        "alpha": (0.001, 0.01, 0.1, 1.0, 10.0),
        "l1_ratio": (0.1, 0.5, 0.9),
    }


def test_categorical_registry_audit_feature_is_not_used_by_numeric_baselines() -> None:
    """Break caught: the real registry's categorical audit field aborts modelling."""
    cohort = _nested_cohort()
    categorical = replace(
        cohort,
        feature_names=(*cohort.feature_names, "candidate_transient_confidence_level"),
        rows=tuple(
            replace(
                row,
                features={
                    **dict(row.features),
                    "candidate_transient_confidence_level": "high",
                },
            )
            for row in cohort.rows
        ),
    )

    results = run_nested_baselines(
        categorical, load_protocol_template(PROTOCOL_PATH)
    )

    for model_name in ("ridge", "elastic_net"):
        assert all(
            "candidate_transient_confidence_level" not in audit.feature_names
            for audit in _evaluation(results, model_name).fold_audits
        )


def test_constant_predictions_report_undefined_correlation_reasons() -> None:
    """Break caught: undefined correlations are emitted as NaN without reasons."""
    cohort = _nested_cohort()
    constant_rows = tuple(replace(row, total_score=50.0) for row in cohort.rows)
    constant = replace(cohort, rows=constant_rows)

    results = run_nested_baselines(
        constant, load_protocol_template(PROTOCOL_PATH)
    )

    for evaluation in results.evaluations:
        assert evaluation.spearman is None
        assert evaluation.pearson is None
        assert evaluation.spearman_reason == "constant_observed_values"
        assert evaluation.pearson_reason == "constant_observed_values"


def test_composite_is_a_cross_fitted_equal_available_dimension_index() -> None:
    """Break caught: weights, global scaling, imputed dilution, or grade calibration."""
    cohort = _composite_cohort()

    result = compute_acoustic_composite(cohort, _composite_protocol())

    assert result.sample_ids == cohort.sample_ids
    np.testing.assert_allclose(result.index_values, (-3.0, -2.0, 6.5, 13.5))
    assert result.available_dimension_counts == (2, 1, 2, 2)
    assert result.feature_directions == {
        "speech_ratio": 1,
        "internal_pauses_per_speech_minute": -1,
    }
    assert result.aggregation == "equal_available_dimensions"
    assert result.scale == "raw_standardized_index"
    assert result.learned_weights is False
    assert result.calibrated_to_target is False
    assert min(result.index_values) < 0.0
    assert result.spearman is not None
    assert result.pearson is not None
    assert result.spearman_reason is None
    assert result.pearson_reason is None

    relabelled = replace(
        cohort,
        rows=tuple(
            replace(row, total_score=100.0 - float(row.total_score))
            for row in cohort.rows
        ),
    )
    assert compute_acoustic_composite(
        relabelled, _composite_protocol()
    ).index_values == result.index_values


def test_composite_rejects_unequal_protocol_weights() -> None:
    """Break caught: an unequal or learned weighting masquerades as the fixed index."""
    with pytest.raises(ValueError, match="equal fixed weights"):
        compute_acoustic_composite(
            _composite_cohort(), _composite_protocol(weights=(0.75, 0.25))
        )


def test_paired_group_bootstrap_is_seeded_percentile_and_group_level() -> None:
    """Break caught: rows/folds are resampled or model intervals are unpaired."""
    reference = _manual_evaluation(
        "ridge", (0.0, 1.0, 5.0, 7.0, 7.0, 11.0)
    )
    comparison = _manual_evaluation(
        "elastic_net", (1.0, 3.0, 3.0, 5.0, 9.0, 9.0)
    )

    first = paired_group_bootstrap(reference, comparison)
    second = paired_group_bootstrap(reference, comparison)

    assert first == second
    assert first.status == "ok"
    assert first.reason is None
    assert first.metric == "mae"
    assert first.method == "paired_group_bootstrap"
    assert first.difference_direction == "comparison_minus_reference"
    assert first.estimate == pytest.approx(1.0 / 6.0)
    assert first.lower == pytest.approx(0.0)
    assert first.upper == pytest.approx(0.5)
    assert first.analysis_unit == "group"
    assert first.paired is True
    assert first.reference_model == "ridge"
    assert first.comparison_model == "elastic_net"
    assert first.replicates == 5_000
    assert first.valid_replicates == 5_000
    assert first.seed == 20260813
    assert first.confidence_level == 0.95
    assert first.interval_method == "percentile"
    assert first.includes_zero is True
    assert "repetitions" not in inspect.signature(paired_group_bootstrap).parameters


def test_group_bootstrap_returns_not_estimable_for_too_few_or_degenerate_groups() -> None:
    """Break caught: an artificial zero-width CI or row bootstrap is reported."""
    reference = _manual_evaluation(
        "ridge", (0.0, 1.0, 5.0, 7.0, 7.0, 11.0)
    )
    one_group = replace(reference, group_ids=("only",) * 6)

    too_few = paired_group_bootstrap(one_group)
    degenerate = paired_group_bootstrap(reference, reference)

    assert too_few.status == "not_estimable"
    assert too_few.reason == "fewer_than_two_independent_groups"
    assert degenerate.status == "not_estimable"
    assert degenerate.reason == "degenerate_bootstrap_distribution"
    assert degenerate.estimate == pytest.approx(0.0)
    assert degenerate.lower is None
    assert degenerate.upper is None
    assert too_few.includes_zero is None
    assert degenerate.includes_zero is None
    with pytest.raises(ValueError, match="seed must remain frozen"):
        paired_group_bootstrap(reference, seed=7)


def test_correlation_bootstrap_fails_closed_when_any_replicate_is_undefined() -> None:
    """Break caught: a partial set of valid correlation draws yields a percentile CI."""
    evaluation = _manual_evaluation(
        "ridge",
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0),
        observed=(0.0, 0.0, 1.0, 1.0, 2.0, 2.0),
    )

    interval = paired_group_bootstrap(evaluation, metric="spearman")

    assert interval.status == "not_estimable"
    assert interval.reason == "not_all_bootstrap_replicates_valid"
    assert 0 < interval.valid_replicates < interval.replicates
    assert interval.replicates == 5_000
    assert interval.lower is None
    assert interval.upper is None


def test_bootstrap_rejects_collapsed_percentiles_despite_nonconstant_draws() -> None:
    """Break caught: rare floating variation hides behind equal percentile bounds."""
    observed = (0.0,) * 10
    group_ids = tuple(f"G{index}" for index in range(10))
    reference = _manual_evaluation(
        "ridge", (1.0,) * 10, group_ids=group_ids, observed=observed
    )
    comparison_predictions = [1.0] * 10
    comparison_predictions[0] = 1.0 + 2.0 * np.finfo(float).eps
    comparison = _manual_evaluation(
        "elastic_net",
        tuple(comparison_predictions),
        group_ids=group_ids,
        observed=observed,
    )

    interval = paired_group_bootstrap(reference, comparison)

    assert interval.status == "not_estimable"
    assert interval.reason == "collapsed_percentile_interval"
    assert interval.valid_replicates == interval.replicates == 5_000
    assert interval.lower is None
    assert interval.upper is None
