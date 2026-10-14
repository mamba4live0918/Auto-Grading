from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from presentation_assessment.research.analysis_data import AnalysisCohort
from presentation_assessment.research.clustering import (
    _holm_adjust,
    export_redacted_cluster_figure,
    export_redacted_cluster_table,
    fit_acoustic_clusters,
    posthoc_cluster_score_test,
)
from presentation_assessment.research.feature_registry import (
    FeatureRegistry,
    load_feature_registry,
)
from presentation_assessment.research.protocol import (
    ProtocolTemplate,
    load_protocol_template,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "audio_feature_registry.v1.yaml"
PROTOCOL_PATH = ROOT / "configs" / "experiment_protocol.template.yaml"
FEATURES = ("speech_ratio", "internal_pauses_per_speech_minute")
CONDITIONAL_FEATURE = "asr_units_per_speech_minute"


@dataclass(frozen=True, slots=True)
class _ScoreGuardedRow:
    sample_id: str
    group_id: str
    features: Mapping[str, float | None]
    transcript_status: str
    whisper_language_code: str | None

    @property
    def total_score(self) -> float:
        raise AssertionError("clustering fit read a forbidden score")


def _registry(feature_names: tuple[str, ...] = FEATURES) -> FeatureRegistry:
    frozen = load_feature_registry(REGISTRY_PATH)
    selected = tuple(frozen.by_name[name] for name in feature_names)
    conditional = frozen.by_name[CONDITIONAL_FEATURE]
    return FeatureRegistry(
        schema_version=1,
        specification="synthetic clustering contract",
        features=(
            selected
            if CONDITIONAL_FEATURE in feature_names
            else (*selected, conditional)
        ),
    )


def _protocol(feature_names: tuple[str, ...] = FEATURES) -> ProtocolTemplate:
    frozen = load_protocol_template(PROTOCOL_PATH)
    clustering = frozen.clustering.model_copy(
        update={"primary_features": feature_names}
    )
    return frozen.model_copy(update={"clustering": clustering})


def _cohort(
    values: list[tuple[float | None, ...]],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    group_ids: tuple[str, ...] | None = None,
    transcript_statuses: tuple[str, ...] | None = None,
    language_codes: tuple[str | None, ...] | None = None,
    purpose: str = "clustering",
) -> AnalysisCohort:
    groups = group_ids or tuple(f"G{index:03d}" for index in range(len(values)))
    statuses = transcript_statuses or tuple("failed" for _ in values)
    languages = language_codes or tuple(None for _ in values)
    rows = tuple(
        _ScoreGuardedRow(
            sample_id=f"S{index:03d}",
            group_id=groups[index],
            features=MappingProxyType(dict(zip(feature_names, row, strict=True))),
            transcript_status=statuses[index],
            whisper_language_code=languages[index],
        )
        for index, row in enumerate(values)
    )
    return AnalysisCohort(
        dataset_id="synthetic-dataset",
        purpose=purpose,  # type: ignore[arg-type]
        feature_names=feature_names,
        rows=rows,  # type: ignore[arg-type]
        exclusions=(),
    )


def _fit(values: list[tuple[float | None, ...]], **cohort_kwargs: object):
    cohort = _cohort(values, **cohort_kwargs)
    names = cohort.feature_names
    return fit_acoustic_clusters(cohort, _protocol(names), _registry(names))


def test_fit_interface_has_no_score_parameter_and_never_reads_row_scores() -> None:
    """Break caught: labels leak into unsupervised fit or cluster-k selection."""
    assert tuple(inspect.signature(fit_acoustic_clusters).parameters) == (
        "cohort",
        "protocol",
        "registry",
    )

    result = _fit([(0.0, 3.0), (0.25, 2.0), (0.5, 1.0), (0.75, 0.0)])

    assert result.sample_count == 4
    assert result.feature_names == FEATURES


def test_frozen_registry_and_protocol_fit_end_to_end() -> None:
    """Break caught: fit silently chooses features outside the frozen registry."""
    protocol = load_protocol_template(PROTOCOL_PATH)
    registry = load_feature_registry(REGISTRY_PATH)
    names = tuple(protocol.clustering.primary_features)
    rows: list[tuple[float, ...]] = []
    for latent in (0.0, 0.1, 0.2, 0.8, 0.9, 1.0):
        values: list[float] = []
        for name in names:
            definition = registry.by_name[name]
            lower = float(definition.minimum or 0.0)
            if definition.maximum is None:
                values.append(lower + 1.0 + latent)
            else:
                upper = float(definition.maximum)
                values.append(lower + (upper - lower) * (0.1 + 0.8 * latent))
        rows.append(tuple(values))
    cohort = _cohort(rows, feature_names=names)

    result = fit_acoustic_clusters(cohort, protocol, registry)

    assert len(result.feature_names) == 11
    assert result.feature_names == names
    assert result.selected_k == 2
    assert result.conditional_feature_status == (
        "excluded_no_successful_asr_rows"
    )


def test_conditional_asr_feature_is_included_for_one_successful_language() -> None:
    """Break caught: a verified single-language ASR rate is always discarded."""
    names = (*FEATURES, CONDITIONAL_FEATURE)
    cohort = _cohort(
        [
            (0.0, 0.0, 10.0),
            (0.1, 0.1, 11.0),
            (0.9, 9.9, None),
            (1.0, 10.0, 20.0),
        ],
        feature_names=names,
        transcript_statuses=("succeeded", "succeeded", "failed", "succeeded"),
        language_codes=("en", "en", None, "en"),
    )

    result = fit_acoustic_clusters(cohort, _protocol(), _registry())

    assert result.feature_names == names
    assert result.conditional_feature_status == "included_single_whisper_language"
    assert result.feature_medians[CONDITIONAL_FEATURE] == pytest.approx(11.0)
    assert result.missing_indicators[2][-1] is True


@pytest.mark.parametrize(
    ("statuses", "languages", "expected_status"),
    [
        (
            ("succeeded", "succeeded", "succeeded"),
            ("en", "fr", "en"),
            "excluded_mixed_whisper_languages",
        ),
        (
            ("succeeded", "succeeded", "succeeded"),
            ("en", None, "en"),
            "excluded_missing_whisper_language",
        ),
        (
            ("failed", "failed", "failed"),
            (None, None, None),
            "excluded_no_successful_asr_rows",
        ),
    ],
)
def test_conditional_asr_feature_exclusion_has_exact_reason(
    statuses: tuple[str, ...],
    languages: tuple[str | None, ...],
    expected_status: str,
) -> None:
    """Break caught: mixed, missing, and absent ASR language states are conflated."""
    names = (*FEATURES, CONDITIONAL_FEATURE)
    cohort = _cohort(
        [(0.0, 0.0, 10.0), (0.5, 1.0, 15.0), (1.0, 2.0, 20.0)],
        feature_names=names,
        transcript_statuses=statuses,
        language_codes=languages,
    )

    result = fit_acoustic_clusters(cohort, _protocol(), _registry())

    assert result.feature_names == FEATURES
    assert result.conditional_feature_status == expected_status


def test_robust_scaling_and_oriented_pca_match_hand_checked_fixture() -> None:
    """Break caught: means/SD or an arbitrary PCA sign replaces the frozen rules."""
    result = _fit([(0.0, 3.0), (0.25, 2.0), (0.5, 1.0), (0.75, 0.0)])

    assert result.feature_medians == {
        "speech_ratio": pytest.approx(0.375),
        "internal_pauses_per_speech_minute": pytest.approx(1.5),
    }
    assert result.feature_iqrs == {
        "speech_ratio": pytest.approx(0.375),
        "internal_pauses_per_speech_minute": pytest.approx(1.5),
    }
    np.testing.assert_allclose(
        result.scaled_values,
        [[-1.0, 1.0], [-1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, -1.0 / 3.0], [1.0, -1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.pca_loadings,
        [[2**-0.5, -(2**-0.5)], [2**-0.5, 2**-0.5]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.pca_coordinates,
        [
            [-(2**0.5), 0.0],
            [-(2**0.5) / 3.0, 0.0],
            [(2**0.5) / 3.0, 0.0],
            [2**0.5, 0.0],
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.pca_explained_variance_ratio, [1.0, 0.0])


def test_missing_values_use_cohort_median_and_stay_audit_only() -> None:
    """Break caught: nulls fail silently or missing indicators enter the PCA."""
    result = _fit([(0.0, 0.0), (None, 1.0), (0.5, 2.0), (1.0, 3.0)])

    assert result.feature_medians["speech_ratio"] == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.asarray(result.scaled_values)[:, 0], [-1.0, 0.0, 0.0, 1.0]
    )
    assert result.missing_indicators == (
        (False, False),
        (True, False),
        (False, False),
        (False, False),
    )
    assert all("__missing" not in name for name in result.feature_names)
    assert np.asarray(result.pca_loadings).shape == (2, 2)


def test_euclidean_distance_and_median_scale_rbf_are_exact() -> None:
    """Break caught: another distance or RBF denominator changes similarity."""
    result = _fit([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)])

    np.testing.assert_allclose(
        result.distance_matrix,
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        atol=1e-12,
    )
    assert result.similarity_scale == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.similarity_matrix,
        [
            [100.0, 100.0 * np.exp(-0.5), 100.0 * np.exp(-2.0)],
            [100.0 * np.exp(-0.5), 100.0, 100.0 * np.exp(-0.5)],
            [100.0 * np.exp(-2.0), 100.0 * np.exp(-0.5), 100.0],
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.distance_matrix, np.asarray(result.distance_matrix).T
    )
    np.testing.assert_allclose(np.diag(result.distance_matrix), 0.0)


def test_identical_inputs_return_explicit_indistinguishable_result() -> None:
    """Break caught: an arbitrary Ward split is reported for identical samples."""
    result = _fit([(0.5, 1.0)] * 4)

    assert result.status == "indistinguishable"
    assert result.selected_k is None
    assert result.cluster_assignments == (1, 1, 1, 1)
    assert result.candidate_silhouettes == ()
    np.testing.assert_allclose(result.similarity_matrix, np.full((4, 4), 100.0))
    np.testing.assert_allclose(result.pca_coordinates, np.zeros((4, 2)))
    np.testing.assert_allclose(result.pca_explained_variance_ratio, [0.0, 0.0])


def test_ward_candidates_selection_and_leave_one_feature_out_stability() -> None:
    """Break caught: wrong k range/linkage or missing feature-omission sensitivity."""
    result = _fit(
        [
            (0.00, 0.00),
            (0.05, 0.02),
            (0.10, 0.01),
            (0.90, 9.90),
            (0.95, 10.00),
            (1.00, 10.10),
        ]
    )

    assert tuple(item.k for item in result.candidate_silhouettes) == (2, 3, 4, 5)
    assert result.selected_k == 2
    assert result.cluster_assignments == (1, 1, 1, 2, 2, 2)
    assert (
        tuple(item.feature_name for item in result.leave_one_feature_out) == FEATURES
    )
    assert all(
        item.adjusted_rand_index == pytest.approx(1.0)
        for item in result.leave_one_feature_out
    )


def test_selection_prefers_smaller_k_within_frozen_point_zero_five_tolerance() -> None:
    """Break caught: raw maximum silhouette wins despite the frozen parsimony rule."""
    raw = np.array(
        [-0.6803, -0.5422, -0.4162, -0.0363, 0.3740, 0.3897, 1.5584]
    )
    normalised = (raw - raw.min()) / (raw.max() - raw.min())
    result = _fit([(float(value), 0.0) for value in normalised])
    silhouettes = {item.k: item.silhouette for item in result.candidate_silhouettes}

    assert max(silhouettes, key=silhouettes.get) == 3
    assert silhouettes[3] - silhouettes[2] == pytest.approx(0.0276, abs=0.0002)
    assert result.selected_k == 2


def test_descriptive_labels_use_frozen_quartiles_and_median_differences() -> None:
    """Break caught: labels become score-derived or free-form post-hoc narratives."""
    result = _fit(
        [(0.00, 1.0), (0.45, 1.0), (0.50, 1.0), (0.55, 1.0), (1.00, 1.0)]
    )
    labels = {
        (item.sample_id, item.feature_name): item
        for item in result.sample_feature_labels
    }
    descriptions = {item.cluster_id: item.label for item in result.cluster_descriptions}

    assert labels[("S000", "speech_ratio")].category == "very_low"
    assert labels[("S000", "speech_ratio")].robust_z == pytest.approx(-5.0)
    assert labels[("S002", "speech_ratio")].category == "typical"
    assert labels[("S004", "speech_ratio")].category == "very_high"
    assert labels[("S000", "internal_pauses_per_speech_minute")].category == "typical"
    assert descriptions == {
        1: "lower speech ratio",
        2: "typical acoustic profile",
        3: "higher speech ratio",
    }


@pytest.mark.parametrize(
    ("cohort", "message"),
    [
        (_cohort([], feature_names=FEATURES), "at least three samples"),
        (
            _cohort([(0.0, 0.0), (0.5, 0.5)], feature_names=FEATURES),
            "at least three samples",
        ),
        (
            _cohort([(0.0,), (0.5,), (1.0,)], feature_names=("speech_ratio",)),
            "at least two clustering features",
        ),
    ],
)
def test_too_small_inputs_fail_with_targeted_reason(
    cohort: AnalysisCohort, message: str
) -> None:
    """Break caught: PCA/Ward fail later with dependency-specific tracebacks."""
    with pytest.raises(ValueError, match=message):
        fit_acoustic_clusters(
            cohort,
            _protocol(cohort.feature_names),
            _registry(cohort.feature_names),
        )


def test_missing_or_invalid_inputs_fail_before_fitting() -> None:
    """Break caught: absent, all-null, non-finite, or wrong-purpose input is coerced."""
    with pytest.raises(ValueError, match="clustering purpose"):
        bad_purpose = _cohort(
            [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)], purpose="quality_control"
        )
        fit_acoustic_clusters(bad_purpose, _protocol(), _registry())

    missing_column = _cohort(
        [(0.0,), (0.5,), (1.0,)], feature_names=("speech_ratio",)
    )
    with pytest.raises(ValueError, match="missing protocol features"):
        fit_acoustic_clusters(missing_column, _protocol(), _registry())

    with pytest.raises(ValueError, match="entirely missing"):
        _fit([(None, 0.0), (None, 1.0), (None, 2.0)])

    with pytest.raises(ValueError, match="finite or missing"):
        _fit([(0.0, 0.0), (0.5, float("inf")), (1.0, 2.0)])


def _posthoc_fixture():
    values = [
        (0.00, 0.00),
        (0.05, 0.05),
        (0.10, 0.10),
        (0.15, 0.15),
        (0.85, 9.85),
        (0.90, 9.90),
        (0.95, 9.95),
        (1.00, 10.00),
    ]
    groups = ("A", "A", "B", "B", "C", "C", "D", "D")
    return _fit(values, group_ids=groups)


def test_posthoc_score_test_is_seeded_group_level_and_cannot_change_k() -> None:
    """Break caught: row shuffling or score-driven re-clustering changes inference."""
    result = _posthoc_fixture()
    scores = pd.Series(
        [10.0, 14.0, 20.0, 24.0, 80.0, 84.0, 90.0, 94.0],
        index=result.sample_ids,
    )

    first = posthoc_cluster_score_test(result, scores, seed=20260813)
    second = posthoc_cluster_score_test(result, scores, seed=20260813)

    assert first == second
    assert first.status == "ok"
    assert first.selected_k == result.selected_k == 2
    assert first.analysis_unit == "group"
    assert first.group_count == 4
    assert first.sample_count == 8
    assert first.repetitions == 10_000
    assert first.multiple_comparison_correction == "holm"
    assert len(first.comparisons) == 1
    comparison = first.comparisons[0]
    assert comparison.mean_difference == pytest.approx(-70.0)
    assert comparison.p_value == pytest.approx(0.33236676332366766)
    assert 0.0 < comparison.p_value <= comparison.adjusted_p_value <= 1.0


def test_posthoc_defaults_to_frozen_ten_thousand_repetitions() -> None:
    """Break caught: the fast-test repetition override becomes the real default."""
    result = _posthoc_fixture()
    scores = {
        sample_id: float(index) for index, sample_id in enumerate(result.sample_ids)
    }

    inference = posthoc_cluster_score_test(result, scores)

    assert inference.repetitions == 10_000
    assert inference.seed == 20260813


def test_posthoc_public_api_rejects_nonfrozen_protocol_metadata() -> None:
    """Break caught: callers override frozen repetitions, seed, or correction."""
    result = _posthoc_fixture()
    scores = {
        sample_id: float(index) for index, sample_id in enumerate(result.sample_ids)
    }

    assert "repetitions" not in inspect.signature(
        posthoc_cluster_score_test
    ).parameters
    with pytest.raises(ValueError, match="seed must remain frozen"):
        posthoc_cluster_score_test(result, scores, seed=7)
    with pytest.raises(ValueError, match="seed must remain frozen"):
        posthoc_cluster_score_test(
            result, scores, seed=20260813.0  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="10,000"):
        posthoc_cluster_score_test(
            replace(result, permutation_repetitions=999), scores
        )
    with pytest.raises(ValueError, match="10,000"):
        posthoc_cluster_score_test(
            replace(result, permutation_repetitions=10_000.0),  # type: ignore[arg-type]
            scores,
        )
    with pytest.raises(ValueError, match="seed metadata"):
        posthoc_cluster_score_test(replace(result, permutation_seed=7), scores)
    with pytest.raises(ValueError, match="seed metadata"):
        posthoc_cluster_score_test(
            replace(result, permutation_seed=20260813.0),  # type: ignore[arg-type]
            scores,
        )
    with pytest.raises(ValueError, match="Holm"):
        posthoc_cluster_score_test(
            replace(result, multiple_comparison_correction="none"), scores
        )


def test_posthoc_missing_and_too_small_inputs_are_explicit() -> None:
    """Break caught: missing scores or dependent groups yield misleading p-values."""
    result = _posthoc_fixture()

    missing = posthoc_cluster_score_test(
        result,
        {result.sample_ids[0]: 10.0, result.sample_ids[4]: 90.0},
    )
    assert missing.status == "not_estimable"
    assert missing.reason == "fewer_than_two_groups_per_cluster"
    assert missing.excluded_sample_count == 6

    split_group = replace(
        result,
        group_ids=("shared", "A", "B", "C", "shared", "D", "E", "F"),
    )
    with pytest.raises(ValueError, match="spans multiple clusters"):
        posthoc_cluster_score_test(
            split_group,
            {
                sample_id: float(index)
                for index, sample_id in enumerate(result.sample_ids)
            },
        )

    with pytest.raises(ValueError, match="finite or missing"):
        posthoc_cluster_score_test(
            result,
            {sample_id: float("inf") for sample_id in result.sample_ids},
        )


def test_posthoc_rejects_cross_cluster_group_before_missing_score_filter() -> None:
    """Break caught: a missing score hides that one frozen group spans clusters."""
    result = _posthoc_fixture()
    split_group = replace(
        result,
        group_ids=("shared", "A", "B", "C", "shared", "D", "E", "F"),
    )
    scores = {
        sample_id: (None if index == 4 else float(index))
        for index, sample_id in enumerate(result.sample_ids)
    }

    with pytest.raises(ValueError, match="spans multiple clusters"):
        posthoc_cluster_score_test(split_group, scores)


def test_holm_adjustment_is_exact_and_monotone_for_four_comparisons() -> None:
    """Break caught: multi-comparison adjustment is unordered or non-monotone."""
    adjusted = _holm_adjust(np.array([0.01, 0.04, 0.03, 0.20]))

    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.20])
    order = np.argsort([0.01, 0.04, 0.03, 0.20])
    assert np.all(np.diff(adjusted[order]) >= 0.0)


def test_redacted_exporters_emit_aggregate_outputs_only(tmp_path: Path) -> None:
    """Break caught: row IDs, group IDs, coordinates, or scores leak to exports."""
    result = _posthoc_fixture()
    table_path = export_redacted_cluster_table(result, tmp_path / "clusters.csv")
    figure_path = export_redacted_cluster_figure(result, tmp_path / "clusters.png")

    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == (
        "dataset_id",
        "status",
        "selected_k",
        "cluster_id",
        "sample_count",
        "sample_percent",
        "descriptive_label",
        "pc1_centroid",
        "pc2_centroid",
    )
    assert sum(int(row["sample_count"]) for row in rows) == result.sample_count
    exported = table_path.read_text(encoding="utf-8")
    assert all(sample_id not in exported for sample_id in result.sample_ids)
    assert all(group_id not in exported for group_id in set(result.group_ids))
    assert "score" not in exported.lower()
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
