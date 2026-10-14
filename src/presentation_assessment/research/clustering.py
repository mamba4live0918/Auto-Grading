"""Deterministic, score-blind acoustic clustering and post-hoc inference."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .analysis_data import AnalysisCohort
from .feature_registry import FeatureRegistry
from .protocol import BoundProtocol, ProtocolTemplate


ClusteringStatus = Literal["ok", "indistinguishable"]
PosthocStatus = Literal["ok", "not_estimable"]
SampleLabelCategory = Literal[
    "very_low", "low", "typical", "high", "very_high", "median_imputed"
]

_MINIMUM_SAMPLE_COUNT = 3
_MINIMUM_FEATURE_COUNT = 2
_FROZEN_CANDIDATE_COUNTS = (2, 3, 4, 5)
_DEFAULT_PERMUTATION_SEED = 20260813
_FROZEN_PERMUTATION_REPETITIONS = 10_000
_VERY_LOW_Z = -1.5
_VERY_HIGH_Z = 1.5


@dataclass(frozen=True, slots=True)
class CandidateSilhouette:
    """One protocol-declared Ward candidate and its Euclidean silhouette."""

    k: int
    silhouette: float


@dataclass(frozen=True, slots=True)
class FeatureStability:
    """Agreement with the selected solution after omitting one feature."""

    feature_name: str
    adjusted_rand_index: float


@dataclass(frozen=True, slots=True)
class SampleFeatureLabel:
    """Frozen, score-independent quartile/robust-z description."""

    sample_id: str
    feature_name: str
    category: SampleLabelCategory
    robust_z: float
    was_missing: bool


@dataclass(frozen=True, slots=True)
class FeatureMedianDifference:
    """Cluster median minus the whole-cohort median in robust-scaled units."""

    feature_name: str
    difference: float


@dataclass(frozen=True, slots=True)
class ClusterDescription:
    """A deterministic label derived only from robust-scaled feature medians."""

    cluster_id: int
    label: str
    feature_differences: tuple[FeatureMedianDifference, ...]


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Immutable numeric and audit outputs from score-blind clustering.

    Matrices are tuples rather than mutable NumPy arrays. Row identifiers and
    group identifiers are retained only for alignment with the separate
    post-hoc procedure; the supplied exporters aggregate over them.
    """

    dataset_id: str
    status: ClusteringStatus
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_medians: Mapping[str, float]
    feature_iqrs: Mapping[str, float]
    constant_features: tuple[str, ...]
    missing_indicators: tuple[tuple[bool, ...], ...]
    scaled_values: tuple[tuple[float, ...], ...]
    pca_coordinates: tuple[tuple[float, float], ...]
    pca_loadings: tuple[tuple[float, ...], tuple[float, ...]]
    pca_explained_variance_ratio: tuple[float, float]
    distance_matrix: tuple[tuple[float, ...], ...]
    similarity_matrix: tuple[tuple[float, ...], ...]
    similarity_scale: float | None
    candidate_silhouettes: tuple[CandidateSilhouette, ...]
    selected_k: int | None
    cluster_assignments: tuple[int, ...]
    leave_one_feature_out: tuple[FeatureStability, ...]
    sample_feature_labels: tuple[SampleFeatureLabel, ...]
    cluster_descriptions: tuple[ClusterDescription, ...]
    conditional_feature_status: str
    smaller_cluster_tolerance: float
    permutation_repetitions: int
    permutation_seed: int
    multiple_comparison_correction: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feature_medians", MappingProxyType(dict(self.feature_medians))
        )
        object.__setattr__(
            self, "feature_iqrs", MappingProxyType(dict(self.feature_iqrs))
        )

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class ClusterScoreSummary:
    """Aggregate score summary computed at the independent-group level."""

    cluster_id: int
    group_count: int
    sample_count: int
    mean_score: float
    median_score: float


@dataclass(frozen=True, slots=True)
class PairwiseScoreComparison:
    """Exploratory two-sided permutation comparison of group means."""

    cluster_a: int
    cluster_b: int
    mean_difference: float
    p_value: float
    adjusted_p_value: float


@dataclass(frozen=True, slots=True)
class PosthocClusterScoreResult:
    """Separate exploratory score inference for an already frozen solution."""

    status: PosthocStatus
    reason: str | None
    selected_k: int | None
    sample_count: int
    excluded_sample_count: int
    group_count: int
    analysis_unit: Literal["group"]
    repetitions: int
    seed: int
    multiple_comparison_correction: str
    exploratory: Literal[True]
    cluster_summaries: tuple[ClusterScoreSummary, ...]
    comparisons: tuple[PairwiseScoreComparison, ...]


def fit_acoustic_clusters(
    cohort: AnalysisCohort,
    protocol: ProtocolTemplate | BoundProtocol,
    registry: FeatureRegistry,
) -> ClusteringResult:
    """Fit the frozen unsupervised pipeline without accepting or reading scores."""
    if not isinstance(cohort, AnalysisCohort):
        raise TypeError("cohort must be an AnalysisCohort")
    if cohort.purpose != "clustering":
        raise ValueError("acoustic clustering requires a clustering purpose cohort")
    if cohort.count < _MINIMUM_SAMPLE_COUNT:
        raise ValueError("acoustic clustering requires at least three samples")
    if not isinstance(registry, FeatureRegistry):
        raise TypeError("registry must be a FeatureRegistry")

    template = _protocol_template(protocol, cohort.dataset_id)
    clustering = template.clustering
    _validate_frozen_clustering_protocol(template, registry)
    rows = tuple(sorted(cohort.rows, key=lambda row: row.sample_id))
    conditional_feature_status, include_conditional = _conditional_feature_policy(
        rows
    )
    feature_names = tuple(clustering.primary_features)
    if include_conditional:
        feature_names = (*feature_names, clustering.conditional_feature)
    missing_features = tuple(
        name for name in feature_names if name not in cohort.feature_names
    )
    if missing_features:
        raise ValueError(
            f"clustering cohort is missing protocol features: {missing_features!r}"
        )
    if len(feature_names) < _MINIMUM_FEATURE_COUNT:
        raise ValueError("PCA requires at least two clustering features")

    sample_ids = tuple(_required_identifier(row.sample_id, "sample_id") for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("clustering sample IDs must be unique")
    group_ids = tuple(_required_identifier(row.group_id, "group_id") for row in rows)
    raw = _numeric_feature_matrix(rows, feature_names)
    missing = np.isnan(raw)
    medians, first_quartiles, third_quartiles, iqrs = _robust_statistics(
        raw, feature_names
    )
    imputed = np.where(missing, medians, raw)
    scales = np.where(iqrs == 0.0, 1.0, iqrs)
    scaled = (imputed - medians) / scales
    constant_features = tuple(
        name for name, iqr in zip(feature_names, iqrs, strict=True) if iqr == 0.0
    )

    distances = squareform(pdist(scaled, metric="euclidean"))
    upper = distances[np.triu_indices(len(rows), k=1)]
    nonzero = upper[upper > 0.0]
    indistinguishable = nonzero.size == 0
    pca_coordinates, pca_loadings, explained = _oriented_pca(
        scaled, indistinguishable=indistinguishable
    )

    if indistinguishable:
        similarity_scale = None
        similarities = np.full(distances.shape, 100.0, dtype=float)
        candidates: tuple[CandidateSilhouette, ...] = ()
        selected_k = None
        assignments = np.ones(len(rows), dtype=int)
        stability: tuple[FeatureStability, ...] = ()
        status: ClusteringStatus = "indistinguishable"
    else:
        similarity_scale = float(np.median(nonzero))
        similarities = 100.0 * np.exp(
            -(distances**2) / (2.0 * similarity_scale**2)
        )
        np.fill_diagonal(similarities, 100.0)
        candidate_labels: dict[int, np.ndarray] = {}
        candidate_items: list[CandidateSilhouette] = []
        maximum_k = min(5, len(rows) - 1)
        for k in range(2, maximum_k + 1):
            labels = _ward_labels(scaled, k, sample_ids)
            score = float(silhouette_score(scaled, labels, metric="euclidean"))
            candidate_labels[k] = labels
            candidate_items.append(CandidateSilhouette(k=k, silhouette=score))
        candidates = tuple(candidate_items)
        best_silhouette = max(item.silhouette for item in candidates)
        selected_k = min(
            item.k
            for item in candidates
            if item.silhouette
            >= best_silhouette - clustering.smaller_cluster_tolerance
        )
        assignments = candidate_labels[selected_k]
        stability = tuple(
            FeatureStability(
                feature_name=name,
                adjusted_rand_index=float(
                    adjusted_rand_score(
                        assignments,
                        _ward_labels(
                            np.delete(scaled, index, axis=1),
                            selected_k,
                            sample_ids,
                        ),
                    )
                ),
            )
            for index, name in enumerate(feature_names)
        )
        status = "ok"

    sample_labels = _sample_feature_labels(
        sample_ids,
        feature_names,
        imputed,
        missing,
        first_quartiles,
        third_quartiles,
        medians,
        iqrs,
    )
    descriptions = _cluster_descriptions(
        assignments, scaled, feature_names
    )
    return ClusteringResult(
        dataset_id=cohort.dataset_id,
        status=status,
        sample_ids=sample_ids,
        group_ids=group_ids,
        feature_names=feature_names,
        feature_medians=dict(zip(feature_names, medians.tolist(), strict=True)),
        feature_iqrs=dict(zip(feature_names, iqrs.tolist(), strict=True)),
        constant_features=constant_features,
        missing_indicators=_bool_matrix(missing),
        scaled_values=_float_matrix(scaled),
        pca_coordinates=tuple(
            (float(row[0]), float(row[1])) for row in pca_coordinates
        ),
        pca_loadings=(
            tuple(float(value) for value in pca_loadings[0]),
            tuple(float(value) for value in pca_loadings[1]),
        ),
        pca_explained_variance_ratio=(float(explained[0]), float(explained[1])),
        distance_matrix=_float_matrix(distances),
        similarity_matrix=_float_matrix(similarities),
        similarity_scale=similarity_scale,
        candidate_silhouettes=candidates,
        selected_k=selected_k,
        cluster_assignments=tuple(int(value) for value in assignments),
        leave_one_feature_out=stability,
        sample_feature_labels=sample_labels,
        cluster_descriptions=descriptions,
        conditional_feature_status=conditional_feature_status,
        smaller_cluster_tolerance=float(clustering.smaller_cluster_tolerance),
        permutation_repetitions=template.uncertainty.permutation_replicates,
        permutation_seed=template.random_seed,
        multiple_comparison_correction=(
            template.uncertainty.multiple_comparison_correction
        ),
    )


def posthoc_cluster_score_test(
    result: ClusteringResult,
    scores: Mapping[str, float | None] | pd.Series,
    seed: int = _DEFAULT_PERMUTATION_SEED,
) -> PosthocClusterScoreResult:
    """Compare scores only after k is frozen, permuting labels at group level."""
    if not isinstance(result, ClusteringResult):
        raise TypeError("result must be a ClusteringResult")
    if type(seed) is not int or seed != _DEFAULT_PERMUTATION_SEED:
        raise ValueError("post-hoc seed must remain frozen at 20260813")
    if (
        type(result.permutation_repetitions) is not int
        or result.permutation_repetitions != _FROZEN_PERMUTATION_REPETITIONS
    ):
        raise ValueError(
            "permutation repetitions metadata must remain frozen at 10,000"
        )
    if (
        type(result.permutation_seed) is not int
        or result.permutation_seed != _DEFAULT_PERMUTATION_SEED
    ):
        raise ValueError("permutation seed metadata must remain frozen at 20260813")
    if result.multiple_comparison_correction != "holm":
        raise ValueError("multiple-comparison correction must remain frozen at Holm")
    replicate_count = _FROZEN_PERMUTATION_REPETITIONS

    if result.selected_k is None:
        return _not_estimable_posthoc(
            result,
            reason="clusters_indistinguishable",
            sample_count=0,
            excluded_sample_count=result.sample_count,
            group_count=0,
            repetitions=replicate_count,
            seed=seed,
        )

    score_map = _score_mapping(scores)
    unknown_ids = tuple(sorted(set(score_map).difference(result.sample_ids)))
    if unknown_ids:
        raise ValueError(f"scores contain unknown sample IDs: {unknown_ids!r}")

    frozen_group_clusters: dict[str, set[int]] = {}
    for group_id, cluster_id in zip(
        result.group_ids, result.cluster_assignments, strict=True
    ):
        frozen_group_clusters.setdefault(group_id, set()).add(cluster_id)
    ambiguous_groups = tuple(
        group_id
        for group_id in sorted(frozen_group_clusters)
        if len(frozen_group_clusters[group_id]) != 1
    )
    if ambiguous_groups:
        raise ValueError(
            f"independent group {ambiguous_groups[0]!r} spans multiple clusters"
        )

    group_rows: dict[str, list[tuple[int, float]]] = {}
    included_count = 0
    for sample_id, group_id, cluster_id in zip(
        result.sample_ids,
        result.group_ids,
        result.cluster_assignments,
        strict=True,
    ):
        if sample_id not in score_map or _is_missing(score_map[sample_id]):
            continue
        score = _finite_score(score_map[sample_id])
        group_rows.setdefault(group_id, []).append((cluster_id, score))
        included_count += 1

    group_records: list[tuple[str, int, float, int]] = []
    for group_id in sorted(group_rows):
        rows = group_rows[group_id]
        cluster_ids = {row[0] for row in rows}
        if len(cluster_ids) != 1:
            raise ValueError(
                f"independent group {group_id!r} spans multiple clusters"
            )
        group_records.append(
            (
                group_id,
                next(iter(cluster_ids)),
                float(np.mean([row[1] for row in rows])),
                len(rows),
            )
        )

    clusters = range(1, result.selected_k + 1)
    group_counts = {
        cluster_id: sum(record[1] == cluster_id for record in group_records)
        for cluster_id in clusters
    }
    if any(group_counts[cluster_id] < 2 for cluster_id in clusters):
        return _not_estimable_posthoc(
            result,
            reason="fewer_than_two_groups_per_cluster",
            sample_count=included_count,
            excluded_sample_count=result.sample_count - included_count,
            group_count=len(group_records),
            repetitions=replicate_count,
            seed=seed,
        )

    labels = np.asarray([record[1] for record in group_records], dtype=int)
    group_scores = np.asarray([record[2] for record in group_records], dtype=float)
    sample_counts = {
        cluster_id: sum(
            record[3] for record in group_records if record[1] == cluster_id
        )
        for cluster_id in clusters
    }
    summaries = tuple(
        ClusterScoreSummary(
            cluster_id=cluster_id,
            group_count=group_counts[cluster_id],
            sample_count=sample_counts[cluster_id],
            mean_score=float(np.mean(group_scores[labels == cluster_id])),
            median_score=float(np.median(group_scores[labels == cluster_id])),
        )
        for cluster_id in clusters
    )

    pairs = tuple(combinations(clusters, 2))
    observed = np.asarray(
        [
            float(np.mean(group_scores[labels == first]))
            - float(np.mean(group_scores[labels == second]))
            for first, second in pairs
        ]
    )
    exceedances = np.zeros(len(pairs), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(replicate_count):
        permuted = rng.permutation(labels)
        permuted_differences = np.asarray(
            [
                float(np.mean(group_scores[permuted == first]))
                - float(np.mean(group_scores[permuted == second]))
                for first, second in pairs
            ]
        )
        exceedances += np.abs(permuted_differences) >= np.abs(observed) - 1e-15
    p_values = (exceedances.astype(float) + 1.0) / (replicate_count + 1.0)
    adjusted = _holm_adjust(p_values)
    comparisons_result = tuple(
        PairwiseScoreComparison(
            cluster_a=first,
            cluster_b=second,
            mean_difference=float(observed[index]),
            p_value=float(p_values[index]),
            adjusted_p_value=float(adjusted[index]),
        )
        for index, (first, second) in enumerate(pairs)
    )
    return PosthocClusterScoreResult(
        status="ok",
        reason=None,
        selected_k=result.selected_k,
        sample_count=included_count,
        excluded_sample_count=result.sample_count - included_count,
        group_count=len(group_records),
        analysis_unit="group",
        repetitions=replicate_count,
        seed=seed,
        multiple_comparison_correction="holm",
        exploratory=True,
        cluster_summaries=summaries,
        comparisons=comparisons_result,
    )


def export_redacted_cluster_table(
    result: ClusteringResult, path: Path
) -> Path:
    """Export one aggregate row per cluster without row IDs, groups, or scores."""
    if not isinstance(result, ClusteringResult):
        raise TypeError("result must be a ClusteringResult")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(result.pca_coordinates, dtype=float)
    assignments = np.asarray(result.cluster_assignments, dtype=int)
    descriptions = {
        item.cluster_id: item.label for item in result.cluster_descriptions
    }
    fieldnames = (
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
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id in sorted(set(result.cluster_assignments)):
            mask = assignments == cluster_id
            count = int(np.count_nonzero(mask))
            centroid = np.mean(coordinates[mask], axis=0)
            writer.writerow(
                {
                    "dataset_id": result.dataset_id,
                    "status": result.status,
                    "selected_k": (
                        "" if result.selected_k is None else result.selected_k
                    ),
                    "cluster_id": cluster_id,
                    "sample_count": count,
                    "sample_percent": f"{100.0 * count / result.sample_count:.6f}",
                    "descriptive_label": descriptions[cluster_id],
                    "pc1_centroid": f"{float(centroid[0]):.12g}",
                    "pc2_centroid": f"{float(centroid[1]):.12g}",
                }
            )
    return destination


def export_redacted_cluster_figure(
    result: ClusteringResult, path: Path
) -> Path:
    """Export aggregate PCA centroids sized by cluster count, never sample dots."""
    if not isinstance(result, ClusteringResult):
        raise TypeError("result must be a ClusteringResult")
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(result.pca_coordinates, dtype=float)
    assignments = np.asarray(result.cluster_assignments, dtype=int)
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for cluster_id in sorted(set(result.cluster_assignments)):
        mask = assignments == cluster_id
        count = int(np.count_nonzero(mask))
        centroid = np.mean(coordinates[mask], axis=0)
        axis.scatter(
            [centroid[0]],
            [centroid[1]],
            s=80.0 + 25.0 * count,
            label=f"Cluster {cluster_id} (n={count})",
        )
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.axvline(0.0, color="#999999", linewidth=0.7)
    axis.set_xlabel(
        f"PC1 ({100.0 * result.pca_explained_variance_ratio[0]:.1f}% variance)"
    )
    axis.set_ylabel(
        f"PC2 ({100.0 * result.pca_explained_variance_ratio[1]:.1f}% variance)"
    )
    axis.set_title(f"Aggregate acoustic cluster centroids; n={result.sample_count}")
    axis.legend()
    figure.savefig(destination, dpi=160, metadata={"Title": "Redacted clustering"})
    plt.close(figure)
    return destination


def _protocol_template(
    protocol: ProtocolTemplate | BoundProtocol, dataset_id: str
) -> ProtocolTemplate:
    if isinstance(protocol, BoundProtocol):
        if protocol.dataset_id != dataset_id:
            raise ValueError("bound protocol dataset ID does not match the cohort")
        return protocol.template
    if isinstance(protocol, ProtocolTemplate):
        return protocol
    raise TypeError("protocol must be a ProtocolTemplate or BoundProtocol")


def _validate_frozen_clustering_protocol(
    template: ProtocolTemplate, registry: FeatureRegistry
) -> None:
    clustering = template.clustering
    if tuple(clustering.candidate_cluster_counts) != _FROZEN_CANDIDATE_COUNTS:
        raise ValueError("candidate cluster counts must remain frozen at 2, 3, 4, 5")
    if not math.isclose(
        clustering.smaller_cluster_tolerance, 0.05, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("smaller-cluster tolerance must remain frozen at 0.05")
    for name in clustering.primary_features:
        definition = registry.by_name.get(name)
        if definition is None:
            raise ValueError(
                f"protocol clustering feature {name!r} is absent from registry"
            )
        if definition.clustering_role != "primary":
            raise ValueError(f"clustering feature {name!r} is not registry-primary")
        if definition.dtype not in {"float", "integer"}:
            raise ValueError(f"clustering feature {name!r} must be numeric")
    registry_primary = tuple(item.name for item in registry.primary_clustering)
    if frozenset(registry_primary) != frozenset(clustering.primary_features):
        raise ValueError(
            "protocol primary features must exactly match registry membership"
        )
    conditional = registry.by_name.get(clustering.conditional_feature)
    if conditional is None:
        raise ValueError(
            "protocol conditional clustering feature is absent from registry"
        )
    if conditional.clustering_role != "conditional_same_language":
        raise ValueError("protocol conditional feature has the wrong registry role")
    if conditional.dtype not in {"float", "integer"}:
        raise ValueError("protocol conditional clustering feature must be numeric")


def _conditional_feature_policy(rows: Sequence[object]) -> tuple[str, bool]:
    successful = tuple(
        row for row in rows if getattr(row, "transcript_status", None) == "succeeded"
    )
    if not successful:
        return "excluded_no_successful_asr_rows", False
    languages = tuple(
        getattr(row, "whisper_language_code", None) for row in successful
    )
    if any(not isinstance(language, str) or not language for language in languages):
        return "excluded_missing_whisper_language", False
    if len(set(languages)) != 1:
        return "excluded_mixed_whisper_languages", False
    return "included_single_whisper_language", True


def _required_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"clustering {label} must be nonblank text")
    return value


def _numeric_feature_matrix(
    rows: Sequence[object], feature_names: Sequence[str]
) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_names)), dtype=float)
    for row_index, row in enumerate(rows):
        features = getattr(row, "features", None)
        if not isinstance(features, Mapping):
            raise ValueError("clustering rows must expose a feature mapping")
        for column_index, name in enumerate(feature_names):
            if name not in features:
                raise ValueError(f"clustering row is missing feature {name!r}")
            value = features[name]
            if _is_missing(value):
                matrix[row_index, column_index] = np.nan
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (Real, np.number)
            ):
                raise ValueError(f"clustering feature {name!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("clustering feature values must be finite or missing")
            matrix[row_index, column_index] = numeric
    return matrix


def _robust_statistics(
    values: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = np.empty(values.shape[1], dtype=float)
    first_quartiles = np.empty(values.shape[1], dtype=float)
    third_quartiles = np.empty(values.shape[1], dtype=float)
    for index, name in enumerate(feature_names):
        observed = values[:, index][~np.isnan(values[:, index])]
        if observed.size == 0:
            raise ValueError(f"clustering feature {name!r} is entirely missing")
        medians[index] = float(np.median(observed))
        first_quartiles[index], third_quartiles[index] = np.percentile(
            observed, [25.0, 75.0]
        )
    iqrs = third_quartiles - first_quartiles
    return medians, first_quartiles, third_quartiles, iqrs


def _oriented_pca(
    scaled: np.ndarray, *, indistinguishable: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indistinguishable:
        loadings = np.zeros((2, scaled.shape[1]), dtype=float)
        loadings[0, 0] = 1.0
        loadings[1, 1] = 1.0
        return (
            np.zeros((scaled.shape[0], 2), dtype=float),
            loadings,
            np.zeros(2, dtype=float),
        )
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(scaled)
    loadings = pca.components_.copy()
    for component in range(2):
        anchor = int(np.argmax(np.abs(loadings[component])))
        if loadings[component, anchor] < 0.0:
            loadings[component] *= -1.0
            coordinates[:, component] *= -1.0
    coordinates[np.abs(coordinates) < 1e-15] = 0.0
    loadings[np.abs(loadings) < 1e-15] = 0.0
    explained = np.nan_to_num(
        pca.explained_variance_ratio_, nan=0.0, posinf=0.0, neginf=0.0
    )
    explained[np.abs(explained) < 1e-15] = 0.0
    return coordinates, loadings, explained


def _ward_labels(
    values: np.ndarray, k: int, sample_ids: Sequence[str]
) -> np.ndarray:
    raw = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward",
    ).fit_predict(values)
    ordered_labels = sorted(
        set(int(value) for value in raw),
        key=lambda label: (
            tuple(float(value) for value in np.mean(values[raw == label], axis=0)),
            min(
                sample_id
                for sample_id, value in zip(sample_ids, raw, strict=True)
                if value == label
            ),
        ),
    )
    remapping = {label: index + 1 for index, label in enumerate(ordered_labels)}
    return np.asarray([remapping[int(value)] for value in raw], dtype=int)


def _sample_feature_labels(
    sample_ids: Sequence[str],
    feature_names: Sequence[str],
    imputed: np.ndarray,
    missing: np.ndarray,
    first_quartiles: np.ndarray,
    third_quartiles: np.ndarray,
    medians: np.ndarray,
    iqrs: np.ndarray,
) -> tuple[SampleFeatureLabel, ...]:
    labels: list[SampleFeatureLabel] = []
    for row_index, sample_id in enumerate(sample_ids):
        for column_index, feature_name in enumerate(feature_names):
            was_missing = bool(missing[row_index, column_index])
            if iqrs[column_index] == 0.0:
                robust_z = 0.0
            else:
                robust_z = float(
                    (imputed[row_index, column_index] - medians[column_index])
                    / iqrs[column_index]
                )
            if was_missing:
                category: SampleLabelCategory = "median_imputed"
            elif iqrs[column_index] == 0.0:
                category = "typical"
            elif robust_z <= _VERY_LOW_Z:
                category = "very_low"
            elif robust_z >= _VERY_HIGH_Z:
                category = "very_high"
            elif imputed[row_index, column_index] <= first_quartiles[column_index]:
                category = "low"
            elif imputed[row_index, column_index] >= third_quartiles[column_index]:
                category = "high"
            else:
                category = "typical"
            labels.append(
                SampleFeatureLabel(
                    sample_id=sample_id,
                    feature_name=feature_name,
                    category=category,
                    robust_z=robust_z,
                    was_missing=was_missing,
                )
            )
    return tuple(labels)


def _cluster_descriptions(
    assignments: np.ndarray,
    scaled: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[ClusterDescription, ...]:
    overall_medians = np.median(scaled, axis=0)
    descriptions: list[ClusterDescription] = []
    for cluster_id in sorted(set(int(value) for value in assignments)):
        differences = (
            np.median(scaled[assignments == cluster_id], axis=0) - overall_medians
        )
        feature_differences = tuple(
            FeatureMedianDifference(feature_name=name, difference=float(difference))
            for name, difference in zip(feature_names, differences, strict=True)
        )
        anchor = int(np.argmax(np.abs(differences)))
        difference = float(differences[anchor])
        if abs(difference) <= 1e-15:
            label = "typical acoustic profile"
        else:
            direction = "higher" if difference > 0.0 else "lower"
            label = f"{direction} {feature_names[anchor].replace('_', ' ')}"
        descriptions.append(
            ClusterDescription(
                cluster_id=cluster_id,
                label=label,
                feature_differences=feature_differences,
            )
        )
    return tuple(descriptions)


def _score_mapping(
    scores: Mapping[str, float | None] | pd.Series,
) -> dict[str, object]:
    if isinstance(scores, pd.Series):
        if not scores.index.is_unique:
            raise ValueError("score sample IDs must be unique")
        items = scores.items()
    elif isinstance(scores, Mapping):
        items = scores.items()
    else:
        raise TypeError("scores must be a sample-ID mapping or pandas Series")
    output: dict[str, object] = {}
    for sample_id, value in items:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("score sample IDs must be nonblank text")
        output[sample_id] = value
    return output


def _finite_score(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.number)):
        raise ValueError("score values must be numeric, finite or missing")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("score values must be finite or missing")
    return numeric


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _not_estimable_posthoc(
    result: ClusteringResult,
    *,
    reason: str,
    sample_count: int,
    excluded_sample_count: int,
    group_count: int,
    repetitions: int,
    seed: int,
) -> PosthocClusterScoreResult:
    return PosthocClusterScoreResult(
        status="not_estimable",
        reason=reason,
        selected_k=result.selected_k,
        sample_count=sample_count,
        excluded_sample_count=excluded_sample_count,
        group_count=group_count,
        analysis_unit="group",
        repetitions=repetitions,
        seed=seed,
        multiple_comparison_correction="holm",
        exploratory=True,
        cluster_summaries=(),
        comparisons=(),
    )


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _float_matrix(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _bool_matrix(values: np.ndarray) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(value) for value in row) for row in values)


__all__ = [
    "CandidateSilhouette",
    "ClusterDescription",
    "ClusterScoreSummary",
    "ClusteringResult",
    "FeatureMedianDifference",
    "FeatureStability",
    "PairwiseScoreComparison",
    "PosthocClusterScoreResult",
    "SampleFeatureLabel",
    "export_redacted_cluster_figure",
    "export_redacted_cluster_table",
    "fit_acoustic_clusters",
    "posthoc_cluster_score_test",
]
