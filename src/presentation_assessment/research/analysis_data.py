"""Strict, privacy-minimised analysis views and fold-local preprocessing."""

from __future__ import annotations

import csv
import json
import math
import re
from numbers import Real
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import dataset_pipeline
from .feature_registry import FeatureDefinition, load_feature_registry
from .protocol import BoundProtocol


AnalysisPurpose = Literal[
    "quality_control",
    "clustering",
    "supervised_total_score",
    "transcript_model",
]
AllNullFeaturePolicy = Literal["exclude"]
FeatureValue = float | int | bool | str | None

_PURPOSES: tuple[AnalysisPurpose, ...] = (
    "quality_control",
    "clustering",
    "supervised_total_score",
    "transcript_model",
)
_MISSING_INDICATOR_SUFFIX = "__missing"
_AUDIT_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_MEASURED_STABILITY_WARNING = re.compile(
    r"^(?P<code>stability_[a-z0-9_]+)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) exceeds "
    r"(?P<threshold>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_WHISPER_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """One immutable joined row without URL, title, or transcript content.

    ``asr_error_code`` is a sanitized QC audit code from the quality table. It
    contains neither transcript text nor an ASR error summary.
    ``whisper_language_code`` is a validated short code; language probability
    and all transcript content remain outside the analysis view.
    """

    sample_id: str
    total_score: float | None
    score_status: str
    terminal_status: str
    acquisition_status: str
    audio_status: str
    transcript_status: str
    feature_status: str
    transcript_available: bool
    whisper_language_code: str | None
    group_id: str
    outer_fold: int
    inner_fold: int
    speaker_id: str
    talk_id: str
    modelling_scope: str
    supervised_eligible: bool
    qc_eligible: bool
    clustering_eligible: bool
    qc_warnings: tuple[str, ...]
    acquisition_error_code: str | None
    audio_error_code: str | None
    asr_error_code: str | None
    feature_error_code: str | None
    missing_feature_count: int
    features: Mapping[str, FeatureValue]
    missing_reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(
            self, "missing_reasons", MappingProxyType(dict(self.missing_reasons))
        )

    @property
    def feature_values(self) -> Mapping[str, FeatureValue]:
        """Compatibility alias naming the registry-bound feature values explicitly."""
        return self.features

    @property
    def asr_status(self) -> str:
        """Alias matching the frozen stage name without exposing transcript content."""
        return self.transcript_status


@dataclass(frozen=True, slots=True)
class CohortExclusion:
    """Explicit reasons a validated sample is absent from one analysis cohort."""

    sample_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    """Immutable joined view of one strictly validated frozen dataset."""

    dataset_id: str
    protocol: BoundProtocol
    feature_names: tuple[str, ...]
    rows: tuple[AnalysisRow, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def feature_frame(self) -> pd.DataFrame:
        """Return a fresh feature projection indexed by validated sample ID."""
        return _feature_frame(self.rows, self.feature_names)


@dataclass(frozen=True, slots=True)
class AnalysisCohort:
    """Immutable eligible rows plus an audit trail for every excluded sample."""

    dataset_id: str
    purpose: AnalysisPurpose
    feature_names: tuple[str, ...]
    rows: tuple[AnalysisRow, ...]
    exclusions: tuple[CohortExclusion, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.rows)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def exclusion_reasons(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {item.sample_id: item.reasons for item in self.exclusions}
        )

    def feature_frame(self) -> pd.DataFrame:
        """Return a fresh feature projection for eligible rows only."""
        return _feature_frame(self.rows, self.feature_names)

    def target_series(self) -> pd.Series:
        """Return a fresh total-score series indexed by validated sample ID."""
        return pd.Series(
            [row.total_score for row in self.rows],
            index=pd.Index(self.sample_ids, name="sample_id"),
            name="total_score",
            dtype="float64",
        )


def load_analysis_dataset(dataset_dir: Path, registry_path: Path) -> AnalysisDataset:
    """Load private frozen tables only after the single strict-loader gate succeeds."""
    root = Path(dataset_dir)
    registry = load_feature_registry(Path(registry_path))
    loaded = dataset_pipeline.load_strict_frozen_dataset(root, registry)

    manifest_rows = _read_csv(root / "manifest.csv", "manifest")
    quality_rows = _read_csv(root / "quality_report.csv", "quality")
    transcript_rows = _read_json_lines(root / "transcripts.jsonl")
    feature_frame = pd.read_parquet(root / "acoustic_features.parquet")
    split_payload = _read_json(root / "splits.json", "split registry")
    raw_assignments = split_payload.get("assignments")
    if not isinstance(raw_assignments, list) or not all(
        isinstance(item, dict) for item in raw_assignments
    ):
        raise ValueError("invalid split assignments")

    feature_names = tuple(definition.name for definition in registry.features)
    if list(feature_frame.columns) != ["sample_id", *feature_names]:
        raise ValueError("analysis feature schema does not exactly match the registry")

    indexed = {
        "manifest": _index_rows(manifest_rows, "manifest"),
        "quality": _index_rows(quality_rows, "quality"),
        "transcripts": _index_rows(transcript_rows, "transcripts"),
        "features": _index_rows(feature_frame.to_dict("records"), "features"),
        "splits": _index_rows(raw_assignments, "splits"),
    }
    sample_ids = frozenset(indexed["manifest"])
    if any(frozenset(rows) != sample_ids for rows in indexed.values()):
        raise ValueError("analysis tables must contain exactly equal sample_id sets")
    if sample_ids != loaded.sample_ids or len(sample_ids) != loaded.row_count:
        raise ValueError("analysis sample IDs do not match the strict frozen dataset")

    rows = tuple(
        _join_row(
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


def select_cohort(dataset: AnalysisDataset, purpose: AnalysisPurpose) -> AnalysisCohort:
    """Select one purpose-specific cohort and retain every exclusion reason."""
    if purpose not in _PURPOSES:
        raise ValueError(f"unsupported analysis purpose: {purpose!r}")
    included: list[AnalysisRow] = []
    exclusions: list[CohortExclusion] = []
    for row in dataset.rows:
        reasons = _exclusion_reasons(row, purpose)
        if reasons:
            exclusions.append(CohortExclusion(row.sample_id, reasons))
        else:
            included.append(row)
    return AnalysisCohort(
        dataset_id=dataset.dataset_id,
        purpose=purpose,
        feature_names=dataset.feature_names,
        rows=tuple(included),
        exclusions=tuple(exclusions),
    )


class RobustFoldPreprocessor:
    """Training-only feature selection, imputation, and robust scaling."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str] | None = None,
        missing_indicator_features: Sequence[str] = (),
        all_null_feature_policy: AllNullFeaturePolicy = "exclude",
    ) -> None:
        if all_null_feature_policy != "exclude":
            raise ValueError("all_null_feature_policy must be 'exclude'")
        self._requested_feature_names = _optional_names(feature_names, "feature_names")
        self._requested_indicators = _names(
            missing_indicator_features, "missing_indicator_features"
        )
        self.all_null_feature_policy = all_null_feature_policy

    def fit(self, training: pd.DataFrame) -> "RobustFoldPreprocessor":
        """Fit all statistics from training rows and no other partition."""
        frame = _require_frame(training)
        if self._requested_feature_names is None:
            feature_names = tuple(
                name
                for name in frame.columns
                if not name.endswith(_MISSING_INDICATOR_SUFFIX)
            )
        else:
            feature_names = self._requested_feature_names
        if not feature_names:
            raise ValueError("at least one model feature is required")
        missing_columns = tuple(name for name in feature_names if name not in frame.columns)
        if missing_columns:
            raise ValueError(f"training frame is missing model features: {missing_columns!r}")
        unknown_indicators = tuple(
            name for name in self._requested_indicators if name not in feature_names
        )
        if unknown_indicators:
            raise ValueError(
                "missing indicator features must name fitted model features: "
                f"{unknown_indicators!r}"
            )

        numeric = _numeric_values(frame, feature_names, "training")
        excluded = tuple(
            name
            for index, name in enumerate(feature_names)
            if not np.any(~np.isnan(numeric[:, index]))
        )
        excluded_set = frozenset(excluded)
        retained = tuple(name for name in feature_names if name not in excluded_set)
        if not retained:
            raise ValueError("all selected features have no finite training values")

        positions = {name: index for index, name in enumerate(feature_names)}
        locations: dict[str, float] = {}
        scales: dict[str, float] = {}
        constants: list[str] = []
        for name in retained:
            values = numeric[:, positions[name]]
            observed = values[~np.isnan(values)]
            location = float(np.median(observed))
            first_quartile, third_quartile = np.percentile(observed, [25.0, 75.0])
            scale = float(third_quartile - first_quartile)
            if scale == 0.0:
                scale = 1.0
                constants.append(name)
            locations[name] = location
            scales[name] = scale

        self.feature_names_in_ = feature_names
        self.feature_names_ = retained
        self.excluded_all_null_features_ = excluded
        self.feature_names_out_ = (
            *retained,
            *(f"{name}{_MISSING_INDICATOR_SUFFIX}" for name in self._requested_indicators),
        )
        self.location_ = MappingProxyType(locations)
        self.scale_ = MappingProxyType(scales)
        self.constant_features_ = tuple(constants)
        return self

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training statistics without refitting on these rows."""
        self._require_fitted()
        frame = _require_frame(values)
        required_names = tuple(
            dict.fromkeys((*self.feature_names_, *self._requested_indicators))
        )
        missing_columns = tuple(
            name for name in required_names if name not in frame.columns
        )
        if missing_columns:
            raise ValueError(f"transform frame is missing model features: {missing_columns!r}")
        numeric = _numeric_values(frame, self.feature_names_, "transform")
        missing = np.isnan(numeric)
        transformed = np.empty_like(numeric, dtype=float)
        for index, name in enumerate(self.feature_names_):
            imputed = np.where(missing[:, index], self.location_[name], numeric[:, index])
            transformed[:, index] = (
                imputed - self.location_[name]
            ) / self.scale_[name]
        result = pd.DataFrame(
            transformed,
            index=frame.index.copy(),
            columns=self.feature_names_,
        )
        for name in self._requested_indicators:
            result[f"{name}{_MISSING_INDICATOR_SUFFIX}"] = (
                frame[name].isna().to_numpy(dtype=float)
            )
        return result.loc[:, self.feature_names_out_]

    def fit_transform(self, training: pd.DataFrame) -> pd.DataFrame:
        """Fit on and transform the same training partition."""
        return self.fit(training).transform(training)

    def audit_missing_indicators(self, values: pd.DataFrame) -> pd.DataFrame:
        """Return missingness flags separately from default model inputs."""
        self._require_fitted()
        frame = _require_frame(values)
        missing_columns = tuple(
            name for name in self.feature_names_in_ if name not in frame.columns
        )
        if missing_columns:
            raise ValueError(f"audit frame is missing model features: {missing_columns!r}")
        numeric = _numeric_values(frame, self.feature_names_in_, "audit")
        return pd.DataFrame(
            np.isnan(numeric),
            index=frame.index.copy(),
            columns=[
                f"{name}{_MISSING_INDICATOR_SUFFIX}"
                for name in self.feature_names_in_
            ],
        )

    def _require_fitted(self) -> None:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("RobustFoldPreprocessor must be fitted before use")


def _read_csv(path: Path, label: str) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"invalid {label} CSV header")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"unable to read {label} table") from error


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("unable to read transcript status table") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("invalid transcript status rows")
    return rows


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {label}")
    return payload


def _index_rows(
    rows: Sequence[Mapping[str, object]], label: str
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise ValueError(f"{label} rows require a valid sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} rows must have unique sample_id values")
        indexed[sample_id] = row
    return indexed


def _join_row(
    sample_id: str,
    *,
    manifest: Mapping[str, object],
    quality: Mapping[str, object],
    transcript: Mapping[str, object],
    features: Mapping[str, object],
    assignment: Mapping[str, object],
    definitions: Sequence[FeatureDefinition],
) -> AnalysisRow:
    feature_status = _required_text(manifest, "feature_status", "manifest")
    transcript_status = _required_text(transcript, "status", "transcripts")
    if transcript_status != _required_text(manifest, "asr_status", "manifest"):
        raise ValueError("analysis transcript status does not match the manifest")
    if quality.get("terminal_status") != manifest.get("terminal_status"):
        raise ValueError("analysis quality status does not match the manifest")

    raw_reasons = quality.get("missing_reasons_json")
    try:
        reasons = json.loads(raw_reasons) if isinstance(raw_reasons, str) else None
    except json.JSONDecodeError as error:
        raise ValueError("invalid per-feature missing reason mapping") from error
    if not isinstance(reasons, dict) or not all(
        isinstance(name, str) and isinstance(reason, str)
        for name, reason in reasons.items()
    ):
        raise ValueError("invalid per-feature missing reason mapping")

    values: dict[str, FeatureValue] = {}
    null_names: set[str] = set()
    definitions_by_name = {definition.name: definition for definition in definitions}
    for definition in definitions:
        raw_value = features.get(definition.name)
        if _is_null(raw_value):
            null_names.add(definition.name)
            reason = reasons.get(definition.name)
            if not isinstance(reason, str):
                raise ValueError("null analysis feature is missing a declared missing reason")
            if feature_status == "succeeded":
                if not definition.missing_policy.allow_missing:
                    raise ValueError("required analysis feature cannot be null")
                if reason not in definition.missing_policy.reason_codes:
                    raise ValueError("feature missing reason is not declared by the registry")
            elif reason != "feature_stage_failed":
                raise ValueError("failed feature stages require feature_stage_failed reasons")
            values[definition.name] = None
            continue
        if definition.name in reasons:
            raise ValueError("non-null analysis feature has a missing reason")
        values[definition.name] = _validate_feature_value(definition, raw_value)
    if set(reasons) != null_names or any(name not in definitions_by_name for name in reasons):
        raise ValueError("missing reason keys must exactly match null analysis features")
    try:
        missing_feature_count = int(str(quality.get("missing_feature_count")))
    except ValueError as error:
        raise ValueError("invalid analysis missing feature count") from error
    if missing_feature_count != len(null_names):
        raise ValueError("analysis missing feature count does not match reasons")

    score_status = _required_text(manifest, "score_status", "manifest")
    raw_score = manifest.get("total_score")
    if raw_score in (None, ""):
        total_score = None
        if score_status == "observed":
            raise ValueError("observed total score cannot be null")
    else:
        try:
            total_score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError("total score must be numeric") from error
        if not math.isfinite(total_score):
            raise ValueError("total score must be finite")
        if score_status != "observed":
            raise ValueError("non-observed score status cannot contain a total score")

    transcription = transcript.get("transcription")
    transcript_text = (
        transcription.get("text") if isinstance(transcription, Mapping) else None
    )
    transcript_available = bool(
        transcript_status == "succeeded"
        and isinstance(transcript_text, str)
        and transcript_text.strip()
    )
    whisper_language_code = _whisper_language_code(
        transcription.get("language") if isinstance(transcription, Mapping) else None,
        transcript_status=transcript_status,
    )
    warnings = _warnings(quality.get("qc_warnings"))
    return AnalysisRow(
        sample_id=sample_id,
        total_score=total_score,
        score_status=score_status,
        terminal_status=_required_text(manifest, "terminal_status", "manifest"),
        acquisition_status=_required_text(manifest, "acquisition_status", "manifest"),
        audio_status=_required_text(manifest, "audio_status", "manifest"),
        transcript_status=transcript_status,
        feature_status=feature_status,
        transcript_available=transcript_available,
        whisper_language_code=whisper_language_code,
        group_id=_required_text(assignment, "group_id", "splits"),
        outer_fold=_integer(assignment, "outer_fold", "splits"),
        inner_fold=_integer(assignment, "inner_fold", "splits"),
        speaker_id=_required_text(assignment, "speaker_id", "splits"),
        talk_id=_required_text(assignment, "talk_id", "splits"),
        modelling_scope=_required_text(assignment, "modelling_scope", "splits"),
        supervised_eligible=_boolean(assignment, "supervised_eligible", "splits"),
        qc_eligible=_boolean(assignment, "qc_eligible", "splits"),
        clustering_eligible=_boolean(assignment, "clustering_eligible", "splits"),
        qc_warnings=warnings,
        acquisition_error_code=_optional_code(
            quality, "acquisition_error_code", "quality"
        ),
        audio_error_code=_optional_code(quality, "audio_error_code", "quality"),
        asr_error_code=_optional_code(quality, "asr_error_code", "quality"),
        feature_error_code=_optional_code(quality, "feature_error_code", "quality"),
        missing_feature_count=missing_feature_count,
        features=values,
        missing_reasons=reasons,
    )


def _required_text(row: Mapping[str, object], name: str, label: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} field {name!r} must be nonblank text")
    return value


def _optional_code(
    row: Mapping[str, object], name: str, label: str
) -> str | None:
    value = row.get(name)
    if value in (None, ""):
        return None
    if not isinstance(value, str) or _AUDIT_CODE.fullmatch(value) is None:
        raise ValueError(f"{label} field {name!r} must be a safe audit code")
    return value


def _whisper_language_code(value: object, *, transcript_status: str) -> str | None:
    if transcript_status != "succeeded" or value in (None, ""):
        return None
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _WHISPER_LANGUAGE_CODE.fullmatch(value) is None
    ):
        raise ValueError("Whisper language must be a safe lowercase language code")
    return value


def _integer(row: Mapping[str, object], name: str, label: str) -> int:
    value = row.get(name)
    if type(value) is not int:
        raise ValueError(f"{label} field {name!r} must be an integer")
    return value


def _boolean(row: Mapping[str, object], name: str, label: str) -> bool:
    value = row.get(name)
    if type(value) is not bool:
        raise ValueError(f"{label} field {name!r} must be boolean")
    return value


def _warnings(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, str):
        raise ValueError("QC warnings must be a semicolon-delimited string")
    warnings: list[str] = []
    for item in (part for part in value.split(";") if part):
        if _AUDIT_CODE.fullmatch(item) is not None:
            warnings.append(item)
            continue
        measured = _MEASURED_STABILITY_WARNING.fullmatch(item)
        if measured is None or not all(
            math.isfinite(float(measured.group(name)))
            for name in ("value", "threshold")
        ):
            raise ValueError("QC warnings must contain safe audit codes")
        warnings.append(measured.group("code"))
    return tuple(warnings)


def _is_null(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _validate_feature_value(
    definition: FeatureDefinition, value: object
) -> FeatureValue:
    if definition.dtype == "boolean":
        if type(value) not in (bool, np.bool_):
            raise ValueError("analysis boolean feature has the wrong type")
        return bool(value)
    if definition.dtype == "string":
        if not isinstance(value, str):
            raise ValueError("analysis string feature has the wrong type")
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("analysis numeric feature has the wrong type") from error
    if not math.isfinite(numeric):
        raise ValueError("analysis numeric feature must be finite")
    if definition.dtype == "integer" and not numeric.is_integer():
        raise ValueError("analysis integer feature has the wrong type")
    if definition.minimum is not None and numeric < float(definition.minimum) - 1e-12:
        raise ValueError("analysis feature is below its registry minimum")
    if definition.maximum is not None and numeric > float(definition.maximum) + 1e-12:
        raise ValueError("analysis feature exceeds its registry maximum")
    return int(numeric) if definition.dtype == "integer" else numeric


def _exclusion_reasons(
    row: AnalysisRow, purpose: AnalysisPurpose
) -> tuple[str, ...]:
    reasons: list[str] = []
    if purpose == "quality_control":
        if row.feature_status != "succeeded":
            reasons.append("feature_status_not_succeeded")
        if not row.qc_eligible:
            reasons.append("qc_not_eligible")
    elif purpose == "clustering":
        if row.feature_status != "succeeded":
            reasons.append("feature_status_not_succeeded")
        if not row.qc_eligible:
            reasons.append("qc_not_eligible")
        if not row.clustering_eligible:
            reasons.append("clustering_not_eligible")
    elif purpose == "supervised_total_score":
        if row.feature_status != "succeeded":
            reasons.append("feature_status_not_succeeded")
        if row.score_status != "observed":
            reasons.append("score_not_observed")
        if row.total_score is None:
            reasons.append("total_score_missing")
        if not row.supervised_eligible:
            reasons.append("supervised_not_eligible")
    else:
        if row.transcript_status != "succeeded":
            reasons.append("transcript_status_not_succeeded")
        if not row.transcript_available:
            reasons.append("transcript_unavailable")
    return tuple(reasons)


def _feature_frame(
    rows: Sequence[AnalysisRow], feature_names: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [{name: row.features[name] for name in feature_names} for row in rows],
        index=pd.Index([row.sample_id for row in rows], name="sample_id"),
        columns=tuple(feature_names),
    )


def _optional_names(
    values: Sequence[str] | None, label: str
) -> tuple[str, ...] | None:
    return None if values is None else _names(values, label)


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of names")
    names = tuple(values)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain nonblank strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must contain unique names")
    return names


def _require_frame(values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("preprocessing input must be a pandas DataFrame")
    if values.empty:
        raise ValueError("preprocessing input must contain at least one row and column")
    if not values.columns.is_unique:
        raise ValueError("preprocessing columns must be unique")
    if any(not isinstance(name, str) or not name for name in values.columns):
        raise ValueError("preprocessing columns must have nonblank string names")
    return values


def _numeric_values(
    frame: pd.DataFrame, feature_names: Sequence[str], label: str
) -> np.ndarray:
    for name in feature_names:
        series = frame[name]
        observed = series[~series.isna()]
        if not is_numeric_dtype(series.dtype) and any(
            not isinstance(value, (Real, np.number)) for value in observed
        ):
            raise ValueError(f"{label} feature {name!r} must be numeric")
    numeric = frame.loc[:, feature_names].to_numpy(dtype=float, copy=True)
    if np.isinf(numeric).any():
        raise ValueError(f"{label} feature values must be finite or null")
    return numeric


__all__ = [
    "AllNullFeaturePolicy",
    "AnalysisCohort",
    "AnalysisDataset",
    "AnalysisPurpose",
    "AnalysisRow",
    "CohortExclusion",
    "RobustFoldPreprocessor",
    "load_analysis_dataset",
    "select_cohort",
]
