"""Strict schema and loader for the frozen acoustic feature registry."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


FeatureFamily = Literal[
    "recording_quality",
    "vad_pause",
    "rhythm",
    "prosody",
    "spectral",
    "transient",
    "normalization_stability",
]
SourceRepresentation = Literal[
    "source_audio_metadata",
    "analysis_wav",
    "analysis_wav_and_vad",
    "analysis_wav_and_asr",
    "analysis_wav_vad_and_asr",
    "all_wav_representations",
]
FeatureDtype = Literal["float", "integer", "boolean", "string"]
ClusteringRole = Literal["primary", "conditional_same_language", "audit_only"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class MissingPolicy(_StrictModel):
    """Whether a missing measurement is valid and how its cause is recorded."""

    allow_missing: bool
    reason_required: bool
    reason_codes: tuple[str, ...]

    @model_validator(mode="after")
    def require_declared_reasons_for_nullable_values(self) -> "MissingPolicy":
        if self.allow_missing and (not self.reason_required or not self.reason_codes):
            raise ValueError("nullable features must require at least one missing reason code")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("missing reason codes must be unique")
        return self


class FeatureDefinition(_StrictModel):
    """One auditable scalar output in the approved acoustic feature map."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    family: FeatureFamily
    unit: str = Field(min_length=1)
    source_representation: SourceRepresentation
    dtype: FeatureDtype
    minimum: float | int | None = None
    maximum: float | int | None = None
    definition: str = Field(min_length=1)
    missing_policy: MissingPolicy
    clustering_role: ClusteringRole
    warning_threshold: float | None = None

    @model_validator(mode="after")
    def validate_numeric_contract(self) -> "FeatureDefinition":
        if self.dtype in {"boolean", "string"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("non-numeric features cannot declare numeric bounds")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("feature minimum cannot exceed maximum")
        if self.warning_threshold is not None and self.warning_threshold < 0:
            raise ValueError("warning threshold cannot be negative")
        return self


class FeatureRegistry(_StrictModel):
    """Versioned collection of unique acoustic feature definitions."""

    schema_version: Literal[1]
    specification: str = Field(min_length=1)
    features: tuple[FeatureDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_feature_names(self) -> "FeatureRegistry":
        names = [item.name for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        return self

    @cached_property
    def by_name(self) -> Mapping[str, FeatureDefinition]:
        return MappingProxyType({item.name: item for item in self.features})

    @cached_property
    def primary_clustering(self) -> tuple[FeatureDefinition, ...]:
        return tuple(item for item in self.features if item.clustering_role == "primary")

    @cached_property
    def conditional_clustering(self) -> tuple[FeatureDefinition, ...]:
        return tuple(
            item
            for item in self.features
            if item.clustering_role == "conditional_same_language"
        )

    @cached_property
    def audit_only(self) -> tuple[FeatureDefinition, ...]:
        return tuple(item for item in self.features if item.clustering_role == "audit_only")


def load_feature_registry(path: Path) -> FeatureRegistry:
    """Load YAML without coercion, rejecting unknown or malformed fields."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Unable to read feature registry: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("feature registry root must be a mapping")
    return FeatureRegistry.model_validate_json(json.dumps(raw, ensure_ascii=False))
