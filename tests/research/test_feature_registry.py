from pathlib import Path

import pytest

from presentation_assessment.research import (
    FeatureDefinition,
    FeatureRegistry,
    load_feature_registry,
)


REGISTRY_PATH = Path("configs/audio_feature_registry.v1.yaml")
EXPECTED_PRIMARY_NAMES = {
    "speech_ratio",
    "internal_pauses_per_speech_minute",
    "internal_pause_duration_median_seconds",
    "speech_segment_duration_median_seconds",
    "acoustic_envelope_peaks_per_speech_minute",
    "envelope_peak_interval_cv",
    "pitch_p10_p90_range_semitones",
    "pitch_std_semitones",
    "rms_robust_cv",
    "spectral_flux_dispersion",
    "candidate_transients_per_speech_minute",
}


def test_registry_types_are_part_of_the_research_public_interface() -> None:
    registry = load_feature_registry(REGISTRY_PATH)

    assert isinstance(registry, FeatureRegistry)
    assert all(isinstance(item, FeatureDefinition) for item in registry.features)


def test_registry_names_are_unique_and_primary_features_are_declared() -> None:
    registry = load_feature_registry(REGISTRY_PATH)

    assert len(registry.by_name) == len(registry.features)
    assert {item.name for item in registry.primary_clustering} == EXPECTED_PRIMARY_NAMES
    assert {item.name for item in registry.conditional_clustering} == {
        "asr_units_per_speech_minute"
    }


def test_registry_name_lookup_cannot_be_mutated_by_callers() -> None:
    registry = load_feature_registry(REGISTRY_PATH)

    with pytest.raises(TypeError):
        registry.by_name["speech_ratio"] = registry.features[0]  # type: ignore[index]

    assert registry.by_name["speech_ratio"].name == "speech_ratio"


def test_speech_minute_features_declare_denominator_failure_reasons() -> None:
    registry = load_feature_registry(REGISTRY_PATH)
    denominator_reasons = {
        "vad_failed",
        "no_speech_detected",
        "zero_speech_duration",
    }

    for name in (
        "speech_segments_per_speech_minute",
        "internal_pauses_per_speech_minute",
        "acoustic_envelope_peaks_per_speech_minute",
        "candidate_transients_per_speech_minute",
    ):
        assert set(registry.by_name[name].missing_policy.reason_codes) == denominator_reasons

    assert set(
        registry.by_name["asr_units_per_speech_minute"].missing_policy.reason_codes
    ) == denominator_reasons | {
        "empty_asr",
        "missing_word_timestamps",
        "unsupported_language_unitization",
    }


def test_speech_minute_stability_features_preserve_denominator_failure_reasons() -> None:
    registry = load_feature_registry(REGISTRY_PATH)
    expected_reasons = {
        "base_feature_missing",
        "normalized_derivative_unavailable",
        "comparison_failed",
        "vad_failed",
        "no_speech_detected",
        "zero_speech_duration",
    }

    for name in (
        "stability_acoustic_envelope_peak_rate_absolute_difference",
        "stability_acoustic_envelope_peak_rate_relative_difference",
        "stability_candidate_transient_rate_absolute_difference",
        "stability_candidate_transient_rate_relative_difference",
    ):
        assert set(registry.by_name[name].missing_policy.reason_codes) == expected_reasons


def test_registry_covers_every_approved_feature_family() -> None:
    registry = load_feature_registry(REGISTRY_PATH)

    assert {item.family for item in registry.features} == {
        "recording_quality",
        "vad_pause",
        "rhythm",
        "prosody",
        "spectral",
        "transient",
        "normalization_stability",
    }
    assert all(item.definition.strip() for item in registry.features)
    assert all(item.missing_policy.reason_required for item in registry.features)


def test_registry_preserves_pitch_missingness_and_stability_tolerances() -> None:
    registry = load_feature_registry(REGISTRY_PATH)

    pitch = registry.by_name["pitch_p10_p90_range_semitones"]
    assert pitch.missing_policy.allow_missing is True
    assert pitch.missing_policy.reason_codes == (
        "insufficient_voiced_frames",
        "pitch_estimation_failed",
    )

    speech_stability = registry.by_name["stability_speech_ratio_absolute_difference"]
    pitch_stability = registry.by_name[
        "stability_pitch_p10_p90_range_semitones_absolute_difference"
    ]
    assert speech_stability.warning_threshold == 0.02
    assert pitch_stability.warning_threshold == 0.5


def test_registry_declares_undefined_and_insufficient_spectral_measurements() -> None:
    """Break caught: undefined silent/one-frame spectra are forced to numeric zero."""
    registry = load_feature_registry(REGISTRY_PATH)

    for name in (
        "spectral_centroid_median_hz",
        "spectral_flux_median",
        "spectral_flux_dispersion",
    ):
        policy = registry.by_name[name].missing_policy
        assert policy.allow_missing is True
        assert policy.reason_codes == (
            "no_valid_spectral_frames",
            "insufficient_frames",
        )


def test_registry_loader_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """schema_version: 1
specification: test
features:
  - &feature
    name: duplicate
    family: rhythm
    unit: count
    source_representation: analysis_wav
    dtype: integer
    minimum: 0
    maximum: null
    definition: first
    missing_policy:
      allow_missing: false
      reason_required: true
      reason_codes: []
    clustering_role: audit_only
    warning_threshold: null
  - *feature
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_feature_registry(path)


def test_registry_loader_forbids_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "extra.yaml"
    path.write_text(
        """schema_version: 1
specification: test
features:
  - name: example
    family: rhythm
    unit: count
    source_representation: analysis_wav
    dtype: integer
    minimum: 0
    maximum: null
    definition: example
    missing_policy:
      allow_missing: false
      reason_required: true
      reason_codes: []
    clustering_role: audit_only
    warning_threshold: null
    undocumented: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="undocumented"):
        load_feature_registry(path)


def test_registry_loader_accepts_omitted_optional_numeric_bounds(tmp_path: Path) -> None:
    path = tmp_path / "unbounded.yaml"
    path.write_text(
        """schema_version: 1
specification: test
features:
  - name: unbounded
    family: rhythm
    unit: ratio
    source_representation: analysis_wav
    dtype: float
    definition: A valid feature with no meaningful numeric limits.
    missing_policy:
      allow_missing: false
      reason_required: true
      reason_codes: []
    clustering_role: audit_only
""",
        encoding="utf-8",
    )

    feature = load_feature_registry(path).features[0]

    assert feature.minimum is None
    assert feature.maximum is None
    assert feature.warning_threshold is None


@pytest.mark.parametrize(
    "field,value",
    [("minimum", ".nan"), ("maximum", ".inf"), ("warning_threshold", ".inf")],
)
def test_registry_loader_rejects_non_finite_numeric_contracts(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "non_finite.yaml"
    path.write_text(
        f"""schema_version: 1
specification: test
features:
  - name: non_finite
    family: rhythm
    unit: ratio
    source_representation: analysis_wav
    dtype: float
    {field}: {value}
    definition: Non-finite numeric contracts are invalid.
    missing_policy:
      allow_missing: false
      reason_required: true
      reason_codes: []
    clustering_role: audit_only
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_feature_registry(path)
