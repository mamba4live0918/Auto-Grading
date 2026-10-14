from pathlib import Path

import pytest

from presentation_assessment.research import (
    BoundProtocol,
    ProtocolTemplate,
    bind_protocol,
    load_protocol_template,
    protocol_id,
)


PROTOCOL_PATH = Path("configs/experiment_protocol.template.yaml")


def test_protocol_type_is_part_of_the_research_public_interface() -> None:
    assert isinstance(load_protocol_template(PROTOCOL_PATH), ProtocolTemplate)


def test_protocol_id_is_canonical_and_binding_is_non_circular() -> None:
    template = load_protocol_template(PROTOCOL_PATH)
    first = protocol_id(template)

    bound = bind_protocol(template, dataset_id="d" * 64, split_sha256="e" * 64)

    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    assert bound.protocol_id == first
    assert bound.dataset_id == "d" * 64
    assert bound.split_sha256 == "e" * 64
    assert bound.template == template
    assert protocol_id(template) == first


def test_bound_protocol_rejects_protocol_id_unrelated_to_embedded_template() -> None:
    template = load_protocol_template(PROTOCOL_PATH)
    unrelated_id = "0" * 64
    assert unrelated_id != protocol_id(template)

    with pytest.raises(ValueError, match="embedded template"):
        BoundProtocol(
            protocol_id=unrelated_id,
            dataset_id="d" * 64,
            split_sha256="e" * 64,
            template=template,
        )


def test_bound_protocol_rejects_non_hex_protocol_id() -> None:
    template = load_protocol_template(PROTOCOL_PATH)

    with pytest.raises(ValueError):
        BoundProtocol(
            protocol_id="g" * 64,
            dataset_id="d" * 64,
            split_sha256="e" * 64,
            template=template,
        )


def test_protocol_freezes_required_analysis_decisions() -> None:
    template = load_protocol_template(PROTOCOL_PATH)

    assert template.random_seed == 20260813
    assert template.target.name == "total_score"
    assert (template.target.scale_min, template.target.scale_max) == (0.0, 100.0)
    assert template.target.rubric_level_enabled is False
    assert template.split.strategy == "nested_grouped_cross_validation"
    assert template.split.no_independent_holdout_claim is True
    assert template.split.group_fields == ("speaker_id", "talk_id")
    assert template.clustering.pca_components == 2
    assert template.clustering.linkage == "ward"
    assert template.clustering.candidate_cluster_counts == (2, 3, 4, 5)
    assert template.clustering.smaller_cluster_tolerance == 0.05
    assert template.evaluation.primary_metric == "mae"
    assert template.uncertainty.bootstrap_replicates == 5000
    assert template.uncertainty.permutation_replicates == 10000
    assert template.uncertainty.multiple_comparison_correction == "holm"
    assert template.qwen.enabled is False
    assert template.qlora.enabled is False
    assert template.qlora.status == "not_run"
    assert template.qlora.smoke_test_interpretation == "method_feasibility_only"


def test_protocol_composite_declares_every_direction_and_fixed_weight() -> None:
    template = load_protocol_template(PROTOCOL_PATH)

    components = template.composite.components
    assert {item.feature for item in components} == {
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
    assert {item.direction for item in components} == {-1, 1}
    assert sum(item.weight for item in components) == pytest.approx(1.0)


def test_protocol_has_no_unresolved_placeholders() -> None:
    template = load_protocol_template(PROTOCOL_PATH)

    def assert_resolved(value: object) -> None:
        if isinstance(value, str):
            assert value.strip()
            assert value.strip().lower() not in {"todo", "tbd", "placeholder", "unknown"}
        elif isinstance(value, dict):
            for item in value.values():
                assert_resolved(item)
        elif isinstance(value, (list, tuple)):
            assert value
            for item in value:
                assert_resolved(item)
        else:
            assert value is not None

    assert_resolved(template.model_dump(mode="json"))


def test_protocol_loader_forbids_unknown_fields(tmp_path: Path) -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    path = tmp_path / "extra.yaml"
    path.write_text(text + "\nundocumented: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undocumented"):
        load_protocol_template(path)


def test_protocol_loader_rejects_non_finite_clustering_tolerance(tmp_path: Path) -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    path = tmp_path / "infinite_tolerance.yaml"
    path.write_text(
        text.replace("smaller_cluster_tolerance: 0.05", "smaller_cluster_tolerance: .inf"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_protocol_template(path)


@pytest.mark.parametrize("dataset_id,split_sha256", [("short", "s" * 64), ("d" * 64, "short")])
def test_protocol_binding_rejects_non_digest_length_identifiers(
    dataset_id: str, split_sha256: str
) -> None:
    template = load_protocol_template(PROTOCOL_PATH)

    with pytest.raises(ValueError):
        bind_protocol(template, dataset_id=dataset_id, split_sha256=split_sha256)


@pytest.mark.parametrize(
    ("dataset_id", "split_sha256"),
    [("g" * 64, "e" * 64), ("d" * 64, "A" * 64)],
)
def test_protocol_binding_rejects_non_lowercase_sha256_identifiers(
    dataset_id: str, split_sha256: str
) -> None:
    """Break caught: digest-shaped but non-SHA-256 identifiers enter the binding."""
    template = load_protocol_template(PROTOCOL_PATH)

    with pytest.raises(ValueError):
        bind_protocol(template, dataset_id=dataset_id, split_sha256=split_sha256)
