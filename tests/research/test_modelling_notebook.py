"""Contracts for the deterministic, privacy-safe modelling-results notebook."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from PIL import Image

import presentation_assessment.research.dataset_pipeline as dataset_pipeline_module
from presentation_assessment.research.canonical import canonical_json_bytes, sha256_file
from presentation_assessment.research.feature_registry import load_feature_registry
from presentation_assessment.research.modelling_notebook_support import (
    FIGURE_FILENAMES,
    aggregate_payloads,
    create_smoke_modelling_dataset,
    finalize_executed_notebook_narrative,
    load_analysis_after_strict_validation,
    render_private_figures,
    run_modelling_analysis,
    validate_model_run_artifacts,
    validate_modelling_results,
    write_private_results,
)
from presentation_assessment.research.notebook_support import (
    validate_executed_notebook_outputs,
)
from scripts.build_modelling_notebook import build_notebook, write_notebook
import scripts.execute_modelling_notebook as execute_modelling_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_modelling_and_results.ipynb"
REQUIRED_SECTIONS = (
    "tl;dr",
    "Context & Methods",
    "Data & QC",
    "Acoustic Clustering",
    "Score Relationships",
    "Baselines",
    "Qwen Calibration",
    "QLoRA Gate",
    "Results",
    "Takeaways",
)
_DATA_OPERATION_NAMES = frozenset(
    {
        "load_strict_frozen_dataset",
        "read_csv",
        "read_json",
        "read_parquet",
        "read_table",
        "open",
    }
)


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _notebook_data_operations(notebook: nbformat.NotebookNode) -> list[str]:
    operations: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        operations.extend(
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for name in [_call_name(node)]
            if name in _DATA_OPERATION_NAMES
        )
    return operations


def _section_titles(notebook: nbformat.NotebookNode) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for match in [re.match(r"^##\s+([^\r\n]+)", cell.source)]
        if match is not None
    )


@pytest.fixture
def invalid_notebook2_dataset(tmp_path: Path) -> Path:
    """Return a dataset-ID-shaped directory that cannot pass strict validation."""
    dataset_root = tmp_path / ("0" * 64)
    dataset_root.mkdir()
    return dataset_root


def test_notebook2_gate_aborts_before_downstream_table_access(
    invalid_notebook2_dataset: Path,
) -> None:
    """Break caught: Notebook 2 reads a table before its strict loader succeeds."""
    assert hasattr(dataset_pipeline_module, "load_strict_frozen_dataset")
    registry = load_feature_registry(
        PROJECT_ROOT / "configs" / "audio_feature_registry.v1.yaml"
    )
    data_accesses: list[Path] = []

    def access_validated_tables(dataset_path: Path) -> None:
        data_accesses.append(dataset_path)

    with pytest.raises(ValueError, match="incomplete"):
        loaded = dataset_pipeline_module.load_strict_frozen_dataset(
            invalid_notebook2_dataset, registry
        )
        access_validated_tables(loaded.path)

    assert data_accesses == []


def test_notebook2_has_reader_order_assumptions_and_one_strict_data_gate() -> None:
    """Break caught: the analytical story or first-data-operation contract drifts."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    assert _section_titles(notebook) == REQUIRED_SECTIONS
    assert any(
        cell.cell_type == "markdown" and "### Key Assumptions" in cell.source
        for cell in notebook.cells
    )
    assert _notebook_data_operations(notebook) == ["load_strict_frozen_dataset"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


def test_notebook2_source_is_deterministic_and_contains_no_install_or_network_cell(
    tmp_path: Path,
) -> None:
    """Break caught: public source depends on hidden state or a live download."""
    first = write_notebook(tmp_path / "first.ipynb")
    second = write_notebook(tmp_path / "second.ipynb")
    source = nbformat.read(NOTEBOOK_PATH, as_version=4)
    built = build_notebook()
    code = "\n".join(
        cell.source for cell in built.cells if cell.cell_type == "code"
    ).lower()

    assert first.read_bytes() == second.read_bytes()
    assert nbformat.writes(source) == nbformat.writes(built)
    assert all(
        token not in code
        for token in (
            "pip install",
            "conda install",
            "curl ",
            "wget ",
            "requests.get",
            "urlopen(",
            "source_url",
            "video_title",
            "transcript_text",
        )
    )
    assert "thesis_dataset_id" in code
    assert "thesis_mode" in code
    assert "thesis_protocol_config" not in code
    assert "protocol_config_path" not in code
    assert "configs/audio_feature_registry.v1.yaml" in code
    assert 'display_aggregate_payload(payloads_by_type["model_comparison_summary"][0])' in code
    assert "figure_paths=figure_paths" in code
    assert "validate_model_run_artifacts" in code


@pytest.fixture(scope="module")
def smoke_modelling_run():
    loaded = create_smoke_modelling_dataset(PROJECT_ROOT)
    registry = load_feature_registry(
        PROJECT_ROOT / "configs" / "audio_feature_registry.v1.yaml"
    )
    dataset = load_analysis_after_strict_validation(loaded, registry)
    return run_modelling_analysis(loaded, dataset, registry)


def test_smoke_run_covers_cohorts_clusters_oof_uncertainty_and_model_gates(
    smoke_modelling_run,
) -> None:
    """Break caught: smoke mode succeeds without exercising an analysis branch."""
    result = smoke_modelling_run
    validation = validate_modelling_results(result)

    assert result.analysis.row_count == 24
    assert result.cohorts["clustering"].count == 24
    assert result.cohorts["supervised_total_score"].count == 24
    assert result.clustering.selected_k in {2, 3, 4, 5}
    assert len(result.clustering.leave_one_feature_out) == len(
        result.clustering.feature_names
    )
    assert all(
        evaluation.oof_prediction_count == result.cohorts["supervised_total_score"].count
        for evaluation in result.baselines.evaluations
    )
    assert all(interval.replicates == 5_000 for interval in result.mae_intervals)
    assert tuple(
        (interval.comparison_model, interval.reference_model)
        for interval in result.mae_difference_intervals
    ) == (
        ("ridge", "training_mean"),
        ("elastic_net", "training_mean"),
        ("ridge", "elastic_net"),
    )
    assert all(
        interval.paired and interval.replicates == interval.valid_replicates == 5_000
        for interval in result.mae_difference_intervals
    )
    assert result.qwen_availability.status == "not_run"
    assert result.qwen_availability.reason == "expected_local_digest_missing"
    assert result.qlora_gate.status == "not_run"
    assert result.qlora_gate.reasons
    assert validation["valid"] is True
    assert validation["one_oof_per_sample"] is True
    assert validation["fold_local_preprocessing"] is True


def test_indistinguishable_clustering_is_valid_with_stability_not_applicable(
    smoke_modelling_run,
) -> None:
    """Break caught: a valid no-separation result fails for omitting meaningless ARIs."""
    indistinguishable = replace(
        smoke_modelling_run,
        clustering=replace(
            smoke_modelling_run.clustering,
            status="indistinguishable",
            selected_k=None,
            candidate_silhouettes=(),
            leave_one_feature_out=(),
            similarity_scale=None,
        ),
        cluster_scores=replace(
            smoke_modelling_run.cluster_scores,
            status="not_estimable",
            reason="clusters_indistinguishable",
            selected_k=None,
            sample_count=0,
            excluded_sample_count=smoke_modelling_run.clustering.sample_count,
            group_count=0,
            cluster_summaries=(),
            comparisons=(),
        ),
    )

    validation = validate_modelling_results(indistinguishable)

    assert validation["valid"] is True
    assert validation["stability_status"] == "not_applicable"


def test_private_artifacts_are_complete_and_aggregate_outputs_are_exact(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: row data escapes text output or required figures are omitted."""
    figure_paths = render_private_figures(smoke_modelling_run, tmp_path / "figures")
    result_paths = write_private_results(
        smoke_modelling_run,
        tmp_path / "results",
        figure_paths=figure_paths,
    )
    payloads = aggregate_payloads(smoke_modelling_run)

    assert set(figure_paths) == set(FIGURE_FILENAMES)
    assert all(path.is_file() and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figure_paths.values())
    assert {path.name for path in result_paths} == {
        "aggregate_results.json",
        "cluster_assignments.jsonl",
        "composite_values.jsonl",
        "oof_predictions.jsonl",
        "cluster_summary.csv",
        "cohort_summary.csv",
        "model_metrics.csv",
        "model_run_manifest.json",
        "result_checksums.sha256",
    }
    assert len(payloads) == 11
    assert [payload["artifact_type"] for payload in payloads].count("model_summary") == 3
    comparison = next(
        payload
        for payload in payloads
        if payload["artifact_type"] == "model_comparison_summary"
    )
    assert comparison["ridge_minus_training_mean_status"] == "ok"
    assert comparison["ridge_minus_training_mean_required_replicates"] == 5_000
    assert comparison["ridge_minus_training_mean_valid_replicates"] == 5_000
    assert type(comparison["ridge_minus_training_mean_includes_zero"]) is bool
    assert comparison["elastic_net_minus_training_mean_status"] == "ok"
    assert comparison["ridge_minus_elastic_net_status"] == "ok"
    assert all("sample_id" not in payload for payload in payloads)
    assert all("total_score" not in payload for payload in payloads)
    assert validate_model_run_artifacts(
        smoke_modelling_run, tmp_path / "results", tmp_path / "figures"
    ) is True


def test_model_run_manifest_records_hashed_reconstructable_audit(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: a model run cannot reproduce folds, preprocessing, or outputs."""
    figure_directory = tmp_path / "figures"
    result_directory = tmp_path / "results"
    figure_paths = render_private_figures(smoke_modelling_run, figure_directory)
    write_private_results(
        smoke_modelling_run, result_directory, figure_paths=figure_paths
    )

    manifest = json.loads(
        (result_directory / "model_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["dataset_id"] == smoke_modelling_run.loaded.dataset_id
    assert manifest["protocol_id"] == smoke_modelling_run.loaded.protocol.protocol_id
    assert manifest["split_sha256"] == smoke_modelling_run.loaded.protocol.split_sha256
    assert manifest["random_seeds"] == {
        "protocol": 20260813,
        "bootstrap": 20260813,
        "cluster_permutation": 20260813,
    }
    assert manifest["feature_registry"] == smoke_modelling_run.feature_registry.model_dump(
        mode="json"
    )
    assert set(manifest["models"]) == {"training_mean", "ridge", "elastic_net"}
    serialized = json.dumps(manifest, sort_keys=True)
    for sample_id in smoke_modelling_run.analysis.sample_ids:
        assert sample_id not in serialized
    for model in manifest["models"].values():
        assert model["outer_folds"]
        assert all(len(value) == 64 for value in model["oof_sample_sha256"])
        for fold in model["outer_folds"]:
            assert all(len(value) == 64 for value in fold["training_sample_sha256"])
            assert all(len(value) == 64 for value in fold["validation_sample_sha256"])
            assert set(fold["location"]) == set(fold["feature_names"])
            assert set(fold["scale"]) == set(fold["feature_names"])
    assert "figures/figure_manifest.json" in manifest["outputs"]
    assert "results/oof_predictions.jsonl" in manifest["outputs"]

    checksum_lines = (
        result_directory / "result_checksums.sha256"
    ).read_text(encoding="utf-8").splitlines()
    assert any(line.endswith("  results/model_run_manifest.json") for line in checksum_lines)
    assert validate_model_run_artifacts(
        smoke_modelling_run, result_directory, figure_directory
    ) is True


def test_model_run_validator_rejects_resigned_fold_audit_tampering(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: edited fold evidence passes after its checksum is refreshed."""
    figure_directory = tmp_path / "figures"
    result_directory = tmp_path / "results"
    figure_paths = render_private_figures(smoke_modelling_run, figure_directory)
    write_private_results(
        smoke_modelling_run, result_directory, figure_paths=figure_paths
    )
    manifest_path = result_directory / "model_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"]["ridge"]["outer_folds"][0]["selected_hyperparameters"] = {
        "alpha": 999.0
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    checksum_path = result_directory / "result_checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(manifest_path)}  results/model_run_manifest.json"
            if line.endswith("  results/model_run_manifest.json")
            else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model run manifest"):
        validate_model_run_artifacts(
            smoke_modelling_run, result_directory, figure_directory
        )


def test_model_run_validator_rejects_resigned_figure_provenance_tampering(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: a rewritten figure caption passes after every hash is refreshed."""
    figure_directory = tmp_path / "figures"
    result_directory = tmp_path / "results"
    figure_paths = render_private_figures(smoke_modelling_run, figure_directory)
    write_private_results(
        smoke_modelling_run, result_directory, figure_paths=figure_paths
    )
    figure_manifest_path = figure_directory / "figure_manifest.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_manifest["figures"]["pca_scatter"]["caption"] = "tampered"
    figure_manifest_path.write_bytes(canonical_json_bytes(figure_manifest))

    model_manifest_path = result_directory / "model_run_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_manifest["outputs"]["figures/figure_manifest.json"] = sha256_file(
        figure_manifest_path
    )
    model_manifest_path.write_bytes(canonical_json_bytes(model_manifest))

    checksum_path = result_directory / "result_checksums.sha256"
    replacements = {
        "figures/figure_manifest.json": sha256_file(figure_manifest_path),
        "results/model_run_manifest.json": sha256_file(model_manifest_path),
    }
    checksum_path.write_text(
        "\n".join(
            sorted(
                f"{replacements.get(line.split('  ', 1)[1], line.split('  ', 1)[0])}  "
                f"{line.split('  ', 1)[1]}"
                for line in checksum_path.read_text(encoding="utf-8").splitlines()
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="figure manifest"):
        validate_model_run_artifacts(
            smoke_modelling_run, result_directory, figure_directory
        )


def test_model_run_validator_rejects_resigned_model_metrics_tampering(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: edited tabular metrics pass after the complete hash chain is refreshed."""
    figure_directory = tmp_path / "figures"
    result_directory = tmp_path / "results"
    figure_paths = render_private_figures(smoke_modelling_run, figure_directory)
    write_private_results(
        smoke_modelling_run, result_directory, figure_paths=figure_paths
    )
    metrics_path = result_directory / "model_metrics.csv"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[2] = "999.0"
    lines[1] = ",".join(fields)
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = result_directory / "model_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["results/model_metrics.csv"] = sha256_file(metrics_path)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    replacements = {
        "results/model_metrics.csv": sha256_file(metrics_path),
        "results/model_run_manifest.json": sha256_file(manifest_path),
    }
    checksum_path = result_directory / "result_checksums.sha256"
    checksum_path.write_text(
        "\n".join(
            sorted(
                f"{replacements.get(line.split('  ', 1)[1], line.split('  ', 1)[0])}  "
                f"{line.split('  ', 1)[1]}"
                for line in checksum_path.read_text(encoding="utf-8").splitlines()
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model metrics output"):
        validate_model_run_artifacts(
            smoke_modelling_run, result_directory, figure_directory
        )


def test_every_aggregate_and_figure_has_standalone_provenance(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: an exported payload or raster cannot be interpreted alone."""
    result = smoke_modelling_run
    payloads = aggregate_payloads(result)
    figure_directory = tmp_path / "figures"
    figure_paths = render_private_figures(result, figure_directory)

    for payload in payloads:
        assert payload["dataset_id"] == result.loaded.dataset_id
        assert payload["protocol_id"] == result.loaded.protocol.protocol_id
        assert payload["included_count"] + payload["excluded_count"] == result.analysis.row_count
        assert isinstance(payload["exclusion_basis"], str) and payload["exclusion_basis"]
        assert isinstance(payload["method"], str) and payload["method"]
        assert isinstance(payload["uncertainty_method"], str) and payload["uncertainty_method"]
        assert "model_identifier" in payload

    clustering = next(
        payload for payload in payloads if payload["artifact_type"] == "clustering_summary"
    )
    assert clustering["conditional_feature_status"] == result.clustering.conditional_feature_status

    figure_manifest = json.loads(
        (figure_directory / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert figure_manifest["schema_version"] == 1
    assert figure_manifest["dataset_id"] == result.loaded.dataset_id
    assert figure_manifest["protocol_id"] == result.loaded.protocol.protocol_id
    assert set(figure_manifest["figures"]) == set(FIGURE_FILENAMES)
    short_id = result.loaded.dataset_id[:12]
    for name, path in figure_paths.items():
        caption = figure_manifest["figures"][name]["caption"]
        assert f"dataset={short_id}" in caption
        assert f"n=24 of {result.analysis.row_count}" in caption
        assert "excluded=0" in caption
        assert "basis=" in caption
        assert "units=" in caption
        assert "method=" in caption
        assert "uncertainty=" in caption
        caption_lines = caption.splitlines()
        assert len(caption_lines) == 4
        assert max(map(len, caption_lines)) <= 92
        assert figure_manifest["figures"][name]["sha256"]
        with Image.open(path) as image:
            assert image.info["Description"] == caption


def test_exact_modelling_aggregate_payloads_pass_the_fail_closed_scanner(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: allowlist is broader than the exact notebook result schemas."""
    payloads = aggregate_payloads(smoke_modelling_run)
    outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"application/json": payload},
            metadata={},
        )
        for payload in payloads
    ]
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "display_aggregate_outputs()", execution_count=1, outputs=outputs
            )
        ]
    )
    executed = tmp_path / "executed.ipynb"
    nbformat.write(notebook, executed)

    assert validate_executed_notebook_outputs(executed) is True

    unsafe = json.loads(json.dumps(payloads[0]))
    unsafe["unapproved_extra"] = 1
    notebook.cells[0].outputs = [
        nbformat.v4.new_output(
            "display_data", data={"application/json": unsafe}, metadata={}
        )
    ]
    nbformat.write(notebook, executed)
    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


def test_executed_narrative_flags_weak_cluster_stability(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: a selected k is described as robust despite negative ARI."""
    payloads = list(aggregate_payloads(smoke_modelling_run))
    clustering = next(
        payload for payload in payloads if payload["artifact_type"] == "clustering_summary"
    )
    clustering["stability_min"] = -0.2
    clustering["stability_median"] = -0.1
    notebook = build_notebook()
    notebook.cells[3].execution_count = 1
    notebook.cells[3].outputs = [
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        )
        for payload in payloads
    ]
    executed = tmp_path / "executed.ipynb"
    nbformat.write(notebook, executed)

    finalize_executed_notebook_narrative(executed)
    finalized = nbformat.read(executed, as_version=4)
    takeaways = next(
        cell.source
        for cell in finalized.cells
        if "narrative-takeaways" in cell.metadata.get("tags", [])
    )

    assert "feature-sensitive" in takeaways
    assert "not a robust natural taxonomy" in takeaways


def _write_payload_notebook(
    path: Path, payloads: list[dict[str, object]]
) -> Path:
    notebook = build_notebook()
    notebook.cells[3].execution_count = 1
    notebook.cells[3].outputs = [
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        )
        for payload in payloads
    ]
    nbformat.write(notebook, path)
    return path


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda payloads: [
                payload
                for payload in payloads
                if payload["artifact_type"] != "cluster_score_summary"
            ],
            "exactly eleven",
        ),
        (
            lambda payloads: [
                (
                    {**payload, "model_name": "ridge"}
                    if payload.get("model_name") == "elastic_net"
                    else payload
                )
                for payload in payloads
            ],
            "exactly one summary for each model",
        ),
        (
            lambda payloads: [
                payload
                for payload in payloads
                if payload.get("model_name") != "elastic_net"
            ],
            "exactly eleven",
        ),
        (
            lambda payloads: [
                *payloads,
                {"artifact_type": "execution_summary", "row_count": 24, "valid": True},
            ],
            "exactly eleven",
        ),
    ],
    ids=("missing-posthoc", "duplicate-model", "missing-model", "unexpected-payload"),
)
def test_executed_narrative_requires_exact_eleven_payload_inventory(
    tmp_path: Path, smoke_modelling_run, mutation, expected: str
) -> None:
    """Break caught: finalization silently ignores missing, duplicate, or extra evidence."""
    payloads = json.loads(json.dumps(aggregate_payloads(smoke_modelling_run)))
    executed = _write_payload_notebook(tmp_path / "executed.ipynb", mutation(payloads))

    with pytest.raises(ValueError, match=expected):
        finalize_executed_notebook_narrative(executed)


def test_executed_narrative_uses_paired_model_differences_not_marginal_overlap(
    tmp_path: Path, smoke_modelling_run
) -> None:
    """Break caught: final prose omits non-significance or implies model superiority."""
    payloads = json.loads(json.dumps(aggregate_payloads(smoke_modelling_run)))
    posthoc = next(
        payload for payload in payloads if payload["artifact_type"] == "cluster_score_summary"
    )
    posthoc.update(
        {
            "minimum_adjusted_p": 0.1504849515048495,
            "significance_threshold": 0.05,
            "statistically_significant": False,
        }
    )
    model_values = {
        "training_mean": (10.0, 9.0, 11.0),
        "ridge": (5.0, 4.0, 6.0),
        "elastic_net": (7.0, 6.5, 8.0),
    }
    for payload in payloads:
        model_name = payload.get("model_name")
        if model_name in model_values:
            mae, lower, upper = model_values[model_name]
            payload.update(
                {"mae": mae, "mae_ci_lower": lower, "mae_ci_upper": upper}
            )
    comparison = next(
        payload
        for payload in payloads
        if payload["artifact_type"] == "model_comparison_summary"
    )
    comparison.update(
        {
            "ridge_minus_training_mean_estimate": -5.0,
            "ridge_minus_training_mean_ci_lower": -6.0,
            "ridge_minus_training_mean_ci_upper": -4.0,
            "ridge_minus_training_mean_includes_zero": False,
            "elastic_net_minus_training_mean_estimate": -3.0,
            "elastic_net_minus_training_mean_ci_lower": -4.0,
            "elastic_net_minus_training_mean_ci_upper": 0.5,
            "elastic_net_minus_training_mean_includes_zero": True,
            "ridge_minus_elastic_net_estimate": -2.0,
            "ridge_minus_elastic_net_ci_lower": -3.0,
            "ridge_minus_elastic_net_ci_upper": 0.25,
            "ridge_minus_elastic_net_includes_zero": True,
        }
    )
    executed = _write_payload_notebook(tmp_path / "executed.ipynb", payloads)

    finalize_executed_notebook_narrative(executed)
    finalized = nbformat.read(executed, as_version=4)
    tl_dr = next(
        cell.source
        for cell in finalized.cells
        if "narrative-tldr" in cell.metadata.get("tags", [])
    )
    takeaways = next(
        cell.source
        for cell in finalized.cells
        if "narrative-takeaways" in cell.metadata.get("tags", [])
    )

    assert (
        "Holm-adjusted p=0.150 was not statistically significant at the declared "
        "descriptive alpha=0.050 threshold and is not compelling evidence of cluster score differences."
        in tl_dr
    )
    assert (
        "Ridge had lower paired MAE than the training-fold mean: difference -5.000 "
        "points, 95% CI [-6.000, -4.000]."
        in tl_dr
    )
    assert "ElasticNet versus the training-fold mean includes zero" in tl_dr
    assert "Ridge versus ElasticNet includes zero" in tl_dr
    assert "marginal intervals overlap" not in tl_dr
    assert "not compelling evidence of cluster score differences" in takeaways
    assert "Ridge had lower paired MAE than the training-fold mean" in takeaways


@pytest.mark.parametrize(
    ("corruption", "expected"),
    (
        ("posthoc-flag", "significance flag"),
        ("paired-zero-flag", "includes-zero flag"),
        ("paired-replicates", "replicate counts"),
    ),
)
def test_executed_narrative_recomputes_and_rejects_inconsistent_inference_flags(
    tmp_path: Path, smoke_modelling_run, corruption: str, expected: str
) -> None:
    """Break caught: narrative trusts caller-supplied p-value or CI-overlap claims."""
    payloads = json.loads(json.dumps(aggregate_payloads(smoke_modelling_run)))
    posthoc = next(
        payload for payload in payloads if payload["artifact_type"] == "cluster_score_summary"
    )
    comparison = next(
        payload
        for payload in payloads
        if payload["artifact_type"] == "model_comparison_summary"
    )
    if corruption == "posthoc-flag":
        posthoc["statistically_significant"] = not posthoc["statistically_significant"]
    elif corruption == "paired-zero-flag":
        comparison["ridge_minus_training_mean_includes_zero"] = not comparison[
            "ridge_minus_training_mean_includes_zero"
        ]
    else:
        comparison["ridge_minus_training_mean_valid_replicates"] = 4_999
    executed = _write_payload_notebook(tmp_path / "executed.ipynb", payloads)

    with pytest.raises(ValueError, match=expected):
        finalize_executed_notebook_narrative(executed)


def test_notebook2_invalid_dataset_stops_before_later_table_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: execution continues to a table operation after loader failure."""
    repository_root = tmp_path / "repository"
    registry_path = repository_root / "configs" / "audio_feature_registry.v1.yaml"
    registry_path.parent.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / "configs" / "audio_feature_registry.v1.yaml", registry_path
    )
    dataset_id = "0" * 64
    (repository_root / "artifacts" / "research" / dataset_id).mkdir(
        parents=True
    )
    marker = repository_root / "table-accessed.marker"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "Path('table-accessed.marker').write_text('unsafe', encoding='utf-8')"
        )
    )
    monkeypatch.setenv("THESIS_REPOSITORY_ROOT", str(repository_root))
    monkeypatch.setenv("THESIS_DATASET_ID", dataset_id)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    client = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(repository_root)}},
    )

    with pytest.raises(CellExecutionError, match="incomplete"):
        client.execute()

    assert not marker.exists()


def test_executed_notebook_destination_must_stay_in_dataset_mode_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an executable copy can be written into notebooks or another run."""
    root = tmp_path / "repository"
    root.mkdir()
    dataset_id = "a" * 64
    private_root = (
        root / "artifacts" / "research" / "modelling" / dataset_id / "full_run"
    )
    source = tmp_path / "source.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), source)

    class ForbiddenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("execution began before destination validation")

    monkeypatch.setattr(execute_modelling_module, "NotebookClient", ForbiddenClient)
    for outside in (
        root / "notebooks" / "02_modelling_and_results.executed.ipynb",
        Path("notebooks/02_modelling_and_results.executed.ipynb"),
        root / "artifacts" / "research" / "modelling" / ("b" * 64) / "full_run" / "wrong.ipynb",
    ):
        with pytest.raises(ValueError, match="private modelling artifact directory"):
            execute_modelling_module.execute_and_validate(
                source,
                outside,
                working_directory=root,
                mode="FULL_RUN",
                dataset_id=dataset_id,
            )
    assert not private_root.exists()


def test_failed_scan_leaves_no_executed_or_temporary_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a fully executed unsafe notebook is persisted before scanning."""
    root = tmp_path / "repository"
    root.mkdir()
    dataset_id = "c" * 64
    private_root = (
        root / "artifacts" / "research" / "modelling" / dataset_id / "full_run"
    )
    final_path = private_root / "custom.executed.ipynb"
    source = tmp_path / "source.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]),
        source,
    )

    class FakeClient:
        def __init__(self, notebook, **kwargs: object) -> None:
            self.notebook = notebook

        def execute(self) -> object:
            self.notebook.cells[0].execution_count = 1
            self.notebook.cells[0].outputs = [
                nbformat.v4.new_output("stream", name="stdout", text="valid=true\n")
            ]
            return self.notebook

    monkeypatch.setattr(execute_modelling_module, "NotebookClient", FakeClient)
    monkeypatch.setattr(
        execute_modelling_module,
        "validate_executed_notebook_outputs",
        lambda path: (_ for _ in ()).throw(ValueError("unsafe scan")),
    )

    with pytest.raises(ValueError, match="unsafe scan"):
        execute_modelling_module.execute_and_validate(
            source,
            final_path,
            working_directory=root,
            mode="FULL_RUN",
            dataset_id=dataset_id,
        )

    assert not final_path.exists()
    assert not list(private_root.glob("*.tmp.ipynb"))
