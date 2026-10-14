"""Contract tests for the deterministic, privacy-safe dataset notebook."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import nbformat
import pytest

from presentation_assessment.research.canonical import canonical_notebook_bytes
from presentation_assessment.research.local_source import (
    LocalAvailability,
    LocalCatalogRow,
    LocalSourceCatalog,
    LocalSourceConfig,
)
from presentation_assessment.research.preflight import PreflightReport
from presentation_assessment.research.source_manifest import (
    FROZEN_LEGACY_VIDEO_IDS,
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)
from presentation_assessment.research.notebook_support import (
    bounded_feature_preview,
    execute_notebook_pipeline,
    freeze_notebook_dataset,
    locate_repository_root,
    prepare_notebook_run,
    redacted_freeze_summary,
    stage_summary,
    summarize_preflight,
    summarize_quality,
    summarize_source_audit,
    validate_executed_notebook_outputs,
    validate_notebook_outputs,
    write_redacted_handoff,
)
import presentation_assessment.research.notebook_support as notebook_support_module
from scripts.build_dataset_notebook import build_notebook, write_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_dataset_pipeline.ipynb"
REQUIRED_SECTIONS = (
    "Goal",
    "Setup",
    "Source Audit",
    "Acquisition",
    "Audio",
    "Whisper",
    "Features",
    "Quality",
    "Freeze",
    "Checks",
    "Next Steps",
)
FROZEN_DATA_FILES = (
    "manifest.csv",
    "transcripts.jsonl",
    "acoustic_features.parquet",
    "quality_report.csv",
    "splits.json",
    "experiment_protocol.json",
    "provenance.json",
    "source_snapshot.json",
    "git_diff.patch",
)
NOTEBOOK1_AGGREGATE_PAYLOADS = (
    {
        "artifact_type": "preflight_summary",
        "mode": "SMOKE_MODE",
        "source_mode": "YOUTUBE",
        "ready": True,
        "network_enabled": False,
        "check_status_counts": {
            "pass": 7,
            "warning": 0,
            "blocked": 0,
            "skipped": 0,
        },
        "warning_check_count": 0,
    },
    {
        "artifact_type": "source_audit_summary",
        "total_rows": 3,
        "observed_label_rows": 0,
        "not_available_label_rows": 3,
        "unresolved_rows": 0,
        "duplicate_source_rows": 0,
        "duplicate_video_rows": 0,
        "invalid_rows": 0,
        "identity_error_rows": 0,
    },
    {
        "artifact_type": "stage_summary",
        "stage": "features",
        "succeeded": 3,
        "failed": 0,
        "skipped": 0,
        "resumed": 3,
        "executed": 0,
    },
    {
        "artifact_type": "feature_preview_summary",
        "previewed_row_count": 3,
        "previewed_feature_count": 6,
        "available_value_count": 18,
        "missing_value_count": 0,
    },
    {
        "artifact_type": "quality_summary",
        "total_rows": 3,
        "feature_succeeded_rows": 3,
        "feature_non_success_rows": 0,
        "rows_with_qc_warnings": 0,
        "recorded_missing_values": 0,
        "terminal_counts": {
            "succeeded": 3,
            "acquisition_failed": 0,
            "audio_failed": 0,
            "asr_failed": 0,
            "feature_failed": 0,
        },
        "pipeline_complete_rows": 3,
        "analysis_usable_counts": {
            "quality_control": 3,
            "clustering": 3,
            "supervised_exploratory": 0,
            "transcript_analysis": 3,
        },
    },
    {
        "artifact_type": "freeze_summary",
        "reused": False,
        "row_count": 3,
        "terminal_counts": {
            "succeeded": 3,
            "acquisition_failed": 0,
            "audio_failed": 0,
            "asr_failed": 0,
            "feature_failed": 0,
        },
        "pipeline_complete_rows": 3,
        "analysis_usable_counts": {
            "quality_control": 3,
            "clustering": 3,
            "supervised_exploratory": 0,
            "transcript_analysis": 3,
        },
    },
    {
        "artifact_type": "validation_summary",
        "row_count": 3,
        "expected_row_count": 3,
        "terminal_records_valid": True,
        "frozen_file_set_valid": True,
        "checksum_count": 9,
        "checksums_valid": True,
        "frozen_row_counts_valid": True,
        "public_source_safe": True,
        "executed_copy_ignored": True,
        "valid": True,
    },
    {
        "artifact_type": "handoff_summary",
        "row_count": 3,
        "terminal_counts": {
            "succeeded": 3,
            "acquisition_failed": 0,
            "audio_failed": 0,
            "asr_failed": 0,
            "feature_failed": 0,
        },
        "pipeline_complete_rows": 3,
        "analysis_usable_counts": {
            "quality_control": 3,
            "clustering": 3,
            "supervised_exploratory": 0,
            "transcript_analysis": 3,
        },
        "valid": True,
    },
)


def _section_titles(notebook: nbformat.NotebookNode) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for match in [re.match(r"^##\s+([^\r\n]+)", cell.source)]
        if match is not None
    )


def test_builder_writes_the_committed_notebook_deterministically(tmp_path: Path) -> None:
    """Break caught: generation depends on timestamps, execution, or prior file state."""
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"

    write_notebook(first)
    write_notebook(second)

    assert first.read_bytes() == second.read_bytes()
    assert canonical_notebook_bytes(first) == canonical_notebook_bytes(NOTEBOOK_PATH)


def test_notebook_has_the_required_reader_facing_section_order() -> None:
    """Break caught: a pipeline stage disappears or becomes out of order."""
    notebook = build_notebook()

    assert _section_titles(notebook) == REQUIRED_SECTIONS
    assert notebook.cells[0].cell_type == "markdown"


def test_every_code_cell_has_one_focused_purpose_and_clean_state() -> None:
    """Break caught: a mixed-purpose or pre-executed cell introduces hidden state."""
    notebook = build_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells
    for cell in code_cells:
        purpose_tags = [
            tag for tag in cell.metadata.get("tags", []) if tag.startswith("purpose-")
        ]
        assert len(purpose_tags) == 1
        assert cell.execution_count is None
        assert cell.outputs == []


def test_notebook_source_is_parameterized_and_reader_safe() -> None:
    """Break caught: the public source leaks row data, local paths, or install actions."""
    notebook = build_notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    source_lower = source.lower()

    assert "THESIS_NOTEBOOK_MODE" in source
    assert "THESIS_ALLOW_YOUTUBE_NETWORK" in source
    assert "THESIS_DATASET_CONFIG" in source
    assert "THESIS_SOURCE_MODE" in source
    assert "THESIS_LOCAL_SOURCE_CONFIG" in source
    assert "SMOKE_MODE" in source
    assert "source_url" not in source
    assert "video_title" not in source
    assert "total_score" not in source
    assert re.search(r"https?://", source, flags=re.IGNORECASE) is None
    assert re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", source) is None
    assert not any(
        marker in source_lower
        for marker in ("%pip", "!pip", "pip install", "conda install", "%conda")
    )


def test_notebook_declares_a_python_311_kernel_without_widgets() -> None:
    """Break caught: handoff silently selects a different runtime or embeds widget state."""
    notebook = build_notebook()

    assert notebook.nbformat == 4
    assert notebook.nbformat_minor == 5
    assert notebook.metadata.kernelspec.name == "python3"
    assert notebook.metadata.language_info.version.startswith("3.11")
    assert "widgets" not in notebook.metadata


def _ready_preflight(root: Path, config_path: Path, mode: str) -> PreflightReport:
    del root, config_path
    return PreflightReport(mode=mode, checks=())


def _write_checksum_fixture(root: Path) -> list[str]:
    root.mkdir()
    lines: list[str] = []
    for filename in FROZEN_DATA_FILES:
        payload = f"fixture:{filename}".encode()
        (root / filename).write_bytes(payload)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {filename}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


@pytest.mark.parametrize("malformation", ["duplicate", "unexpected", "traversal", "absolute"])
def test_checksum_validation_rejects_non_exact_or_unsafe_filenames(
    tmp_path: Path, malformation: str
) -> None:
    """Break caught: nine valid hashes can omit or escape a required frozen file."""
    frozen = tmp_path / "frozen"
    lines = _write_checksum_fixture(frozen)
    if malformation == "duplicate":
        lines[-1] = lines[0]
    elif malformation == "unexpected":
        extra = frozen / "unexpected.bin"
        extra.write_bytes(b"unexpected")
        lines[-1] = f"{hashlib.sha256(extra.read_bytes()).hexdigest()}  unexpected.bin"
    elif malformation == "traversal":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        lines[-1] = f"{hashlib.sha256(outside.read_bytes()).hexdigest()}  ../outside.bin"
    else:
        outside = tmp_path / "absolute.bin"
        outside.write_bytes(b"absolute")
        lines[-1] = f"{hashlib.sha256(outside.read_bytes()).hexdigest()}  {outside}"
    (frozen / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    count, valid = notebook_support_module._verify_checksums(frozen)

    assert count == 9
    assert valid is False


def test_checksum_validation_rejects_a_missing_declaration(tmp_path: Path) -> None:
    """Break caught: an incomplete checksum manifest is accepted as a complete freeze."""
    frozen = tmp_path / "frozen"
    lines = _write_checksum_fixture(frozen)
    (frozen / "checksums.sha256").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    count, valid = notebook_support_module._verify_checksums(frozen)

    assert count == 8
    assert valid is False


def test_repository_root_discovery_uses_project_markers() -> None:
    """Break caught: execution depends on the shell starting in one exact directory."""
    assert locate_repository_root(PROJECT_ROOT / "notebooks") == PROJECT_ROOT


def test_full_run_fails_closed_without_explicit_network_opt_in() -> None:
    """Break caught: selecting FULL_RUN alone silently enables external acquisition."""
    with pytest.raises(PermissionError, match="explicit network opt-in"):
        prepare_notebook_run(
            PROJECT_ROOT,
            mode="FULL_RUN",
            network_opt_in="0",
            config_path="configs/notebook_pipeline.yaml",
            preflight_runner=_ready_preflight,
        )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("acquisition", frozenset({"acquisition", "audio", "asr", "features"})),
        ("audio", frozenset({"audio", "asr", "features"})),
        ("asr", frozenset({"asr", "features"})),
        ("features", frozenset({"features"})),
        ("all", frozenset({"acquisition", "audio", "asr", "features"})),
    ],
)
def test_force_stage_environment_selects_exact_stage_and_downstream(
    requested: str, expected: frozenset[str]
) -> None:
    """Break caught: FORCE_STAGE is silently fixed to features."""
    assert notebook_support_module._force_stages_from_environment(
        "FORCE_STAGE", {"THESIS_FORCE_STAGE": requested}
    ) == expected


def test_force_stage_mode_fails_closed_without_a_valid_exact_stage() -> None:
    """Break caught: missing or misspelled force intent triggers an arbitrary retry."""
    with pytest.raises(ValueError, match="THESIS_FORCE_STAGE"):
        notebook_support_module._force_stages_from_environment(
            "FORCE_STAGE", {"THESIS_FORCE_STAGE": "whisper"}
        )


def test_legacy_cache_upgrade_requires_explicit_one_time_opt_in() -> None:
    """Break caught: schema-v1 cache compatibility stays silently enabled forever."""
    assert notebook_support_module._legacy_cache_opt_in(
        {"THESIS_ALLOW_LEGACY_STAGE_CACHE": "1"}
    ) is True
    assert notebook_support_module._legacy_cache_opt_in({}) is False


def test_smoke_pipeline_is_offline_frozen_and_reader_safe() -> None:
    """Break caught: smoke mode needs network/model data or exposes private row fields."""
    context = prepare_notebook_run(
        PROJECT_ROOT,
        mode="SMOKE_MODE",
        network_opt_in="1",
        config_path="configs/notebook_pipeline.yaml",
        preflight_runner=_ready_preflight,
    )

    assert context.allow_network is False
    preflight = summarize_preflight(context)
    source_audit = summarize_source_audit(context)
    assert preflight["artifact_type"] == "preflight_summary"
    assert preflight["ready"] is True
    assert source_audit["artifact_type"] == "source_audit_summary"
    assert source_audit["total_rows"] == 3

    result = execute_notebook_pipeline(context)
    assert len(result.samples) == 3
    for stage in ("acquisition", "audio", "asr", "features"):
        cache_summary = stage_summary(result, stage)
        resumed = sum(
            getattr(sample, stage).resumed for sample in result.samples
        )
        assert cache_summary == {
            "artifact_type": "stage_summary",
            "stage": stage,
            "succeeded": 3,
            "failed": 0,
            "skipped": 0,
            "resumed": resumed,
            "executed": len(result.samples) - resumed,
        }

    preview = bounded_feature_preview(result, limit=2, feature_limit=3)
    assert preview == {
        "artifact_type": "feature_preview_summary",
        "previewed_row_count": 2,
        "previewed_feature_count": 3,
        "available_value_count": 6,
        "missing_value_count": 0,
    }
    quality = summarize_quality(result)
    assert quality["artifact_type"] == "quality_summary"
    assert quality["total_rows"] == 3
    assert quality["feature_succeeded_rows"] == 3
    assert quality["pipeline_complete_rows"] == 3
    assert quality["analysis_usable_counts"] == {
        "quality_control": 3,
        "clustering": 3,
        "supervised_exploratory": 0,
        "transcript_analysis": 3,
    }

    frozen = freeze_notebook_dataset(result)
    freeze_summary = redacted_freeze_summary(frozen, result)
    validation = validate_notebook_outputs(context, result, frozen)
    handoff = write_redacted_handoff(context, result, frozen, validation)

    assert freeze_summary["artifact_type"] == "freeze_summary"
    assert validation["artifact_type"] == "validation_summary"
    assert handoff["artifact_type"] == "handoff_summary"
    assert validation["valid"] is True
    assert validation["row_count"] == 3
    assert validation["checksum_count"] == 9
    assert handoff["analysis_usable_counts"] == quality["analysis_usable_counts"]
    handoff_path = (
        PROJECT_ROOT
        / "artifacts"
        / "research"
        / "handoffs"
        / "01_dataset_pipeline.smoke_mode.json"
    )
    persisted_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert persisted_handoff["next_notebook"] == "02_modelling_and_results.ipynb"
    assert not Path(str(persisted_handoff["artifact_path"])).is_absolute()

    displayed = json.dumps(
        {
            "source": source_audit,
            "preview": preview,
            "quality": quality,
            "freeze": freeze_summary,
            "validation": validation,
            "handoff": handoff,
        },
        sort_keys=True,
    )
    assert "http://" not in displayed
    assert "https://" not in displayed
    assert "source_url" not in displayed
    assert "video_title" not in displayed
    assert "total_score" not in displayed
    assert str(PROJECT_ROOT) not in displayed


def test_handoff_rejects_validation_that_is_not_exactly_true(tmp_path: Path) -> None:
    """Break caught: an invalid freeze is handed to Notebook 2."""
    context = prepare_notebook_run(
        PROJECT_ROOT,
        mode="SMOKE_MODE",
        network_opt_in="0",
        config_path="configs/notebook_pipeline.yaml",
        preflight_runner=_ready_preflight,
    )
    result = execute_notebook_pipeline(context)
    frozen = freeze_notebook_dataset(result)
    isolated_context = replace(context, repository_root=tmp_path)

    with pytest.raises(ValueError, match="strict validation"):
        write_redacted_handoff(
            isolated_context, result, frozen, {"valid": False}
        )

    assert not (tmp_path / "artifacts").exists()


def test_executed_notebook_output_scanner_rejects_private_payload(
    tmp_path: Path,
) -> None:
    """Break caught: ignored executed notebooks are trusted without scanning outputs."""
    private_url = "https" + "://" + "youtu.be/" + "realVideo42"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                source="print('redacted')",
                execution_count=1,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text=private_url)],
            )
        ]
    )
    executed = tmp_path / "executed.ipynb"
    nbformat.write(notebook, executed)

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


def _write_executed_output(
    path: Path, output: nbformat.NotebookNode
) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                source="display_aggregate_result()",
                execution_count=1,
                outputs=[output],
            )
        ]
    )
    nbformat.write(notebook, path)
    return path


@pytest.mark.parametrize(
    "output",
    [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text=(
                "sample_id,video_title,total_score\n"
                "N001,Synthetic private presentation,91\n"
            ),
        ),
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="N001,Synthetic private presentation,91",
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "text/plain": (
                    "sample_id  video_title                    total_score\n"
                    "N001       Synthetic private presentation 91\n"
                )
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": "c2FmZS1pbWFnZQ==",
                "text/plain": (
                    "presentation title | score\n"
                    "Synthetic private keynote | 91\n"
                ),
            },
        ),
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text=(
                "Good morning everyone, and thank you for joining me.\n"
                "Today I will explain the approach and the findings.\n"
                "Finally, I will summarize what we learned.\n"
            ),
        ),
        nbformat.v4.new_output(
            "display_data",
            data={"application/octet-stream": "opaque-row-payload"},
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "application/json": {
                    "sample_id": "N001",
                    "duration_seconds": 90.0,
                }
            },
        ),
    ],
    ids=(
        "stream-csv",
        "headerless-stream-row",
        "text-plain-table",
        "unsafe-image-companion",
        "unkeyed-transcript-lines",
        "unknown-mime",
        "identity-json",
    ),
)
def test_executed_notebook_output_scanner_rejects_unkeyed_row_payloads(
    tmp_path: Path, output: nbformat.NotebookNode
) -> None:
    """Break caught: unkeyed row data bypasses JSON-key privacy checks."""
    executed = _write_executed_output(tmp_path / "executed.ipynb", output)

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


@pytest.mark.parametrize(
    "output",
    [
        nbformat.v4.new_output(
            "display_data",
            data={
                "application/json": {
                    "artifact_type": "execution_summary",
                    "row_count": 3,
                    "valid": True,
                }
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(
                    b"\x89PNG\r\n\x1a\nsynthetic"
                ).decode("ascii")
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/jpeg": base64.b64encode(
                    b"\xff\xd8\xffsynthetic"
                ).decode("ascii")
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(
                    b"\x89PNG\r\n\x1a\nsynthetic"
                ).decode("ascii"),
                "text/plain": "<Figure size 640x480 with 1 Axes>",
            },
        ),
    ],
    ids=(
        "aggregate-json",
        "png",
        "jpeg",
        "matplotlib-png",
    ),
)
def test_executed_notebook_output_scanner_allows_bounded_aggregate_outputs(
    tmp_path: Path, output: nbformat.NotebookNode
) -> None:
    """Break caught: fail-closed scanning also blocks approved aggregate outputs."""
    executed = _write_executed_output(tmp_path / "executed.ipynb", output)

    assert validate_executed_notebook_outputs(executed) is True


@pytest.mark.parametrize(
    "payload",
    [
        {
            "artifact_type": "modelling_dataset_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "strict_frozen_dataset",
            "method": "strict_frozen_dataset_validation",
            "uncertainty_method": "not_applicable",
            "model_identifier": "not_applicable",
            "row_count": 24,
            "qc_count": 24,
            "qc_excluded_count": 0,
            "clustering_count": 24,
            "clustering_excluded_count": 0,
            "supervised_count": 24,
            "supervised_excluded_count": 0,
            "transcript_count": 24,
            "transcript_excluded_count": 0,
        },
        {
            "artifact_type": "clustering_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "clustering_cohort_policy",
            "method": "ward_euclidean_robust_scaled_pca",
            "uncertainty_method": "leave_one_feature_out_adjusted_rand_index",
            "model_identifier": "not_applicable",
            "status": "ok",
            "sample_count": 24,
            "feature_count": 12,
            "selected_k": 2,
            "silhouette": 0.41,
            "stability_min": 0.72,
            "stability_median": 0.94,
            "similarity_scale": 3.2,
            "conditional_feature_included": True,
            "conditional_feature_status": "included_single_whisper_language",
        },
        {
            "artifact_type": "cluster_score_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "observed_scores_after_cluster_freeze",
            "method": "posthoc_group_permutation_holm",
            "uncertainty_method": "group_permutation_10000_seed_20260813_holm",
            "model_identifier": "not_applicable",
            "status": "ok",
            "selected_k": 2,
            "sample_count": 24,
            "excluded_sample_count": 0,
            "group_count": 24,
            "comparison_count": 1,
            "minimum_adjusted_p": 0.04,
            "significance_threshold": 0.05,
            "statistically_significant": True,
        },
        {
            "artifact_type": "composite_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "supervised_total_score_cohort_policy",
            "method": "cross_fitted_equal_available_dimensions",
            "uncertainty_method": "descriptive_correlations_no_interval",
            "model_identifier": "not_applicable",
            "sample_count": 24,
            "feature_count": 11,
            "spearman": 0.4,
            "pearson": 0.5,
            "learned_weights": False,
            "calibrated_to_target": False,
        },
        {
            "artifact_type": "model_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "supervised_total_score_cohort_policy",
            "method": "nested_grouped_cross_validation",
            "uncertainty_method": "paired_group_bootstrap_percentile_5000_seed_20260813_95pct",
            "model_identifier": "sklearn_linear_model_ridge",
            "model_name": "ridge",
            "sample_count": 24,
            "oof_prediction_count": 24,
            "mae": 4.1,
            "rmse": 5.2,
            "spearman": 0.4,
            "pearson": 0.5,
            "mae_ci_status": "ok",
            "mae_ci_lower": 3.1,
            "mae_ci_upper": 5.1,
            "fold_count": 5,
            "fold_local_preprocessing": True,
        },
        {
            "artifact_type": "model_comparison_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "supervised_total_score_cohort_policy",
            "method": "paired_model_mae_differences",
            "uncertainty_method": "paired_group_bootstrap_percentile_5000_seed_20260813_95pct",
            "model_identifier": "training_mean_ridge_elastic_net",
            "ridge_minus_training_mean_estimate": -0.4,
            "ridge_minus_training_mean_status": "ok",
            "ridge_minus_training_mean_ci_lower": -1.1,
            "ridge_minus_training_mean_ci_upper": 0.3,
            "ridge_minus_training_mean_valid_replicates": 5000,
            "ridge_minus_training_mean_required_replicates": 5000,
            "ridge_minus_training_mean_includes_zero": True,
            "elastic_net_minus_training_mean_estimate": -0.2,
            "elastic_net_minus_training_mean_status": "ok",
            "elastic_net_minus_training_mean_ci_lower": -0.9,
            "elastic_net_minus_training_mean_ci_upper": 0.5,
            "elastic_net_minus_training_mean_valid_replicates": 5000,
            "elastic_net_minus_training_mean_required_replicates": 5000,
            "elastic_net_minus_training_mean_includes_zero": True,
            "ridge_minus_elastic_net_estimate": -0.2,
            "ridge_minus_elastic_net_status": "ok",
            "ridge_minus_elastic_net_ci_lower": -0.6,
            "ridge_minus_elastic_net_ci_upper": 0.2,
            "ridge_minus_elastic_net_valid_replicates": 5000,
            "ridge_minus_elastic_net_required_replicates": 5000,
            "ridge_minus_elastic_net_includes_zero": True,
        },
        {
            "artifact_type": "qwen_gate_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "transcript_model_cohort_policy",
            "method": "local_qwen_identity_gate",
            "uncertainty_method": "not_applicable",
            "model_identifier": "qwen3.5:9b-q4_K_M",
            "status": "not_run",
            "reason_code": "expected_local_digest_missing",
            "warning": True,
            "attempt_count": 0,
            "calibration_fit": False,
        },
        {
            "artifact_type": "qlora_gate_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "transcript_model_cohort_policy",
            "method": "authority_bound_qlora_gate",
            "uncertainty_method": "not_applicable",
            "model_identifier": "not_applicable",
            "status": "not_run",
            "reason_count": 2,
            "ready": False,
            "training_started": False,
            "metrics_available": False,
            "qlora_reasons": {
                "authority_entry_missing": True,
                "protocol_disables_qlora": True,
            },
        },
        {
            "artifact_type": "modelling_validation_summary",
            "dataset_id": "a" * 64,
            "protocol_id": "b" * 64,
            "included_count": 24,
            "excluded_count": 0,
            "exclusion_basis": "strict_frozen_dataset",
            "method": "fail_closed_modelling_validation",
            "uncertainty_method": "not_applicable",
            "model_identifier": "not_applicable",
            "valid": True,
            "row_count": 24,
            "selected_k": 2,
            "stability_checked": True,
            "stability_status": "passed",
            "cohort_counts_valid": True,
            "one_oof_per_sample": True,
            "fold_local_preprocessing": True,
            "bootstrap_replicates": 5000,
            "bootstrap_valid": True,
            "paired_model_differences_valid": True,
            "cluster_scores_posthoc": True,
            "mae_intervals_overlap": True,
            "mae_overlap_lower": 3.1,
            "mae_overlap_upper": 5.1,
            "qwen_not_run": True,
            "qlora_not_run": True,
        },
    ],
    ids=(
        "dataset",
        "clustering",
        "cluster-score",
        "composite",
        "model",
        "model-comparison",
        "qwen",
        "qlora",
        "validation",
    ),
)
def test_executed_notebook_output_scanner_allows_exact_modelling_aggregates(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """Break caught: Notebook 2 cannot emit one of its reviewed aggregate schemas."""
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        ),
    )

    assert validate_executed_notebook_outputs(executed) is True


@pytest.mark.parametrize("payload", NOTEBOOK1_AGGREGATE_PAYLOADS)
def test_executed_notebook_output_scanner_allows_exact_notebook_one_aggregates(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """Break caught: a reviewed Notebook 1 aggregate cannot pass its own scanner."""
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        ),
    )

    assert validate_executed_notebook_outputs(executed) is True


@pytest.mark.parametrize(
    ("payload", "missing_key"),
    tuple(
        (payload, key)
        for payload in NOTEBOOK1_AGGREGATE_PAYLOADS
        for key in payload
        if key != "artifact_type"
    ),
)
def test_notebook_one_aggregate_allowlist_rejects_every_missing_required_key(
    tmp_path: Path, payload: dict[str, object], missing_key: str
) -> None:
    """Break caught: a partial Notebook 1 aggregate passes as complete evidence."""
    incomplete = dict(payload)
    incomplete.pop(missing_key)
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": incomplete}, metadata={}
        ),
    )

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


@pytest.mark.parametrize(
    ("artifact_type", "nested_key"),
    (
        ("preflight_summary", "check_status_counts"),
        ("quality_summary", "terminal_counts"),
        ("quality_summary", "analysis_usable_counts"),
    ),
)
def test_notebook_one_aggregate_rejects_incomplete_nested_counts(
    tmp_path: Path, artifact_type: str, nested_key: str
) -> None:
    """Break caught: an omitted zero-count category makes totals ambiguous."""
    payload = next(
        dict(candidate)
        for candidate in NOTEBOOK1_AGGREGATE_PAYLOADS
        if candidate["artifact_type"] == artifact_type
    )
    nested = dict(payload[nested_key])
    nested.pop(next(iter(nested)))
    payload[nested_key] = nested
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        ),
    )

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


@pytest.mark.parametrize(
    ("artifact_type", "missing_key"),
    tuple(
        (artifact_type, key)
        for artifact_type in (
            "modelling_dataset_summary",
            "clustering_summary",
            "cluster_score_summary",
            "composite_summary",
            "model_summary",
            "qwen_gate_summary",
            "qlora_gate_summary",
            "modelling_validation_summary",
        )
        for key in sorted(
            notebook_support_module._AGGREGATE_SCHEMAS[artifact_type]
            - {"artifact_type"}
        )
    ),
)
def test_modelling_aggregate_allowlist_rejects_every_missing_required_key(
    tmp_path: Path, artifact_type: str, missing_key: str
) -> None:
    """Break caught: subset-only schemas accept incomplete Notebook 2 evidence."""
    schema = notebook_support_module._AGGREGATE_SCHEMAS[artifact_type]
    payload: dict[str, object] = {}
    for key in schema:
        if key == "artifact_type":
            payload[key] = artifact_type
        elif key == "dataset_id":
            payload[key] = "a" * 64
        elif key == "status":
            payload[key] = "ok"
        elif key == "model_name":
            payload[key] = "ridge"
        elif key == "mae_ci_status":
            payload[key] = "ok"
        elif key == "reason_code":
            payload[key] = "expected_local_digest_missing"
        elif key == "qlora_reasons":
            payload[key] = {}
        elif key in {
            "conditional_feature_included",
            "statistically_significant",
            "learned_weights",
            "calibrated_to_target",
            "fold_local_preprocessing",
            "warning",
            "calibration_fit",
            "ready",
            "training_started",
            "metrics_available",
            "valid",
            "stability_checked",
            "cohort_counts_valid",
            "one_oof_per_sample",
            "bootstrap_valid",
            "cluster_scores_posthoc",
            "mae_intervals_overlap",
            "qwen_not_run",
            "qlora_not_run",
        }:
            payload[key] = True
        else:
            payload[key] = 1
    payload.pop(missing_key)
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        ),
    )

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


def test_modelling_aggregate_allowlist_rejects_unknown_gate_reason(
    tmp_path: Path,
) -> None:
    """Break caught: a free-form model-gate reason bypasses the output policy."""
    payload = {
        "artifact_type": "qlora_gate_summary",
        "status": "not_run",
        "reason_count": 1,
        "ready": False,
        "training_started": False,
        "metrics_available": False,
        "qlora_reasons": {"invented_reason": True},
    }
    executed = _write_executed_output(
        tmp_path / "executed.ipynb",
        nbformat.v4.new_output(
            "display_data", data={"application/json": payload}, metadata={}
        ),
    )

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


@pytest.mark.parametrize(
    "output",
    [
        nbformat.v4.new_output(
            "stream", name="stdout", text="completed_rows=3\nvalid=true"
        ),
        nbformat.v4.new_output(
            "stream", name="stdout", text="Arbitrary prose that looks harmless."
        ),
        nbformat.v4.new_output(
            "stream", name="stdout", text="label,value"
        ),
        nbformat.v4.new_output(
            "stream", name="stdout", text="artifact_path=/tmp/synthetic/output"
        ),
        nbformat.v4.new_output(
            "stream", name="stdout", text="artifact_uri=s3://synthetic-bucket/result"
        ),
        nbformat.v4.new_output(
            "display_data",
            data={"application/json": {"row_count": 3, "valid": True}},
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "application/json": {
                    "artifact_type": "execution_summary",
                    "arbitrary_key": 3,
                }
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "application/json": {
                    "artifact_type": "preflight_summary",
                    "mode": "arbitrary free-form mode",
                }
            },
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(
                    b"\x89PNG\r\n\x1a\nsynthetic"
                ).decode("ascii"),
                "text/plain": "valid=true",
            },
        ),
        nbformat.v4.new_output(
            "display_data", data={"image/png": "not valid base64 or an image"}
        ),
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": base64.b64encode(b"plain synthetic prose").decode(
                    "ascii"
                )
            },
        ),
        nbformat.v4.new_output(
            "display_data", data={"image/svg+xml": "<svg></svg>"}
        ),
    ],
    ids=(
        "legacy-aggregate-stream",
        "arbitrary-prose",
        "single-delimiter-row",
        "posix-path",
        "non-http-uri",
        "json-without-discriminator",
        "json-arbitrary-key",
        "json-arbitrary-string",
        "image-with-legacy-aggregate-text",
        "invalid-base64",
        "wrong-image-magic",
        "svg",
    ),
)
def test_executed_notebook_output_scanner_rejects_non_aggregate_outputs(
    tmp_path: Path, output: nbformat.NotebookNode
) -> None:
    """Break caught: output passes without proving an approved aggregate shape."""
    executed = _write_executed_output(tmp_path / "executed.ipynb", output)

    with pytest.raises(ValueError, match="private output"):
        validate_executed_notebook_outputs(executed)


def test_execution_entrypoint_saves_then_privacy_validates_outputs(
    tmp_path: Path,
) -> None:
    """Break caught: the documented executor never scans the saved notebook."""
    from scripts.execute_dataset_notebook import execute_and_validate

    source = tmp_path / "source.ipynb"
    output = (
        PROJECT_ROOT
        / "artifacts"
        / "research"
        / "executed"
        / "01_dataset_pipeline.executor-test.ipynb"
    )
    output.unlink(missing_ok=True)
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "from IPython.display import display\n"
                    "display({'application/json': {"
                    "'artifact_type': 'execution_summary', "
                    "'row_count': 3, 'valid': True}}, raw=True)"
                )
            ]
        ),
        source,
    )

    try:
        result = execute_and_validate(
            source, output, working_directory=PROJECT_ROOT
        )

        assert result == output
        assert output.is_file()
        assert validate_executed_notebook_outputs(output) is True
    finally:
        output.unlink(missing_ok=True)


def test_dataset_executed_destination_is_validated_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: Notebook 1 can execute toward a public repository path."""
    import scripts.execute_dataset_notebook as execute_dataset_module

    source = tmp_path / "source.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), source)

    class ForbiddenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("execution began before destination validation")

    monkeypatch.setattr(
        execute_dataset_module, "NotebookClient", ForbiddenClient
    )

    with pytest.raises(ValueError, match="private executed-notebook directory"):
        execute_dataset_module.execute_and_validate(
            source,
            PROJECT_ROOT / "notebooks" / "01_dataset_pipeline.executed.ipynb",
            working_directory=PROJECT_ROOT,
        )


def test_dataset_failed_scan_leaves_no_final_or_temporary_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an unsafe Notebook 1 copy is promoted before scanning."""
    import scripts.execute_dataset_notebook as execute_dataset_module

    source = tmp_path / "source.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]),
        source,
    )
    private_root = PROJECT_ROOT / "artifacts" / "research" / "executed"
    output = private_root / "01_dataset_pipeline.failed-scan-test.ipynb"
    output.unlink(missing_ok=True)

    class FakeClient:
        def __init__(self, notebook, **kwargs: object) -> None:
            self.notebook = notebook

        def execute(self) -> object:
            self.notebook.cells[0].execution_count = 1
            self.notebook.cells[0].outputs = [
                nbformat.v4.new_output(
                    "stream", name="stdout", text="unapproved diagnostic"
                )
            ]
            return self.notebook

    monkeypatch.setattr(execute_dataset_module, "NotebookClient", FakeClient)

    try:
        with pytest.raises(ValueError, match="private output"):
            execute_dataset_module.execute_and_validate(
                source,
                output,
                working_directory=PROJECT_ROOT,
            )

        assert not output.exists()
        assert not list(private_root.glob("*.tmp.ipynb"))
    finally:
        output.unlink(missing_ok=True)


def test_notebook_one_executes_with_only_exact_json_aggregate_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: Notebook 1 emits reprs, streams, or incomplete stage totals."""
    from scripts.execute_dataset_notebook import execute_and_validate

    output = (
        PROJECT_ROOT
        / "artifacts"
        / "research"
        / "executed"
        / "01_dataset_pipeline.regression.ipynb"
    )
    output.unlink(missing_ok=True)
    monkeypatch.setenv("THESIS_NOTEBOOK_MODE", "SMOKE_MODE")
    monkeypatch.setenv("THESIS_ALLOW_YOUTUBE_NETWORK", "0")
    monkeypatch.setenv(
        "THESIS_DATASET_CONFIG", "configs/notebook_pipeline.yaml"
    )
    monkeypatch.setenv("THESIS_ALLOW_LEGACY_STAGE_CACHE", "0")
    monkeypatch.delenv("THESIS_FORCE_STAGE", raising=False)

    try:
        execute_and_validate(
            NOTEBOOK_PATH,
            output,
            working_directory=PROJECT_ROOT,
        )
        notebook = nbformat.read(output, as_version=4)
        payload_types: list[str] = []
        for cell in notebook.cells:
            for saved_output in cell.get("outputs", []):
                assert saved_output.get("output_type") == "display_data"
                data = saved_output.get("data")
                assert isinstance(data, dict)
                assert set(data) == {"application/json"}
                payload = data["application/json"]
                assert isinstance(payload, dict)
                artifact_type = payload.get("artifact_type")
                assert isinstance(artifact_type, str)
                assert artifact_type in notebook_support_module._NOTEBOOK1_AGGREGATE_TYPES
                assert set(payload) == set(
                    notebook_support_module._AGGREGATE_SCHEMAS[artifact_type]
                )
                payload_types.append(artifact_type)

        assert payload_types.count("preflight_summary") == 1
        assert payload_types.count("source_audit_summary") == 1
        assert payload_types.count("stage_summary") == 4
        assert payload_types.count("feature_preview_summary") == 1
        assert payload_types.count("quality_summary") == 1
        assert payload_types.count("freeze_summary") == 1
        assert payload_types.count("validation_summary") == 1
        assert payload_types.count("handoff_summary") == 1
        assert len(payload_types) == 11
        assert validate_executed_notebook_outputs(output) is True
    finally:
        output.unlink(missing_ok=True)


def test_pipeline_wrapper_suppresses_third_party_stream_leaks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Break caught: acquisition diagnostics leak a URL or local path into outputs."""
    context = prepare_notebook_run(
        PROJECT_ROOT,
        mode="SMOKE_MODE",
        network_opt_in="0",
        config_path="configs/notebook_pipeline.yaml",
        preflight_runner=_ready_preflight,
    )
    actual_acquire = context.pipeline_config.adapters.acquire

    def leaking_acquire(*args: object, **kwargs: object) -> object:
        print("diagnostic https://private.invalid/item")
        # privacy-scan: synthetic-fixture
        print(r"diagnostic C:\private\item", file=sys.stderr)
        return actual_acquire(*args, **kwargs)

    leaking_adapters = replace(
        context.pipeline_config.adapters,
        acquire=leaking_acquire,
    )
    leaking_config = replace(
        context.pipeline_config,
        adapters=leaking_adapters,
        force_stages=frozenset({"acquisition", "asr"}),
    )

    execute_notebook_pipeline(replace(context, pipeline_config=leaking_config))

    captured = capsys.readouterr()
    displayed = captured.out + captured.err
    assert displayed == ""
