"""Build the public modelling-results notebook from deterministic source cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "02_modelling_and_results.ipynb"


def _markdown(source: str, *tags: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        source=source.strip(), metadata={"tags": list(tags)} if tags else {}
    )


def _code(purpose: str, source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(
        source=source.strip(),
        execution_count=None,
        outputs=[],
        metadata={"tags": [f"purpose-{purpose}"]},
    )


def build_notebook() -> nbformat.NotebookNode:
    """Return a clean analytical notebook with stable cells and metadata."""
    cells = [
        _markdown(
            """
## tl;dr

This section is populated in the private executed copy only after all aggregate
outputs pass analytical and privacy validation.
""",
            "narrative-tldr",
        ),
        _markdown(
            """
## Context & Methods

This notebook describes acoustic structure and exploratory association with the
advisor-provided overall score. It consumes one immutable dataset through the
strict frozen-dataset gate, uses the frozen cohort and split definitions, and
runs the reviewed clustering, composite, nested-CV baseline, Qwen, and QLoRA
interfaces without duplicating their algorithms.

### Key Assumptions

- The frozen dataset, registry, protocol, and split digests are the analysis authority.
- Clustering is fitted before scores are made available to the post-hoc procedure.
- Supervised preprocessing and tuning are fitted within training folds only.
- The acoustic composite is a fixed raw standardized index, not a learned or 0–100 calibrated grade.
- Speaker identity is not reliably available, so OOF estimates are exploratory and not speaker-independent generalisation evidence.
- The source target is a broad advisor total score; causal, rubric-dimension, personality, and sensitive-attribute claims are out of scope.
- Qwen and QLoRA remain not run unless every reviewed local identity, label, split, artifact, and hardware gate passes.
"""
        ),
        _code(
            "setup",
            """
import os
from pathlib import Path

from presentation_assessment.research.dataset_pipeline import load_strict_frozen_dataset
from presentation_assessment.research.feature_registry import load_feature_registry
from presentation_assessment.research.modelling_notebook_support import (
    aggregate_payloads,
    display_aggregate_payload,
    display_private_png,
    load_analysis_after_strict_validation,
    modelling_artifact_directory,
    render_private_figures,
    resolve_repository_relative,
    run_modelling_analysis,
    validate_model_run_artifacts,
    validate_modelling_results,
    write_private_results,
)

repository_root = Path(os.environ.get("THESIS_REPOSITORY_ROOT", Path.cwd())).resolve(strict=True)
notebook_mode = os.environ.get("THESIS_MODE", "SMOKE_MODE").strip().upper()
dataset_id = os.environ.get("THESIS_DATASET_ID", "").strip()
feature_registry_path = resolve_repository_relative(
    repository_root,
    os.environ.get("THESIS_FEATURE_REGISTRY", "configs/audio_feature_registry.v1.yaml"),
    "feature registry",
)
feature_registry = load_feature_registry(feature_registry_path)
dataset_root = repository_root / "artifacts" / "research" / dataset_id
""",
        ),
        _code(
            "strict-loader",
            """
strict_dataset = load_strict_frozen_dataset(
    dataset_root,
    feature_registry,
    expected_dataset_id=dataset_id,
)
""",
        ),
        _markdown(
            """
## Data & QC

Analysis-specific cohort counts retain every exclusion. The histogram uses the
complete declared 0–100 source scale; individual scores appear only inside the
rendered raster and ignored private audit artifacts.
"""
        ),
        _code(
            "analysis-run",
            """
analysis_dataset = load_analysis_after_strict_validation(strict_dataset, feature_registry)
modelling_result = run_modelling_analysis(strict_dataset, analysis_dataset, feature_registry)
validation_payload = validate_modelling_results(modelling_result)
if validation_payload.get("valid") is not True:
    raise RuntimeError("Modelling result validation failed")
artifact_directory = modelling_artifact_directory(repository_root, dataset_id, notebook_mode)
figure_paths = render_private_figures(modelling_result, artifact_directory / "figures")
private_result_paths = write_private_results(
    modelling_result,
    artifact_directory / "results",
    figure_paths=figure_paths,
)
validate_model_run_artifacts(
    modelling_result,
    artifact_directory / "results",
    artifact_directory / "figures",
)
notebook_payloads = aggregate_payloads(modelling_result)
payloads_by_type = {}
for payload in notebook_payloads:
    payloads_by_type.setdefault(payload["artifact_type"], []).append(payload)
""",
        ),
        _code(
            "data-summary",
            """
display_aggregate_payload(payloads_by_type["modelling_dataset_summary"][0])
display_private_png(figure_paths["target_histogram"])
""",
        ),
        _markdown(
            """
## Acoustic Clustering

Ward clustering is score-blind. Candidate k is selected by the frozen
silhouette/tolerance rule, and leave-one-feature-out adjusted Rand index reports
solution stability. PCA markers duplicate colour encoding for accessibility.
"""
        ),
        _code(
            "clustering-results",
            """
display_aggregate_payload(payloads_by_type["clustering_summary"][0])
display_private_png(figure_paths["pca_scatter"])
display_private_png(figure_paths["similarity_heatmap"])
display_private_png(figure_paths["ward_dendrogram"])
""",
        ),
        _markdown(
            """
## Score Relationships

Scores enter only after the cluster solution is frozen. Group-level
permutations are exploratory. The fixed acoustic composite uses declared
directions and fold-local robust scaling without learned weights.
"""
        ),
        _code(
            "score-relationships",
            """
display_aggregate_payload(payloads_by_type["cluster_score_summary"][0])
display_aggregate_payload(payloads_by_type["composite_summary"][0])
display_private_png(figure_paths["composite_scatter"])
""",
        ),
        _markdown(
            """
## Baselines

Training-fold mean, Ridge, and ElasticNet receive one OOF prediction per
eligible observation. Outer and inner preprocessing/tuning evidence is retained
in ignored private results; metrics use score points and correlations.
"""
        ),
        _code(
            "baseline-results",
            """
for model_payload in payloads_by_type["model_summary"]:
    display_aggregate_payload(model_payload)
display_aggregate_payload(payloads_by_type["model_comparison_summary"][0])
display_private_png(figure_paths["oof_scatter"])
""",
        ),
        _markdown(
            """
## Qwen Calibration

The local Qwen gate requires a predeclared exact model digest. Without verified
raw outputs, affine calibration is also not fitted; no placeholder prediction
is created.
"""
        ),
        _code(
            "qwen-gate",
            "display_aggregate_payload(payloads_by_type[\"qwen_gate_summary\"][0])",
        ),
        _markdown(
            """
## QLoRA Gate

QLoRA gate evaluation is policy evidence only. A failed gate produces no
training command, checkpoint, or effectiveness metric, and never launches a
training process.
"""
        ),
        _code(
            "qlora-gate",
            "display_aggregate_payload(payloads_by_type[\"qlora_gate_summary\"][0])",
        ),
        _markdown(
            """
## Results

The validation record reconciles cohort counts, selected k and stability,
post-hoc ordering, exactly-once OOF coverage, fold-local evidence, bootstrap
metadata, and model-gate outcomes. MAE intervals resample whole frozen groups.
"""
        ),
        _code(
            "validation-results",
            """
display_aggregate_payload(payloads_by_type["modelling_validation_summary"][0])
display_private_png(figure_paths["mae_comparison"])
""",
        ),
        _markdown(
            """
## Takeaways

This section is populated in the private executed copy only after all executed
outputs pass privacy scanning and analytical validation.
""",
            "narrative-takeaways",
        ),
    ]
    cell_ids = (
        "tldr",
        "context-methods",
        "setup",
        "strict-loader",
        "data-qc",
        "analysis-run",
        "data-summary",
        "acoustic-clustering",
        "clustering-results",
        "score-relationships",
        "score-results",
        "baselines",
        "baseline-results",
        "qwen-calibration",
        "qwen-gate",
        "qlora-gate-section",
        "qlora-gate",
        "results",
        "validation-results",
        "takeaways",
    )
    for cell, cell_id in zip(cells, cell_ids, strict=True):
        cell["id"] = cell_id
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9",
            },
        },
    )


def write_notebook(path: Path = DEFAULT_OUTPUT) -> Path:
    """Write the deterministic notebook with no execution state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    write_notebook(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
