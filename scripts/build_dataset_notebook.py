"""Build the public dataset-pipeline notebook from deterministic source cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "01_dataset_pipeline.ipynb"


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source.strip())


def _code(purpose: str, source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(
        source=source.strip(),
        execution_count=None,
        outputs=[],
        metadata={"tags": [f"purpose-{purpose}"]},
    )


def build_notebook() -> nbformat.NotebookNode:
    """Return a clean tutorial notebook with stable cells and metadata."""
    cells = [
        _markdown(
            """
## Goal

Build an auditable frozen acoustic dataset through the reviewed pipeline. The
default is `SMOKE_MODE`; private-data and network processing require an explicit
authorised full-run environment. Reader-visible results are aggregate or
redacted.
"""
        ),
        _markdown(
            """
## Setup

Resolve the repository, read the source/runtime parameters, and run the local
preflight. This notebook never installs packages or downloads model weights.
`FORCE_STAGE` additionally requires `THESIS_FORCE_STAGE` to name acquisition,
audio, ASR, features, or all. The one-time legacy-cache migration is opt-in via
`THESIS_ALLOW_LEGACY_STAGE_CACHE=1`.
"""
        ),
        _code(
            "setup",
            """
import os
from pathlib import Path
from presentation_assessment.research.notebook_support import (
    bounded_feature_preview,
    display_aggregate_payload,
    execute_notebook_pipeline,
    freeze_notebook_dataset,
    locate_repository_root,
    prepare_notebook_run,
    redacted_freeze_summary,
    stage_summary,
    summarize_preflight,
    summarize_quality,
    summarize_source_audit,
    validate_notebook_outputs,
    write_redacted_handoff,
)

repository_root = locate_repository_root(Path.cwd())
notebook_mode = os.environ.get("THESIS_NOTEBOOK_MODE", "SMOKE_MODE")
network_opt_in = os.environ.get("THESIS_ALLOW_YOUTUBE_NETWORK", "0")
dataset_config = os.environ.get("THESIS_DATASET_CONFIG", "configs/notebook_pipeline.yaml")
source_mode = os.environ.get("THESIS_SOURCE_MODE", "YOUTUBE")
local_source_config = os.environ.get("THESIS_LOCAL_SOURCE_CONFIG", "")
run_context = prepare_notebook_run(
    repository_root,
    mode=notebook_mode,
    network_opt_in=network_opt_in,
    config_path=dataset_config,
    source_mode=source_mode,
    local_source_config=local_source_config or None,
)
display_aggregate_payload(summarize_preflight(run_context))
""",
        ),
        _markdown(
            """
## Source Audit

Confirm that every intended row is countable before any acquisition starts.
Only aggregate availability and verification counts are displayed.
"""
        ),
        _code(
            "source-audit",
            "source_audit = summarize_source_audit(run_context)\n"
            "display_aggregate_payload(source_audit)",
        ),
        _markdown(
            """
## Acquisition

Run the tested resumable orchestration. Smoke mode is synthetic; authorised
full runs use either explicit YouTube acquisition or validated local files.
"""
        ),
        _code(
            "pipeline-run",
            "pipeline_result = execute_notebook_pipeline(run_context)\n"
            "display_aggregate_payload(stage_summary(pipeline_result, \"acquisition\"))",
        ),
        _markdown(
            """
## Audio

Report terminal audio-stage counts. Failed inputs remain represented by an
explicit terminal state.
"""
        ),
        _code(
            "audio-summary",
            "display_aggregate_payload(stage_summary(pipeline_result, \"audio\"))",
        ),
        _markdown(
            """
## Whisper

Report local transcription-stage counts. Full runs bind the exact validated
snapshot and reuse one model instance for the batch.
"""
        ),
        _code(
            "whisper-summary",
            "display_aggregate_payload(stage_summary(pipeline_result, \"asr\"))",
        ),
        _markdown(
            """
## Features

Report terminal feature-stage counts and a bounded aggregate availability
preview. No sample identifiers or feature values are displayed.
"""
        ),
        _code(
            "feature-preview",
            "display_aggregate_payload(stage_summary(pipeline_result, \"features\"))\n"
            "feature_preview = bounded_feature_preview(pipeline_result)\n"
            "display_aggregate_payload(feature_preview)",
        ),
        _markdown(
            """
## Quality

Summarize feature availability, missingness, and warnings without exposing
row-level source material.
"""
        ),
        _code(
            "quality-summary",
            "quality_summary = summarize_quality(pipeline_result)\n"
            "display_aggregate_payload(quality_summary)",
        ),
        _markdown(
            """
## Freeze

Create or reuse the content-addressed immutable dataset and show only its
public-safe identity and aggregate terminal counts.
"""
        ),
        _code(
            "dataset-freeze",
            "frozen_dataset = freeze_notebook_dataset(pipeline_result)\n"
            "display_aggregate_payload("
            "redacted_freeze_summary(frozen_dataset, pipeline_result))",
        ),
        _markdown(
            """
## Checks

Verify row completeness, required terminal records, frozen hashes, and the
privacy boundary before handoff.
"""
        ),
        _code(
            "validation-checks",
            "validation = validate_notebook_outputs("
            "run_context, pipeline_result, frozen_dataset)\n"
            "display_aggregate_payload(validation)",
        ),
        _markdown(
            """
## Next Steps

Write a redacted local handoff summary for the next notebook. The executed full
copy and all row-level artifacts stay inside ignored research artifacts.
"""
        ),
        _code(
            "handoff-summary",
            """if validation.get("valid") is not True:
    raise RuntimeError("Strict frozen-dataset validation failed")
handoff = write_redacted_handoff(run_context, pipeline_result, frozen_dataset, validation)
display_aggregate_payload(handoff)""",
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
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
    """Write the deterministic notebook with LF newlines and no executed state."""
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
