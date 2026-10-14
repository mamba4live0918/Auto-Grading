"""Execute Notebook 2 offline, persist its private copy, and fail closed on output."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterator, Mapping

import nbformat
from nbclient import NotebookClient

from presentation_assessment.research.modelling_notebook_support import (
    create_smoke_modelling_dataset,
    finalize_executed_notebook_narrative,
    modelling_artifact_directory,
)
from presentation_assessment.research.notebook_support import (
    validate_executed_notebook_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "notebooks" / "02_modelling_and_results.ipynb"


def execute_and_validate(
    source: Path = DEFAULT_SOURCE,
    output: Path | None = None,
    *,
    working_directory: Path = PROJECT_ROOT,
    mode: str = "SMOKE_MODE",
    dataset_id: str | None = None,
) -> Path:
    """Execute top-to-bottom, scan outputs, then write evidence-derived narrative."""
    root = Path(working_directory).resolve(strict=True)
    normalized_mode = str(mode).strip().upper()
    if normalized_mode not in {"SMOKE_MODE", "FULL_RUN"}:
        raise ValueError("mode must be SMOKE_MODE or FULL_RUN")
    if normalized_mode == "SMOKE_MODE":
        resolved_dataset_id = create_smoke_modelling_dataset(root).dataset_id
    else:
        resolved_dataset_id = str(dataset_id or "").strip()
        if re.fullmatch(r"[0-9a-f]{64}", resolved_dataset_id) is None:
            raise ValueError("FULL_RUN requires a lowercase SHA-256 dataset ID")
    artifact_directory = modelling_artifact_directory(
        root, resolved_dataset_id, normalized_mode
    )
    executed_path = _private_executed_notebook_path(
        root, artifact_directory, output
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(Path(source), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    execution_error: BaseException | None = None
    environment = {
        "THESIS_REPOSITORY_ROOT": str(root),
        "THESIS_MODE": normalized_mode,
        "THESIS_DATASET_ID": resolved_dataset_id,
        "THESIS_ALLOW_NETWORK": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with _temporary_environment(environment):
        try:
            client.execute()
        except BaseException as error:
            execution_error = error
    if execution_error is not None:
        raise execution_error
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{executed_path.stem}-",
            suffix=".tmp.ipynb",
            dir=executed_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        nbformat.write(notebook, temporary_path)
        validate_executed_notebook_outputs(temporary_path)
        finalize_executed_notebook_narrative(temporary_path)
        validate_executed_notebook_outputs(temporary_path)
        os.replace(temporary_path, executed_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return executed_path


def _private_executed_notebook_path(
    repository_root: Path,
    artifact_directory: Path,
    output: Path | None,
) -> Path:
    """Resolve an executed copy only beneath its ignored dataset/mode root."""
    root = Path(repository_root).resolve(strict=True)
    private_root = Path(artifact_directory).resolve()
    candidate = (
        private_root / "02_modelling_and_results.executed.ipynb"
        if output is None
        else Path(output)
    )
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if private_root not in resolved.parents or resolved.suffix.lower() != ".ipynb":
        raise ValueError(
            "executed notebook destination must remain beneath the private modelling artifact directory"
        )
    return resolved


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--working-directory", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("SMOKE_MODE", "FULL_RUN"),
        default=os.environ.get("THESIS_MODE", "SMOKE_MODE").strip().upper(),
    )
    parser.add_argument("--dataset-id", default=os.environ.get("THESIS_DATASET_ID"))
    arguments = parser.parse_args()
    execute_and_validate(
        arguments.source,
        arguments.output,
        working_directory=arguments.working_directory,
        mode=arguments.mode,
        dataset_id=arguments.dataset_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
