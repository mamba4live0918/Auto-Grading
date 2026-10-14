"""Execute Notebook 1, save its outputs, then enforce the output privacy gate."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

from presentation_assessment.research.notebook_support import (
    validate_executed_notebook_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "notebooks" / "01_dataset_pipeline.ipynb"


def execute_and_validate(
    source: Path,
    output: Path,
    *,
    working_directory: Path,
) -> Path:
    """Execute privately, scan a temporary copy, then atomically promote it."""
    root = Path(working_directory).resolve(strict=True)
    executed_path = _private_executed_notebook_path(root, output)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    execution_error: BaseException | None = None
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
        os.replace(temporary_path, executed_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return executed_path


def _private_executed_notebook_path(
    repository_root: Path, output: Path
) -> Path:
    """Resolve an ignored Notebook 1 destination before execution starts."""
    root = Path(repository_root).resolve(strict=True)
    private_root = (root / "artifacts" / "research" / "executed").resolve()
    candidate = Path(output)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if private_root not in resolved.parents or resolved.suffix.lower() != ".ipynb":
        raise ValueError(
            "executed notebook destination must remain beneath the private "
            "executed-notebook directory"
        )
    relative = resolved.relative_to(root).as_posix()
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("private executed-notebook ignore policy is unavailable") from error
    if not ignored:
        raise ValueError("private executed-notebook destination must be git-ignored")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research/executed/01_dataset_pipeline.full.ipynb"),
    )
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    execute_and_validate(
        arguments.source,
        arguments.output,
        working_directory=arguments.working_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
