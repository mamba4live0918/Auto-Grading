import json
import subprocess
from pathlib import Path

import nbformat
import pytest
import presentation_assessment.research.provenance as provenance

from presentation_assessment.research.canonical import (
    canonical_json_bytes,
    canonical_notebook_bytes,
    sha256_bytes,
    sha256_file,
)
from presentation_assessment.research.provenance import (
    snapshot_source_tree,
    write_source_snapshot,
)


def run_git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def initialise_repository(root: Path) -> None:
    run_git(root, "init", "--quiet")
    run_git(root, "config", "user.email", "tests@example.invalid")
    run_git(root, "config", "user.name", "Research Tests")


def commit_empty_head(root: Path) -> None:
    run_git(root, "commit", "--quiet", "--allow-empty", "-m", "initial source")


def write_notebook(path: Path, *, execution_count: int, output_text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('stable')",
                id="stable-code-cell",
                execution_count=execution_count,
                outputs=[nbformat.v4.new_output("stream", text=output_text)],
            )
        ]
    )
    nbformat.write(notebook, path)
    return path


def test_snapshot_records_relevant_tracked_and_untracked_hashes(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    tracked_path = tmp_path / "src" / "module.py"
    tracked_path.parent.mkdir()
    tracked_path.write_text("value = 1\n", encoding="utf-8")
    run_git(tmp_path, "add", "src/module.py")
    run_git(tmp_path, "commit", "--quiet", "-m", "initial source")
    tracked_path.write_text("value = 2\n", encoding="utf-8")
    run_git(tmp_path, "add", "src/module.py")
    untracked_path = tmp_path / "configs" / "local.yaml"
    untracked_path.parent.mkdir()
    untracked_path.write_text("mode: smoke\n", encoding="utf-8")

    snapshot = snapshot_source_tree(tmp_path)

    expected_files = {
        "configs/local.yaml": sha256_file(untracked_path),
        "src/module.py": sha256_file(tracked_path),
    }
    assert snapshot.git_commit == run_git(tmp_path, "rev-parse", "HEAD").decode().strip()
    assert snapshot.git_dirty is True
    assert snapshot.files == expected_files
    assert snapshot.untracked_files == {"configs/local.yaml": expected_files["configs/local.yaml"]}
    assert snapshot.tree_sha256 == sha256_bytes(canonical_json_bytes(expected_files))
    assert snapshot.git_diff_patch == run_git(tmp_path, "diff", "--binary", "HEAD")
    assert snapshot.git_diff_sha256 == sha256_bytes(run_git(tmp_path, "diff", "--binary", "HEAD"))


def test_snapshot_uses_canonical_notebook_hashes_and_writes_json(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    commit_empty_head(tmp_path)
    notebook_path = write_notebook(
        tmp_path / "notebooks" / "analysis.ipynb", execution_count=7, output_text="run output"
    )

    snapshot = snapshot_source_tree(tmp_path)
    output_path = tmp_path / "source_snapshot.json"
    write_source_snapshot(snapshot, output_path)

    assert snapshot.files == {"notebooks/analysis.ipynb": sha256_bytes(canonical_notebook_bytes(notebook_path))}
    assert snapshot.untracked_files == snapshot.files
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "files": snapshot.files,
        "git_commit": snapshot.git_commit,
        "git_diff_sha256": snapshot.git_diff_sha256,
        "git_dirty": snapshot.git_dirty,
        "tree_sha256": snapshot.tree_sha256,
        "untracked_files": snapshot.untracked_files,
    }
    patch_path = output_path.with_name("git_diff.patch")
    assert patch_path.read_bytes() == snapshot.git_diff_patch
    assert snapshot.git_diff_sha256 == sha256_bytes(patch_path.read_bytes())


def test_snapshot_excludes_git_ignored_private_files(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    commit_empty_head(tmp_path)
    public_path = tmp_path / "configs" / "pipeline.yaml"
    public_path.parent.mkdir()
    public_path.write_text("mode: smoke\n", encoding="utf-8")
    private_path = tmp_path / "configs" / "private" / "advisor_scores.csv"
    private_path.parent.mkdir()
    private_path.write_text("student,score\nprivate,100\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("configs/private/\n", encoding="utf-8")

    snapshot = snapshot_source_tree(tmp_path)

    assert snapshot.files == {"configs/pipeline.yaml": sha256_file(public_path)}
    assert "configs/private/advisor_scores.csv" not in snapshot.files
    assert snapshot.untracked_files == snapshot.files


def test_snapshot_rejects_a_non_repository(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as raised:
        provenance.snapshot_source_tree(tmp_path)

    assert raised.value.code == "not_repository"


def test_snapshot_rejects_a_repository_without_head(tmp_path: Path) -> None:
    initialise_repository(tmp_path)

    with pytest.raises(RuntimeError) as raised:
        provenance.snapshot_source_tree(tmp_path)

    assert raised.value.code == "missing_head"
