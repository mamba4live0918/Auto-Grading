"""Source-tree snapshots that preserve dirty working-tree provenance."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_json_bytes, canonical_notebook_bytes, sha256_bytes, sha256_file


_SOURCE_DIRECTORIES = ("src", "configs", "scripts", "environments", "notebooks")


class ProvenanceError(RuntimeError):
    """A structured failure to establish reconstructible Git provenance."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class SourceSnapshot:
    """Git and content identities for source material used by a research run."""

    git_commit: str
    git_dirty: bool
    tree_sha256: str
    files: dict[str, str]
    git_diff_patch: bytes
    untracked_files: dict[str, str]

    @property
    def git_diff_sha256(self) -> str:
        """Return the identity of the reconstruction patch."""
        return sha256_bytes(self.git_diff_patch)


def _git_output(
    root: Path, *args: str, failure_code: str = "git_command_failed"
) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise ProvenanceError("git_unavailable", "Git is unavailable for provenance") from error
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProvenanceError(failure_code, "Git provenance command could not complete") from error
    if result.returncode != 0:
        raise ProvenanceError(failure_code, "Git provenance command failed")
    return result.stdout


def _git_paths(root: Path, *args: str) -> set[str]:
    return {
        path.decode("utf-8")
        for path in _git_output(root, *args).split(b"\0")
        if path
    }


def _is_relevant_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return "private" not in {part.lower() for part in path.parts} and any(
        relative_path.startswith(f"{directory}/") for directory in _SOURCE_DIRECTORIES
    )


def _relevant_files(root: Path) -> dict[str, Path]:
    candidate_paths = _git_paths(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    )
    return {
        relative_path: root / relative_path
        for relative_path in sorted(candidate_paths)
        if _is_relevant_path(relative_path) and (root / relative_path).is_file()
    }


def _file_hash(path: Path) -> str:
    if path.suffix == ".ipynb":
        return sha256_bytes(canonical_notebook_bytes(path))
    return sha256_file(path)


def _untracked_paths(root: Path) -> set[str]:
    return _git_paths(root, "ls-files", "-z", "--others", "--exclude-standard")


def _require_repository_with_head(root: Path) -> str:
    top_level = _git_output(
        root,
        "rev-parse",
        "--show-toplevel",
        failure_code="not_repository",
    ).decode("utf-8").strip()
    try:
        if Path(top_level).resolve() != root:
            raise ProvenanceError("not_repository", "Input must be the repository root")
    except OSError as error:
        raise ProvenanceError("not_repository", "Repository root could not be resolved") from error
    commit = _git_output(
        root,
        "rev-parse",
        "--verify",
        "HEAD",
        failure_code="missing_head",
    ).decode("utf-8").strip()
    if not commit:
        raise ProvenanceError("missing_head", "Repository has no HEAD commit")
    return commit


def snapshot_source_tree(root: Path) -> SourceSnapshot:
    """Capture relevant source hashes plus Git's dirty working-tree material."""
    root = root.resolve()
    git_commit = _require_repository_with_head(root)
    relevant_paths = _relevant_files(root)
    files = {relative_path: _file_hash(path) for relative_path, path in relevant_paths.items()}
    untracked_paths = _untracked_paths(root)
    untracked_files = {
        relative_path: file_hash
        for relative_path, file_hash in files.items()
        if relative_path in untracked_paths
    }
    git_diff = _git_output(root, "diff", "--binary", "HEAD")
    git_status = _git_output(root, "status", "--porcelain")

    return SourceSnapshot(
        git_commit=git_commit,
        git_dirty=bool(git_status),
        tree_sha256=sha256_bytes(canonical_json_bytes(files)),
        files=files,
        git_diff_patch=git_diff,
        untracked_files=untracked_files,
    )


def write_source_snapshot(snapshot: SourceSnapshot, path: Path) -> None:
    """Write canonical metadata and its sibling Git reconstruction patch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "git_diff.patch").write_bytes(snapshot.git_diff_patch)
    metadata = {
        "git_commit": snapshot.git_commit,
        "git_dirty": snapshot.git_dirty,
        "tree_sha256": snapshot.tree_sha256,
        "files": snapshot.files,
        "git_diff_sha256": snapshot.git_diff_sha256,
        "untracked_files": snapshot.untracked_files,
    }
    path.write_bytes(canonical_json_bytes(metadata))
