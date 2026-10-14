import json
import os
from pathlib import Path

import presentation_assessment.research.artifacts as artifacts
from presentation_assessment.research.artifacts import (
    StageMetadata,
    atomic_write_json,
    cache_matches,
)


def test_temporary_file_is_never_treated_as_a_complete_cache_entry(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    interrupted_temporary_file = tmp_path / ".metadata.json-interrupted.tmp"
    interrupted_temporary_file.write_text(
        json.dumps({"input_sha256": "input", "config_sha256": "config"}), encoding="utf-8"
    )

    assert cache_matches(metadata_path, "input", "config") is False


def test_matching_hashes_reuse_a_completed_artifact(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    atomic_write_json(
        metadata_path,
        StageMetadata(input_sha256="input", config_sha256="config").to_dict(),
    )

    assert cache_matches(metadata_path, "input", "config") is True


def test_mismatched_hashes_rebuild_a_completed_artifact(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    atomic_write_json(
        metadata_path,
        StageMetadata(input_sha256="old-input", config_sha256="config").to_dict(),
    )

    assert cache_matches(metadata_path, "new-input", "config") is False


def test_json_replacement_uses_a_temporary_file_in_the_target_directory(
    tmp_path: Path, monkeypatch
) -> None:
    metadata_path = tmp_path / "nested" / "metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text('{"version":"old"}', encoding="utf-8")
    replacement: dict[str, Path] = {}
    real_replace = os.replace

    def capture_replace(source: Path | str, destination: Path | str) -> None:
        replacement["source"] = Path(source)
        replacement["destination"] = Path(destination)
        assert metadata_path.read_text(encoding="utf-8") == '{"version":"old"}'
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", capture_replace)

    atomic_write_json(metadata_path, {"version": "new"})

    assert replacement["source"].parent == metadata_path.parent
    assert replacement["destination"] == metadata_path
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {"version": "new"}


def test_text_replacement_uses_a_temporary_file_in_the_target_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: transcript text is partially overwritten on interrupted output."""
    transcript_path = tmp_path / "nested" / "transcript.txt"
    transcript_path.parent.mkdir()
    transcript_path.write_text("old\n", encoding="utf-8")
    replacement: dict[str, Path] = {}
    real_replace = os.replace

    def capture_replace(source: Path | str, destination: Path | str) -> None:
        replacement["source"] = Path(source)
        replacement["destination"] = Path(destination)
        assert transcript_path.read_text(encoding="utf-8") == "old\n"
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", capture_replace)

    artifacts.atomic_write_text(transcript_path, "new\n")

    assert replacement["source"].parent == transcript_path.parent
    assert replacement["destination"] == transcript_path
    assert transcript_path.read_text(encoding="utf-8") == "new\n"
