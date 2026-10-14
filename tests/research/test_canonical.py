from pathlib import Path

import nbformat
import pytest

from presentation_assessment.research.canonical import (
    canonical_json_bytes,
    canonical_notebook_bytes,
    sha256_bytes,
    sha256_file,
)


def make_notebook(*, execution_count: int, output_text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('stable')",
                id="stable-code-cell",
                execution_count=execution_count,
                outputs=[nbformat.v4.new_output("stream", text=output_text)],
            )
        ],
        metadata={"widgets": {"application/vnd.jupyter.widget-state+json": {}}},
    )


def write_notebook(path: Path, notebook: nbformat.NotebookNode) -> Path:
    nbformat.write(notebook, path)
    return path


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})


def test_canonical_json_normalises_unicode_and_line_endings() -> None:
    assert canonical_json_bytes({"text": "e\u0301\r\nnext\rlast"}) == canonical_json_bytes(
        {"text": "é\nnext\nlast"}
    )


def test_canonical_json_normalises_dictionary_keys() -> None:
    assert canonical_json_bytes({"e\u0301\rkey": 1}) == canonical_json_bytes(
        {"é\nkey": 1}
    )


def test_canonical_json_rejects_normalised_key_collisions() -> None:
    with pytest.raises(ValueError, match="key collision"):
        canonical_json_bytes({"é": 1, "e\u0301": 2})


def test_canonical_json_normalises_crlf_dictionary_keys() -> None:
    assert canonical_json_bytes({"line\r\nkey": 1}) == canonical_json_bytes(
        {"line\nkey": 1}
    )


def test_sha256_helpers_hash_raw_bytes(tmp_path: Path) -> None:
    payload = b"\x00binary\r\n"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_bytes(payload) == "cc209d417c245b0f79ed422778b6961c80f318ea0f6c834b6075bed54b2f0ae8"
    assert sha256_file(path) == "cc209d417c245b0f79ed422778b6961c80f318ea0f6c834b6075bed54b2f0ae8"


def test_notebook_hash_ignores_outputs_and_execution_counts(tmp_path: Path) -> None:
    left = make_notebook(execution_count=1, output_text="first")
    right = make_notebook(execution_count=99, output_text="second")
    left_path = write_notebook(tmp_path / "left.ipynb", left)
    right_path = write_notebook(tmp_path / "right.ipynb", right)

    assert canonical_notebook_bytes(left_path) == canonical_notebook_bytes(right_path)
