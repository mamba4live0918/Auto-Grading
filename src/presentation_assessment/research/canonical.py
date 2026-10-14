"""Canonical representations and SHA-256 identities for research artifacts."""

import hashlib
import unicodedata
from pathlib import Path

import nbformat
import rfc8785


def _normalise(value: object) -> object:
    if isinstance(value, str):
        return _normalise_text(value)
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        normalised: dict[str, object] = {}
        for key, item in value.items():
            normalised_key = _normalise_text(str(key))
            if normalised_key in normalised:
                raise ValueError("dictionary key collision after canonical normalization")
            normalised[normalised_key] = _normalise(item)
        return normalised
    return value


def _normalise_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def canonical_json_bytes(value: object) -> bytes:
    """Return an RFC 8785 JSON representation with normalised text."""
    return rfc8785.dumps(_normalise(value))


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's raw bytes."""
    return sha256_bytes(path.read_bytes())


def canonical_notebook_bytes(path: Path) -> bytes:
    """Return a notebook source identity that ignores executed state."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)
    notebook.metadata.pop("widgets", None)
    return canonical_json_bytes(notebook)
