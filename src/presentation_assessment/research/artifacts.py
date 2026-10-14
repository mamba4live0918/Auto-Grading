"""Atomic, hash-addressed metadata helpers for resumable research stages."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .canonical import canonical_json_bytes


@dataclass(frozen=True)
class StageMetadata:
    """The stable inputs required to decide whether a stage may be reused."""

    input_sha256: str
    config_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the metadata fields stored beside an artifact."""
        return {
            "input_sha256": self.input_sha256,
            "config_sha256": self.config_sha256,
        }


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace ``path`` with canonical JSON written in its directory."""
    _atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_text(path: Path, value: str) -> None:
    """Atomically replace ``path`` with UTF-8 text written in its directory."""
    _atomic_write_bytes(path, value.encode("utf-8"))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}-",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def cache_matches(path: Path, input_sha256: str, config_sha256: str) -> bool:
    """Return whether a completed metadata file matches the required cache key."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("input_sha256") == input_sha256
        and payload.get("config_sha256") == config_sha256
    )
