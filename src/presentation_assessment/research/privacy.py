"""Local-data authorisation gates and staged-content privacy checks."""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AuthorisationError(ValueError):
    """Raised when a local data-authorisation record cannot permit a full run."""


class DataAuthorisation(BaseModel):
    """The explicit local record required before processing private research data."""

    model_config = ConfigDict(extra="forbid", strict=True)

    authorised_by: str = Field(min_length=1)
    source: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    data_scope: list[str] = Field(min_length=1)
    permitted_access: list[str] = Field(min_length=1)
    storage: str = Field(min_length=1)
    retention_until: date
    deletion_plan: str = Field(min_length=1)
    ethics_status: str = Field(min_length=1)
    acknowledged_at: datetime

    @field_validator(
        "authorised_by",
        "source",
        "purpose",
        "storage",
        "deletion_plan",
        "ethics_status",
    )
    @classmethod
    def _require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("data_scope", "permitted_access")
    @classmethod
    def _require_known_values(cls, values: list[str], info: object) -> list[str]:
        if not values:
            raise ValueError("must not be empty")
        cleaned = [value.strip() for value in values]
        if any(not value or value.lower() in {"unknown", "unspecified"} for value in cleaned):
            raise ValueError("must contain known, non-empty values")
        return cleaned

    @field_validator("acknowledged_at")
    @classmethod
    def _require_aware_acknowledgement(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("must include a timezone")
        return value


@dataclass(frozen=True)
class PrivacyViolation:
    """One privacy rule violated by a path proposed for Git staging."""

    path: Path
    code: str
    detail: str


_PRIVATE_PATH_PARTS = {"private"}
_RAW_MEDIA_SUFFIXES = {
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".wav",
    ".webm",
}
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
_TEXT_PAYLOAD_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
_PAYLOAD_ROOTS = {"artifacts", "data", "docs", "notebooks"}
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_YOUTUBE_URL = re.compile(
    r"https?://[^\s\"']*(?:youtube(?:-nocookie)?\.com|youtu\.be)[^\s\"']*",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"']+")
_UNC_PATH = re.compile(  # privacy-scan: synthetic-fixture
    r"(?<!\\)\\\\[^\\/\s]+\\[^\r\n\"']+"
)
_PRIVATE_SCORE = re.compile(
    r"\b(?:advisor|assessor|individual|private)[_-]?scores?\b"
    r"|\b(?:student|participant|candidate|speaker|person)[_-]?(?:id|name)?\b[^\n]*\bscore\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_AUTHORISATION_STATUSES = frozenset(
    {"user_authorised_local_research", "institutional_approval_confirmed"}
)
_SYNTHETIC_EMAIL_DOMAINS = frozenset({"example.com", "example.invalid"})
_SYNTHETIC_FIXTURE_MARKER = "privacy-scan: synthetic-fixture"
_MARKER_ALLOWED_PATHS = frozenset(
    {
        "src/presentation_assessment/research/notebook_support.py",
        "src/presentation_assessment/research/privacy.py",
        "tests/research/test_dataset_notebook.py",
        "tests/research/test_dataset_pipeline.py",
        "tests/research/test_privacy.py",
    }
)


def load_authorisation(path: Path) -> DataAuthorisation:
    """Load a complete local authorisation record without inferring ethics approval."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuthorisationError(f"Missing data_authorisation.yaml: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise AuthorisationError(f"Unable to read data authorisation: {path}") from error

    if not isinstance(payload, dict):
        raise AuthorisationError("data authorisation must be a mapping")

    try:
        return DataAuthorisation.model_validate(payload)
    except ValidationError as error:
        raise AuthorisationError(str(error)) from error


def require_full_run_authorisation(path: Path, now: datetime) -> DataAuthorisation:
    """Return a current authorisation or refuse a private-data full run."""
    authorisation = load_authorisation(path)
    now_utc = _as_utc(now)
    if authorisation.retention_until < now_utc.date():
        raise AuthorisationError("data retention has expired")
    if authorisation.acknowledged_at > now_utc:
        raise AuthorisationError("data authorisation is acknowledged in the future")
    if authorisation.ethics_status not in _AFFIRMATIVE_AUTHORISATION_STATUSES:
        raise AuthorisationError("FULL_RUN requires an explicit affirmative authorisation status")
    return authorisation


def find_staging_violations(
    root: Path,
    staged_paths: Sequence[Path],
    *,
    staged_blob_reader: Callable[[Path, Path], bytes | None] | None = None,
) -> list[PrivacyViolation]:
    """Return privacy violations in files proposed for Git staging."""
    repository_root = root.resolve()
    read_staged_blob = staged_blob_reader or _read_staged_blob
    violations: list[PrivacyViolation] = []

    for staged_path in staged_paths:
        if staged_path.drive or staged_path.root or staged_path.is_absolute():
            violations.append(_violation(staged_path, "absolute_path", "absolute paths cannot be staged"))
            continue
        if ".." in staged_path.parts:
            violations.append(
                _violation(staged_path, "outside_repository", "path leaves the repository")
            )
            continue
        relative_path = Path(staged_path.as_posix())

        path_violation = _path_violation(relative_path)
        if path_violation is not None:
            violations.append(_violation(relative_path, *path_violation))
            continue

        if not _is_scannable_payload(relative_path):
            continue

        content = read_staged_blob(repository_root, relative_path)
        if content is None:
            violations.append(
                _violation(
                    relative_path,
                    "staged_blob_unreadable",
                    "staged content could not be read from the Git index",
                )
            )
            continue
        violations.extend(_content_violations(relative_path, content))

    return violations


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AuthorisationError("current time must include a timezone")
    return value.astimezone(timezone.utc)


def _path_violation(path: Path) -> tuple[str, str] | None:
    parts = {part.lower() for part in path.parts}
    if parts & _PRIVATE_PATH_PARTS:
        return "private_path", "private directories must remain local"
    if path.suffix.lower() in _RAW_MEDIA_SUFFIXES:
        return "raw_media", "raw media must remain local"
    if {".ipynb_checkpoints", "checkpoints"} & parts or path.suffix.lower() in _CHECKPOINT_SUFFIXES:
        return "checkpoint", "model checkpoints must remain local"
    if path.suffix.lower() == ".jsonl":
        return "transcript_jsonl", "transcript JSONL must remain local"
    return None


def _is_redacted_fixture(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    return len(parts) >= 2 and parts[0:2] == ("tests", "fixtures")


def _is_scannable_payload(path: Path) -> bool:
    return path.parts[0].lower() in _PAYLOAD_ROOTS or path.suffix.lower() in _TEXT_PAYLOAD_SUFFIXES


def _read_staged_blob(root: Path, path: Path) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f":{path.as_posix()}"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _content_violations(path: Path, payload: bytes) -> list[PrivacyViolation]:
    content = payload.decode("utf-8", errors="ignore")
    violations: list[PrivacyViolation] = []
    if any(
        not _match_is_explicitly_synthetic(path, content, match, kind="email")
        for match in _EMAIL.finditer(content)
    ):
        violations.append(_violation(path, "email_address", "email addresses must not be staged"))

    checks = (
        ("youtube_url", _YOUTUBE_URL, "YouTube URLs must not be staged"),
        ("local_path", _WINDOWS_ABSOLUTE_PATH, "local absolute paths must not be staged"),
        ("local_path", _UNC_PATH, "local absolute paths must not be staged"),
        (  # privacy-scan: synthetic-fixture
            "private_score",
            _PRIVATE_SCORE,
            "individual scores must not be staged",
        ),
    )
    found_codes: set[str] = set()
    for code, pattern, detail in checks:
        if code in found_codes:
            continue
        matches = tuple(pattern.finditer(content))
        if any(
            not _match_is_explicitly_synthetic(path, content, match, kind=code)
            for match in matches
        ):
            violations.append(_violation(path, code, detail))
            found_codes.add(code)
    if (  # privacy-scan: synthetic-fixture
        "private_score" not in found_codes
        and _contains_row_level_scores(path, content)
    ):
        violations.append(  # privacy-scan: synthetic-fixture
            _violation(path, "private_score", "individual scores must not be staged")
        )
        found_codes.add("private_score")  # privacy-scan: synthetic-fixture
    if path.suffix.lower() == ".py":
        for code in _python_semantic_violation_codes(path, content) - found_codes:
            detail = (
                "individual scores must not be staged"
                if code == "private_score"  # privacy-scan: synthetic-fixture
                else "transcript payloads must not be staged"
            )
            violations.append(_violation(path, code, detail))
    return violations


def _python_semantic_violation_codes(path: Path, content: str) -> set[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"python_parse_error"}
    codes: set[str] = set()
    for node in ast.walk(tree):
        if _python_node_is_marked(path, content, getattr(node, "lineno", 0)):
            continue
        if isinstance(node, ast.Dict):
            pairs = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if (
                {"sample_id", "total_score"} <= pairs.keys()
                and isinstance(pairs["sample_id"], ast.Constant)
                and not _ast_is_blank(pairs["sample_id"])
                and isinstance(pairs["total_score"], ast.Constant)
                and not _ast_is_blank(pairs["total_score"])
            ):
                codes.add("private_score")  # privacy-scan: synthetic-fixture
            for key in ("transcript", "transcription", "text"):
                value = pairs.get(key)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if _private_transcript_literal(value.value):
                        codes.add("transcript_payload")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and any(
                    isinstance(target, ast.Name)
                    and "transcript" in target.id.lower()
                    for target in targets
                )
                and _private_transcript_literal(value.value)
            ):
                codes.add("transcript_payload")
    return codes


def _ast_is_blank(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and value.value in {None, ""}


def _private_transcript_literal(value: str) -> bool:
    normalized = " ".join(value.split()).lower()
    return len(normalized) >= 20 and not any(
        marker in normalized
        for marker in ("synthetic", "redacted", "example", "historical")
    )


def _python_node_is_marked(path: Path, content: str, lineno: int) -> bool:
    if lineno < 1:
        return False
    lines = content.splitlines()
    selected = lines[max(0, lineno - 2) : lineno]
    return _marker_allowed_path(path) and any(
        _SYNTHETIC_FIXTURE_MARKER in line for line in selected
    )


def _marker_allowed_path(path: Path) -> bool:
    return path.as_posix().lower() in _MARKER_ALLOWED_PATHS


def _match_is_explicitly_synthetic(
    path: Path, content: str, match: re.Match[str], *, kind: str
) -> bool:
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    line = content[line_start : None if line_end < 0 else line_end]
    previous_end = max(0, line_start - 1)
    previous_start = content.rfind("\n", 0, previous_end) + 1
    previous_line = content[previous_start:previous_end]
    if (
        path.suffix.lower() == ".py"
        and _marker_allowed_path(path)
        and _SYNTHETIC_FIXTURE_MARKER in (line + previous_line)
    ):
        return True
    value = match.group().rstrip(".,;)").lower()
    if kind == "email":
        domain = value.rsplit("@", maxsplit=1)[-1]
        return domain in _SYNTHETIC_EMAIL_DOMAINS
    if kind == "youtube_url":
        video_id_match = re.search(r"(?:[?&]v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", value)
        if video_id_match is None:
            return False
        video_id = video_id_match.group(1).lower()
        return len(set(video_id)) == 1 or video_id.startswith(("smoke", "synthetic"))
    return False


def _contains_row_level_scores(path: Path, content: str) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        reader = csv.DictReader(io.StringIO(content), delimiter="\t" if suffix == ".tsv" else ",")
        fieldnames = {field.strip().lower() for field in (reader.fieldnames or [])}
        if {"sample_id", "total_score"} <= fieldnames:
            return any(
                str(row.get("sample_id", "")).strip()
                and str(row.get("total_score", "")).strip()
                for row in reader
            )
        return False
    if suffix not in {".json", ".jsonl"}:
        return False
    try:
        payloads = (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if suffix == ".jsonl"
            else [json.loads(content)]
        )
    except (json.JSONDecodeError, TypeError):
        return False
    return any(_has_score_record(payload) for payload in payloads)


def _has_score_record(value: object) -> bool:
    if isinstance(value, list):
        return any(_has_score_record(item) for item in value)
    if not isinstance(value, dict):
        return False
    keys = {str(key).strip().lower(): item for key, item in value.items()}
    if {"sample_id", "total_score"} <= keys.keys():
        return bool(str(keys["sample_id"]).strip() and str(keys["total_score"]).strip())
    return any(_has_score_record(item) for item in value.values())


def _violation(path: Path, code: str, detail: str) -> PrivacyViolation:
    return PrivacyViolation(path=path, code=code, detail=detail)
