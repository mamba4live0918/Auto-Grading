"""Read-only local prerequisite checks for thesis research runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from .canonical import canonical_json_bytes, sha256_bytes
from .cuda_runtime import inspect_cuda_runtime
from .local_manifest_migration import migrate_manifest_to_local_catalog
from .local_source import load_local_catalog, load_local_source_config
from .privacy import AuthorisationError, require_full_run_authorisation
from .source_manifest import load_source_manifest, validate_source_rows


class PreflightStatus(str, Enum):
    """Outcome of one prerequisite check."""

    PASS = "pass"
    BLOCKED = "blocked"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PreflightCheck:
    """One named, non-mutating prerequisite observation."""

    name: str
    status: PreflightStatus
    detail: str
    remediation: str
    required: bool


@dataclass(frozen=True)
class PreflightReport:
    """The ordered outcome of a mode-specific prerequisite inspection."""

    mode: str
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(
            check.status is PreflightStatus.PASS for check in self.checks if check.required
        )

    def by_name(self, name: str) -> PreflightCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


class PreflightProbes(Protocol):
    """Observations kept injectable so tests never inspect local tooling or models."""

    def command_version(self, command: str) -> str | None: ...

    def disk_free_bytes_at(self, path: Path) -> int: ...

    def directory_exists(self, path: Path) -> bool: ...

    def python_environment_matches(self, root: Path) -> bool: ...

    def lock_identity_matches(self, root: Path) -> bool: ...

    def local_whisper_snapshot(
        self, model_name: str, revision: str, snapshot_sha256: str
    ) -> str | None: ...

    def local_qwen_digest(self) -> str | None: ...

    def cuda_runtime_ready(self) -> bool: ...

    def directory_is_writable(self, path: Path) -> bool: ...

    def media_duration(self, path: Path) -> float: ...


class SystemPreflightProbes:
    """Read-only local probes; none install packages, download models, or create files."""

    def __init__(
        self, process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    ) -> None:
        self._process_runner = process_runner

    def command_version(self, command: str) -> str | None:
        executable = shutil.which(command)
        if executable is None:
            return None
        try:
            result = self._process_runner(
                [executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.splitlines()[0] if result.stdout else executable

    def disk_free_bytes_at(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def directory_exists(self, path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    def python_environment_matches(self, root: Path) -> bool:
        try:
            expected = _venv_python(root).resolve()
            running = Path(sys.executable).resolve()
        except OSError:
            return False
        return running == expected and sys.version_info[:2] == (3, 11)

    def lock_identity_matches(self, root: Path) -> bool:
        environment = os.environ.copy()
        environment["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv311")
        commands = (
            ["uv", "lock", "--check", "--offline"],
            [
                "uv",
                "sync",
                "--locked",
                "--offline",
                "--dry-run",
                "--check",
                "--extra",
                "dev",
                "--extra",
                "real",
                "--extra",
                "research",
            ],
            ["uv", "pip", "check", "--python", str(_venv_python(root))],
        )
        try:
            results = [
                self._process_runner(
                    command,
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=environment,
                )
                for command in commands
            ]
        except (OSError, subprocess.TimeoutExpired):
            return False
        return all(result.returncode == 0 for result in results)

    def local_whisper_snapshot(
        self, model_name: str, revision: str, snapshot_sha256: str
    ) -> str | None:
        explicit_path = os.environ.get("PRESENTATION_ASSESSMENT_WHISPER_SNAPSHOT")
        cache_model_directory = _hf_cache_model_directory(model_name)
        if cache_model_directory is None or not _safe_snapshot_revision(revision):
            return None
        cache_root = _huggingface_hub_cache()
        candidate = (
            Path(explicit_path)
            if explicit_path
            else cache_root / cache_model_directory / "snapshots" / revision
        )
        return _validated_whisper_snapshot(
            candidate,
            cache_root,
            cache_model_directory,
            revision,
            snapshot_sha256,
        )

    def local_qwen_digest(self) -> str | None:
        return os.environ.get("PRESENTATION_ASSESSMENT_QWEN_DIGEST") or None

    def cuda_runtime_ready(self) -> bool:
        return inspect_cuda_runtime().ready

    def directory_is_writable(self, path: Path) -> bool:
        try:
            existing = path
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            return existing.is_dir() and os.access(existing, os.W_OK)
        except OSError:
            return False

    def media_duration(self, path: Path) -> float:
        from .audio_pipeline import probe_audio

        return probe_audio(path).duration_seconds


_MODES = frozenset({"SMOKE_MODE", "RESUME", "FORCE_STAGE", "FULL_RUN"})
_FULL_DATA_MODES = frozenset({"RESUME", "FORCE_STAGE", "FULL_RUN"})
_MINIMUM_FREE_DISK_BYTES = 40 * 1024**3
_MANIFEST_COUNT_KEYS = ("total_rows", "observed_score_rows", "not_available_rows")
_WHISPER_SNAPSHOT_FILES = ("config.json", "model.bin", "tokenizer.json")


def run_preflight(
    root: Path,
    config_path: Path,
    mode: str,
    *,
    probes: PreflightProbes | None = None,
    source_mode: str = "YOUTUBE",
    local_source_config: Path | None = None,
) -> PreflightReport:
    """Inspect local prerequisites without installing, downloading, or writing data."""
    if mode not in _MODES:
        raise ValueError(f"unsupported preflight mode: {mode}")
    normalized_source_mode = str(source_mode).strip().upper()
    if normalized_source_mode not in {"YOUTUBE", "LOCAL"}:
        raise ValueError(f"unsupported source mode: {normalized_source_mode}")

    try:
        root = root.resolve()
    except OSError:
        return _unresolved_root_report(mode)
    resolved_config_path = config_path if config_path.is_absolute() else root / config_path
    active_probes = probes or SystemPreflightProbes()
    full_data_mode = mode in _FULL_DATA_MODES
    config, config_check = _load_config(resolved_config_path, root)
    checks: list[PreflightCheck] = [
        _check(
            "repository_root",
            _path_is_dir(root),
            "repository root is available",
            "Use a local repository root before running research.",
        ),
        config_check,
        _check_command("ffmpeg", active_probes),
        _check_command("ffprobe", active_probes),
        _check_disk(root, active_probes),
        _check(
            "venv311",
            _safe_bool(lambda: active_probes.directory_exists(root / ".venv311")),
            ".venv311 is available",
            "Create or restore the locked .venv311 environment.",
        ),
        _check(
            "python_environment",
            _safe_bool(lambda: active_probes.python_environment_matches(root)),
            "running Python is repository .venv311 Python 3.11",
            "Run preflight with the repository .venv311 Python 3.11 interpreter.",
        ),
        _check(
            "lock_identity",
            _safe_bool(lambda: active_probes.lock_identity_matches(root)),
            "dependency lock identity is available",
            "Run uv lock --check and restore the matching uv.lock before research.",
        ),
    ]

    if full_data_mode and config is not None:
        checks.extend(_check_private_data(root, config))
    else:
        checks.extend(
            (
                _skipped("data_authorisation", "Not needed for smoke mode."),
                _skipped("source_manifest", "Not needed for smoke mode."),
            )
        )

    if full_data_mode and normalized_source_mode == "LOCAL":
        checks.append(
            _check_local_source(
                local_source_config, active_probes, root, config
            )
        )
    else:
        checks.append(_skipped("local_source", "Not needed for this source mode."))

    checks.append(_check_cuda_runtime(full_data_mode, active_probes))
    checks.append(_check_whisper(config, full_data_mode, active_probes))
    checks.append(_check_qwen(mode, active_probes))
    checks.append(
        _check_artifact_access(
            root,
            config,
            active_probes,
            key="research_artifact_root",
            name="artifact_write_access",
            detail="research artifact location is writable",
        )
    )
    checks.append(
        _check_artifact_access(
            root,
            config,
            active_probes,
            key="model_artifact_root",
            name="model_artifact_write_access",
            detail="model artifact location is writable",
        )
    )
    return PreflightReport(mode=mode, checks=tuple(checks))


def _load_config(path: Path, root: Path) -> tuple[dict[str, Any] | None, PreflightCheck]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, _blocked(
            "configuration",
            "configuration is unavailable or invalid",
            "Restore a readable notebook pipeline configuration.",
        )
    if not isinstance(payload, dict):
        return None, _blocked(
            "configuration",
            "configuration must be a mapping",
            "Restore a mapping-based notebook pipeline configuration.",
        )
    if not _configured_paths_are_valid(root, payload):
        return None, _blocked(
            "configuration",
            "configured paths are outside their approved repository locations",
            "Use repository-local private-data and artifact paths from the approved layout.",
        )
    return payload, _passed("configuration", "configuration is readable")


def _check_command(command: str, probes: PreflightProbes) -> PreflightCheck:
    try:
        available = probes.command_version(command) is not None
    except (OSError, subprocess.TimeoutExpired):
        available = False
    return _check(
        command,
        available,
        f"{command} is available locally",
        f"Install {command} locally before a research run; preflight will not install it.",
    )


def _check_disk(root: Path, probes: PreflightProbes) -> PreflightCheck:
    try:
        free_bytes = probes.disk_free_bytes_at(root)
    except (OSError, subprocess.TimeoutExpired):
        free_bytes = -1
    return _check(
        "free_disk",
        free_bytes >= _MINIMUM_FREE_DISK_BYTES,
        "at least 40 GiB free disk is available",
        "Free at least 40 GiB of local disk space before research.",
    )


def _check_private_data(root: Path, config: dict[str, Any]) -> tuple[PreflightCheck, PreflightCheck]:
    authorisation_path = _configured_path(root, config, "data_authorisation")
    manifest_path = _configured_path(root, config, "source_manifest")
    try:
        require_full_run_authorisation(authorisation_path, datetime.now(timezone.utc))
        authorisation_check = _passed(
            "data_authorisation", "current local data authorisation is recorded"
        )
    except AuthorisationError:
        authorisation_check = _blocked(
            "data_authorisation",
            "current local data authorisation is missing or invalid",
            "Record valid local authorisation before processing private research data.",
        )

    try:
        audit = validate_source_rows(load_source_manifest(manifest_path))
        expected_counts = _configured_manifest_counts(config)
        valid = not (
            audit.invalid_rows
            or audit.identity_errors
            or audit.unresolved_rows
            or audit.duplicate_urls
            or audit.duplicate_video_ids
            or expected_counts is None
            or any(getattr(audit, key) != value for key, value in expected_counts.items())
        )
    except (OSError, ValueError, TypeError):
        valid = False
    manifest_check = _check(
        "source_manifest",
        valid,
        "source manifest is complete and verified",
        "Repair the private source manifest before processing local research data.",
    )
    return authorisation_check, manifest_check


def _check_local_source(
    config_path: Path | None,
    probes: PreflightProbes,
    root: Path,
    dataset_config: dict[str, Any] | None,
) -> PreflightCheck:
    valid = False
    if config_path is not None and Path(config_path).is_absolute():
        try:
            duration_probe = getattr(probes, "media_duration")
            catalog = load_local_catalog(
                load_local_source_config(Path(config_path)),
                duration_probe=duration_probe,
            )
            if dataset_config is None:
                raise ValueError("dataset configuration is unavailable")
            old_rows = load_source_manifest(
                _configured_path(root, dataset_config, "source_manifest")
            )
            migration = migrate_manifest_to_local_catalog(old_rows, catalog.rows)
            valid = (
                len(catalog.rows) == 85
                and catalog.available_count == 83
                and catalog.unavailable_count == 2
                and len(migration.active_rows) == 85
                and len(migration.retired_rows) == 15
            )
        except (AttributeError, OSError, TypeError, ValueError):
            valid = False
    return _check(
        "local_source",
        valid,
        "local source config, catalog, counts, root, files, hashes, and probes are valid",
        "Repair the private local source config or catalog before processing local files.",
    )


def _check_whisper(
    config: dict[str, Any] | None, required: bool, probes: PreflightProbes
) -> PreflightCheck:
    if not required:
        return _skipped("whisper_snapshot", "Not needed for smoke mode.")
    model_name = str(config.get("whisper_model", "")) if config is not None else ""
    revision = str(config.get("whisper_revision", "")) if config is not None else ""
    snapshot_sha256 = (
        str(config.get("whisper_snapshot_sha256", "")) if config is not None else ""
    )
    local_only = config is not None and config.get("whisper_local_files_only") is True
    settings_valid = config is not None and _whisper_settings_are_valid(config)
    try:
        has_snapshot = bool(
            model_name
            and revision
            and _is_sha256(snapshot_sha256)
            and local_only
            and settings_valid
            and probes.local_whisper_snapshot(model_name, revision, snapshot_sha256)
        )
    except (OSError, subprocess.TimeoutExpired):
        has_snapshot = False
    return _check(
        "whisper_snapshot",
        has_snapshot,
        "local Whisper snapshot is available",
        "Provision the approved Whisper snapshot locally; preflight will not download it.",
    )


def _check_cuda_runtime(required: bool, probes: PreflightProbes) -> PreflightCheck:
    if not required:
        return _skipped("cuda_runtime", "Not needed for smoke mode.")
    try:
        ready = probes.cuda_runtime_ready()
    except (OSError, RuntimeError):
        ready = False
    return _check(
        "cuda_runtime",
        ready,
        "pinned local CUDA and cuDNN runtime DLLs are available",
        "Restore the pinned local NVIDIA runtime packages from the locked environment.",
    )


def _check_qwen(mode: str, probes: PreflightProbes) -> PreflightCheck:
    if mode == "SMOKE_MODE":
        return _skipped("qwen_digest", "Not needed for smoke mode.")
    try:
        digest = probes.local_qwen_digest()
    except (OSError, subprocess.TimeoutExpired):
        digest = None
    if digest is None:
        return PreflightCheck(
            "qwen_digest",
            PreflightStatus.WARNING,
            "optional local Qwen digest is unavailable",
            "Set the recorded local Qwen digest before an optional Qwen stage.",
            required=False,
        )
    return PreflightCheck(
        "qwen_digest",
        PreflightStatus.PASS,
        "optional local Qwen digest is available",
        "",
        required=False,
    )


def _check_artifact_access(
    root: Path,
    config: dict[str, Any] | None,
    probes: PreflightProbes,
    *,
    key: str,
    name: str,
    detail: str,
) -> PreflightCheck:
    defaults = {
        "research_artifact_root": "artifacts/research",
        "model_artifact_root": "artifacts/models",
    }
    artifact_root = _configured_path(root, config, key, defaults[key])
    writable = _safe_bool(lambda: probes.directory_is_writable(artifact_root))
    return _check(
        name,
        writable,
        detail,
        "Restore local write access to the configured artifact location.",
    )


def _configured_path(
    root: Path, config: dict[str, Any] | None, key: str, default: str = ""
) -> Path:
    value = default if config is None else config.get(key, default)
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _configured_paths_are_valid(root: Path, config: dict[str, Any]) -> bool:
    constraints = {
        "data_authorisation": (Path("data/private/data_authorisation.yaml"), True),
        "source_manifest": (Path("data/private"), False),
        "research_artifact_root": (Path("artifacts/research"), False),
        "model_artifact_root": (Path("artifacts/models"), False),
    }
    for key, (approved, exact) in constraints.items():
        raw_value = config.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            return False
        relative = Path(raw_value)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            return False
        if exact:
            if relative != approved:
                return False
        elif relative == approved:
            if key == "source_manifest":
                return False
        elif approved not in relative.parents:
            return False
        try:
            (root / relative).resolve().relative_to(root)
        except (OSError, ValueError):
            return False
    return True


def _configured_manifest_counts(config: dict[str, Any]) -> dict[str, int] | None:
    raw_counts = config.get("source_manifest_expected_counts")
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(_MANIFEST_COUNT_KEYS):
        return None
    if any(type(raw_counts[key]) is not int or raw_counts[key] < 0 for key in _MANIFEST_COUNT_KEYS):
        return None
    return {key: raw_counts[key] for key in _MANIFEST_COUNT_KEYS}


def _huggingface_hub_cache() -> Path:
    configured_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured_hub:
        return Path(configured_hub)
    configured_home = os.environ.get("HF_HOME")
    if configured_home:
        return Path(configured_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_model_directory(model_name: str) -> str | None:
    parts = model_name.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        return None
    return f"models--{parts[0]}--{parts[1]}"


def _safe_snapshot_revision(revision: str) -> bool:
    return bool(revision and Path(revision).name == revision and revision not in {".", ".."})


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _whisper_settings_are_valid(config: dict[str, Any]) -> bool:
    return (
        config.get("whisper_device") in {"cpu", "cuda"}
        and isinstance(config.get("whisper_compute_type"), str)
        and bool(config["whisper_compute_type"].strip())
        and type(config.get("whisper_beam_size")) is int
        and config["whisper_beam_size"] > 0
        and type(config.get("whisper_vad")) is bool
        and type(config.get("whisper_word_timestamps")) is bool
    )


def _venv_python(root: Path) -> Path:
    return root / ".venv311" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _validated_whisper_snapshot(
    candidate: Path,
    cache_root: Path,
    cache_model_directory: str,
    revision: str,
    expected_digest: str,
) -> str | None:
    try:
        resolved = candidate.resolve()
        expected_snapshots = (cache_root / cache_model_directory / "snapshots").resolve()
        if resolved.parent != expected_snapshots or resolved.name != revision:
            return None
        if not resolved.is_dir() or any(
            not (resolved / filename).is_file() or (resolved / filename).stat().st_size == 0
            for filename in _WHISPER_SNAPSHOT_FILES
        ):
            return None
        if canonical_directory_digest(resolved) != expected_digest:
            return None
    except OSError:
        return None
    return str(resolved)


def canonical_directory_digest(root: Path) -> str:
    """Hash a directory from canonical relative-path-to-file-digest metadata."""
    file_hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return sha256_bytes(canonical_json_bytes(file_hashes))


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_bool(observation: Callable[[], bool]) -> bool:
    try:
        return observation()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _unresolved_root_report(mode: str) -> PreflightReport:
    blocked = lambda name: _blocked(
        name,
        "repository root could not be resolved",
        "Restore access to the local repository root before research.",
    )
    full_data_mode = mode in _FULL_DATA_MODES
    checks = [
        blocked("repository_root"),
        blocked("configuration"),
        blocked("ffmpeg"),
        blocked("ffprobe"),
        blocked("free_disk"),
        blocked("venv311"),
        blocked("python_environment"),
        blocked("lock_identity"),
        blocked("data_authorisation") if full_data_mode else _skipped("data_authorisation", "Not needed for smoke mode."),
        blocked("source_manifest") if full_data_mode else _skipped("source_manifest", "Not needed for smoke mode."),
        blocked("whisper_snapshot") if full_data_mode else _skipped("whisper_snapshot", "Not needed for smoke mode."),
        _skipped("qwen_digest", "Repository root is unavailable."),
        blocked("artifact_write_access"),
        blocked("model_artifact_write_access"),
    ]
    return PreflightReport(mode=mode, checks=tuple(checks))


def _check(name: str, passed: bool, detail: str, remediation: str) -> PreflightCheck:
    return _passed(name, detail) if passed else _blocked(name, detail, remediation)


def _passed(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name, PreflightStatus.PASS, detail, "", required=True)


def _blocked(name: str, detail: str, remediation: str) -> PreflightCheck:
    return PreflightCheck(name, PreflightStatus.BLOCKED, detail, remediation, required=True)


def _skipped(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name, PreflightStatus.SKIPPED, detail, "", required=False)
