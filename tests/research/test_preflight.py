from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import io
import subprocess

import pytest
import yaml

from presentation_assessment.research.canonical import canonical_json_bytes, sha256_bytes
from presentation_assessment.research.local_source import LOCAL_CATALOG_FIELDS
from presentation_assessment.research.source_manifest import FROZEN_LEGACY_VIDEO_IDS
from presentation_assessment.research.preflight import (
    PreflightStatus,
    SystemPreflightProbes,
    run_preflight,
)


class FakeProbes:
    """Deterministic local-machine observations for preflight tests."""

    def __init__(self) -> None:
        self.commands = {"ffmpeg": "6.1", "ffprobe": "6.1"}
        self.free_disk_bytes = 100 * 1024**3
        self.venv_exists = True
        self.python_matches = True
        self.lock_matches = True
        self.whisper_snapshot: str | None = "local-whisper-digest"
        self.cuda_runtime = True
        self.qwen_digest: str | None = None
        self.artifact_writable = True
        self.disk_error: OSError | None = None
        self.whisper_request: tuple[str, str, str] | None = None
        self.media_duration_seconds = 12.5
        self.media_probe_paths: list[Path] = []

    def command_version(self, command: str) -> str | None:
        return self.commands.get(command)

    def disk_free_bytes_at(self, path: Path) -> int:
        if self.disk_error is not None:
            raise self.disk_error
        return self.free_disk_bytes

    def directory_exists(self, path: Path) -> bool:
        return self.venv_exists

    def lock_identity_matches(self, root: Path) -> bool:
        return self.lock_matches

    def python_environment_matches(self, root: Path) -> bool:
        return self.python_matches

    def local_whisper_snapshot(
        self, model_name: str, revision: str, snapshot_sha256: str
    ) -> str | None:
        self.whisper_request = (model_name, revision, snapshot_sha256)
        return self.whisper_snapshot

    def local_qwen_digest(self) -> str | None:
        return self.qwen_digest

    def cuda_runtime_ready(self) -> bool:
        return self.cuda_runtime

    def directory_is_writable(self, path: Path) -> bool:
        return self.artifact_writable

    def media_duration(self, path: Path) -> float:
        self.media_probe_paths.append(path)
        return self.media_duration_seconds


def _write_config(root: Path) -> Path:
    private = root / "data" / "private"
    private.mkdir(parents=True)
    (private / "data_authorisation.yaml").write_text(
        "\n".join(
            (
                "authorised_by: Researcher",
                "source: Advisor-provided table",
                "purpose: Local research",
                "data_scope:",
                "  - private_advisor_video_scores",
                "permitted_access:",
                "  - researcher",
                "storage: Local storage",
                "retention_until: 2027-12-31",
                "deletion_plan: Delete private inputs at completion",
                "ethics_status: user_authorised_local_research",
                "acknowledged_at: 2026-08-01T00:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )
    (private / "advisor_video_scores.csv").write_text(
        "source_row_id,source_page,sample_id,legacy_sample_id,source_url,video_id,video_title,total_score,score_status,source_course,verification_status,verification_note\n"
        "row-1,1,N001,,https://www.youtube.com/watch?v=abcdefghijk,abcdefghijk,Redacted example,81.5,observed,,verified,\n",
        encoding="utf-8",
    )
    config = root / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_authorisation": "data/private/data_authorisation.yaml",
                "source_manifest": "data/private/advisor_video_scores.csv",
                "research_artifact_root": "artifacts/research",
                "model_artifact_root": "artifacts/models",
                "whisper_model": "local/whisper",
                "whisper_revision": "a" * 40,
                "whisper_snapshot_sha256": "a" * 64,
                "whisper_local_files_only": True,
                "whisper_device": "cuda",
                "whisper_compute_type": "float16",
                "whisper_beam_size": 5,
                "whisper_vad": True,
                "whisper_word_timestamps": True,
                "source_manifest_expected_counts": {
                    "total_rows": 1,
                    "observed_score_rows": 1,
                    "not_available_rows": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _write_local_source_config(
    root: Path,
    *,
    source_sha256: str | None = None,
    unavailable_count: int = 2,
) -> tuple[Path, Path]:
    private_root = root.parent / f"{root.name}-external-videos"
    private_root.mkdir()
    frozen = list(FROZEN_LEGACY_VIDEO_IDS.items())
    common_ids = [video_id for video_id, _ in frozen[:11]] + [
        f"c{index:010d}" for index in range(1, 60)
    ]
    retired_ids = [f"r{index:010d}" for index in range(1, 16)]
    local_only = [video_id for video_id, _ in frozen[11:]] + [
        f"n{index:010d}" for index in range(1, 12)
    ]
    catalog_ids = [*common_ids, *local_only]
    available_count = 85 - unavailable_count
    catalog_path = root.parent / f"{root.name}-local-catalog.csv"
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=LOCAL_CATALOG_FIELDS, lineterminator="\n")
    writer.writeheader()
    first_media_path: Path | None = None
    for index, video_id in enumerate(catalog_ids, 1):
        available = index <= available_count
        media_path = private_root / f"{video_id}.mp4"
        if available:
            media_path.write_bytes(f"synthetic local media {index}".encode())
            if first_media_path is None:
                first_media_path = media_path
            digest = hashlib.sha256(media_path.read_bytes()).hexdigest()
            if index == 1 and source_sha256 is not None:
                digest = source_sha256
        else:
            digest = ""
        writer.writerow(
            {
                "local_row_id": f"local-r{index:03d}",
                "source_url": f"https://www.youtube.com/watch?v={video_id}",  # privacy-scan: synthetic-fixture
                "video_id": video_id,
                "video_title": f"Synthetic local preflight source {index}",
                "total_score": "81.5" if available else "",
                "duration_seconds": "12.5" if available else "",
                "local_filename": media_path.name if available else "",
                "source_sha256": digest,
                "status": "available" if available else "unavailable",
            }
        )
    catalog_path.write_text(handle.getvalue(), encoding="utf-8")

    manifest_path = root / "data" / "private" / "advisor_video_scores.csv"
    manifest_fields = (
        "source_row_id",
        "source_page",
        "sample_id",
        "legacy_sample_id",
        "source_url",
        "video_id",
        "video_title",
        "total_score",
        "score_status",
        "source_course",
        "verification_status",
        "verification_note",
    )
    manifest_handle = io.StringIO(newline="")
    manifest_writer = csv.DictWriter(
        manifest_handle, fieldnames=manifest_fields, lineterminator="\n"
    )
    manifest_writer.writeheader()
    old_ids = [*common_ids, *retired_ids]
    for index, video_id in enumerate(old_ids, 1):
        frozen_id = FROZEN_LEGACY_VIDEO_IDS.get(video_id)
        if frozen_id is not None:
            sample_id = frozen_id
        elif index <= 70:
            sample_id = f"N{index - 11:03d}"
        else:
            sample_id = f"N{index - 11:03d}"
        observed = index <= 83
        manifest_writer.writerow(
            {
                "source_row_id": f"old-r{index:03d}",
                "source_page": "synthetic",
                "sample_id": sample_id,
                "legacy_sample_id": sample_id if sample_id.startswith("S") else "",
                "source_url": f"https://www.youtube.com/watch?v={video_id}",  # privacy-scan: synthetic-fixture
                "video_id": video_id,
                "video_title": f"Synthetic old source {index}",
                "total_score": "70" if observed else "",
                "score_status": "observed" if observed else "not_available",
                "source_course": "",
                "verification_status": "verified",
                "verification_note": "",
            }
        )
    manifest_path.write_text(manifest_handle.getvalue(), encoding="utf-8")
    dataset_config = root / "config.yaml"
    dataset_payload = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    dataset_payload["source_manifest_expected_counts"] = {
        "total_rows": 85,
        "observed_score_rows": 83,
        "not_available_rows": 2,
    }
    dataset_config.write_text(yaml.safe_dump(dataset_payload), encoding="utf-8")

    local_config_path = root.parent / f"{root.name}-local-source.yaml"
    local_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "video_root": str(private_root.resolve()),
                "catalog_path": str(catalog_path.resolve()),
                "expected_total_rows": 85,
                "expected_available_rows": available_count,
                "expected_unavailable_rows": unavailable_count,
                "duration_tolerance_seconds": 0.25,
            }
        ),
        encoding="utf-8",
    )
    assert first_media_path is not None
    return local_config_path, first_media_path


def test_local_preflight_blocks_hash_mismatch_with_aggregate_detail(tmp_path: Path) -> None:
    """Break caught: changed private media passes preflight or exposes its path in diagnostics."""
    config = _write_config(tmp_path)
    local_config, media_path = _write_local_source_config(
        tmp_path, source_sha256="0" * 64
    )

    report = run_preflight(
        tmp_path,
        config,
        "RESUME",
        probes=FakeProbes(),
        source_mode="LOCAL",
        local_source_config=local_config,
    )

    check = report.by_name("local_source")
    assert check.status is PreflightStatus.BLOCKED
    assert str(media_path) not in check.detail
    assert str(media_path.parent) not in check.remediation


def test_local_preflight_blocks_non_83_plus_2_catalog_composition(tmp_path: Path) -> None:
    """Break caught: preflight accepts a local dataset with 84 available and one unavailable."""
    config = _write_config(tmp_path)
    local_config, _ = _write_local_source_config(tmp_path, unavailable_count=1)

    report = run_preflight(
        tmp_path,
        config,
        "RESUME",
        probes=FakeProbes(),
        source_mode="LOCAL",
        local_source_config=local_config,
    )

    assert report.by_name("local_source").status is PreflightStatus.BLOCKED


def test_full_run_blocks_missing_local_whisper(tmp_path: Path) -> None:
    """A full run must not substitute a download for its approved local ASR snapshot."""
    config = _write_config(tmp_path)
    probes = FakeProbes()
    probes.whisper_snapshot = None

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=probes)

    assert not report.ready
    assert report.by_name("whisper_snapshot").status is PreflightStatus.BLOCKED


def test_full_run_binds_the_configured_whisper_snapshot_and_inference_settings(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    probes = FakeProbes()

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=probes)

    assert report.by_name("whisper_snapshot").status is PreflightStatus.PASS
    assert probes.whisper_request == ("local/whisper", "a" * 40, "a" * 64)


def test_full_run_blocks_an_incomplete_local_cuda_runtime(tmp_path: Path) -> None:
    """A full run must stop before Whisper when required split DLLs are absent."""
    config = _write_config(tmp_path)
    probes = FakeProbes()
    probes.cuda_runtime = False

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=probes)

    assert not report.ready
    check = report.by_name("cuda_runtime")
    assert check.status is PreflightStatus.BLOCKED
    assert "install" not in check.remediation.lower()
    assert "network" not in check.remediation.lower()


def test_preflight_blocks_a_different_running_python_environment(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    probes = FakeProbes()
    probes.python_matches = False

    report = run_preflight(tmp_path, config, "SMOKE_MODE", probes=probes)

    assert not report.ready
    assert report.by_name("python_environment").status is PreflightStatus.BLOCKED


def test_smoke_mode_needs_no_network_or_models(tmp_path: Path) -> None:
    """Smoke mode remains usable when neither optional local model is present."""
    config = _write_config(tmp_path)
    probes = FakeProbes()
    probes.whisper_snapshot = None
    probes.qwen_digest = None

    report = run_preflight(tmp_path, config, "SMOKE_MODE", probes=probes)

    assert report.ready
    assert report.by_name("whisper_snapshot").status is PreflightStatus.SKIPPED
    assert report.by_name("cuda_runtime").status is PreflightStatus.SKIPPED
    assert report.by_name("qwen_digest").status is PreflightStatus.SKIPPED


def test_full_run_blocks_duplicate_normalised_video_id(tmp_path: Path) -> None:
    """A complete-looking manifest cannot acquire the same normalized video twice."""
    config = _write_config(tmp_path)
    manifest = tmp_path / "data" / "private" / "advisor_video_scores.csv"
    first_row = manifest.read_text(encoding="utf-8").splitlines()[1]
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + first_row.replace("row-1", "row-2") + "\n",
        encoding="utf-8",
    )

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("source_manifest").status is PreflightStatus.BLOCKED


def test_full_run_blocks_manifest_count_drift(tmp_path: Path) -> None:
    """A manifest must equal the immutable, non-sensitive configured audit counts."""
    config = _write_config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["source_manifest_expected_counts"]["total_rows"] = 85
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("source_manifest").status is PreflightStatus.BLOCKED


def test_full_run_blocks_manifest_identity_errors(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    manifest = tmp_path / "data" / "private" / "advisor_video_scores.csv"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("N001", "invalid-id"),
        encoding="utf-8",
    )

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("source_manifest").status is PreflightStatus.BLOCKED


def test_full_run_requires_local_only_whisper_configuration(tmp_path: Path) -> None:
    """A local snapshot is insufficient if the config permits a model download."""
    config = _write_config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["whisper_local_files_only"] = False
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("whisper_snapshot").status is PreflightStatus.BLOCKED


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data_authorisation", "data/private/other.yaml"),
        ("source_manifest", "data/public/manifest.csv"),
        ("source_manifest", "../private/manifest.csv"),
        ("research_artifact_root", "C:/research/artifacts"),
        ("research_artifact_root", "artifacts/elsewhere"),
        ("model_artifact_root", "artifacts/research/models"),
    ],
)
def test_preflight_rejects_unapproved_configured_paths(
    tmp_path: Path, key: str, value: str
) -> None:
    config = _write_config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload[key] = value
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = run_preflight(tmp_path, config, "FULL_RUN", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("configuration").status is PreflightStatus.BLOCKED


def test_system_lock_probe_checks_exact_extras_and_environment_without_mutation(
    tmp_path: Path,
) -> None:
    """Lock identity verifies the bound environment, exact extras, and consistency."""
    invocations: list[tuple[list[str], Path]] = []

    def run_process(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        invocations.append((arguments, kwargs["cwd"]))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    probes = SystemPreflightProbes(process_runner=run_process)

    assert probes.lock_identity_matches(tmp_path)
    assert invocations == [
        (["uv", "lock", "--check", "--offline"], tmp_path),
        (
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
            tmp_path,
        ),
        (
            [
                "uv",
                "pip",
                "check",
                "--python",
                str(tmp_path / ".venv311" / "Scripts" / "python.exe"),
            ],
            tmp_path,
        ),
    ]


def test_system_whisper_probe_rejects_arbitrary_override_and_accepts_complete_snapshot(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Whisper overrides must remain complete snapshots of the configured HF model."""
    model_name = "Systran/faster-whisper-large-v3"
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    arbitrary = tmp_path / "mimic" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc"
    arbitrary.mkdir(parents=True)
    for filename in ("config.json", "model.bin", "tokenizer.json"):
        (arbitrary / filename).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PRESENTATION_ASSESSMENT_WHISPER_SNAPSHOT", str(arbitrary))

    probes = SystemPreflightProbes()

    file_digest = sha256_bytes(b"{}")
    expected_digest = sha256_bytes(
        canonical_json_bytes(
            {
                "config.json": file_digest,
                "model.bin": file_digest,
                "tokenizer.json": file_digest,
            }
        )
    )

    assert probes.local_whisper_snapshot(model_name, "abc", expected_digest) is None

    snapshot = hf_home / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    for filename in ("config.json", "model.bin", "tokenizer.json"):
        (snapshot / filename).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PRESENTATION_ASSESSMENT_WHISPER_SNAPSHOT", str(snapshot))

    assert probes.local_whisper_snapshot(model_name, "abc", expected_digest) == str(snapshot)
    assert probes.local_whisper_snapshot(model_name, "different", expected_digest) is None
    assert probes.local_whisper_snapshot(model_name, "abc", "0" * 64) is None


def test_probe_failures_become_blocked_checks(tmp_path: Path) -> None:
    """Missing roots and failed disk probes must not crash a preflight report."""
    config = _write_config(tmp_path)
    probes = FakeProbes()
    probes.disk_error = OSError("disk unavailable")

    report = run_preflight(tmp_path / "missing-root", config, "SMOKE_MODE", probes=probes)

    assert not report.ready
    assert report.by_name("repository_root").status is PreflightStatus.BLOCKED
    assert report.by_name("free_disk").status is PreflightStatus.BLOCKED


def test_system_command_timeout_is_reported_as_unavailable() -> None:
    """Timeouts from an external probe are treated as absent local tooling."""
    def run_process(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, 5)

    probes = SystemPreflightProbes(process_runner=run_process)

    assert probes.command_version("ffmpeg") is None


def test_root_resolution_failure_returns_a_blocked_report(tmp_path: Path, monkeypatch: object) -> None:
    """An inaccessible root must yield structured status rather than an exception."""
    config = _write_config(tmp_path)
    original_resolve = Path.resolve

    def fail_root_resolution(path: Path, *args: object, **kwargs: object) -> Path:
        if path == tmp_path:
            raise OSError("root unavailable")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_root_resolution)

    report = run_preflight(tmp_path, config, "SMOKE_MODE", probes=FakeProbes())

    assert not report.ready
    assert report.by_name("repository_root").status is PreflightStatus.BLOCKED
    assert all(check.status is not PreflightStatus.PASS for check in report.checks if check.required)
