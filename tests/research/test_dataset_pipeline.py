"""Synthetic integration tests for orchestration, resume, and dataset freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pandas as pd
import pytest
import presentation_assessment.research.dataset_pipeline as dataset_pipeline_module
import presentation_assessment.research.notebook_support as notebook_support_module

from presentation_assessment.research.dataset_pipeline import (
    DatasetPipelineConfig,
    DatasetStageAdapters,
    DatasetStageResult,
    freeze_dataset,
    run_dataset_pipeline,
    stage_implementation_digests,
)
from presentation_assessment.research.feature_registry import load_feature_registry
from presentation_assessment.research.canonical import canonical_json_bytes, sha256_bytes
from presentation_assessment.research.local_source import (
    LocalAudioAdapter,
    LocalAvailability,
    LocalCatalogRow,
    LocalFileAcquirer,
    LocalSourceCatalog,
    LocalSourceConfig,
    LocalSourceResolver,
)
from presentation_assessment.research.protocol import load_protocol_template, protocol_id
from presentation_assessment.research.provenance import SourceSnapshot
from presentation_assessment.research.source_manifest import (
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


_FROZEN_OUTPUTS = {
    "manifest.csv",
    "transcripts.jsonl",
    "acoustic_features.parquet",
    "quality_report.csv",
    "splits.json",
    "experiment_protocol.json",
    "provenance.json",
    "source_snapshot.json",
    "git_diff.patch",
    "checksums.sha256",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _copy_implementation_tree_with_change(
    destination_root: Path, changed_filename: str
) -> Path:
    relative_paths = {
        relative_path
        for paths in dataset_pipeline_module._STAGE_IMPLEMENTATION_FILES.values()
        for relative_path in paths
    }
    relative_paths.add(
        Path("src/presentation_assessment/research/dataset_pipeline.py")
    )
    changed_path: Path | None = None
    for relative_path in relative_paths:
        source = PROJECT_ROOT / relative_path
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if relative_path.name == changed_filename:
            changed_path = destination
    assert changed_path is not None
    changed_path.write_text(
        changed_path.read_text(encoding="utf-8") + "\n# implementation drift\n",
        encoding="utf-8",
    )
    return destination_root


def test_task_7_interfaces_are_public() -> None:
    """Break caught: notebook callers cannot import the frozen Task 7 surface."""
    from presentation_assessment.research import (
        DatasetRunResult,
        SplitAssignment,
        SplitRegistry,
        create_exploratory_splits,
        freeze_dataset,
        run_dataset_pipeline,
    )

    assert all(
        item is not None
        for item in (
            DatasetRunResult,
            SplitAssignment,
            SplitRegistry,
            create_exploratory_splits,
            freeze_dataset,
            run_dataset_pipeline,
        )
    )


def _source_row(sample_id: str, video_id: str, score: float | None = 70.0) -> SourceRow:
    return SourceRow(
        source_row_id=f"row-{sample_id}",
        source_page="synthetic",
        sample_id=sample_id,
        legacy_sample_id="",
        # privacy-scan: synthetic-fixture
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        video_id=video_id,
        video_title="synthetic",
        total_score=score,
        score_status=(ScoreStatus.OBSERVED if score is not None else ScoreStatus.NOT_AVAILABLE),
        source_course="synthetic",
        verification_status=VerificationStatus.VERIFIED,
        verification_note="synthetic",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_complete_frozen_candidate(root: Path) -> None:
    assert {path.name for path in root.iterdir()} == _FROZEN_OUTPUTS
    checksum_lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 9
    for line in checksum_lines:
        expected, filename = line.split("  ", 1)
        assert _digest(root / filename) == expected


def _refresh_checksum(root: Path, filename: str) -> None:
    checksum_path = root / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            f"{_digest(root / filename)}  {filename}"
            if line.endswith(f"  {filename}")
            else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )


class SyntheticStages:
    """File-writing stage adapters with deterministic terminal outcomes."""

    def __init__(
        self,
        *,
        acquisition_failures: set[str] | None = None,
        asr_failures: set[str] | None = None,
        feature_failures: set[str] | None = None,
    ) -> None:
        self.acquisition_failures = acquisition_failures or set()
        self.asr_failures = asr_failures or set()
        self.feature_failures = feature_failures or set()
        self.acquisition_calls: list[str] = []
        self.audio_calls: list[str] = []
        self.asr_batches: list[tuple[str, ...]] = []
        self.feature_calls: list[tuple[str, bool]] = []

    def acquire(self, row: SourceRow, output_root: Path, allow_network: bool) -> DatasetStageResult:
        self.acquisition_calls.append(row.sample_id)
        if row.sample_id in self.acquisition_failures:
            return DatasetStageResult(
                sample_id=row.sample_id,
                status="failed",
                error_code="private_video",
                error_summary=f"unavailable at {row.source_url}",
            )
        path = output_root / f"{row.sample_id}.source"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{row.sample_id}".encode())
        return DatasetStageResult(
            sample_id=row.sample_id,
            status="succeeded",
            payload={"source_path": path, "source_sha256": _digest(path)},
        )

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        sample_id = output_dir.parent.name
        self.audio_calls.append(sample_id)
        payload: dict[str, object] = {
            "source_channel_count": 1,
            "analysis_channel_count": 1,
        }
        for name in ("analysis", "peak_normalized", "loudness_normalized"):
            path = output_dir / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_path.read_bytes() + name.encode())
            payload[f"{name}_path"] = path
            payload[f"{name}_sha256"] = _digest(path)
        return DatasetStageResult(sample_id=sample_id, status="succeeded", payload=payload)

    def transcribe_many(self, items: Sequence[Any]) -> list[DatasetStageResult]:
        sample_ids = tuple(item.sample_id for item in items)
        self.asr_batches.append(sample_ids)
        return [
            DatasetStageResult(
                sample_id=item.sample_id,
                status="failed",
                error_code="transcription_error",
                error_summary="synthetic ASR failure",
            )
            if item.sample_id in self.asr_failures
            else DatasetStageResult(
                sample_id=item.sample_id,
                status="succeeded",
                payload={
                    "transcription": {
                        "text": f"transcript {item.sample_id}",
                        "language": "en",
                        "segments": [],
                    },
                    "provenance": {"model_name": "synthetic-whisper"},
                },
            )
            for item in items
        ]

    def features(
        self, sample_id: str, audio_paths: dict[str, Path], transcript: object, registry: object
    ) -> DatasetStageResult:
        self.feature_calls.append((sample_id, transcript is not None))
        if sample_id in self.feature_failures:
            return DatasetStageResult(
                sample_id=sample_id,
                status="failed",
                error_code="feature_error",
                error_summary="synthetic feature failure",
            )
        feature_values: dict[str, object] = {}
        for definition in registry.features:
            if definition.dtype == "boolean":
                value: object = False
            elif definition.dtype == "string":
                value = "synthetic"
            elif definition.dtype == "integer":
                value = int(definition.minimum or 0)
            else:
                value = float(definition.minimum or 0.0)
            feature_values[definition.name] = value
        return DatasetStageResult(
            sample_id=sample_id,
            status="succeeded",
            payload={
                "features": feature_values,
                "missing_reasons": {},
                "qc_warnings": ["synthetic_qc"] if sample_id == "N003" else [],
            },
        )

    def adapters(self) -> DatasetStageAdapters:
        return DatasetStageAdapters(
            acquire=self.acquire,
            audio=self.audio,
            transcribe_many=self.transcribe_many,
            features=self.features,
        )


class InterruptingStreamStages(SyntheticStages):
    """Yield one terminal ASR result, then simulate an external interruption."""

    def __init__(self, interruption: BaseException | None = None) -> None:
        super().__init__()
        self.interrupt = True
        self.interruption = interruption or KeyboardInterrupt(
            "synthetic native stall interruption"
        )

    def transcribe_many(self, items: Sequence[Any]) -> object:
        batch = tuple(items)
        self.asr_batches.append(tuple(item.sample_id for item in batch))

        def results() -> object:
            for index, item in enumerate(batch):
                if index == 1 and self.interrupt:
                    raise self.interruption
                yield DatasetStageResult(
                    sample_id=item.sample_id,
                    status="succeeded",
                    payload={
                        "transcription": {
                            "text": f"transcript {item.sample_id}",
                            "language": "en",
                            "segments": [],
                        },
                        "provenance": {"model_name": "synthetic-whisper"},
                    },
                )

        return results()


class MalformedSuccessStages(SyntheticStages):
    """Return one structurally invalid success at a selected adapter boundary."""

    def __init__(self, malformed_stage: str) -> None:
        super().__init__()
        self.malformed_stage = malformed_stage

    def acquire(self, row: SourceRow, output_root: Path, allow_network: bool) -> DatasetStageResult:
        if self.malformed_stage == "acquisition" and row.sample_id == "N001":
            self.acquisition_calls.append(row.sample_id)
            return DatasetStageResult(sample_id=row.sample_id, status="succeeded", payload={})
        return super().acquire(row, output_root, allow_network)

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        sample_id = output_dir.parent.name
        if self.malformed_stage == "audio" and sample_id == "N001":
            self.audio_calls.append(sample_id)
            return DatasetStageResult(sample_id=sample_id, status="succeeded", payload={})
        return super().audio(source_path, output_dir)

    def transcribe_many(self, items: Sequence[Any]) -> list[DatasetStageResult]:
        results = super().transcribe_many(items)
        if self.malformed_stage != "asr":
            return results
        return [
            DatasetStageResult(
                sample_id=result.sample_id,
                status="succeeded",
                payload={"transcription": "malformed", "provenance": {}},
            )
            if result.sample_id == "N001"
            else result
            for result in results
        ]

    def features(
        self, sample_id: str, audio_paths: dict[str, Path], transcript: object, registry: object
    ) -> DatasetStageResult:
        if self.malformed_stage == "features" and sample_id == "N001":
            self.feature_calls.append((sample_id, transcript is not None))
            return DatasetStageResult(
                sample_id=sample_id,
                status="succeeded",
                payload={"features": [0.5], "missing_reasons": {}, "qc_warnings": []},
            )
        return super().features(sample_id, audio_paths, transcript, registry)


class DishonestDigestStages(SyntheticStages):
    """Write valid artifacts while supplying deliberately incorrect digests."""

    def acquire(self, row: SourceRow, output_root: Path, allow_network: bool) -> DatasetStageResult:
        result = super().acquire(row, output_root, allow_network)
        return replace(result, payload={**result.payload, "source_sha256": "0" * 64})

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        result = super().audio(source_path, output_dir)
        payload = dict(result.payload)
        for name in ("analysis", "peak_normalized", "loudness_normalized"):
            payload[f"{name}_sha256"] = "0" * 64
        return replace(result, payload=payload)


class MissingSourceChannelStages(SyntheticStages):
    """Return valid WAVs while omitting required pre-downmix provenance."""

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        result = super().audio(source_path, output_dir)
        payload = dict(result.payload)
        payload.pop("source_channel_count")
        return replace(result, payload=payload)


class StereoSourceStages(SyntheticStages):
    """Expose stereo source provenance while producing canonical mono WAVs."""

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        del source_path
        sample_id = output_dir.parent.name
        self.audio_calls.append(sample_id)
        payload: dict[str, object] = {
            "source_channel_count": 2,
            "analysis_channel_count": 1,
        }
        for name in ("analysis", "peak_normalized", "loudness_normalized"):
            path = output_dir / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\x00\x00" * 64_000)
            payload[f"{name}_path"] = path
            payload[f"{name}_sha256"] = _digest(path)
        return DatasetStageResult(
            sample_id=sample_id, status="succeeded", payload=payload
        )

    def features(
        self,
        sample_id: str,
        audio_paths: dict[str, Path],
        transcript: object,
        registry: object,
    ) -> DatasetStageResult:
        self.feature_calls.append((sample_id, transcript is not None))
        return dataset_pipeline_module._extract_feature_stage(
            sample_id, audio_paths, transcript, registry
        )


class ChangingAnalysisStages(SyntheticStages):
    """Change analysis bytes when the audio implementation is recomputed."""

    def audio(self, source_path: Path, output_dir: Path) -> DatasetStageResult:
        result = super().audio(source_path, output_dir)
        if len(self.audio_calls) > 1:
            payload = dict(result.payload)
            for name in ("analysis", "peak_normalized", "loudness_normalized"):
                path = Path(payload[f"{name}_path"])
                path.write_bytes(path.read_bytes() + b"changed-analysis")
                payload[f"{name}_sha256"] = _digest(path)
            return replace(result, payload=payload)
        return result


def _snapshot(_: Path) -> SourceSnapshot:
    files = {"src/synthetic.py": "c" * 64}
    return SourceSnapshot(
        git_commit="a" * 40,
        git_dirty=False,
        tree_sha256=sha256_bytes(canonical_json_bytes(files)),
        files=files,
        git_diff_patch=b"synthetic patch\n",
        untracked_files={},
    )


def _config(tmp_path: Path, stages: SyntheticStages, rows: Sequence[SourceRow]) -> DatasetPipelineConfig:
    project_root = Path(__file__).resolve().parents[2]
    return DatasetPipelineConfig(
        repository_root=tmp_path,
        staging_dir=tmp_path / "artifacts" / "staging" / "synthetic-run",
        artifact_root=tmp_path / "artifacts" / "research",
        rows=tuple(rows),
        protocol_template=load_protocol_template(
            project_root / "configs" / "experiment_protocol.template.yaml"
        ),
        feature_registry=load_feature_registry(
            project_root / "configs" / "audio_feature_registry.v1.yaml"
        ),
        pipeline_parameters={"schema_version": 1, "mode": "synthetic"},
        stage_parameters={
            "acquisition": {"revision": 1},
            "audio": {"revision": 1},
            "asr": {"revision": 1},
            "features": {"revision": 1},
        },
        whisper_fingerprint={"model": "synthetic-whisper", "snapshot_sha256": "d" * 64},
        implementation_digests={
            "acquisition": "1" * 64,
            "audio": "2" * 64,
            "asr": "3" * 64,
            "features": "4" * 64,
        },
        adapters=stages.adapters(),
        allow_network=False,
        source_snapshotter=_snapshot,
    )


def _local_pipeline_config(
    tmp_path: Path,
    *,
    availability: LocalAvailability = LocalAvailability.AVAILABLE,
) -> tuple[DatasetPipelineConfig, Path, LocalSourceCatalog]:
    """Build one catalog-backed pipeline without touching external media."""
    private_root = tmp_path / "private-catalog-media"
    private_root.mkdir(parents=True)
    video_id = "abcdefghijk"
    media_path = private_root / "approved-corpus-file.mp4"
    if availability is LocalAvailability.AVAILABLE:
        media_path.write_bytes(b"synthetic catalog media")
        source_sha256 = _digest(media_path)
        local_filename = media_path.name
        duration_seconds: float | None = 8.0
    else:
        source_sha256 = ""
        local_filename = ""
        duration_seconds = None
    catalog = LocalSourceCatalog(
        config=LocalSourceConfig(
            private_root,
            tmp_path / "unused.csv",
            1,
            int(availability is LocalAvailability.AVAILABLE),
            int(availability is LocalAvailability.UNAVAILABLE),
            0.25,
        ),
        rows=(
            LocalCatalogRow(
                local_row_id="local-r001",
                source_url=(
                    f"https://www.youtube.com/watch?v={video_id}"  # privacy-scan: synthetic-fixture
                ),
                video_id=video_id,
                video_title="Synthetic catalog source",
                total_score=72.0,
                duration_seconds=duration_seconds,
                local_filename=local_filename,
                source_sha256=source_sha256,
                status=availability,
            ),
        ),
        identity_sha256="a" * 64,
    )
    resolver = LocalSourceResolver(catalog)
    stages = SyntheticStages()
    base = _config(
        tmp_path,
        stages,
        (replace(_source_row("N001", video_id), source_row_id="local-r001"),),
    )
    return (
        replace(
            base,
            adapters=replace(
                stages.adapters(),
                acquire=LocalFileAcquirer(resolver).acquire,
                audio=LocalAudioAdapter(resolver, builder=stages.audio),
            ),
            stage_parameters={
                **base.stage_parameters,
                "acquisition": {
                    "adapter": "local_file",
                    "local_source_identity_sha256": catalog.identity_sha256,
                },
            },
        ),
        media_path,
        catalog,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"pipeline_parameters": {"ambiguous": {"alpha", "beta"}}},
        {"pipeline_parameters": {"ambiguous": frozenset({"alpha", "beta"})}},
        {"stage_parameters": {"features": {"ambiguous": {"alpha", "beta"}}}},
        {"whisper_fingerprint": {"ambiguous": {"alpha", "beta"}}},
    ],
)
def test_config_rejects_ambiguous_sets(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    """Break caught: set iteration order silently changes cache or dataset hashes."""
    config = _config(
        tmp_path,
        SyntheticStages(),
        (_source_row("N001", "aaaaaaaaaaa"),),
    )

    with pytest.raises(TypeError, match="sets are not supported"):
        replace(config, **changes)


def test_set_rejection_is_identical_across_python_hash_seeds() -> None:
    """Break caught: process hash seed selects a different canonical set ordering."""
    script = (
        "from presentation_assessment.research.dataset_pipeline import _canonical_value;"
        "\ntry: _canonical_value({'alpha','beta','gamma','delta'})"
        "\nexcept TypeError as error: print(type(error).__name__ + ':' + str(error))"
        "\nelse: print('accepted')"
    )
    outputs = []
    for seed in ("1", "987654"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(completed.stdout.strip())

    assert outputs == [
        "TypeError:sets are not supported in canonical configuration",
        "TypeError:sets are not supported in canonical configuration",
    ]


def test_pipeline_isolates_samples_batches_asr_and_redacts_stage_metadata(tmp_path: Path) -> None:
    """Break caught: one stage failure aborts later rows or leaks a private URL."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
        _source_row("N003", "ccccccccccc", score=None),
    )
    stages = SyntheticStages(acquisition_failures={"N002"}, asr_failures={"N003"})

    result = run_dataset_pipeline(_config(tmp_path, stages, rows))

    assert result.terminal_counts == {"succeeded": 1, "acquisition_failed": 1, "asr_failed": 1}
    assert [sample.sample_id for sample in result.samples] == ["N001", "N002", "N003"]
    assert stages.acquisition_calls == ["N001", "N002", "N003"]
    assert stages.audio_calls == ["N001", "N003"]
    assert stages.asr_batches == [("N001", "N003")]
    assert stages.feature_calls == [("N001", True), ("N003", False)]
    metadata = "\n".join(
        path.read_text(encoding="utf-8") for path in result.staging_dir.rglob("*.json")
    )
    assert "youtube.com" not in metadata
    assert "https://" not in metadata
    assert str(tmp_path) not in metadata
    assert "[redacted-url]" in metadata


def test_local_acquisition_flows_to_audio_without_persisting_private_root(
    tmp_path: Path,
) -> None:
    """Break caught: the reviewed pipeline requires a repository-local downloaded source path."""
    private_root = tmp_path.parent / f"{tmp_path.name}-private-videos"
    private_root.mkdir()
    video_id = "abcdefghijk"
    media_path = private_root / f"{video_id}.mp4"
    media_path.write_bytes(b"synthetic local pipeline media")
    source_sha256 = _digest(media_path)
    catalog_row = LocalCatalogRow(
        local_row_id="local-r001",
        source_url=f"https://www.youtube.com/watch?v={video_id}",  # privacy-scan: synthetic-fixture
        video_id=video_id,
        video_title="Synthetic local pipeline source",
        total_score=72.0,
        duration_seconds=8.0,
        local_filename=media_path.name,
        source_sha256=source_sha256,
        status=LocalAvailability.AVAILABLE,
    )
    catalog = LocalSourceCatalog(
        config=LocalSourceConfig(private_root, tmp_path / "unused.csv", 1, 1, 0, 0.25),
        rows=(catalog_row,),
        identity_sha256="a" * 64,
    )
    resolver = LocalSourceResolver(catalog)
    stages = SyntheticStages()
    base = _config(
        tmp_path,
        stages,
        (
            replace(
                _source_row("N001", video_id),
                source_row_id="local-r001",
            ),
        ),
    )
    adapters = replace(
        stages.adapters(),
        acquire=LocalFileAcquirer(resolver).acquire,
        audio=LocalAudioAdapter(resolver, builder=stages.audio),
    )
    config = replace(
        base,
        adapters=adapters,
        allow_network=True,
        stage_parameters={
            **base.stage_parameters,
            "acquisition": {
                "adapter": "local_file",
                "local_source_identity_sha256": catalog.identity_sha256,
            },
            "audio": {
                "revision": 1,
                "local_source_identity_sha256": catalog.identity_sha256,
            },
        },
    )

    result = run_dataset_pipeline(config)

    assert result.samples[0].terminal_status == "succeeded"
    assert stages.audio_calls == ["N001"]
    acquisition = result.samples[0].acquisition.payload
    assert acquisition["source_kind"] == "local_file"
    assert acquisition["local_basename"] == media_path.name
    assert acquisition["duration_seconds"] == 8.0
    metadata = "\n".join(
        path.read_text(encoding="utf-8") for path in result.staging_dir.rglob("*.json")
    )
    assert str(private_root) not in metadata
    assert str(media_path) not in metadata
    assert "source_path" not in acquisition
    assert notebook_support_module._sample_records_valid(result.samples[0]) is True

    tampered_references = (
        {"source_locator": f"local-file/local-r999/{media_path.name}"},
        {"local_basename": "other approved.mp4"},
        {
            "local_basename": "other approved.mp4",
            "source_locator": "local-file/local-r001/other approved.mp4",
        },
        {"source_sha256": "b" * 64},
    )
    for updates in tampered_references:
        tampered_acquisition = replace(
            result.samples[0].acquisition,
            payload={**acquisition, **updates},
        )
        tampered_sample = replace(result.samples[0], acquisition=tampered_acquisition)
        assert notebook_support_module._sample_records_valid(tampered_sample) is False

    acquisition_metadata = (
        result.staging_dir / "metadata" / "N001" / "acquisition.json"
    )
    for updates in tampered_references:
        cached_payload = json.loads(acquisition_metadata.read_text(encoding="utf-8"))
        cached_payload["payload"].update(updates)
        acquisition_metadata.write_text(
            json.dumps(cached_payload, sort_keys=True), encoding="utf-8"
        )
        rebuilt = run_dataset_pipeline(config)
        assert rebuilt.samples[0].acquisition.resumed is False
        assert rebuilt.samples[0].terminal_status == "succeeded"

    resumed = run_dataset_pipeline(config)
    assert resumed.samples[0].resumed_stages == (
        "acquisition",
        "audio",
        "asr",
        "features",
    )

    frozen = freeze_dataset(result)
    for field, value in (
        ("source_kind", "youtube"),
        ("local_basename", "other approved.mp4"),
        ("source_sha256", "b" * 64),
    ):
        tampered_root = tmp_path / f"tampered-{field}" / frozen.dataset_id
        shutil.copytree(frozen.path, tampered_root)
        manifest_path = tampered_root / "manifest.csv"
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0][field] = value
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        _refresh_checksum(tampered_root, "manifest.csv")
        with pytest.raises(ValueError, match="frozen source identity"):
            dataset_pipeline_module.load_frozen_dataset(
                tampered_root, result.feature_registry
            )

    changed_identity = "b" * 64
    with pytest.raises(ValueError, match="does not match the local resolver"):
        replace(
            config,
            stage_parameters={
                **config.stage_parameters,
                "acquisition": {
                    "adapter": "local_file",
                    "local_source_identity_sha256": changed_identity,
                },
            },
        )
    changed_catalog = replace(catalog, identity_sha256=changed_identity)
    changed_resolver = LocalSourceResolver(changed_catalog)
    changed_adapters = replace(
        stages.adapters(),
        acquire=LocalFileAcquirer(changed_resolver).acquire,
        audio=LocalAudioAdapter(changed_resolver, builder=stages.audio),
    )
    changed = run_dataset_pipeline(
        replace(
            config,
            adapters=changed_adapters,
            stage_parameters={
                **config.stage_parameters,
                "acquisition": {
                    "adapter": "local_file",
                    "local_source_identity_sha256": changed_identity,
                },
                "audio": {
                    "revision": 1,
                    "local_source_identity_sha256": changed_identity,
                },
            },
        )
    )
    assert changed.samples[0].resumed_stages == ()


def test_local_dataset_config_requires_catalog_identity_in_stage_key(
    tmp_path: Path,
) -> None:
    """Break caught: local acquisition cache can be keyed without catalog/config identity."""
    stages = SyntheticStages()
    config = _config(
        tmp_path, stages, (_source_row("N001", "abcdefghijk"),)
    )

    with pytest.raises(ValueError, match="local_source_identity_sha256"):
        replace(
            config,
            stage_parameters={
                **config.stage_parameters,
                "acquisition": {"adapter": "local_file"},
            },
        )


def test_local_cache_cannot_switch_to_youtube_branch(tmp_path: Path) -> None:
    """Break caught: a local cache payload selects the YouTube validation branch."""
    config, _, _ = _local_pipeline_config(tmp_path)
    completed = run_dataset_pipeline(config)
    metadata_path = completed.staging_dir / "metadata" / "N001" / "acquisition.json"
    repository_file = tmp_path / "forged-youtube-source.mp4"
    repository_file.write_bytes(b"forged repository source")
    cached = json.loads(metadata_path.read_text(encoding="utf-8"))
    cached["payload"] = {
        "source_kind": "youtube",
        "source_path": repository_file.relative_to(tmp_path).as_posix(),
        "source_sha256": _digest(repository_file),
        "attempts": 1,
    }
    metadata_path.write_text(json.dumps(cached), encoding="utf-8")

    rerun = run_dataset_pipeline(config)

    assert rerun.samples[0].acquisition.resumed is False
    assert rerun.samples[0].acquisition.payload["source_kind"] == "local_file"


@pytest.mark.parametrize(
    ("declaration", "local_owner"),
    [("local_file", False), ("yt-dlp", True)],
)
def test_local_adapter_declaration_and_owner_must_be_equivalent(
    tmp_path: Path, declaration: str, local_owner: bool
) -> None:
    """Break caught: the declared source adapter differs from the acquisition owner."""
    config, _, catalog = _local_pipeline_config(tmp_path)
    adapters = config.adapters if local_owner else SyntheticStages().adapters()

    with pytest.raises(ValueError, match="local.*adapter"):
        replace(
            config,
            adapters=adapters,
            stage_parameters={
                **config.stage_parameters,
                "acquisition": {
                    "adapter": declaration,
                    "local_source_identity_sha256": catalog.identity_sha256,
                },
            },
        )


@pytest.mark.parametrize(
    ("source_row_id", "video_id"),
    [("local-r999", "abcdefghijk"), ("local-r001", "bbbbbbbbbbb")],
)
def test_local_config_rejects_rows_without_catalog_identity(
    tmp_path: Path, source_row_id: str, video_id: str
) -> None:
    """Break caught: a resolver-rejected row receives a fabricated local identity."""
    config, _, _ = _local_pipeline_config(tmp_path)
    invalid_row = replace(
        config.rows[0], source_row_id=source_row_id, video_id=video_id
    )

    with pytest.raises(ValueError, match="local source identity"):
        replace(config, rows=(invalid_row,))


def test_live_local_byte_corruption_keeps_catalog_identity_for_audit(
    tmp_path: Path,
) -> None:
    """Break caught: failed live verification erases the catalog-approved identity."""
    config, media_path, catalog = _local_pipeline_config(tmp_path)
    declared = catalog.rows[0]
    media_path.write_bytes(b"changed after catalog construction")

    result = run_dataset_pipeline(config)
    sample = result.samples[0]

    assert sample.acquisition.status == "failed"
    assert sample.acquisition.error_code == "local_source_hash_mismatch"
    assert sample.trusted_source_identity["local_basename"] == declared.local_filename
    assert sample.trusted_source_identity["source_sha256"] == declared.source_sha256


def test_terminal_and_frozen_local_identities_require_catalog_authority(
    tmp_path: Path,
) -> None:
    """Break caught: terminal or frozen records authenticate identities against themselves."""
    success_config, _, _ = _local_pipeline_config(tmp_path / "success")
    success = run_dataset_pipeline(success_config)
    original = dict(success.samples[0].trusted_source_identity)
    tampered_success = replace(
        success.samples[0],
        trusted_source_identity={**original, "catalog_identity_sha256": "b" * 64},
    )
    assert (
        notebook_support_module._sample_records_valid(
            tampered_success, expected_identity=original
        )
        is False
    )

    unavailable_config, _, _ = _local_pipeline_config(
        tmp_path / "unavailable", availability=LocalAvailability.UNAVAILABLE
    )
    unavailable = run_dataset_pipeline(unavailable_config)
    unavailable_original = dict(unavailable.samples[0].trusted_source_identity)
    tampered_unavailable = replace(
        unavailable.samples[0],
        trusted_source_identity={
            **unavailable_original,
            "catalog_identity_sha256": "b" * 64,
        },
    )
    assert (
        notebook_support_module._sample_records_valid(
            tampered_unavailable, expected_identity=unavailable_original
        )
        is False
    )

    frozen = freeze_dataset(success)
    candidate = tmp_path / "forged-candidate"
    shutil.copytree(frozen.path, candidate)
    manifest_path = candidate / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
        fieldnames = list(manifest_rows[0])
    manifest_rows[0]["catalog_identity_sha256"] = "b" * 64
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    provenance_path = candidate / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    identity_inputs = provenance["dataset_identity_inputs"]
    identity_inputs["terminal_samples"][0]["trusted_source"]["catalog_identity_sha256"] = (
        "b" * 64
    )
    identity_inputs["logical_table_sha256"]["manifest.csv"] = _digest(manifest_path)
    provenance_path.write_bytes(canonical_json_bytes(provenance))
    forged_dataset_id = sha256_bytes(canonical_json_bytes(identity_inputs))
    protocol_path = candidate / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset_id"] = forged_dataset_id
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    forged_root = tmp_path / "forged" / forged_dataset_id
    forged_root.parent.mkdir()
    candidate.rename(forged_root)
    for filename in ("manifest.csv", "provenance.json", "experiment_protocol.json"):
        _refresh_checksum(forged_root, filename)

    dataset_pipeline_module.load_strict_frozen_dataset(
        forged_root,
        success.feature_registry,
        expected_dataset_id=forged_root.name,
    )
    with pytest.raises(ValueError, match="expected source identity"):
        dataset_pipeline_module.load_strict_frozen_dataset(
            forged_root,
            success.feature_registry,
            expected_dataset_id=forged_root.name,
            expected_source_identities={success.samples[0].sample_id: original},
        )


def test_notebook_validation_forwards_resolver_identity_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: Notebook 1 authenticates frozen local records only by their own ID."""
    config, _, _ = _local_pipeline_config(tmp_path)
    result = run_dataset_pipeline(config)
    frozen = freeze_dataset(result)
    context = SimpleNamespace(
        pipeline_config=config,
        expected_row_count=len(result.samples),
        repository_root=tmp_path,
    )
    actual_loader = notebook_support_module.load_strict_frozen_dataset
    loader_calls: list[dict[str, object]] = []

    def observed_loader(*args: object, **kwargs: object) -> object:
        loader_calls.append(dict(kwargs))
        return actual_loader(*args, **kwargs)

    monkeypatch.setattr(
        notebook_support_module, "load_strict_frozen_dataset", observed_loader
    )

    checks = notebook_support_module.validate_notebook_outputs(context, result, frozen)

    assert checks["terminal_records_valid"] is True
    assert checks["frozen_file_set_valid"] is True
    assert loader_calls == [
        {
            "expected_dataset_id": frozen.dataset_id,
            "expected_split_sha256": frozen.protocol.split_sha256,
            "expected_source_identities": {
                result.samples[0].sample_id: result.samples[0].trusted_source_identity
            },
        }
    ]


@pytest.mark.parametrize("malformed_stage", ["acquisition", "audio", "asr", "features"])
def test_malformed_fresh_success_fails_only_that_sample(
    tmp_path: Path, malformed_stage: str
) -> None:
    """Break caught: malformed success escapes its stage and aborts later samples."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
    )
    stages = MalformedSuccessStages(malformed_stage)

    result = run_dataset_pipeline(_config(tmp_path, stages, rows))

    first, second = result.samples
    expected_terminal = "feature_failed" if malformed_stage == "features" else f"{malformed_stage}_failed"
    assert first.terminal_status == expected_terminal
    assert getattr(first, malformed_stage).status == "failed"
    assert second.terminal_status == "succeeded"
    assert stages.acquisition_calls == ["N001", "N002"]


def test_adapter_supplied_file_digests_are_recomputed_from_bytes(tmp_path: Path) -> None:
    """Break caught: a dishonest adapter digest controls grouping and dataset identity."""
    stages = DishonestDigestStages()

    result = run_dataset_pipeline(
        _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    )
    sample = result.samples[0]

    source_path = tmp_path / str(sample.acquisition.payload["source_path"])
    analysis_path = tmp_path / str(sample.audio.payload["analysis_path"])
    assert sample.source_sha256 == _digest(source_path)
    assert sample.source_sha256 != "0" * 64
    assert sample.audio.payload["analysis_sha256"] == _digest(analysis_path)


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "mystery"},
        {"status": "failed", "error_code": None, "error_summary": None},
        {"status": "succeeded", "payload": {"features": [], "missing_reasons": {}}},
    ],
)
def test_invalid_cached_terminal_schema_is_rebuilt(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    """Break caught: an illegal cached status/payload is treated as terminal and resumed."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    metadata_path = first.staging_dir / "metadata" / "N001" / "features.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.update(mutation)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    second = run_dataset_pipeline(config)

    assert "features" not in second.samples[0].resumed_stages
    assert second.samples[0].features.status == "succeeded"
    assert stages.feature_calls == [("N001", True), ("N001", True)]


@pytest.mark.parametrize(
    "stage,invalid_payload,call_attribute",
    [
        ("acquisition", {"source_sha256": "0" * 64}, "acquisition_calls"),
        ("audio", {"analysis_sha256": "0" * 64}, "audio_calls"),
        (
            "asr",
            {"transcription": "malformed", "provenance": {}},
            "asr_batches",
        ),
        (
            "features",
            {"features": [], "missing_reasons": {}, "qc_warnings": []},
            "feature_calls",
        ),
    ],
)
def test_malformed_cached_success_payload_is_rebuilt(
    tmp_path: Path,
    stage: str,
    invalid_payload: dict[str, object],
    call_attribute: str,
) -> None:
    """Break caught: stage-specific malformed cached success bypasses its adapter."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    metadata_path = first.staging_dir / "metadata" / "N001" / f"{stage}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["payload"] = invalid_payload
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls_before = len(getattr(stages, call_attribute))

    second = run_dataset_pipeline(config)

    assert stage not in second.samples[0].resumed_stages
    assert len(getattr(stages, call_attribute)) == calls_before + 1
    assert getattr(second.samples[0], stage).status == "succeeded"


def test_resume_requires_matching_input_and_stage_configuration_hashes(tmp_path: Path) -> None:
    """Break caught: stale stage output is reused after a relevant hash changes."""
    rows = (_source_row("N001", "aaaaaaaaaaa"),)
    stages = SyntheticStages()
    config = _config(tmp_path, stages, rows)

    first = run_dataset_pipeline(config)
    second = run_dataset_pipeline(config)
    changed = replace(
        config,
        stage_parameters={**config.stage_parameters, "features": {"revision": 2}},
    )
    third = run_dataset_pipeline(changed)

    assert first.samples[0].resumed_stages == ()
    assert second.samples[0].resumed_stages == ("acquisition", "audio", "asr", "features")
    assert third.samples[0].resumed_stages == ("acquisition", "audio", "asr")
    assert stages.acquisition_calls == ["N001"]
    assert stages.audio_calls == ["N001"]
    assert stages.asr_batches == [("N001",)]
    assert stages.feature_calls == [("N001", True), ("N001", True)]


def test_network_permission_does_not_change_acquisition_cache_identity(
    tmp_path: Path,
) -> None:
    """Break caught: switching FULL_RUN to safe RESUME retries every acquisition."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    run_dataset_pipeline(replace(config, allow_network=True))

    resumed = run_dataset_pipeline(replace(config, allow_network=False))

    assert resumed.samples[0].resumed_stages == (
        "acquisition",
        "audio",
        "asr",
        "features",
    )
    assert stages.acquisition_calls == ["N001"]


@pytest.mark.parametrize(
    ("forced", "expected_resumed"),
    [
        ("acquisition", ()),
        ("audio", ("acquisition",)),
        ("asr", ("acquisition", "audio")),
        ("features", ("acquisition", "audio", "asr")),
        ("all", ()),
    ],
)
def test_force_stage_invalidates_the_requested_stage_and_every_downstream_stage(
    tmp_path: Path, forced: str, expected_resumed: tuple[str, ...]
) -> None:
    """Break caught: FORCE_STAGE is hardcoded or leaves stale downstream outputs."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    run_dataset_pipeline(config)

    forced_result = run_dataset_pipeline(
        replace(config, force_stages=frozenset({forced}))
    )

    assert forced_result.samples[0].resumed_stages == expected_resumed


@pytest.mark.parametrize(
    ("changed_stage", "expected_resumed"),
    [
        ("acquisition", ("asr",)),
        ("audio", ("acquisition", "asr")),
        ("asr", ("acquisition", "audio")),
        ("features", ("acquisition", "audio", "asr")),
    ],
)
def test_stage_implementation_digest_drift_invalidates_only_that_stage_and_downstream(
    tmp_path: Path, changed_stage: str, expected_resumed: tuple[str, ...]
) -> None:
    """Break caught: cache reuse ignores a relevant implementation change."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    run_dataset_pipeline(config)
    changed_digests = dict(config.implementation_digests)
    changed_digests[changed_stage] = "f" * 64

    changed = run_dataset_pipeline(
        replace(config, implementation_digests=changed_digests)
    )

    assert changed.samples[0].resumed_stages == expected_resumed
    metadata = json.loads(
        (
            config.staging_dir
            / "metadata"
            / "N001"
            / f"{changed_stage}.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["implementation_sha256"] == "f" * 64


def test_stage_implementation_digests_ignore_unrelated_notebook_changes(
    tmp_path: Path,
) -> None:
    """Break caught: a whole-tree hash forces an unnecessary Whisper rerun."""
    project_root = Path(__file__).resolve().parents[2]
    for relative_paths in dataset_pipeline_module._STAGE_IMPLEMENTATION_FILES.values():
        for relative_path in relative_paths:
            source = project_root / relative_path
            destination = tmp_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    before = stage_implementation_digests(tmp_path)

    unrelated = tmp_path / "notebooks" / "reader_notes.ipynb"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("{}", encoding="utf-8")
    after = stage_implementation_digests(tmp_path)

    assert after == before


def test_orchestration_change_invalidates_every_stage(tmp_path: Path) -> None:
    """Break caught: orchestration drift reuses any durable stage cache."""
    before = stage_implementation_digests(PROJECT_ROOT)
    changed = _copy_implementation_tree_with_change(tmp_path, "dataset_pipeline.py")

    after = stage_implementation_digests(changed)

    assert all(
        before[name] != after[name]
        for name in ("acquisition", "audio", "asr", "features")
    )


def test_audio_implementation_change_invalidates_audio_and_downstream_only(
    tmp_path: Path,
) -> None:
    """Break caught: audio drift leaves downstream ASR or feature caches reusable."""
    before = stage_implementation_digests(PROJECT_ROOT)
    changed = _copy_implementation_tree_with_change(tmp_path, "audio_pipeline.py")

    after = stage_implementation_digests(changed)

    assert before["acquisition"] == after["acquisition"]
    assert all(
        before[name] != after[name]
        for name in ("audio", "asr", "features")
    )


def test_local_source_implementation_change_invalidates_every_dependent_stage(
    tmp_path: Path,
) -> None:
    """Break caught: local resolver or hash logic changes while durable cache stays trusted."""
    before = stage_implementation_digests(PROJECT_ROOT)
    changed = _copy_implementation_tree_with_change(tmp_path, "local_source.py")

    after = stage_implementation_digests(changed)

    assert all(
        before[name] != after[name]
        for name in ("acquisition", "audio", "asr", "features")
    )


def test_asr_implementation_change_invalidates_asr_and_features_only(
    tmp_path: Path,
) -> None:
    """Break caught: ASR drift leaks into upstream cache identity or spares features."""
    before = stage_implementation_digests(PROJECT_ROOT)
    changed = _copy_implementation_tree_with_change(tmp_path, "whisper_local.py")

    after = stage_implementation_digests(changed)

    assert before["acquisition"] == after["acquisition"]
    assert before["audio"] == after["audio"]
    assert before["asr"] != after["asr"]
    assert before["features"] != after["features"]


def test_feature_implementation_change_invalidates_features_only(
    tmp_path: Path,
) -> None:
    """Break caught: feature drift changes an upstream durable-stage identity."""
    before = stage_implementation_digests(PROJECT_ROOT)
    changed = _copy_implementation_tree_with_change(tmp_path, "acoustic_features.py")

    after = stage_implementation_digests(changed)

    assert all(
        before[name] == after[name] for name in ("acquisition", "audio", "asr")
    )
    assert before["features"] != after["features"]


def test_explicit_legacy_cache_compatibility_upgrades_without_rerunning_stage(
    tmp_path: Path,
) -> None:
    """Break caught: the one-time strict refresh discards every schema-v1 cache."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    metadata_root = first.staging_dir / "metadata" / "N001"
    legacy_records = {
        stage: json.loads((metadata_root / f"{stage}.json").read_text(encoding="utf-8"))
        for stage in ("acquisition", "audio", "asr", "features")
    }
    for record in legacy_records.values():
        record["schema_version"] = 1
        record.pop("implementation_sha256")
    legacy_records["audio"]["input_sha256"] = sha256_bytes(
        canonical_json_bytes(legacy_records["acquisition"])
    )
    legacy_records["asr"]["input_sha256"] = sha256_bytes(
        canonical_json_bytes(legacy_records["audio"])
    )
    legacy_records["features"]["input_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "audio": sha256_bytes(canonical_json_bytes(legacy_records["audio"])),
                "asr": sha256_bytes(canonical_json_bytes(legacy_records["asr"])),
            }
        )
    )
    for stage, record in legacy_records.items():
        (metadata_root / f"{stage}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    resumed = run_dataset_pipeline(
        replace(
            config,
            allow_legacy_stage_cache=True,
            legacy_compatible_stages=frozenset({"acquisition", "asr"}),
        )
    )

    assert resumed.samples[0].resumed_stages == ("acquisition", "asr")
    assert stages.acquisition_calls == ["N001"]
    assert stages.audio_calls == ["N001", "N001"]
    assert stages.asr_batches == [("N001",)]
    assert stages.feature_calls == [("N001", True), ("N001", True)]
    for stage in ("acquisition", "asr"):
        upgraded = json.loads(
            (metadata_root / f"{stage}.json").read_text(encoding="utf-8")
        )
        assert upgraded["schema_version"] == 2
        assert upgraded["implementation_sha256"] == config.implementation_digests[stage]


def test_legacy_asr_is_not_reused_when_recomputed_analysis_hash_changes(
    tmp_path: Path,
) -> None:
    """Break caught: one-time compatibility hides a measured analysis-WAV change."""
    stages = ChangingAnalysisStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    metadata_root = first.staging_dir / "metadata" / "N001"
    for stage in ("acquisition", "audio", "asr", "features"):
        path = metadata_root / f"{stage}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["schema_version"] = 1
        record.pop("implementation_sha256")
        path.write_text(json.dumps(record), encoding="utf-8")

    resumed = run_dataset_pipeline(
        replace(
            config,
            allow_legacy_stage_cache=True,
            legacy_compatible_stages=frozenset({"acquisition", "asr"}),
        )
    )

    assert resumed.samples[0].asr.resumed is False
    assert stages.asr_batches == [("N001",), ("N001",)]


def test_unbound_legacy_asr_cache_is_not_reused_by_default(tmp_path: Path) -> None:
    """Break caught: schema-v1 ASR is trusted without a prior implementation digest."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    metadata_root = first.staging_dir / "metadata" / "N001"
    for stage in ("audio", "asr"):
        metadata_path = metadata_root / f"{stage}.json"
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["schema_version"] = 1
        record.pop("implementation_sha256")
        metadata_path.write_text(json.dumps(record), encoding="utf-8")

    rerun = run_dataset_pipeline(
        replace(config, allow_legacy_stage_cache=True)
    )

    assert rerun.samples[0].asr.resumed is False
    assert stages.asr_batches == [("N001",), ("N001",)]


def test_registry_channel_count_uses_source_channels_not_mono_analysis(
    tmp_path: Path,
) -> None:
    """Break caught: channel_count is measured from analysis.wav after downmix."""
    stages = StereoSourceStages()

    result = run_dataset_pipeline(
        _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    )

    sample = result.samples[0]
    assert sample.audio.payload["source_channel_count"] == 2
    assert sample.audio.payload["analysis_channel_count"] == 1
    assert sample.features.payload["features"]["channel_count"] == 2


def test_audio_success_without_source_channel_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    """Break caught: source channel_count silently falls back to mono analysis.wav."""
    result = run_dataset_pipeline(
        _config(
            tmp_path,
            MissingSourceChannelStages(),
            (_source_row("N001", "aaaaaaaaaaa"),),
        )
    )

    assert result.samples[0].terminal_status == "audio_failed"
    assert result.samples[0].audio.error_code == "invalid_audio_result"


def test_source_identity_change_rebuilds_acquisition_and_dependent_audio(tmp_path: Path) -> None:
    """Break caught: a changed source-manifest row reuses stale acquisition metadata."""
    row = _source_row("N001", "aaaaaaaaaaa")
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (row,))
    run_dataset_pipeline(config)

    changed = replace(config, rows=(replace(row, video_title="changed source identity"),))
    result = run_dataset_pipeline(changed)

    assert "acquisition" not in result.samples[0].resumed_stages
    assert "audio" not in result.samples[0].resumed_stages
    assert stages.acquisition_calls == ["N001", "N001"]
    assert stages.audio_calls == ["N001", "N001"]


@pytest.mark.parametrize("stage,path_key", [("acquisition", "source_path"), ("audio", "analysis_path")])
def test_corrupted_cached_artifact_is_rebuilt(
    tmp_path: Path, stage: str, path_key: str
) -> None:
    """Break caught: cache metadata matches while referenced artifact bytes changed."""
    stages = SyntheticStages()
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    record = getattr(first.samples[0], stage)
    artifact_path = tmp_path / str(record.payload[path_key])
    artifact_path.write_bytes(b"corrupted")

    second = run_dataset_pipeline(config)

    assert stage not in second.samples[0].resumed_stages
    if stage == "acquisition":
        assert stages.acquisition_calls == ["N001", "N001"]
    else:
        assert stages.audio_calls == ["N001", "N001"]


def test_matching_upstream_failure_and_skipped_terminal_records_resume(tmp_path: Path) -> None:
    """Break caught: matching terminal skips are rewritten instead of resumed exactly."""
    rows = (_source_row("N001", "aaaaaaaaaaa"),)
    stages = SyntheticStages(acquisition_failures={"N001"})
    config = _config(tmp_path, stages, rows)

    first = run_dataset_pipeline(config)
    second = run_dataset_pipeline(config)

    assert first.samples[0].terminal_status == "acquisition_failed"
    assert first.samples[0].resumed_stages == ()
    assert second.samples[0].resumed_stages == ("acquisition", "audio", "asr", "features")
    assert stages.acquisition_calls == ["N001"]


@pytest.mark.parametrize("forced_stage", ["acquisition", "asr"])
def test_forced_acquisition_or_asr_explicitly_retries_terminal_failure(
    tmp_path: Path, forced_stage: str
) -> None:
    """Break caught: forced expensive stages keep reusing a valid terminal failure."""
    stages = SyntheticStages(
        acquisition_failures={"N001"} if forced_stage == "acquisition" else None,
        asr_failures={"N001"} if forced_stage == "asr" else None,
    )
    config = _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    first = run_dataset_pipeline(config)
    stages.acquisition_failures.clear()
    stages.asr_failures.clear()

    retried = run_dataset_pipeline(
        replace(config, force_stages=frozenset({forced_stage}))
    )

    assert first.samples[0].terminal_status == f"{forced_stage}_failed"
    assert retried.samples[0].terminal_status == "succeeded"
    assert forced_stage not in retried.samples[0].resumed_stages


def test_streamed_asr_result_is_cached_before_a_later_external_interruption(
    tmp_path: Path,
) -> None:
    """Break caught: one late ASR stall loses every earlier completed transcript."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
    )
    stages = InterruptingStreamStages()
    config = _config(tmp_path, stages, rows)

    with pytest.raises(KeyboardInterrupt, match="synthetic native stall interruption"):
        run_dataset_pipeline(config)

    first_metadata = config.staging_dir / "metadata" / "N001" / "asr.json"
    second_metadata = config.staging_dir / "metadata" / "N002" / "asr.json"
    assert json.loads(first_metadata.read_text(encoding="utf-8"))["status"] == "succeeded"
    assert not second_metadata.exists()

    stages.interrupt = False
    resumed = run_dataset_pipeline(config)

    assert stages.asr_batches == [("N001", "N002"), ("N002",)]
    assert resumed.samples[0].asr.resumed is True
    assert resumed.samples[1].asr.status == "succeeded"


def test_iterator_runtime_error_preserves_yields_and_leaves_misses_retryable(
    tmp_path: Path,
) -> None:
    """Break caught: a batch abort fabricates terminal failures for unyielded rows."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
    )
    stages = InterruptingStreamStages(RuntimeError("synthetic iterator abort"))
    config = _config(tmp_path, stages, rows)

    with pytest.raises(RuntimeError, match="synthetic iterator abort"):
        run_dataset_pipeline(config)

    first_metadata = config.staging_dir / "metadata" / "N001" / "asr.json"
    second_metadata = config.staging_dir / "metadata" / "N002" / "asr.json"
    assert json.loads(first_metadata.read_text(encoding="utf-8"))["status"] == "succeeded"
    assert not second_metadata.exists()

    stages.interrupt = False
    resumed = run_dataset_pipeline(config)

    assert stages.asr_batches == [("N001", "N002"), ("N002",)]
    assert resumed.samples[0].asr.resumed is True
    assert resumed.samples[1].asr.status == "succeeded"


def test_asr_progress_is_emitted_only_after_each_durable_terminal_record(
    tmp_path: Path,
) -> None:
    """Break caught: reader progress advances before the matching ASR cache exists."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
    )
    stages = SyntheticStages()
    events: list[dict[str, object]] = []
    base_config = _config(tmp_path, stages, rows)
    config = replace(base_config, progress_callback=lambda event: events.append(dict(event)))

    run_dataset_pipeline(config)

    assert events == [
        {
            "stage": "asr",
            "completed": 1,
            "total": 2,
            "status_counts": {"succeeded": 1},
        },
        {
            "stage": "asr",
            "completed": 2,
            "total": 2,
            "status_counts": {"succeeded": 2},
        },
    ]
    for completed, sample_id in enumerate(("N001", "N002"), start=1):
        metadata = config.staging_dir / "metadata" / sample_id / "asr.json"
        assert metadata.is_file()
        assert events[completed - 1]["completed"] == completed


def test_feature_failure_is_terminal_without_stopping_later_samples(tmp_path: Path) -> None:
    """Break caught: a feature exception prevents a later sample from reaching features."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
    )
    stages = SyntheticStages(feature_failures={"N001"})

    result = run_dataset_pipeline(_config(tmp_path, stages, rows))

    assert result.terminal_counts == {"feature_failed": 1, "succeeded": 1}
    assert stages.feature_calls == [("N001", True), ("N002", True)]


def test_quality_report_preserves_canonical_per_feature_missing_reasons(
    tmp_path: Path,
) -> None:
    """Break caught: quality_report.csv retains only a missing-value count."""
    stages = SyntheticStages(feature_failures={"N001"})
    run_result = run_dataset_pipeline(
        _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    )

    frozen = freeze_dataset(run_result)

    with (frozen.path / "quality_report.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        quality = next(csv.DictReader(handle))
    reasons = json.loads(quality["missing_reasons_json"])
    assert set(reasons) == set(run_result.feature_registry.by_name)
    assert set(reasons.values()) == {"feature_stage_failed"}
    assert int(quality["missing_feature_count"]) == len(
        run_result.feature_registry.features
    )


def test_feature_failed_labelled_sample_is_not_supervised_eligible(tmp_path: Path) -> None:
    """Break caught: a labelled terminal feature failure enters supervised folds."""
    stages = SyntheticStages(feature_failures={"N001"})
    run_result = run_dataset_pipeline(
        _config(tmp_path, stages, (_source_row("N001", "aaaaaaaaaaa"),))
    )

    frozen = freeze_dataset(run_result)

    assignment = frozen.split_registry.assignments[0]
    assert assignment.modelling_scope == "supervised_exploratory"
    assert assignment.supervised_eligible is False


def test_freeze_writes_a_non_circular_immutable_dataset(tmp_path: Path) -> None:
    """Break caught: the bound protocol feeds back into ID calculation or outputs are incomplete."""
    rows = (
        _source_row("N001", "aaaaaaaaaaa"),
        _source_row("N002", "bbbbbbbbbbb"),
        _source_row("N003", "ccccccccccc", score=None),
    )
    stages = SyntheticStages(acquisition_failures={"N002"}, asr_failures={"N003"})
    run_result = run_dataset_pipeline(_config(tmp_path, stages, rows))

    frozen = freeze_dataset(run_result)

    assert frozen.path.name == frozen.dataset_id
    assert len(frozen.dataset_id) == 64
    assert set("0123456789abcdef") >= set(frozen.dataset_id)
    assert {path.name for path in frozen.path.iterdir()} == _FROZEN_OUTPUTS

    bound = json.loads((frozen.path / "experiment_protocol.json").read_text(encoding="utf-8"))
    splits = json.loads((frozen.path / "splits.json").read_text(encoding="utf-8"))
    provenance = json.loads((frozen.path / "provenance.json").read_text(encoding="utf-8"))
    assert bound["dataset_id"] == frozen.dataset_id
    assert bound["protocol_id"] == protocol_id(run_result.protocol_template)
    assert bound["split_sha256"] == frozen.split_registry.sha256
    assert splits["evaluation_claim"] == "exploratory_no_speaker_holdout"
    assert "experiment_protocol.json" not in provenance["dataset_identity_inputs"]

    with (frozen.path / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 3
    assert manifest_rows[2]["modelling_scope"] == "unsupervised_only"
    assert "youtube.com" not in (frozen.path / "manifest.csv").read_text(encoding="utf-8")
    assert len(pd.read_parquet(frozen.path / "acoustic_features.parquet")) == 3

    checksum_lines = (frozen.path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 9
    for line in checksum_lines:
        expected, filename = line.split("  ", 1)
        assert _digest(frozen.path / filename) == expected

    reused = freeze_dataset(run_result)
    assert reused.dataset_id == frozen.dataset_id
    assert reused.path == frozen.path
    assert reused.reused is True


def test_candidate_is_complete_and_valid_before_single_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: immutable destination is visible before protocol/checksums exist."""
    run_result = run_dataset_pipeline(
        _config(
            tmp_path,
            SyntheticStages(),
            (_source_row("N001", "aaaaaaaaaaa"),),
        )
    )
    observed: list[tuple[Path, Path]] = []

    def inspect_then_promote(candidate: Path, destination: Path) -> None:
        _assert_complete_frozen_candidate(candidate)
        observed.append((candidate, destination))
        os.replace(candidate, destination)

    monkeypatch.setattr(
        dataset_pipeline_module,
        "_promote_candidate",
        inspect_then_promote,
        raising=False,
    )

    frozen = freeze_dataset(run_result)

    assert observed == [(run_result.staging_dir / "frozen-candidate", frozen.path)]
    _assert_complete_frozen_candidate(frozen.path)


def test_promotion_interruption_leaves_complete_candidate_and_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: interruption publishes or strands a partially frozen dataset."""
    run_result = run_dataset_pipeline(
        _config(
            tmp_path,
            SyntheticStages(),
            (_source_row("N001", "aaaaaaaaaaa"),),
        )
    )
    attempted_destinations: list[Path] = []

    def interrupt(candidate: Path, destination: Path) -> None:
        _assert_complete_frozen_candidate(candidate)
        attempted_destinations.append(destination)
        raise OSError("synthetic promotion interruption")

    monkeypatch.setattr(
        dataset_pipeline_module,
        "_promote_candidate",
        interrupt,
        raising=False,
    )

    with pytest.raises(OSError, match="synthetic promotion interruption"):
        freeze_dataset(run_result)

    assert len(attempted_destinations) == 1
    assert not attempted_destinations[0].exists()
    _assert_complete_frozen_candidate(run_result.staging_dir / "frozen-candidate")


def test_incomplete_existing_destination_fails_closed_without_deleting_content(
    tmp_path: Path,
) -> None:
    """Break caught: collision recovery overwrites or deletes unrelated destination data."""
    run_result = run_dataset_pipeline(
        _config(
            tmp_path,
            SyntheticStages(),
            (_source_row("N001", "aaaaaaaaaaa"),),
        )
    )
    frozen = freeze_dataset(run_result)
    (frozen.path / "checksums.sha256").unlink()
    sentinel = frozen.path / "unrelated-user-data.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable dataset path exists"):
        freeze_dataset(run_result)

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert frozen.path.is_dir()


def test_semantically_corrupt_existing_destination_fails_closed(
    tmp_path: Path,
) -> None:
    """Break caught: recomputed checksums allow invalid frozen artifact formats to escape."""
    run_result = run_dataset_pipeline(
        _config(
            tmp_path,
            SyntheticStages(),
            (_source_row("N001", "aaaaaaaaaaa"),),
        )
    )
    frozen = freeze_dataset(run_result)
    parquet = frozen.path / "acoustic_features.parquet"
    parquet.write_bytes(b"not a parquet file")
    checksum_path = frozen.path / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            f"{_digest(parquet)}  acoustic_features.parquet"
            if line.endswith("  acoustic_features.parquet")
            else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="immutable dataset path exists"):
        freeze_dataset(run_result)

    assert parquet.read_bytes() == b"not a parquet file"


def test_private_frozen_manifest_preserves_source_audit_and_terminal_reasons(
    tmp_path: Path,
) -> None:
    """Break caught: freeze discards the source-row trace needed to audit one sample."""
    row = replace(
        _source_row("N001", "aaaaaaaaaaa", score=88.5),
        source_row_id="advisor-row-17",
        legacy_sample_id="S17",
        source_page="page-7",
        video_title="Synthetic audited title",
        source_course="COMM-9000",
        verification_note="verified twice, locally",
    )
    stages = SyntheticStages(asr_failures={"N001"})
    run_result = run_dataset_pipeline(_config(tmp_path, stages, (row,)))

    frozen = freeze_dataset(run_result)

    with (frozen.path / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_row = next(csv.DictReader(handle))
    assert manifest_row == {  # privacy-scan: synthetic-fixture
        "source_row_id": "advisor-row-17",
        "sample_id": "N001",
        "legacy_sample_id": "S17",
        "source_page": "page-7",
        "video_id": "aaaaaaaaaaa",
        "video_title": "Synthetic audited title",
        "source_course": "COMM-9000",
        "total_score": "88.5",
        "score_status": "observed",
        "verification_status": "verified",
        "verification_note": "verified twice, locally",
        "source_kind": "youtube",
        "local_basename": "",
        "catalog_identity_sha256": "",
        "source_duration_seconds": "",
        "source_sha256": run_result.samples[0].source_sha256,
        "terminal_status": "asr_failed",
        "acquisition_status": "succeeded",
        "acquisition_error_code": "",
        "acquisition_error_summary": "",
        "audio_status": "succeeded",
        "audio_error_code": "",
        "audio_error_summary": "",
        "asr_status": "failed",
        "asr_error_code": "transcription_error",
        "asr_error_summary": "synthetic ASR failure",
        "feature_status": "succeeded",
        "feature_error_code": "",
        "feature_error_summary": "",
        "speaker_id": "unknown",
        "talk_id": "aaaaaaaaaaa",
        "group_id": frozen.split_registry.assignments[0].group_id,
        "outer_fold": "0",
        "inner_fold": "0",
        "modelling_scope": "supervised_exploratory",
        "supervised_eligible": "True",
        "qc_eligible": "True",
        "clustering_eligible": "True",
    }
    manifest_text = (frozen.path / "manifest.csv").read_text(encoding="utf-8")
    assert row.source_url not in manifest_text


def _strict_frozen_fixture(tmp_path: Path) -> tuple[object, object]:
    stages = SyntheticStages(asr_failures={"N002"})
    run_result = run_dataset_pipeline(
        _config(
            tmp_path,
            stages,
            (
                _source_row("N001", "aaaaaaaaaaa"),
                _source_row("N002", "bbbbbbbbbbb"),
            ),
        )
    )
    return run_result, freeze_dataset(run_result)


def test_public_strict_frozen_loader_validates_exact_tables_and_sample_sets(
    tmp_path: Path,
) -> None:
    """Break caught: notebooks rely on separate shallow frozen-data validators."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)

    loaded = dataset_pipeline_module.load_frozen_dataset(
        frozen.path, run_result.feature_registry
    )

    assert loaded.dataset_id == frozen.dataset_id
    assert loaded.row_count == 2
    assert loaded.checksum_count == 9
    assert loaded.sample_ids == frozenset({"N001", "N002"})


def test_strict_loader_rejects_duplicate_sample_ids_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    """Break caught: equal row counts hide a duplicated and omitted sample."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    manifest_path = frozen.path / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[1]["sample_id"] = rows[0]["sample_id"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksum(frozen.path, "manifest.csv")

    with pytest.raises(ValueError, match="unique sample_id"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


@pytest.mark.parametrize("case", ["missing", "mismatched"])
def test_strict_loader_rejects_missing_or_mismatched_sample_id_sets(
    tmp_path: Path, case: str
) -> None:
    """Break caught: table counts match while the represented samples differ."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    transcript_path = frozen.path / "transcripts.jsonl"
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    if case == "missing":
        rows.pop()
    else:
        rows[-1]["sample_id"] = "N999"
    transcript_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    _refresh_checksum(frozen.path, "transcripts.jsonl")

    with pytest.raises(ValueError, match="equal sample_id sets"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_strict_loader_rejects_terminal_status_inconsistency(
    tmp_path: Path,
) -> None:
    """Break caught: manifest terminal status contradicts its stage outcomes."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    manifest_path = frozen.path / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["terminal_status"] = "audio_failed"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksum(frozen.path, "manifest.csv")

    with pytest.raises(ValueError, match="terminal status"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_strict_loader_rejects_split_eligibility_inconsistency(
    tmp_path: Path,
) -> None:
    """Break caught: splits declare analysis eligibility inconsistent with manifest status."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    splits_path = frozen.path / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits["assignments"][0]["qc_eligible"] = False
    splits_path.write_bytes(canonical_json_bytes(splits))
    split_sha = sha256_bytes(canonical_json_bytes(splits))
    protocol_path = frozen.path / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["split_sha256"] = split_sha
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    _refresh_checksum(frozen.path, "splits.json")
    _refresh_checksum(frozen.path, "experiment_protocol.json")

    with pytest.raises(ValueError, match="eligibility"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_strict_loader_rejects_feature_schema_drift(
    tmp_path: Path,
) -> None:
    """Break caught: unregistered feature columns silently enter Notebook 2."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    feature_path = frozen.path / "acoustic_features.parquet"
    frame = pd.read_parquet(feature_path)
    frame["unregistered_feature"] = 1.0
    frame.to_parquet(feature_path, index=False)
    _refresh_checksum(frozen.path, "acoustic_features.parquet")

    with pytest.raises(ValueError, match="feature schema"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_strict_loader_rejects_null_feature_without_matching_reason(
    tmp_path: Path,
) -> None:
    """Break caught: null feature values lose their per-feature audit reason."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    feature_path = frozen.path / "acoustic_features.parquet"
    frame = pd.read_parquet(feature_path)
    frame.loc[0, "speech_ratio"] = None
    frame.to_parquet(feature_path, index=False)
    _refresh_checksum(frozen.path, "acoustic_features.parquet")

    with pytest.raises(ValueError, match="missing reason"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_strict_loader_rejects_non_finite_feature_values(
    tmp_path: Path,
) -> None:
    """Break caught: infinity passes through Parquet into analysis."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    feature_path = frozen.path / "acoustic_features.parquet"
    frame = pd.read_parquet(feature_path)
    frame.loc[0, "speech_ratio"] = float("inf")
    frame.to_parquet(feature_path, index=False)
    _refresh_checksum(frozen.path, "acoustic_features.parquet")

    with pytest.raises(ValueError, match="finite"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )


def test_table_tampering_plus_checksum_refresh_still_fails_dataset_identity(
    tmp_path: Path,
) -> None:
    """Break caught: checksums can be refreshed without changing the directory ID."""
    run_result, frozen = _strict_frozen_fixture(tmp_path)
    quality_path = frozen.path / "quality_report.csv"
    with quality_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["qc_warnings"] = "synthetic_tamper"
    with quality_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksum(frozen.path, "quality_report.csv")

    with pytest.raises(ValueError, match="dataset identity"):
        dataset_pipeline_module.load_frozen_dataset(
            frozen.path, run_result.feature_registry
        )
