"""Resumable orchestration and two-phase immutable research-dataset freeze."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from presentation_assessment.transcription import (
    TranscriptionResult,
    transcription_from_dict,
)

from .acoustic_features import (
    FeatureBundle,
    compare_normalization_stability,
    extract_acoustic_features,
)
from .acquisition import AcquisitionResult, YouTubeAcquirer
from .artifacts import atomic_write_json, atomic_write_text
from .audio_pipeline import AudioArtifacts, build_audio_representations
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .feature_registry import FeatureRegistry
from .local_source import (
    LocalAudioAdapter,
    LocalFileAcquirer,
    LocalSourceReference,
    LocalSourceResolver,
    TrustedSourceIdentity,
    local_source_reference_valid,
)
from .protocol import BoundProtocol, ProtocolTemplate, bind_protocol, protocol_id
from .provenance import SourceSnapshot, snapshot_source_tree, write_source_snapshot
from .quality import read_pcm16_wav
from .source_manifest import SourceRow
from .splits import SplitRegistry, create_exploratory_splits
from .vad import run_webrtc_vad
from .whisper_local import (
    LocalWhisperTranscriber,
    ModelFactory,
    WhisperRuntimeConfig,
    WhisperTranscriptionItem,
    WhisperTranscriptionResult,
)


_STAGES = ("acquisition", "audio", "asr", "features")
_SUCCESS = "succeeded"
_SUMMARY_LIMIT = 240
_SAFE_ERROR_CODE = re.compile(r"[a-z0-9_]{1,64}\Z")
_URL = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"']+")
_FROZEN_FILENAMES = (
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
)
_CHECKSUM_FILENAMES = tuple(
    sorted(filename for filename in _FROZEN_FILENAMES if filename != "checksums.sha256")
)
_STAGE_IMPLEMENTATION_FILES = {
    "acquisition": (
        Path("src/presentation_assessment/research/dataset_pipeline.py"),
        Path("src/presentation_assessment/research/acquisition.py"),
        Path("src/presentation_assessment/research/local_source.py"),
        Path("src/presentation_assessment/research/local_manifest_migration.py"),
        Path("src/presentation_assessment/research/artifacts.py"),
        Path("src/presentation_assessment/research/source_manifest.py"),
    ),
    "audio": (
        Path("src/presentation_assessment/research/dataset_pipeline.py"),
        Path("src/presentation_assessment/research/audio_pipeline.py"),
    ),
    "asr": (
        Path("src/presentation_assessment/research/dataset_pipeline.py"),
        Path("src/presentation_assessment/research/whisper_local.py"),
        Path("src/presentation_assessment/research/cuda_runtime.py"),
        Path("src/presentation_assessment/transcription.py"),
    ),
    "features": (
        Path("src/presentation_assessment/research/dataset_pipeline.py"),
        Path("src/presentation_assessment/research/vad.py"),
        Path("src/presentation_assessment/research/quality.py"),
        Path("src/presentation_assessment/research/acoustic_features.py"),
        Path("src/presentation_assessment/research/feature_registry.py"),
        Path("configs/audio_feature_registry.v1.yaml"),
    ),
}


@dataclass(frozen=True)
class DatasetStageResult:
    """Adapter-facing terminal result containing only stage output, not cache keys."""

    status: str
    sample_id: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    error_code: str | None = None
    error_summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status).lower())
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class DatasetStageRecord:
    """Durable hash-keyed outcome used for exact resume decisions."""

    stage: str
    sample_id: str
    status: str
    input_sha256: str
    config_sha256: str
    implementation_sha256: str
    payload: Mapping[str, object]
    error_code: str | None
    error_summary: str | None
    resumed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, object]:
        """Return persisted terminal metadata without run-local resume state."""
        return {
            "schema_version": 2,
            "stage": self.stage,
            "sample_id": self.sample_id,
            "terminal": True,
            "status": self.status,
            "input_sha256": self.input_sha256,
            "config_sha256": self.config_sha256,
            "implementation_sha256": self.implementation_sha256,
            "payload": dict(self.payload),
            "error_code": self.error_code,
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True)
class DatasetStageAdapters:
    """Injected boundaries around the Task 2–6 implementations."""

    acquire: Callable[[SourceRow, Path, bool], object]
    audio: Callable[[object, Path], object]
    transcribe_many: Callable[[Sequence[WhisperTranscriptionItem]], object]
    features: Callable[[str, dict[str, Path], TranscriptionResult | None, FeatureRegistry], object]

    @classmethod
    def production(
        cls,
        *,
        whisper_config: WhisperRuntimeConfig,
        acquirer: YouTubeAcquirer | None = None,
        local_source_resolver: LocalSourceResolver | None = None,
        model_factory: ModelFactory | None = None,
    ) -> "DatasetStageAdapters":
        """Compose the reviewed Task 2–6 implementations without reimplementing them."""
        if acquirer is not None and local_source_resolver is not None:
            raise ValueError("choose either a YouTube acquirer or a local source resolver")
        actual_acquirer: object = (
            LocalFileAcquirer(local_source_resolver)
            if local_source_resolver is not None
            else (acquirer or YouTubeAcquirer())
        )
        transcriber = LocalWhisperTranscriber(whisper_config, model_factory=model_factory)
        return cls(
            acquire=actual_acquirer.acquire,  # type: ignore[attr-defined]
            audio=(
                LocalAudioAdapter(local_source_resolver)
                if local_source_resolver is not None
                else build_audio_representations
            ),
            transcribe_many=transcriber.iter_transcriptions,
            features=_extract_feature_stage,
        )


@dataclass(frozen=True)
class DatasetPipelineConfig:
    """All in-memory inputs and injected effects for one resumable dataset run."""

    repository_root: Path
    staging_dir: Path
    artifact_root: Path
    rows: Sequence[SourceRow]
    protocol_template: ProtocolTemplate
    feature_registry: FeatureRegistry
    pipeline_parameters: Mapping[str, object]
    stage_parameters: Mapping[str, object]
    whisper_fingerprint: Mapping[str, object]
    implementation_digests: Mapping[str, str]
    adapters: DatasetStageAdapters
    allow_network: bool = False
    allow_legacy_stage_cache: bool = False
    legacy_compatible_stages: frozenset[str] = frozenset({"acquisition"})
    force_stages: frozenset[str] = frozenset()
    progress_callback: Callable[[Mapping[str, object]], None] | None = None
    source_snapshotter: Callable[[Path], SourceSnapshot] = snapshot_source_tree

    def __post_init__(self) -> None:
        root = Path(self.repository_root).resolve(strict=True)
        staging = Path(self.staging_dir).resolve()
        artifacts = Path(self.artifact_root).resolve()
        _require_descendant(staging, root, "staging_dir")
        _require_descendant(artifacts, root, "artifact_root")
        invalid_forces = set(self.force_stages) - {*_STAGES, "all"}
        if invalid_forces:
            raise ValueError(f"unknown force stages: {sorted(invalid_forces)}")
        invalid_legacy_stages = set(self.legacy_compatible_stages) - {
            "acquisition",
            "asr",
        }
        if invalid_legacy_stages:
            raise ValueError(
                f"unknown legacy-compatible stages: {sorted(invalid_legacy_stages)}"
            )
        object.__setattr__(self, "repository_root", root)
        object.__setattr__(self, "staging_dir", staging)
        object.__setattr__(self, "artifact_root", artifacts)
        object.__setattr__(self, "rows", tuple(self.rows))
        _canonical_value(self.pipeline_parameters)
        _canonical_value(self.stage_parameters)
        _canonical_value(self.whisper_fingerprint)
        acquisition_parameters = self.stage_parameters.get("acquisition", {})
        declared_local_adapter = (
            isinstance(acquisition_parameters, Mapping)
            and acquisition_parameters.get("adapter") == "local_file"
        )
        if declared_local_adapter and re.fullmatch(
            r"[0-9a-f]{64}",
            str(acquisition_parameters.get("local_source_identity_sha256", "")),
        ) is None:
            raise ValueError(
                "local_file acquisition requires local_source_identity_sha256"
            )
        acquire_owner = getattr(self.adapters.acquire, "__self__", None)
        local_owner = isinstance(acquire_owner, LocalFileAcquirer)
        if declared_local_adapter != local_owner:
            raise ValueError(
                "local acquisition adapter declaration must match LocalFileAcquirer owner"
            )
        if (
            declared_local_adapter
            and local_owner
            and acquisition_parameters.get("local_source_identity_sha256")
            != acquire_owner.catalog_identity_sha256
        ):
            raise ValueError(
                "local_source_identity_sha256 does not match the local resolver"
            )
        if local_owner and any(
            acquire_owner.trusted_identity(row) is None for row in self.rows
        ):
            raise ValueError("local source identity does not match the configured catalog")
        object.__setattr__(self, "pipeline_parameters", MappingProxyType(dict(self.pipeline_parameters)))
        object.__setattr__(self, "stage_parameters", MappingProxyType(dict(self.stage_parameters)))
        object.__setattr__(self, "whisper_fingerprint", MappingProxyType(dict(self.whisper_fingerprint)))
        digests = dict(self.implementation_digests)
        if set(digests) != set(_STAGES) or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in digests.values()
        ):
            raise ValueError("implementation_digests must contain one lowercase SHA-256 per stage")
        object.__setattr__(self, "implementation_digests", MappingProxyType(digests))
        object.__setattr__(
            self, "legacy_compatible_stages", frozenset(self.legacy_compatible_stages)
        )
        requested_forces = set(self.force_stages)
        if "all" in requested_forces:
            expanded_forces = set(_STAGES)
        else:
            expanded_forces = {
                downstream
                for requested in requested_forces
                for downstream in _STAGES[_STAGES.index(requested) :]
            }
        object.__setattr__(self, "force_stages", frozenset(expanded_forces))
        configured_seed = self.pipeline_parameters.get("random_seed")
        if configured_seed is not None and configured_seed != self.protocol_template.random_seed:
            raise ValueError("pipeline and protocol random seeds must match")


@dataclass(frozen=True)
class DatasetSampleResult:
    """One source row with terminal outcomes retained for every pipeline stage."""

    source_row: SourceRow
    acquisition: DatasetStageRecord
    audio: DatasetStageRecord
    asr: DatasetStageRecord
    features: DatasetStageRecord
    trusted_source_identity: TrustedSourceIdentity | Mapping[str, str]
    terminal_status: str

    def __post_init__(self) -> None:
        trusted = self.trusted_source_identity
        if not isinstance(trusted, TrustedSourceIdentity):
            trusted = TrustedSourceIdentity(**dict(trusted))
        object.__setattr__(self, "trusted_source_identity", trusted)

    @property
    def sample_id(self) -> str:
        return self.source_row.sample_id

    @property
    def video_id(self) -> str:
        return self.source_row.video_id

    @property
    def source_sha256(self) -> str | None:
        value = self.acquisition.payload.get("source_sha256")
        return str(value) if isinstance(value, str) and value else None

    @property
    def resumed_stages(self) -> tuple[str, ...]:
        return tuple(
            record.stage
            for record in (self.acquisition, self.audio, self.asr, self.features)
            if record.resumed
        )

    @property
    def qc_eligible(self) -> bool:
        return self.features.status == _SUCCESS

    @property
    def clustering_eligible(self) -> bool:
        return self.features.status == _SUCCESS

    def split_row(self) -> dict[str, object]:
        """Return the limitation-aware fields consumed by split creation."""
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "source_sha256": self.source_sha256,
            "speaker_id": "unknown",
            "talk_id": self.video_id or "unknown",
            "total_score": self.source_row.total_score,
            "score_status": self.source_row.score_status.value,
            "supervised_eligible": (
                self.features.status == _SUCCESS
                and self.source_row.score_status.value == "observed"
                and self.source_row.total_score is not None
            ),
            "qc_eligible": self.qc_eligible,
            "clustering_eligible": self.clustering_eligible,
        }

    def to_identity_dict(self) -> dict[str, object]:
        """Return private-URL-free terminal content for dataset identity."""
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "total_score": self.source_row.total_score,
            "score_status": self.source_row.score_status.value,
            "terminal_status": self.terminal_status,
            "trusted_source": dict(self.trusted_source_identity),
            "stages": {
                record.stage: record.to_dict()
                for record in (self.acquisition, self.audio, self.asr, self.features)
            },
        }


@dataclass(frozen=True)
class DatasetRunResult:
    """Completed resumable staging run, ready for two-phase freeze."""

    repository_root: Path
    staging_dir: Path
    artifact_root: Path
    samples: tuple[DatasetSampleResult, ...]
    protocol_template: ProtocolTemplate
    feature_registry: FeatureRegistry
    pipeline_parameters: Mapping[str, object]
    whisper_fingerprint: Mapping[str, object]
    source_manifest_sha256: str
    source_snapshotter: Callable[[Path], SourceSnapshot]

    @property
    def terminal_counts(self) -> dict[str, int]:
        """Count terminal row outcomes without dropping failed samples."""
        counts: Counter[str] = Counter(sample.terminal_status for sample in self.samples)
        return dict(counts)


@dataclass(frozen=True)
class FrozenDatasetResult:
    """Immutable dataset location and its post-freeze bindings."""

    dataset_id: str
    path: Path
    protocol: BoundProtocol
    split_registry: SplitRegistry
    reused: bool = False


@dataclass(frozen=True)
class LoadedFrozenDataset:
    """Strictly validated frozen input shared by both research notebooks."""

    path: Path
    dataset_id: str
    protocol: BoundProtocol
    row_count: int
    checksum_count: int
    sample_ids: frozenset[str]


@dataclass
class _SampleState:
    row: SourceRow
    acquisition: DatasetStageRecord
    audio: DatasetStageRecord
    trusted_source_identity: Mapping[str, str]
    legacy_analysis_sha256: str | None = None
    asr: DatasetStageRecord | None = None
    features: DatasetStageRecord | None = None


class _AudioFeatureInputs(dict[str, Path]):
    """Path mapping with source/analysis channel provenance for feature extraction."""

    def __init__(
        self,
        *args: object,
        source_channel_count: int | None = None,
        analysis_channel_count: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.source_channel_count = source_channel_count
        self.analysis_channel_count = analysis_channel_count


def run_dataset_pipeline(config: DatasetPipelineConfig | Mapping[str, object]) -> DatasetRunResult:
    """Run acquisition → audio → one ASR batch → features with sample isolation."""
    resolved = _coerce_config(config)
    protocol_id(resolved.protocol_template)
    resolved.staging_dir.mkdir(parents=True, exist_ok=True)
    metadata_root = resolved.staging_dir / "metadata"
    work_root = resolved.staging_dir / "work"
    states: list[_SampleState] = []

    for row in resolved.rows:
        trusted_source = _trusted_source_identity(resolved, row)
        sample_root = work_root / row.sample_id
        acquisition_input = _source_row_sha256(row)
        acquisition = _run_individual_stage(
            config=resolved,
            stage="acquisition",
            sample_id=row.sample_id,
            input_sha256=acquisition_input,
            metadata_path=metadata_root / row.sample_id / "acquisition.json",
            output_root=sample_root,
            operation=lambda row=row, sample_root=sample_root: resolved.adapters.acquire(
                row, sample_root, resolved.allow_network
            ),
            expected_source_identity=trusted_source,
        )
        trusted_source = _bind_acquired_source_hash(trusted_source, acquisition)
        if acquisition.status != _SUCCESS:
            audio = _record_skipped(
                resolved,
                "audio",
                row.sample_id,
                _record_sha256(acquisition),
                metadata_root / row.sample_id / "audio.json",
                "acquisition_failed",
            )
        else:
            source_input = _acquired_source_input(resolved.repository_root, acquisition)
            audio_metadata_path = metadata_root / row.sample_id / "audio.json"
            legacy_analysis_sha256 = (
                _legacy_cached_analysis_sha256(audio_metadata_path)
                if resolved.allow_legacy_stage_cache
                else None
            )
            audio = _run_individual_stage(
                config=resolved,
                stage="audio",
                sample_id=row.sample_id,
                input_sha256=_record_sha256(acquisition),
                metadata_path=audio_metadata_path,
                output_root=sample_root / "audio",
                operation=lambda source_input=source_input, sample_root=sample_root: resolved.adapters.audio(
                    source_input, sample_root / "audio"
                ),
            )
        states.append(
            _SampleState(
                row=row,
                acquisition=acquisition,
                audio=audio,
                trusted_source_identity=trusted_source,
                legacy_analysis_sha256=(
                    legacy_analysis_sha256
                    if acquisition.status == _SUCCESS
                    else None
                ),
            )
        )

    _run_asr_batch(resolved, states, metadata_root)

    for state in states:
        assert state.asr is not None
        if state.audio.status != _SUCCESS:
            state.features = _record_skipped(
                resolved,
                "features",
                state.row.sample_id,
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            "audio": _record_sha256(state.audio),
                            "asr": _record_sha256(state.asr),
                        }
                    )
                ),
                metadata_root / state.row.sample_id / "features.json",
                "audio_failed",
            )
            continue
        audio_paths = _audio_paths(resolved.repository_root, state.audio)
        transcript = _transcription_from_record(state.asr)
        feature_input = sha256_bytes(
            canonical_json_bytes(
                {"audio": _record_sha256(state.audio), "asr": _record_sha256(state.asr)}
            )
        )
        state.features = _run_individual_stage(
            config=resolved,
            stage="features",
            sample_id=state.row.sample_id,
            input_sha256=feature_input,
            metadata_path=metadata_root / state.row.sample_id / "features.json",
            output_root=work_root / state.row.sample_id / "features",
            operation=lambda state=state, audio_paths=audio_paths, transcript=transcript: (
                resolved.adapters.features(
                    state.row.sample_id,
                    audio_paths,
                    transcript,
                    resolved.feature_registry,
                )
            ),
        )

    samples = tuple(_finalize_sample(state) for state in states)
    return DatasetRunResult(
        repository_root=resolved.repository_root,
        staging_dir=resolved.staging_dir,
        artifact_root=resolved.artifact_root,
        samples=samples,
        protocol_template=resolved.protocol_template,
        feature_registry=resolved.feature_registry,
        pipeline_parameters=resolved.pipeline_parameters,
        whisper_fingerprint=resolved.whisper_fingerprint,
        source_manifest_sha256=sha256_bytes(
            canonical_json_bytes([_source_row_identity(row) for row in resolved.rows])
        ),
        source_snapshotter=resolved.source_snapshotter,
    )


def stage_implementation_digests(repository_root: Path) -> dict[str, str]:
    """Hash each stage implementation plus its cumulative upstream identity."""
    root = Path(repository_root).resolve(strict=True)
    digests: dict[str, str] = {}
    upstream_digest: str | None = None
    for stage in _STAGES:
        relative_paths = _STAGE_IMPLEMENTATION_FILES[stage]
        file_hashes = {
            path.as_posix(): sha256_file(root / path)
            for path in relative_paths
        }
        identity: dict[str, object] = {"files": file_hashes}
        if upstream_digest is not None:
            identity["upstream_implementation_sha256"] = upstream_digest
        digests[stage] = sha256_bytes(canonical_json_bytes(identity))
        upstream_digest = digests[stage]
    return digests


def freeze_dataset(run_result: DatasetRunResult) -> FrozenDatasetResult:
    """Freeze a staging result without feeding the bound protocol into its ID."""
    pre_protocol_id = protocol_id(run_result.protocol_template)
    split_registry = create_exploratory_splits(
        [sample.split_row() for sample in run_result.samples],
        seed=run_result.protocol_template.random_seed,
    )
    source_snapshot = run_result.source_snapshotter(run_result.repository_root)
    if not isinstance(source_snapshot, SourceSnapshot):
        raise TypeError("source_snapshotter must return SourceSnapshot")

    candidate = run_result.staging_dir / "frozen-candidate"
    _reset_candidate(candidate, run_result.staging_dir)
    candidate.mkdir(parents=True)
    _write_manifest(candidate / "manifest.csv", run_result.samples, split_registry)
    _write_transcripts(candidate / "transcripts.jsonl", run_result.samples)
    _write_features(
        candidate / "acoustic_features.parquet",
        run_result.samples,
        run_result.feature_registry,
    )
    _write_quality(
        candidate / "quality_report.csv",
        run_result.samples,
        run_result.feature_registry,
    )
    atomic_write_json(candidate / "splits.json", split_registry.to_dict())
    write_source_snapshot(source_snapshot, candidate / "source_snapshot.json")

    identity_inputs = {
        "schema_version": 1,
        "source_manifest_sha256": run_result.source_manifest_sha256,
        "terminal_samples": [sample.to_identity_dict() for sample in run_result.samples],
        "pipeline_configuration_sha256": sha256_bytes(
            canonical_json_bytes(_canonical_value(run_result.pipeline_parameters))
        ),
        "feature_registry_sha256": sha256_bytes(
            canonical_json_bytes(run_result.feature_registry.model_dump(mode="json"))
        ),
        "protocol_id": pre_protocol_id,
        "split_registry": split_registry.to_dict(),
        "whisper_fingerprint": _safe_metadata_value(
            run_result.whisper_fingerprint, run_result.repository_root, run_result.repository_root
        ),
        "source_snapshot": _source_snapshot_payload(source_snapshot),
        "logical_table_sha256": {
            filename: sha256_file(candidate / filename)
            for filename in (
                "manifest.csv",
                "transcripts.jsonl",
                "acoustic_features.parquet",
                "quality_report.csv",
                "splits.json",
            )
        },
    }
    dataset_id = sha256_bytes(canonical_json_bytes(identity_inputs))
    atomic_write_json(
        candidate / "provenance.json",
        {
            "schema_version": 1,
            "dataset_identity_algorithm": "sha256_rfc8785",
            "dataset_identity_inputs": identity_inputs,
            "terminal_counts": run_result.terminal_counts,
        },
    )

    bound = bind_protocol(
        run_result.protocol_template,
        dataset_id=dataset_id,
        split_sha256=split_registry.sha256,
    )
    atomic_write_json(
        candidate / "experiment_protocol.json", bound.model_dump(mode="json")
    )
    _write_checksums(candidate)
    load_frozen_dataset(
        candidate,
        run_result.feature_registry,
        expected_dataset_id=dataset_id,
        expected_split_sha256=split_registry.sha256,
    )

    destination = run_result.artifact_root / dataset_id
    run_result.artifact_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_existing_freeze(
            destination,
            dataset_id,
            split_registry.sha256,
            run_result.feature_registry,
        )
        shutil.rmtree(candidate)
        existing_protocol = BoundProtocol.model_validate_json(
            (destination / "experiment_protocol.json").read_text(encoding="utf-8")
        )
        return FrozenDatasetResult(
            dataset_id=dataset_id,
            path=destination,
            protocol=existing_protocol,
            split_registry=split_registry,
            reused=True,
        )

    _promote_candidate(candidate, destination)
    return FrozenDatasetResult(
        dataset_id=dataset_id,
        path=destination,
        protocol=bound,
        split_registry=split_registry,
        reused=False,
    )


def _coerce_config(config: DatasetPipelineConfig | Mapping[str, object]) -> DatasetPipelineConfig:
    if isinstance(config, DatasetPipelineConfig):
        return config
    if isinstance(config, Mapping):
        return DatasetPipelineConfig(**dict(config))  # type: ignore[arg-type]
    raise TypeError("config must be DatasetPipelineConfig or a mapping")


def _trusted_source_identity(
    config: DatasetPipelineConfig, row: SourceRow
) -> TrustedSourceIdentity:
    acquire_owner = getattr(config.adapters.acquire, "__self__", None)
    if isinstance(acquire_owner, LocalFileAcquirer):
        trusted = acquire_owner.trusted_identity(row)
        if trusted is not None:
            return trusted
        raise ValueError("local source identity does not match the configured catalog")
    return TrustedSourceIdentity(
        source_kind="youtube",
        source_row_id=row.source_row_id,
        video_id=row.video_id,
        local_basename="",
        source_sha256="",
        catalog_identity_sha256="",
    )


def _bind_acquired_source_hash(
    trusted: TrustedSourceIdentity, acquisition: DatasetStageRecord
) -> TrustedSourceIdentity:
    if trusted.get("source_kind") == "local_file" or acquisition.status != _SUCCESS:
        return trusted
    source_sha256 = acquisition.payload.get("source_sha256")
    if not isinstance(source_sha256, str):
        return trusted
    return TrustedSourceIdentity(
        source_kind=trusted.source_kind,
        source_row_id=trusted.source_row_id,
        video_id=trusted.video_id,
        local_basename=trusted.local_basename,
        source_sha256=source_sha256,
        catalog_identity_sha256=trusted.catalog_identity_sha256,
    )


def _run_individual_stage(
    *,
    config: DatasetPipelineConfig,
    stage: str,
    sample_id: str,
    input_sha256: str,
    metadata_path: Path,
    output_root: Path,
    operation: Callable[[], object],
    expected_source_identity: TrustedSourceIdentity | None = None,
) -> DatasetStageRecord:
    config_sha256 = _stage_config_sha256(config, stage)
    implementation_sha256 = config.implementation_digests[stage]
    if stage not in config.force_stages:
        cached = _load_cached_stage(
            metadata_path,
            stage,
            sample_id,
            input_sha256,
            config_sha256,
            implementation_sha256,
            config.repository_root,
            allow_legacy=(
                config.allow_legacy_stage_cache
                and stage in config.legacy_compatible_stages
            ),
            legacy_config_sha256s=_legacy_stage_config_sha256s(config, stage),
            expected_source_identity=expected_source_identity,
        )
        if cached is not None:
            return cached
    try:
        result = _coerce_stage_result(
            operation(),
            stage=stage,
            sample_id=sample_id,
            output_root=output_root,
            repository_root=config.repository_root,
        )
    except Exception:
        result = DatasetStageResult(
            sample_id=sample_id,
            status="failed",
            error_code=f"{stage}_error",
            error_summary=f"{stage.capitalize()} stage failed",
        )
    record = DatasetStageRecord(
        stage=stage,
        sample_id=sample_id,
        status=result.status,
        input_sha256=input_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        payload=result.payload,
        error_code=result.error_code,
        error_summary=result.error_summary,
    )
    record = _validated_or_failed(
        record,
        config.repository_root,
        expected_source_identity=expected_source_identity,
    )
    atomic_write_json(metadata_path, record.to_dict())
    return record


def _run_asr_batch(
    config: DatasetPipelineConfig, states: Sequence[_SampleState], metadata_root: Path
) -> None:
    misses: list[tuple[_SampleState, str, str, Path]] = []
    items: list[WhisperTranscriptionItem] = []
    for state in states:
        metadata_path = metadata_root / state.row.sample_id / "asr.json"
        input_sha256 = (
            _record_sha256(state.audio)
            if state.audio.status != _SUCCESS
            else str(state.audio.payload.get("analysis_sha256", ""))
        )
        if state.audio.status != _SUCCESS:
            state.asr = _record_skipped(
                config,
                "asr",
                state.row.sample_id,
                input_sha256,
                metadata_path,
                "audio_failed",
            )
            continue
        config_sha256 = _stage_config_sha256(config, "asr")
        cached = None
        if "asr" not in config.force_stages:
            cached = _load_cached_stage(
                metadata_path,
                "asr",
                state.row.sample_id,
                input_sha256,
                config_sha256,
                config.implementation_digests["asr"],
                config.repository_root,
                allow_legacy=(
                    config.allow_legacy_stage_cache
                    and "asr" in config.legacy_compatible_stages
                    and state.legacy_analysis_sha256 == input_sha256
                ),
                legacy_config_sha256s=_legacy_stage_config_sha256s(config, "asr"),
            )
        if cached is not None:
            state.asr = cached
            continue
        analysis_path = _required_payload_path(config.repository_root, state.audio, "analysis_path")
        misses.append((state, input_sha256, config_sha256, metadata_path))
        items.append(WhisperTranscriptionItem(state.row.sample_id, analysis_path))

    if not misses:
        return
    pending = {
        state.row.sample_id: (state, input_sha256, config_sha256, metadata_path)
        for state, input_sha256, config_sha256, metadata_path in misses
    }
    status_counts: Counter[str] = Counter()
    completed = 0
    expected_sample_ids = [state.row.sample_id for state, *_ in misses]
    raw_results = config.adapters.transcribe_many(tuple(items))
    for sample_id, raw in _iter_batch_results(raw_results, expected_sample_ids):
        target = pending.pop(sample_id, None)
        if target is None:
            continue
        state, input_sha256, config_sha256, metadata_path = target
        state.asr = _persist_asr_result(
            config,
            state,
            input_sha256,
            config_sha256,
            metadata_path,
            raw,
        )
        completed += 1
        status_counts[state.asr.status] += 1
        _emit_progress(config, completed, len(misses), status_counts)
    for state, input_sha256, config_sha256, metadata_path in pending.values():
        state.asr = _persist_asr_result(
            config,
            state,
            input_sha256,
            config_sha256,
            metadata_path,
            None,
        )
        completed += 1
        status_counts[state.asr.status] += 1
        _emit_progress(config, completed, len(misses), status_counts)


def _persist_asr_result(
    config: DatasetPipelineConfig,
    state: _SampleState,
    input_sha256: str,
    config_sha256: str,
    metadata_path: Path,
    raw: object | None,
) -> DatasetStageRecord:
    sample_id = state.row.sample_id
    if raw is None:
        result = DatasetStageResult(
            sample_id=sample_id,
            status="failed",
            error_code="asr_error",
            error_summary="ASR batch did not return a terminal sample result",
        )
    else:
        try:
            result = _coerce_stage_result(
                raw,
                stage="asr",
                sample_id=sample_id,
                output_root=metadata_path.parent,
                repository_root=config.repository_root,
            )
        except Exception:
            result = DatasetStageResult(
                sample_id=sample_id,
                status="failed",
                error_code="asr_error",
                error_summary="ASR result could not be validated",
            )
    record = DatasetStageRecord(
        stage="asr",
        sample_id=sample_id,
        status=result.status,
        input_sha256=input_sha256,
        config_sha256=config_sha256,
        implementation_sha256=config.implementation_digests["asr"],
        payload=result.payload,
        error_code=result.error_code,
        error_summary=result.error_summary,
    )
    record = _validated_or_failed(record, config.repository_root)
    atomic_write_json(metadata_path, record.to_dict())
    return record


def _emit_progress(
    config: DatasetPipelineConfig,
    completed: int,
    total: int,
    status_counts: Counter[str],
) -> None:
    callback = config.progress_callback
    if callback is None:
        return
    event: dict[str, object] = {
        "stage": "asr",
        "completed": completed,
        "total": total,
        "status_counts": dict(sorted(status_counts.items())),
    }
    try:
        callback(event)
    except Exception:
        return


def _record_skipped(
    config: DatasetPipelineConfig,
    stage: str,
    sample_id: str,
    input_sha256: str,
    metadata_path: Path,
    reason: str,
) -> DatasetStageRecord:
    config_sha256 = _stage_config_sha256(config, stage)
    implementation_sha256 = config.implementation_digests[stage]
    if stage not in config.force_stages:
        cached = _load_cached_stage(
            metadata_path,
            stage,
            sample_id,
            input_sha256,
            config_sha256,
            implementation_sha256,
            config.repository_root,
            allow_legacy=(
                config.allow_legacy_stage_cache
                and stage in config.legacy_compatible_stages
            ),
            legacy_config_sha256s=_legacy_stage_config_sha256s(config, stage),
        )
        if cached is not None:
            return cached
    record = DatasetStageRecord(
        stage=stage,
        sample_id=sample_id,
        status="skipped",
        input_sha256=input_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        payload={},
        error_code="upstream_failure",
        error_summary=f"Stage skipped because {reason}",
    )
    atomic_write_json(metadata_path, record.to_dict())
    return record


def _load_cached_stage(
    path: Path,
    stage: str,
    sample_id: str,
    input_sha256: str,
    config_sha256: str,
    implementation_sha256: str,
    repository_root: Path,
    *,
    allow_legacy: bool = False,
    legacy_config_sha256s: Sequence[str] = (),
    expected_source_identity: TrustedSourceIdentity | None = None,
) -> DatasetStageRecord | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        schema_version = raw.get("schema_version") if isinstance(raw, dict) else None
        legacy = schema_version == 1 and allow_legacy
        accepted_config_hashes = {config_sha256}
        accepted_config_hashes.update(legacy_config_sha256s)
        if (
            not isinstance(raw, dict)
            or schema_version not in ({1, 2} if allow_legacy else {2})
            or raw.get("terminal") is not True
            or raw.get("stage") != stage
            or raw.get("sample_id") != sample_id
            or (
                raw.get("input_sha256") != input_sha256
                and not (legacy and stage != "acquisition")
            )
            or raw.get("config_sha256") not in accepted_config_hashes
            or (
                not legacy
                and raw.get("implementation_sha256") != implementation_sha256
            )
            or not isinstance(raw.get("payload"), dict)
        ):
            return None
        record = DatasetStageRecord(
            stage=stage,
            sample_id=sample_id,
            status=str(raw["status"]),
            input_sha256=input_sha256,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            payload=raw["payload"],
            error_code=_optional_string(raw.get("error_code")),
            error_summary=_optional_string(raw.get("error_summary")),
            resumed=True,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not _stage_record_valid(
        record,
        repository_root,
        expected_source_identity=expected_source_identity,
    ):
        return None
    if legacy:
        atomic_write_json(path, record.to_dict())
    return record


def _validated_or_failed(
    record: DatasetStageRecord,
    repository_root: Path,
    *,
    expected_source_identity: TrustedSourceIdentity | None = None,
) -> DatasetStageRecord:
    if _stage_record_valid(
        record,
        repository_root,
        expected_source_identity=expected_source_identity,
    ):
        return record
    return DatasetStageRecord(
        stage=record.stage,
        sample_id=record.sample_id,
        status="failed",
        input_sha256=record.input_sha256,
        config_sha256=record.config_sha256,
        implementation_sha256=record.implementation_sha256,
        payload={},
        error_code=f"invalid_{record.stage}_result",
        error_summary=f"{record.stage.capitalize()} stage returned an invalid terminal result",
    )


def _stage_record_valid(
    record: DatasetStageRecord,
    repository_root: Path,
    *,
    expected_source_identity: TrustedSourceIdentity | None = None,
) -> bool:
    if record.status not in {_SUCCESS, "failed", "skipped"}:
        return False
    if record.status in {"failed", "skipped"}:
        return _valid_terminal_error(record)
    if record.error_code is not None or record.error_summary is not None:
        return False
    required = {
        "acquisition": ("source_sha256",),
        "audio": (
            "analysis_path",
            "analysis_sha256",
            "source_channel_count",
            "analysis_channel_count",
        ),
        "asr": ("transcription", "provenance"),
        "features": ("features", "missing_reasons", "qc_warnings"),
    }
    if any(key not in record.payload for key in required[record.stage]):
        return False
    if record.stage == "acquisition":
        source_kind = record.payload.get("source_kind")
        local_expected = (
            expected_source_identity is not None
            and expected_source_identity.source_kind == "local_file"
        )
        if local_expected:
            duration = record.payload.get("duration_seconds")
            source_sha256 = record.payload.get("source_sha256")
            if (
                source_kind != "local_file"
                or
                not local_source_reference_valid(
                    record.payload,
                    expected_local_row_id=expected_source_identity.source_row_id,
                    expected_basename=expected_source_identity.local_basename,
                    expected_source_sha256=expected_source_identity.source_sha256,
                )
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or float(duration) < 0
                or not isinstance(source_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
            ):
                return False
        elif source_kind == "local_file" or not isinstance(
            record.payload.get("source_path"), str
        ):
            return False
    for key, value in record.payload.items():
        if not key.endswith("_path") or value is None:
            continue
        try:
            path = _absolute_relative_path(repository_root, str(value))
        except ValueError:
            return False
        if not path.is_file():
            return False
        expected = record.payload.get(f"{key[:-5]}_sha256")
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or sha256_file(path) != expected
        ):
            return False
    if record.stage == "audio":
        return (
            type(record.payload.get("source_channel_count")) is int
            and int(record.payload["source_channel_count"]) >= 1
            and record.payload.get("analysis_channel_count") == 1
        )
    if record.stage == "asr":
        return _valid_transcription_payload(record.payload)
    if record.stage == "features":
        return _valid_feature_payload(record.payload)
    return True


def _valid_terminal_error(record: DatasetStageRecord) -> bool:
    return (
        isinstance(record.error_code, str)
        and _SAFE_ERROR_CODE.fullmatch(record.error_code) is not None
        and isinstance(record.error_summary, str)
        and 0 < len(record.error_summary) <= _SUMMARY_LIMIT
        and _URL.search(record.error_summary) is None
        and _WINDOWS_PATH.search(record.error_summary) is None
    )


def _valid_transcription_payload(payload: Mapping[str, object]) -> bool:
    transcription = payload.get("transcription")
    provenance = payload.get("provenance")
    if (
        not isinstance(transcription, dict)
        or not isinstance(provenance, dict)
        or not isinstance(transcription.get("text"), str)
        or not isinstance(transcription.get("segments"), list)
    ):
        return False
    language = transcription.get("language")
    if language is not None and not isinstance(language, str):
        return False
    try:
        transcription_from_dict(transcription)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _valid_feature_payload(payload: Mapping[str, object]) -> bool:
    features = payload.get("features")
    reasons = payload.get("missing_reasons")
    warnings = payload.get("qc_warnings")
    if not isinstance(features, dict) or not isinstance(reasons, dict) or not isinstance(warnings, list):
        return False
    return (
        all(isinstance(name, str) for name in features)
        and all(isinstance(name, str) and isinstance(reason, str) for name, reason in reasons.items())
        and all(isinstance(warning, str) for warning in warnings)
    )


def _coerce_stage_result(
    raw: object,
    *,
    stage: str,
    sample_id: str,
    output_root: Path,
    repository_root: Path,
) -> DatasetStageResult:
    if isinstance(raw, DatasetStageResult):
        status = raw.status
        raw_sample_id = raw.sample_id
        payload: object = raw.payload
        error_code = raw.error_code
        error_summary = raw.error_summary
    elif isinstance(raw, AcquisitionResult):
        status = raw.status.value
        raw_sample_id = raw.sample_id
        payload = {
            "source_sha256": raw.source_sha256,
            "attempts": raw.attempts,
            "source_kind": raw.source_kind,
        }
        if raw.source_path is not None:
            payload["source_path"] = raw.source_path
        if raw.metadata_path is not None:
            payload["metadata_path"] = raw.metadata_path
        if raw.source_locator is not None:
            payload["source_locator"] = raw.source_locator
        if raw.local_basename is not None:
            payload["local_basename"] = raw.local_basename
        if raw.duration_seconds is not None:
            payload["duration_seconds"] = raw.duration_seconds
        error_code = raw.error_code
        error_summary = raw.error_summary
    elif isinstance(raw, AudioArtifacts):
        status = raw.analysis.status.value
        raw_sample_id = sample_id
        payload = _audio_artifact_payload(raw)
        error_code = raw.analysis.error_code
        error_summary = raw.analysis.error_summary
    elif isinstance(raw, WhisperTranscriptionResult):
        status = raw.status.value
        raw_sample_id = raw.sample_id
        payload = {
            "transcription": None if raw.transcription is None else asdict(raw.transcription),
            "provenance": asdict(raw.provenance),
        }
        error_code = raw.error_code
        error_summary = raw.error_summary
    elif isinstance(raw, FeatureBundle):
        status = _SUCCESS
        raw_sample_id = sample_id
        payload = _feature_bundle_payload(raw)
        error_code = None
        error_summary = None
    elif isinstance(raw, Mapping):
        status = str(raw.get("status", "failed"))
        raw_sample_id = _optional_string(raw.get("sample_id")) or sample_id
        payload = raw.get("payload", {})
        error_code = _optional_string(raw.get("error_code"))
        error_summary = _optional_string(raw.get("error_summary"))
    else:
        raise TypeError("stage adapters must return a supported terminal result")
    if raw_sample_id not in (None, sample_id):
        raise ValueError("stage result sample_id mismatch")
    normalized_status = _normalise_status(stage, status)
    safe_payload = _safe_metadata_value(payload, output_root, repository_root)
    if not isinstance(safe_payload, dict):
        raise ValueError("stage payload must be a mapping")
    safe_payload = _add_file_hashes(safe_payload, repository_root)
    safe_code, safe_summary = _safe_error(normalized_status, error_code, error_summary, stage)
    return DatasetStageResult(
        sample_id=sample_id,
        status=normalized_status,
        payload=safe_payload,
        error_code=safe_code,
        error_summary=safe_summary,
    )


def _normalise_status(stage: str, status: object) -> str:
    value = str(status.value if isinstance(status, Enum) else status).lower()
    successes = {
        "acquisition": {"succeeded", "acquired", "cached"},
        "audio": {"succeeded", "created"},
        "asr": {"succeeded"},
        "features": {"succeeded", "created"},
    }
    if value in successes[stage]:
        return _SUCCESS
    if value == "skipped":
        return "skipped"
    return "failed"


def _safe_error(
    status: str,
    error_code: str | None,
    error_summary: str | None,
    stage: str,
) -> tuple[str | None, str | None]:
    if status == _SUCCESS:
        return None, None
    code = str(error_code or f"{stage}_error").lower().strip()
    if _SAFE_ERROR_CODE.fullmatch(code) is None:
        code = f"{stage}_error"
    summary = " ".join(str(error_summary or f"{stage.capitalize()} stage failed").split())
    summary = _URL.sub("[redacted-url]", summary)
    summary = _WINDOWS_PATH.sub("[redacted-path]", summary)
    return code, summary[:_SUMMARY_LIMIT]


def _safe_metadata_value(value: object, base: Path, repository_root: Path) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _safe_metadata_value(value.value, base, repository_root)
    if isinstance(value, Path):
        return _relative_artifact_path(value, base, repository_root)
    if isinstance(value, str):
        return _WINDOWS_PATH.sub("[redacted-path]", _URL.sub("[redacted-url]", value))
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.endswith("_path") or key.endswith("_dir"):
                safe[key] = None if item is None else _relative_artifact_path(item, base, repository_root)
            else:
                safe[key] = _safe_metadata_value(item, base, repository_root)
        return safe
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not supported in artifact metadata")
    if isinstance(value, (list, tuple)):
        return [_safe_metadata_value(item, base, repository_root) for item in value]
    if is_dataclass(value):
        return _safe_metadata_value(asdict(value), base, repository_root)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _safe_metadata_value(item_method(), base, repository_root)
    return str(value)


def _relative_artifact_path(value: object, base: Path, repository_root: Path) -> str:
    path = Path(str(value))
    candidate = path if path.is_absolute() else base / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("artifact paths must remain repository-relative") from error
    return relative.as_posix()


def _add_file_hashes(payload: dict[str, object], repository_root: Path) -> dict[str, object]:
    result = dict(payload)
    for key, value in payload.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        try:
            path = _absolute_relative_path(repository_root, value)
        except ValueError:
            continue
        if path.is_file():
            result[f"{key[:-5]}_sha256"] = sha256_file(path)
    return result


def _iter_batch_results(
    raw: object, expected_sample_ids: Sequence[str]
) -> Iterator[tuple[str, object]]:
    if isinstance(raw, Mapping):
        for sample_id, result in raw.items():
            yield str(sample_id), result
        return
    if not isinstance(raw, Sequence):
        raw = iter(raw)  # type: ignore[arg-type]
    for index, item in enumerate(raw):  # type: ignore[arg-type]
        sample_id = getattr(item, "sample_id", None)
        if sample_id is None and isinstance(item, Mapping):
            sample_id = item.get("sample_id")
        if sample_id is None and index < len(expected_sample_ids):
            sample_id = expected_sample_ids[index]
        if sample_id is not None:
            yield str(sample_id), item


def _audio_artifact_payload(raw: AudioArtifacts) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_duration_seconds": raw.source_duration_seconds,
        "source_channel_count": raw.source_channel_count,
        "analysis_channel_count": raw.analysis_channel_count,
    }
    for representation in (raw.analysis, raw.peak_normalized, raw.loudness_normalized):
        payload[f"{representation.name}_status"] = representation.status.value
        payload[f"{representation.name}_path"] = representation.path
        payload[f"{representation.name}_error_code"] = representation.error_code
    return payload


def _feature_bundle_payload(bundle: FeatureBundle) -> dict[str, object]:
    return {
        "features": dict(bundle.values),
        "missing_reasons": dict(bundle.missing_reasons),
        "candidate_transients": [asdict(item) for item in bundle.candidate_transients],
        "qc_warnings": list(bundle.qc_warnings),
    }


def _extract_feature_stage(
    sample_id: str,
    audio_paths: dict[str, Path],
    transcript: TranscriptionResult | None,
    registry: FeatureRegistry,
) -> DatasetStageResult:
    bundles: dict[str, FeatureBundle] = {}
    for name in ("analysis", "peak_normalized", "loudness_normalized"):
        path = audio_paths.get(name)
        if path is None:
            continue
        decoded = read_pcm16_wav(path)
        if decoded.channel_count != 1:
            raise ValueError("frozen audio representations must be mono")
        vad = run_webrtc_vad(decoded.samples.reshape(-1), decoded.sample_rate_hz)
        bundles[name] = extract_acoustic_features(path, transcript, vad, registry)
    if "analysis" not in bundles:
        raise ValueError("analysis representation is required")
    bundle = bundles["analysis"]
    if {"peak_normalized", "loudness_normalized"} <= bundles.keys():
        bundle = compare_normalization_stability(
            bundle, bundles["peak_normalized"], bundles["loudness_normalized"]
        )
    source_channel_count = getattr(audio_paths, "source_channel_count", None)
    if type(source_channel_count) is int and source_channel_count >= 1:
        values = dict(bundle.values)
        values["channel_count"] = source_channel_count
        bundle = FeatureBundle(
            values=values,
            missing_reasons=bundle.missing_reasons,
            candidate_transients=bundle.candidate_transients,
            qc_warnings=bundle.qc_warnings,
        )
    return DatasetStageResult(
        sample_id=sample_id,
        status=_SUCCESS,
        payload=_feature_bundle_payload(bundle),
    )


def _stage_config_sha256(config: DatasetPipelineConfig, stage: str) -> str:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "parameters": _canonical_value(config.stage_parameters.get(stage, {})),
    }
    acquisition_parameters = config.stage_parameters.get("acquisition", {})
    if isinstance(acquisition_parameters, Mapping):
        local_identity = acquisition_parameters.get("local_source_identity_sha256")
        if local_identity is not None:
            payload["local_source_identity_sha256"] = _canonical_value(local_identity)
    if stage == "asr":
        payload["whisper_fingerprint"] = _canonical_value(config.whisper_fingerprint)
    elif stage == "features":
        payload["feature_registry"] = config.feature_registry.model_dump(mode="json")
    return sha256_bytes(canonical_json_bytes(payload))


def _legacy_stage_config_sha256s(
    config: DatasetPipelineConfig, stage: str
) -> tuple[str, ...]:
    if stage == "acquisition":
        base = {
            "schema_version": 1,
            "stage": stage,
            "parameters": _canonical_value(config.stage_parameters.get(stage, {})),
        }
        return tuple(
            sha256_bytes(canonical_json_bytes({**base, "allow_network": value}))
            for value in (False, True)
        )
    legacy_revisions = {"audio": 3, "asr": 2}
    if stage not in legacy_revisions:
        return (_stage_config_sha256(config, stage),)
    parameters = dict(config.stage_parameters.get(stage, {}))
    parameters["implementation_revision"] = legacy_revisions[stage]
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "parameters": _canonical_value(parameters),
    }
    if stage == "asr":
        payload["whisper_fingerprint"] = _canonical_value(config.whisper_fingerprint)
    return (sha256_bytes(canonical_json_bytes(payload)),)


def _legacy_cached_analysis_sha256(path: Path) -> str | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = raw.get("payload", {}).get("analysis_sha256")
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if raw.get("schema_version") != 1 or raw.get("status") != _SUCCESS:
        return None
    return str(value) if re.fullmatch(r"[0-9a-f]{64}", str(value)) else None


def _record_sha256(record: DatasetStageRecord) -> str:
    return sha256_bytes(canonical_json_bytes(record.to_dict()))


def _source_row_sha256(row: SourceRow) -> str:
    return sha256_bytes(canonical_json_bytes(_source_row_identity(row)))


def _source_row_identity(row: SourceRow) -> dict[str, object]:
    return _canonical_value(asdict(row))  # type: ignore[return-value]


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not supported in canonical configuration")
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    return str(value)


def _required_payload_path(
    repository_root: Path, record: DatasetStageRecord, key: str
) -> Path:
    value = record.payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"successful {record.stage} result is missing {key}")
    path = _absolute_relative_path(repository_root, value)
    if not path.is_file():
        raise ValueError(f"successful {record.stage} artifact is unavailable")
    return path


def _acquired_source_input(
    repository_root: Path, record: DatasetStageRecord
) -> Path | LocalSourceReference:
    if record.payload.get("source_kind") == "local_file":
        locator = record.payload.get("source_locator")
        source_sha256 = record.payload.get("source_sha256")
        if not isinstance(locator, str) or not isinstance(source_sha256, str):
            raise ValueError("successful local acquisition is missing its opaque source reference")
        return LocalSourceReference(locator=locator, source_sha256=source_sha256)
    return _required_payload_path(repository_root, record, "source_path")


def _absolute_relative_path(repository_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact metadata path must be repository-relative")
    path = (repository_root / relative).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("artifact metadata path leaves repository") from error
    return path


def _audio_paths(repository_root: Path, record: DatasetStageRecord) -> dict[str, Path]:
    paths: _AudioFeatureInputs = _AudioFeatureInputs(
        source_channel_count=(
            record.payload.get("source_channel_count")
            if type(record.payload.get("source_channel_count")) is int
            else None
        ),
        analysis_channel_count=(
            record.payload.get("analysis_channel_count")
            if type(record.payload.get("analysis_channel_count")) is int
            else None
        ),
    )
    for name in ("analysis", "peak_normalized", "loudness_normalized"):
        value = record.payload.get(f"{name}_path")
        if isinstance(value, str):
            path = _absolute_relative_path(repository_root, value)
            if path.is_file():
                paths[name] = path
    if "analysis" not in paths:
        raise ValueError("successful audio stage has no analysis artifact")
    return paths


def _transcription_from_record(record: DatasetStageRecord) -> TranscriptionResult | None:
    if record.status != _SUCCESS:
        return None
    raw = record.payload.get("transcription")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("successful ASR result has invalid transcription payload")
    return transcription_from_dict(raw)


def _finalize_sample(state: _SampleState) -> DatasetSampleResult:
    assert state.asr is not None and state.features is not None
    if state.acquisition.status != _SUCCESS:
        terminal = "acquisition_failed"
    elif state.audio.status != _SUCCESS:
        terminal = "audio_failed"
    elif state.features.status != _SUCCESS:
        terminal = "feature_failed"
    elif state.asr.status != _SUCCESS:
        terminal = "asr_failed"
    else:
        terminal = "succeeded"
    return DatasetSampleResult(
        source_row=state.row,
        acquisition=state.acquisition,
        audio=state.audio,
        asr=state.asr,
        features=state.features,
        trusted_source_identity=state.trusted_source_identity,
        terminal_status=terminal,
    )


def _write_manifest(
    path: Path, samples: Sequence[DatasetSampleResult], registry: SplitRegistry
) -> None:
    assignments = {assignment.sample_id: assignment for assignment in registry.assignments}
    fieldnames = [
        "source_row_id",
        "sample_id",
        "legacy_sample_id",
        "source_page",
        "video_id",
        "video_title",
        "source_course",
        "total_score",
        "score_status",
        "verification_status",
        "verification_note",
        "source_kind",
        "local_basename",
        "catalog_identity_sha256",
        "source_duration_seconds",
        "source_sha256",
        "terminal_status",
        "acquisition_status",
        "acquisition_error_code",
        "acquisition_error_summary",
        "audio_status",
        "audio_error_code",
        "audio_error_summary",
        "asr_status",
        "asr_error_code",
        "asr_error_summary",
        "feature_status",
        "feature_error_code",
        "feature_error_summary",
        "speaker_id",
        "talk_id",
        "group_id",
        "outer_fold",
        "inner_fold",
        "modelling_scope",
        "supervised_eligible",
        "qc_eligible",
        "clustering_eligible",
    ]
    rows: list[dict[str, object]] = []
    for sample in samples:
        assignment = assignments[sample.sample_id]
        rows.append(
            {
                "source_row_id": sample.source_row.source_row_id,
                "sample_id": sample.sample_id,
                "legacy_sample_id": sample.source_row.legacy_sample_id,
                "source_page": sample.source_row.source_page,
                "video_id": sample.video_id,
                "video_title": sample.source_row.video_title,
                "source_course": sample.source_row.source_course,
                "total_score": "" if sample.source_row.total_score is None else sample.source_row.total_score,
                "score_status": sample.source_row.score_status.value,
                "verification_status": sample.source_row.verification_status.value,
                "verification_note": sample.source_row.verification_note,
                "source_kind": sample.trusted_source_identity.get("source_kind", ""),
                "local_basename": sample.trusted_source_identity.get(
                    "local_basename", ""
                ),
                "catalog_identity_sha256": sample.trusted_source_identity.get(
                    "catalog_identity_sha256", ""
                ),
                "source_duration_seconds": sample.acquisition.payload.get(
                    "duration_seconds", ""
                ),
                "source_sha256": sample.trusted_source_identity.get(
                    "source_sha256", ""
                ),
                "terminal_status": sample.terminal_status,
                "acquisition_status": sample.acquisition.status,
                "acquisition_error_code": sample.acquisition.error_code or "",
                "acquisition_error_summary": sample.acquisition.error_summary or "",
                "audio_status": sample.audio.status,
                "audio_error_code": sample.audio.error_code or "",
                "audio_error_summary": sample.audio.error_summary or "",
                "asr_status": sample.asr.status,
                "asr_error_code": sample.asr.error_code or "",
                "asr_error_summary": sample.asr.error_summary or "",
                "feature_status": sample.features.status,
                "feature_error_code": sample.features.error_code or "",
                "feature_error_summary": sample.features.error_summary or "",
                "speaker_id": assignment.speaker_id,
                "talk_id": assignment.talk_id,
                "group_id": assignment.group_id,
                "outer_fold": assignment.outer_fold,
                "inner_fold": assignment.inner_fold,
                "modelling_scope": assignment.modelling_scope,
                "supervised_eligible": assignment.supervised_eligible,
                "qc_eligible": assignment.qc_eligible,
                "clustering_eligible": assignment.clustering_eligible,
            }
        )
    _write_csv(path, fieldnames, rows)


def _write_transcripts(path: Path, samples: Sequence[DatasetSampleResult]) -> None:
    lines = []
    for sample in samples:
        lines.append(
            canonical_json_bytes(
                {
                    "sample_id": sample.sample_id,
                    "status": sample.asr.status,
                    "transcription": sample.asr.payload.get("transcription"),
                    "provenance": sample.asr.payload.get("provenance"),
                    "error_code": sample.asr.error_code,
                    "error_summary": sample.asr.error_summary,
                }
            ).decode("utf-8")
        )
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def _write_features(
    path: Path,
    samples: Sequence[DatasetSampleResult],
    registry: FeatureRegistry,
) -> None:
    import pandas as pd

    names = [definition.name for definition in registry.features]
    rows = [
        {"sample_id": sample.sample_id, **{name: _feature_values(sample).get(name) for name in names}}
        for sample in samples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["sample_id", *names]).to_parquet(path, index=False)


def _write_quality(
    path: Path,
    samples: Sequence[DatasetSampleResult],
    registry: FeatureRegistry,
) -> None:
    fieldnames = [
        "sample_id",
        "terminal_status",
        "acquisition_error_code",
        "audio_error_code",
        "asr_error_code",
        "feature_error_code",
        "qc_warnings",
        "missing_feature_count",
        "missing_reasons_json",
    ]
    rows = []
    for sample in samples:
        warnings = sample.features.payload.get("qc_warnings", [])
        raw_reasons = sample.features.payload.get("missing_reasons", {})
        reasons = dict(raw_reasons) if isinstance(raw_reasons, Mapping) else {}
        if sample.features.status != _SUCCESS:
            reasons = {
                definition.name: "feature_stage_failed"
                for definition in registry.features
            }
        rows.append(
            {
                "sample_id": sample.sample_id,
                "terminal_status": sample.terminal_status,
                "acquisition_error_code": sample.acquisition.error_code or "",
                "audio_error_code": sample.audio.error_code or "",
                "asr_error_code": sample.asr.error_code or "",
                "feature_error_code": sample.features.error_code or "",
                "qc_warnings": ";".join(str(item) for item in warnings)
                if isinstance(warnings, list)
                else "",
                "missing_feature_count": len(reasons),
                "missing_reasons_json": canonical_json_bytes(
                    dict(sorted(reasons.items()))
                ).decode("utf-8"),
            }
        )
    _write_csv(path, fieldnames, rows)


def _write_csv(path: Path, fieldnames: list[str], rows: Sequence[Mapping[str, object]]) -> None:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, handle.getvalue())


def _feature_values(sample: DatasetSampleResult) -> dict[str, object]:
    values = sample.features.payload.get("features", {})
    return dict(values) if isinstance(values, Mapping) else {}


def _source_snapshot_payload(snapshot: SourceSnapshot) -> dict[str, object]:
    return {
        "git_commit": snapshot.git_commit,
        "git_dirty": snapshot.git_dirty,
        "tree_sha256": snapshot.tree_sha256,
        "files": snapshot.files,
        "git_diff_sha256": snapshot.git_diff_sha256,
        "untracked_files": snapshot.untracked_files,
    }


def _write_checksums(root: Path) -> None:
    lines = [f"{sha256_file(root / filename)}  {filename}" for filename in _CHECKSUM_FILENAMES]
    atomic_write_text(root / "checksums.sha256", "\n".join(lines) + "\n")


def _verify_existing_freeze(
    root: Path,
    dataset_id: str,
    split_sha256: str,
    registry: FeatureRegistry,
) -> None:
    try:
        load_frozen_dataset(
            root,
            registry,
            expected_dataset_id=dataset_id,
            expected_split_sha256=split_sha256,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as error:
        raise FileExistsError("immutable dataset path exists with different content") from error


def load_strict_frozen_dataset(
    root: Path,
    registry: FeatureRegistry,
    *,
    expected_dataset_id: str | None = None,
    expected_split_sha256: str | None = None,
    expected_source_identities: Mapping[
        str, TrustedSourceIdentity | Mapping[str, str]
    ]
    | None = None,
) -> LoadedFrozenDataset:
    """Load one frozen dataset only after every identity and table contract passes."""
    root = Path(root).resolve(strict=True)
    dataset_id = expected_dataset_id or root.name
    if re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None:
        raise ValueError("frozen dataset directory must be a lowercase SHA-256 dataset ID")
    if not root.is_dir() or {path.name for path in root.iterdir()} != set(_FROZEN_FILENAMES):
        raise ValueError("frozen dataset is incomplete")
    protocol = BoundProtocol.model_validate_json(
        (root / "experiment_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.dataset_id != dataset_id or (
        expected_split_sha256 is not None
        and protocol.split_sha256 != expected_split_sha256
    ):
        raise ValueError("frozen protocol binding mismatch")

    checksum_lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != len(_CHECKSUM_FILENAMES):
        raise ValueError("invalid frozen checksum registry")
    declarations = [line.split("  ", 1) for line in checksum_lines]
    if any(
        len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None
        for parts in declarations
    ):
        raise ValueError("invalid frozen checksum registry")
    recorded = {parts[1]: parts[0] for parts in declarations}
    if set(recorded) != set(_CHECKSUM_FILENAMES):
        raise ValueError("invalid frozen checksum filenames")
    if any(sha256_file(root / filename) != recorded[filename] for filename in recorded):
        raise ValueError("frozen artifact checksum mismatch")

    splits_payload = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json_bytes(splits_payload)) != protocol.split_sha256:
        raise ValueError("split registry digest mismatch")
    assignments = splits_payload.get("assignments") if isinstance(splits_payload, dict) else None
    if not isinstance(assignments, list) or not all(isinstance(item, dict) for item in assignments):
        raise ValueError("invalid split assignments")

    manifest_rows, manifest_fields = _read_frozen_csv(root / "manifest.csv")
    if not {
        "source_row_id",
        "sample_id",
        "video_id",
        "source_kind",
        "local_basename",
        "source_sha256",
        "catalog_identity_sha256",
    } <= set(manifest_fields):
        raise ValueError("invalid frozen source identity schema")
    quality_rows, quality_fields = _read_frozen_csv(root / "quality_report.csv")
    required_quality_fields = {
        "sample_id",
        "terminal_status",
        "acquisition_error_code",
        "audio_error_code",
        "asr_error_code",
        "feature_error_code",
        "qc_warnings",
        "missing_feature_count",
        "missing_reasons_json",
    }
    if set(quality_fields) != required_quality_fields:
        raise ValueError("invalid frozen quality reason schema")
    transcript_rows = [
        json.loads(line)
        for line in (root / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(item, dict) for item in transcript_rows):
        raise ValueError("invalid frozen transcript record")
    import pandas as pd

    features = pd.read_parquet(root / "acoustic_features.parquet")
    expected_feature_columns = [
        "sample_id", *(definition.name for definition in registry.features)
    ]
    if list(features.columns) != expected_feature_columns:
        raise ValueError("frozen feature schema does not match the registry")

    id_sets = {
        "manifest": _unique_frozen_ids(manifest_rows, "manifest"),
        "transcripts": _unique_frozen_ids(transcript_rows, "transcripts"),
        "features": _unique_frozen_ids(features.to_dict("records"), "features"),
        "quality": _unique_frozen_ids(quality_rows, "quality"),
        "splits": _unique_frozen_ids(assignments, "splits"),
    }
    sample_ids = id_sets["manifest"]
    if any(ids != sample_ids for ids in id_sets.values()):
        raise ValueError("frozen tables must contain exactly equal sample_id sets")
    if not sample_ids:
        raise ValueError("frozen dataset must contain at least one sample")
    expected_identities = _expected_source_identity_map(expected_source_identities)
    if not set(expected_identities) <= sample_ids:
        raise ValueError("expected source identity sample set mismatch")

    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    quality_by_id = {row["sample_id"]: row for row in quality_rows}
    transcript_by_id = {str(row["sample_id"]): row for row in transcript_rows}
    assignment_by_id = {str(row["sample_id"]): row for row in assignments}
    for sample_id in sorted(sample_ids):
        manifest = manifest_by_id[sample_id]
        quality = quality_by_id[sample_id]
        transcript = transcript_by_id[sample_id]
        assignment = assignment_by_id[sample_id]
        expected_terminal = _terminal_status_from_manifest(manifest)
        if manifest.get("terminal_status") != expected_terminal:
            raise ValueError("manifest terminal status is inconsistent with stage statuses")
        if quality.get("terminal_status") != expected_terminal:
            raise ValueError("quality terminal status mismatch")
        if transcript.get("status") != manifest.get("asr_status"):
            raise ValueError("transcript status mismatch")
        _validate_split_assignment(manifest, assignment)

    feature_rows = {str(row["sample_id"]): row for row in features.to_dict("records")}
    for sample_id, row in feature_rows.items():
        manifest = manifest_by_id[sample_id]
        quality = quality_by_id[sample_id]
        try:
            reasons = json.loads(quality["missing_reasons_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid per-feature missing reason mapping") from error
        if not isinstance(reasons, dict) or not all(
            isinstance(name, str) and isinstance(reason, str)
            for name, reason in reasons.items()
        ):
            raise ValueError("invalid per-feature missing reason mapping")
        null_names: set[str] = set()
        for definition in registry.features:
            value = row[definition.name]
            if pd.isna(value):
                null_names.add(definition.name)
                reason = reasons.get(definition.name)
                if not isinstance(reason, str):
                    raise ValueError("null feature is missing a declared missing reason")
                if (
                    manifest.get("feature_status") == _SUCCESS
                    and reason not in definition.missing_policy.reason_codes
                ):
                    raise ValueError("feature missing reason is not declared by the registry")
                continue
            if definition.name in reasons:
                raise ValueError("non-null feature has a missing reason")
            _validate_frozen_feature_value(definition, value)
        if set(reasons) != null_names:
            raise ValueError("missing reason keys must exactly match null features")
        if int(quality["missing_feature_count"]) != len(null_names):
            raise ValueError("missing feature count does not match missing reasons")

    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    identity_inputs = provenance.get("dataset_identity_inputs")
    if not isinstance(identity_inputs, dict):
        raise ValueError("missing dataset identity inputs")
    terminal_samples = identity_inputs.get("terminal_samples")
    if not isinstance(terminal_samples, list) or not all(
        isinstance(item, dict) for item in terminal_samples
    ):
        raise ValueError("missing frozen source identity inputs")
    terminal_by_id = {
        str(item.get("sample_id")): item for item in terminal_samples
    }
    if set(terminal_by_id) != sample_ids:
        raise ValueError("frozen source identity sample set mismatch")
    for sample_id, manifest in manifest_by_id.items():
        trusted = terminal_by_id[sample_id].get("trusted_source")
        if not isinstance(trusted, dict):
            raise ValueError("missing frozen source identity")
        frozen_identity = {
            "source_kind": manifest.get("source_kind", ""),
            "source_row_id": manifest.get("source_row_id", ""),
            "video_id": manifest.get("video_id", ""),
            "local_basename": manifest.get("local_basename", ""),
            "source_sha256": manifest.get("source_sha256", ""),
            "catalog_identity_sha256": manifest.get(
                "catalog_identity_sha256", ""
            ),
        }
        if frozen_identity != trusted:
            raise ValueError("frozen source identity does not match dataset identity")
        expected_identity = expected_identities.get(sample_id)
        if expected_identity is not None and (
            frozen_identity != expected_identity.to_dict()
            or trusted != expected_identity.to_dict()
        ):
            raise ValueError("expected source identity does not match frozen dataset")
    snapshot = json.loads((root / "source_snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("git_diff_sha256") != sha256_file(root / "git_diff.patch"):
        raise ValueError("source snapshot patch digest mismatch")
    files = snapshot.get("files")
    if not isinstance(files, dict) or snapshot.get("tree_sha256") != sha256_bytes(
        canonical_json_bytes(files)
    ):
        raise ValueError("source snapshot tree digest mismatch")
    if identity_inputs.get("source_snapshot") != snapshot:
        raise ValueError("dataset identity source snapshot mismatch")
    if identity_inputs.get("protocol_id") != protocol.protocol_id:
        raise ValueError("dataset identity protocol mismatch")
    if identity_inputs.get("split_registry") != splits_payload:
        raise ValueError("dataset identity split mismatch")
    table_hashes = identity_inputs.get("logical_table_sha256")
    expected_table_hashes = {
        filename: sha256_file(root / filename)
        for filename in (
            "manifest.csv",
            "transcripts.jsonl",
            "acoustic_features.parquet",
            "quality_report.csv",
            "splits.json",
        )
    }
    if table_hashes != expected_table_hashes:
        raise ValueError("dataset identity logical table hash mismatch")
    if sha256_bytes(canonical_json_bytes(identity_inputs)) != dataset_id:
        raise ValueError("dataset identity digest mismatch")
    terminal_counts = dict(
        Counter(row["terminal_status"] for row in manifest_rows)
    )
    if provenance.get("terminal_counts") != terminal_counts:
        raise ValueError("dataset terminal counts mismatch")
    return LoadedFrozenDataset(
        path=root,
        dataset_id=dataset_id,
        protocol=protocol,
        row_count=len(manifest_rows),
        checksum_count=len(checksum_lines),
        sample_ids=frozenset(sample_ids),
    )


def load_frozen_dataset(
    root: Path,
    registry: FeatureRegistry,
    *,
    expected_dataset_id: str | None = None,
    expected_split_sha256: str | None = None,
    expected_source_identities: Mapping[
        str, TrustedSourceIdentity | Mapping[str, str]
    ]
    | None = None,
) -> LoadedFrozenDataset:
    """Compatibility name for the shared strict frozen-dataset loader."""
    return load_strict_frozen_dataset(
        root,
        registry,
        expected_dataset_id=expected_dataset_id,
        expected_split_sha256=expected_split_sha256,
        expected_source_identities=expected_source_identities,
    )


def _expected_source_identity_map(
    expected_source_identities: Mapping[
        str, TrustedSourceIdentity | Mapping[str, str]
    ]
    | None,
) -> dict[str, TrustedSourceIdentity]:
    if expected_source_identities is None:
        return {}
    expected: dict[str, TrustedSourceIdentity] = {}
    for sample_id, identity in expected_source_identities.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("expected source identity sample ID is invalid")
        if isinstance(identity, TrustedSourceIdentity):
            expected[sample_id] = identity
            continue
        try:
            expected[sample_id] = TrustedSourceIdentity(**dict(identity))
        except (TypeError, ValueError) as error:
            raise ValueError("expected source identity is invalid") from error
    return expected


def _read_frozen_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError("invalid frozen CSV table")
    return rows, fields


def _unique_frozen_ids(rows: Sequence[Mapping[str, object]], table: str) -> frozenset[str]:
    ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{table} must contain unique sample_id values")
    return frozenset(ids)


def _terminal_status_from_manifest(row: Mapping[str, str]) -> str:
    if row.get("acquisition_status") != _SUCCESS:
        return "acquisition_failed"
    if row.get("audio_status") != _SUCCESS:
        return "audio_failed"
    if row.get("feature_status") != _SUCCESS:
        return "feature_failed"
    if row.get("asr_status") != _SUCCESS:
        return "asr_failed"
    return _SUCCESS


def _csv_bool(value: object) -> bool:
    if value not in {"True", "False"}:
        raise ValueError("frozen eligibility flags must be exact booleans")
    return value == "True"


def _validate_split_assignment(
    manifest: Mapping[str, str], assignment: Mapping[str, object]
) -> None:
    for name in (
        "group_id",
        "speaker_id",
        "talk_id",
        "video_id",
        "modelling_scope",
    ):
        if str(assignment.get(name, "")) != manifest.get(name):
            raise ValueError("manifest and split assignment mismatch")
    for name in ("outer_fold", "inner_fold"):
        if str(assignment.get(name, "")) != manifest.get(name):
            raise ValueError("manifest and split assignment mismatch")
    expected_eligible = {
        "supervised_eligible": (
            manifest.get("feature_status") == _SUCCESS
            and manifest.get("score_status") == "observed"
            and bool(manifest.get("total_score"))
        ),
        "qc_eligible": manifest.get("feature_status") == _SUCCESS,
        "clustering_eligible": manifest.get("feature_status") == _SUCCESS,
    }
    for name, expected in expected_eligible.items():
        if _csv_bool(manifest.get(name)) is not expected or assignment.get(name) is not expected:
            raise ValueError("split eligibility is inconsistent with frozen stage status")


def _validate_frozen_feature_value(definition: object, value: object) -> None:
    dtype = getattr(definition, "dtype")
    if dtype == "boolean":
        if type(value) is not bool:
            raise ValueError("frozen boolean feature has the wrong type")
        return
    if dtype == "string":
        if not isinstance(value, str):
            raise ValueError("frozen string feature has the wrong type")
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("frozen numeric feature has the wrong type") from error
    if not math.isfinite(numeric):
        raise ValueError("frozen numeric feature must be finite")
    if dtype == "integer" and not numeric.is_integer():
        raise ValueError("frozen integer feature has the wrong type")
    minimum = getattr(definition, "minimum")
    maximum = getattr(definition, "maximum")
    if minimum is not None and numeric < float(minimum) - 1e-12:
        raise ValueError("frozen feature is below its registry minimum")
    if maximum is not None and numeric > float(maximum) + 1e-12:
        raise ValueError("frozen feature exceeds its registry maximum")


def _promote_candidate(candidate: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("immutable dataset path already exists")
    os.rename(candidate, destination)


def _reset_candidate(candidate: Path, staging_dir: Path) -> None:
    resolved_candidate = candidate.resolve()
    resolved_staging = staging_dir.resolve()
    _require_descendant(resolved_candidate, resolved_staging, "freeze candidate")
    if resolved_candidate.exists():
        shutil.rmtree(resolved_candidate)


def _require_descendant(path: Path, parent: Path, name: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise ValueError(f"{name} must remain within the repository workspace") from error
    if path == parent:
        raise ValueError(f"{name} must not be the repository root")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
