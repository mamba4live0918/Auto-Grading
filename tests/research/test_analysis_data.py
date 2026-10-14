"""Synthetic tests for immutable analysis cohorts and fold-local preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

import presentation_assessment.research.analysis_data as analysis_data_module
from presentation_assessment.research.analysis_data import (
    AnalysisCohort,
    AnalysisDataset,
    RobustFoldPreprocessor,
    load_analysis_dataset,
    select_cohort,
)
from presentation_assessment.research.canonical import canonical_json_bytes, sha256_bytes
from presentation_assessment.research.dataset_pipeline import (
    DatasetPipelineConfig,
    DatasetStageAdapters,
    DatasetStageResult,
    freeze_dataset,
    run_dataset_pipeline,
)
from presentation_assessment.research.feature_registry import load_feature_registry
from presentation_assessment.research.protocol import load_protocol_template
from presentation_assessment.research.provenance import SourceSnapshot
from presentation_assessment.research.source_manifest import (
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "audio_feature_registry.v1.yaml"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_row(sample_id: str, video_id: str, score: float | None) -> SourceRow:
    return SourceRow(
        source_row_id=f"row-{sample_id}",
        source_page="synthetic-page",
        sample_id=sample_id,
        legacy_sample_id="",
        # privacy-scan: synthetic-fixture
        source_url="https" + "://" + "www.youtube.com/watch?v=" + video_id,
        video_id=video_id,
        video_title="SYNTHETIC_PRIVATE_TITLE",
        total_score=score,
        score_status=ScoreStatus.OBSERVED if score is not None else ScoreStatus.NOT_AVAILABLE,
        source_course="synthetic-course",
        verification_status=VerificationStatus.VERIFIED,
        verification_note="synthetic-only",
    )


class _SyntheticAnalysisStages:
    """Deterministic adapters covering success, failure, and explained missingness."""

    def acquire(
        self, row: SourceRow, output_root: Path, allow_network: bool
    ) -> DatasetStageResult:
        del allow_network
        if row.sample_id == "N003":
            return DatasetStageResult(
                sample_id=row.sample_id,
                status="failed",
                error_code="synthetic_unavailable",
                error_summary="synthetic acquisition failure",
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
        return [
            DatasetStageResult(
                sample_id=item.sample_id,
                status="succeeded",
                payload={
                    "transcription": {
                        "text": f"SYNTHETIC_PRIVATE_TRANSCRIPT_{item.sample_id}",
                        "language": "en",
                        "segments": [],
                    },
                    "provenance": {"model_name": "synthetic-whisper"},
                },
            )
            for item in items
        ]

    def features(
        self,
        sample_id: str,
        audio_paths: dict[str, Path],
        transcript: object,
        registry: object,
    ) -> DatasetStageResult:
        del audio_paths, transcript
        values: dict[str, object] = {}
        for definition in registry.features:
            if definition.dtype == "boolean":
                value: object = False
            elif definition.dtype == "string":
                value = "synthetic"
            elif definition.dtype == "integer":
                value = int(definition.minimum or 0)
            else:
                value = float(definition.minimum or 0.0)
            values[definition.name] = value
        reasons: dict[str, str] = {}
        if sample_id == "N004":
            values["integrated_loudness_lufs"] = None
            reasons["integrated_loudness_lufs"] = "silent_audio"
        return DatasetStageResult(
            sample_id=sample_id,
            status="succeeded",
            payload={
                "features": values,
                "missing_reasons": reasons,
                "qc_warnings": ["synthetic_warning"] if sample_id == "N004" else [],
            },
        )

    def adapters(self) -> DatasetStageAdapters:
        return DatasetStageAdapters(
            acquire=self.acquire,
            audio=self.audio,
            transcribe_many=self.transcribe_many,
            features=self.features,
        )


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


@pytest.fixture
def frozen_analysis_fixture(tmp_path: Path) -> tuple[Path, Path]:
    registry = load_feature_registry(REGISTRY_PATH)
    rows = (
        _source_row("N001", "aaaaaaaaaaa", 80.0),
        _source_row("N002", "bbbbbbbbbbb", None),
        _source_row("N003", "ccccccccccc", 40.0),
        _source_row("N004", "ddddddddddd", 60.0),
    )
    config = DatasetPipelineConfig(
        repository_root=tmp_path,
        staging_dir=tmp_path / "artifacts" / "staging" / "analysis-test",
        artifact_root=tmp_path / "artifacts" / "research",
        rows=rows,
        protocol_template=load_protocol_template(
            PROJECT_ROOT / "configs" / "experiment_protocol.template.yaml"
        ),
        feature_registry=registry,
        pipeline_parameters={"schema_version": 1, "mode": "synthetic-analysis"},
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
        adapters=_SyntheticAnalysisStages().adapters(),
        allow_network=False,
        source_snapshotter=_snapshot,
    )
    frozen = freeze_dataset(run_dataset_pipeline(config))
    return frozen.path, REGISTRY_PATH


def test_analysis_loader_calls_strict_gate_once_and_builds_immutable_safe_rows(
    frozen_analysis_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: analysis reads tables outside one strict gate or exposes private text."""
    dataset_dir, registry_path = frozen_analysis_fixture
    real_loader = analysis_data_module.dataset_pipeline.load_strict_frozen_dataset
    calls = 0

    def counted_loader(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(
        analysis_data_module.dataset_pipeline,
        "load_strict_frozen_dataset",
        counted_loader,
    )

    dataset = load_analysis_dataset(dataset_dir, registry_path)

    assert isinstance(dataset, AnalysisDataset)
    assert calls == 1
    assert dataset.sample_ids == ("N001", "N002", "N003", "N004")
    assert dataset.row_count == 4
    assert tuple(dataset.feature_frame().columns) == dataset.feature_names
    assert dataset.rows[3].missing_reasons["integrated_loudness_lufs"] == "silent_audio"
    assert dataset.rows[3].missing_feature_count == 1
    assert dataset.rows[2].acquisition_error_code == "synthetic_unavailable"
    assert dataset.rows[2].feature_error_code == "upstream_failure"
    assert dataset.rows[3].transcript_available is True
    assert tuple(row.whisper_language_code for row in dataset.rows) == (
        "en",
        "en",
        None,
        "en",
    )
    assert not hasattr(dataset.rows[0], "transcript_text")
    assert not hasattr(dataset.rows[0], "language_probability")
    public_representation = repr(dataset)
    assert "SYNTHETIC_PRIVATE_TITLE" not in public_representation
    assert "SYNTHETIC_PRIVATE_TRANSCRIPT" not in public_representation
    with pytest.raises(FrozenInstanceError):
        dataset.rows[0].sample_id = "changed"  # type: ignore[misc]
    first_projection = dataset.feature_frame()
    first_projection.iloc[0, 0] = 999.0
    assert dataset.feature_frame().iloc[0, 0] != 999.0


def test_all_four_cohorts_have_expected_members_and_explicit_exclusions(
    frozen_analysis_fixture: tuple[Path, Path],
) -> None:
    """Break caught: eligibility collapses QC, unsupervised, supervised, and ASR gates."""
    dataset = load_analysis_dataset(*frozen_analysis_fixture)

    quality = select_cohort(dataset, "quality_control")
    clustering = select_cohort(dataset, "clustering")
    supervised = select_cohort(dataset, "supervised_total_score")
    transcript = select_cohort(dataset, "transcript_model")

    assert all(
        isinstance(item, AnalysisCohort)
        for item in (quality, clustering, supervised, transcript)
    )
    assert (quality.sample_ids, quality.count) == (("N001", "N002", "N004"), 3)
    assert (clustering.sample_ids, clustering.count) == (("N001", "N002", "N004"), 3)
    assert (supervised.sample_ids, supervised.count) == (("N001", "N004"), 2)
    assert (transcript.sample_ids, transcript.count) == (("N001", "N002", "N004"), 3)
    assert quality.exclusion_reasons == {
        "N003": ("feature_status_not_succeeded", "qc_not_eligible")
    }
    assert clustering.exclusion_reasons == {
        "N003": (
            "feature_status_not_succeeded",
            "qc_not_eligible",
            "clustering_not_eligible",
        )
    }
    assert supervised.exclusion_reasons == {
        "N002": ("score_not_observed", "total_score_missing", "supervised_not_eligible"),
        "N003": ("feature_status_not_succeeded", "supervised_not_eligible"),
    }
    assert transcript.exclusion_reasons == {
        "N003": ("transcript_status_not_succeeded", "transcript_unavailable")
    }
    assert supervised.target_series().to_dict() == {"N001": 80.0, "N004": 60.0}


def test_analysis_warning_parser_redacts_stability_measurements_to_audit_codes() -> None:
    """Break caught: persisted stability measurements block the strict analysis view."""
    persisted = (
        "low_snr;"
        "stability_pause_ratio_absolute_difference=0.0215237 exceeds 0.02;"
        "stability_acoustic_envelope_peak_rate_relative_difference=0.112818 exceeds 0.1"
    )

    assert analysis_data_module._warnings(persisted) == (
        "low_snr",
        "stability_pause_ratio_absolute_difference",
        "stability_acoustic_envelope_peak_rate_relative_difference",
    )


def test_analysis_loader_rejects_private_text_in_language_code_if_gate_is_bypassed(
    frozen_analysis_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: arbitrary transcript text is retained as a language label."""
    dataset_dir, registry_path = frozen_analysis_fixture
    registry = load_feature_registry(registry_path)
    loaded = analysis_data_module.dataset_pipeline.load_strict_frozen_dataset(
        dataset_dir, registry
    )
    transcript_path = dataset_dir / "transcripts.jsonl"
    records = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["transcription"]["language"] = "SYNTHETIC_PRIVATE_TRANSCRIPT"
    transcript_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis_data_module.dataset_pipeline,
        "load_strict_frozen_dataset",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(ValueError, match="safe lowercase language code"):
        load_analysis_dataset(dataset_dir, registry_path)


@pytest.mark.parametrize("corruption", ["duplicate", "nonfinite", "unexplained_null"])
def test_analysis_loader_revalidates_features_if_strict_gate_is_bypassed(
    frozen_analysis_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Break caught: a bypassed lower gate lets corrupt feature rows enter analysis."""
    dataset_dir, registry_path = frozen_analysis_fixture
    registry = load_feature_registry(registry_path)
    loaded = analysis_data_module.dataset_pipeline.load_strict_frozen_dataset(
        dataset_dir, registry
    )
    feature_path = dataset_dir / "acoustic_features.parquet"
    frame = pd.read_parquet(feature_path)
    if corruption == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
        expected = "unique sample_id"
    elif corruption == "nonfinite":
        frame.loc[0, "speech_ratio"] = float("inf")
        expected = "finite"
    else:
        frame.loc[0, "speech_ratio"] = None
        expected = "missing reason"
    frame.to_parquet(feature_path, index=False)
    monkeypatch.setattr(
        analysis_data_module.dataset_pipeline,
        "load_strict_frozen_dataset",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(ValueError, match=expected):
        load_analysis_dataset(dataset_dir, registry_path)


def test_validation_values_do_not_change_training_statistics() -> None:
    """Break caught: validation values leak into fitted imputation or scale statistics."""
    train = pd.DataFrame({"a": [1.0, 2.0, None]})
    validation_a = pd.DataFrame({"a": [3.0]})
    validation_b = pd.DataFrame({"a": [3000.0]})

    fitted = RobustFoldPreprocessor().fit(train)
    location_before = fitted.location_
    scale_before = fitted.scale_
    first_shape = fitted.transform(validation_a).shape
    second_shape = fitted.transform(validation_b).shape

    assert fitted.location_ == location_before == {"a": 1.5}
    assert fitted.scale_ == scale_before == {"a": 0.5}
    assert first_shape == second_shape == (1, 1)


def test_preprocessor_imputes_scales_records_constants_and_keeps_indicators_audit_only() -> None:
    """Break caught: constants divide by zero or missing flags silently enter the model."""
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, None],
            "constant": [4.0, 4.0, None],
            "a__missing": [0, 0, 1],
        },
        index=["one", "two", "three"],
    )
    fitted = RobustFoldPreprocessor().fit(train)

    transformed = fitted.transform(train)
    audit = fitted.audit_missing_indicators(train)

    assert fitted.feature_names_in_ == ("a", "constant")
    assert fitted.feature_names_ == ("a", "constant")
    assert fitted.excluded_all_null_features_ == ()
    assert fitted.feature_names_out_ == ("a", "constant")
    assert fitted.location_ == {"a": 1.5, "constant": 4.0}
    assert fitted.scale_ == {"a": 0.5, "constant": 1.0}
    assert fitted.constant_features_ == ("constant",)
    assert list(transformed.columns) == ["a", "constant"]
    assert transformed.loc["three"].to_dict() == {"a": 0.0, "constant": 0.0}
    assert audit["a__missing"].tolist() == [False, False, True]


def test_protocol_can_explicitly_promote_selected_missing_indicators() -> None:
    """Break caught: explicit indicator inputs are ignored or all audit flags are promoted."""
    train = pd.DataFrame({"a": [1.0, None], "b": [2.0, None]})

    fitted = RobustFoldPreprocessor(missing_indicator_features=("a",)).fit(train)
    transformed = fitted.transform(train)

    assert fitted.feature_names_out_ == ("a", "b", "a__missing")
    assert transformed["a__missing"].tolist() == [0.0, 1.0]
    assert "b__missing" not in transformed


def test_all_null_training_feature_stays_excluded_when_validation_has_values() -> None:
    """Break caught: validation availability changes the training-selected schema."""
    training = pd.DataFrame({"all_null": [None, None], "usable": [1.0, 2.0]})
    validation = pd.DataFrame({"all_null": [9000.0], "usable": [3.0]})

    fitted = RobustFoldPreprocessor().fit(training)
    transformed = fitted.transform(validation)

    assert fitted.feature_names_in_ == ("all_null", "usable")
    assert fitted.feature_names_ == ("usable",)
    assert fitted.excluded_all_null_features_ == ("all_null",)
    assert fitted.location_ == {"usable": 1.5}
    assert fitted.scale_ == {"usable": 0.5}
    assert list(transformed.columns) == ["usable"]
    assert transformed.iloc[0].to_dict() == {"usable": 3.0}


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"a": [1.0, float("inf")]}), "finite"),
        (
            pd.DataFrame({"a": [None, None]}),
            "all selected features have no finite training values",
        ),
    ],
)
def test_preprocessor_rejects_invalid_training_values(
    frame: pd.DataFrame, message: str
) -> None:
    """Break caught: undefined training statistics silently produce invalid model inputs."""
    with pytest.raises(ValueError, match=message):
        RobustFoldPreprocessor().fit(frame)
