from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import presentation_assessment.research.whisper_local as whisper_local

from presentation_assessment.research.whisper_local import (
    LocalWhisperTranscriber,
    WhisperRuntimeConfig,
    WhisperTranscriptionItem,
    WhisperTranscriptionStatus,
    transcribe_many,
)
from presentation_assessment.transcription import load_transcription, write_transcription


MODEL_NAME = "Systran/faster-whisper-large-v3"
REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"


class FakeWhisperModel:
    def __init__(self, failing_names: set[str] | None = None) -> None:
        self.failing_names = failing_names or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(self, audio_path: str, **kwargs: Any) -> tuple[list[object], object]:
        self.calls.append((audio_path, kwargs))
        if Path(audio_path).name in self.failing_names:
            raise RuntimeError(f"decoder failed for C:\\private\\{Path(audio_path).name}")
        word = SimpleNamespace(word=" hello ", start=0.1254, end=0.7555, probability=0.98765)
        segment = SimpleNamespace(
            start=0.1254,
            end=1.5555,
            text=" Hello world ",
            avg_logprob=-0.12545,
            no_speech_prob=0.01234,
            words=[word],
        )
        info = SimpleNamespace(language="en", language_probability=0.99876, duration=1.55555)
        return [segment], info


def snapshot_digest(snapshot: Path) -> str:
    file_hashes = {
        path.relative_to(snapshot).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    canonical = json.dumps(
        file_hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def runtime_config(
    tmp_path: Path, *, expected_snapshot_digest: str | None = None
) -> WhisperRuntimeConfig:
    snapshot = tmp_path / "hub" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b'{"model":"synthetic"}')
    (snapshot / "model.bin").write_bytes(b"synthetic model weights")
    (snapshot / "tokenizer.json").write_bytes(b'{"tokenizer":"synthetic"}')
    return WhisperRuntimeConfig.from_mapping(
        {
            "whisper_model": MODEL_NAME,
            "whisper_revision": REVISION,
            "whisper_snapshot_sha256": (
                expected_snapshot_digest or snapshot_digest(snapshot)
            ),
            "whisper_local_files_only": True,
            "whisper_device": "cuda",
            "whisper_compute_type": "float16",
            "whisper_beam_size": 5,
            "whisper_vad": True,
            "whisper_word_timestamps": True,
        },
        validated_snapshot_path=snapshot.resolve(),
    )


def test_batch_reuses_one_model_created_from_the_validated_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a model ID/download path or one model per sample reaches the factory."""
    config = runtime_config(tmp_path)
    first = tmp_path / "N001.wav"
    second = tmp_path / "N002.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    model = FakeWhisperModel()
    factory_calls: list[tuple[str, dict[str, Any]]] = []
    digest_calls: list[Path] = []
    from presentation_assessment.research import whisper_local

    real_digest = whisper_local.canonical_directory_digest

    def counting_digest(path: Path) -> str:
        digest_calls.append(path)
        return real_digest(path)

    monkeypatch.setattr(whisper_local, "canonical_directory_digest", counting_digest)

    def factory(model_path: str, **kwargs: Any) -> FakeWhisperModel:
        factory_calls.append((model_path, kwargs))
        return model

    results = LocalWhisperTranscriber(config, model_factory=factory).transcribe_many(
        [
            WhisperTranscriptionItem("N001", first),
            WhisperTranscriptionItem("N002", second),
        ]
    )

    assert factory_calls == [
        (
            str(config.validated_snapshot_path),
            {"device": "cuda", "compute_type": "float16", "local_files_only": True},
        )
    ]
    assert digest_calls == [config.validated_snapshot_path]
    assert [result.status for result in results] == [
        WhisperTranscriptionStatus.SUCCEEDED,
        WhisperTranscriptionStatus.SUCCEEDED,
    ]
    assert [call[1] for call in model.calls] == [
        {"language": None, "beam_size": 5, "vad_filter": True, "word_timestamps": True},
        {"language": None, "beam_size": 5, "vad_filter": True, "word_timestamps": True},
    ]
    first_result = results[0]
    assert first_result.transcription is not None
    assert first_result.transcription.text == "Hello world"
    assert first_result.transcription.segments[0].words[0].probability == 0.9877
    assert first_result.provenance.model_name == MODEL_NAME
    assert first_result.provenance.revision == REVISION
    assert first_result.provenance.snapshot_sha256 == config.snapshot_sha256
    assert first_result.provenance.device == "cuda"
    assert first_result.provenance.compute_type == "float16"
    assert first_result.provenance.beam_size == 5
    assert first_result.provenance.vad_filter is True
    assert first_result.provenance.word_timestamps is True
    assert first_result.provenance.requested_language is None
    assert first_result.provenance.detected_language == "en"
    assert first_result.provenance.language_probability == 0.9988
    assert first_result.provenance.audio_duration_s == 1.5555
    assert first_result.provenance.started_at_utc.endswith("Z")
    assert first_result.provenance.completed_at_utc.endswith("Z")
    assert first_result.provenance.elapsed_s >= 0.0
    assert str(config.validated_snapshot_path) not in repr(first_result.provenance)


def test_iter_transcriptions_yields_each_result_before_starting_the_next_item(
    tmp_path: Path,
) -> None:
    """Break caught: completed transcriptions remain buffered behind a later stall."""
    config = runtime_config(tmp_path)
    first = tmp_path / "N001.wav"
    second = tmp_path / "N002.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    model = FakeWhisperModel()
    transcriber = LocalWhisperTranscriber(
        config, model_factory=lambda *args, **kwargs: model
    )

    streamed = transcriber.iter_transcriptions(
        [WhisperTranscriptionItem("N001", first), WhisperTranscriptionItem("N002", second)]
    )
    first_result = next(streamed)

    assert first_result.sample_id == "N001"
    assert [Path(call[0]).name for call in model.calls] == ["N001.wav"]

    second_result = next(streamed)
    assert second_result.sample_id == "N002"
    assert [Path(call[0]).name for call in model.calls] == ["N001.wav", "N002.wav"]
    with pytest.raises(StopIteration):
        next(streamed)


def test_tampered_snapshot_fails_before_model_construction(tmp_path: Path) -> None:
    """Break caught: snapshot contents change after config binding but still reach Whisper."""
    config = runtime_config(tmp_path)
    (config.validated_snapshot_path / "model.bin").write_bytes(b"tampered weights")
    factory_calls: list[str] = []

    result = transcribe_many(
        [("N001", tmp_path / "N001.wav")],
        config=config,
        model_factory=lambda model_path, **kwargs: factory_calls.append(model_path),
    )[0]

    assert result.status is WhisperTranscriptionStatus.FAILED
    assert result.error_code == "snapshot_validation_error"
    assert result.error_summary == "Local Whisper snapshot validation failed"
    assert factory_calls == []


def test_reused_transcriber_revalidates_snapshot_before_each_batch(tmp_path: Path) -> None:
    """Break caught: a successful cached digest masks tampering before a later batch."""
    config = runtime_config(tmp_path)
    model = FakeWhisperModel()
    factory_calls: list[str] = []

    def factory(model_path: str, **kwargs: Any) -> FakeWhisperModel:
        factory_calls.append(model_path)
        return model

    transcriber = LocalWhisperTranscriber(config, model_factory=factory)
    first = transcriber.transcribe_many([("N001", tmp_path / "N001.wav")])
    (config.validated_snapshot_path / "model.bin").write_bytes(b"tampered after batch one")

    second = transcriber.transcribe_many([("N002", tmp_path / "N002.wav")])

    assert first[0].status is WhisperTranscriptionStatus.SUCCEEDED
    assert second[0].status is WhisperTranscriptionStatus.FAILED
    assert second[0].error_code == "snapshot_validation_error"
    assert factory_calls == [str(config.validated_snapshot_path)]
    assert len(model.calls) == 1


def test_arbitrary_revision_named_directory_cannot_claim_an_unrelated_digest(
    tmp_path: Path,
) -> None:
    """Break caught: a format-valid configured digest is trusted without hashing files."""
    config = runtime_config(tmp_path, expected_snapshot_digest="0" * 64)
    factory_calls: list[str] = []

    result = transcribe_many(
        [("N001", tmp_path / "N001.wav")],
        config=config,
        model_factory=lambda model_path, **kwargs: factory_calls.append(model_path),
    )[0]

    assert result.status is WhisperTranscriptionStatus.FAILED
    assert result.error_code == "snapshot_validation_error"
    assert factory_calls == []


def test_one_failed_sample_is_terminal_and_later_samples_continue(tmp_path: Path) -> None:
    """Break caught: one decoder exception aborts the rest of the batch or leaks a path."""
    config = runtime_config(tmp_path)
    paths = [tmp_path / name for name in ("N001.wav", "N002.wav", "N003.wav")]
    for path in paths:
        path.write_bytes(b"audio")
    model = FakeWhisperModel(failing_names={"N002.wav"})

    results = transcribe_many(
        [("N001", paths[0]), ("N002", paths[1]), ("N003", paths[2])],
        config=config,
        model_factory=lambda *args, **kwargs: model,
    )

    assert [result.sample_id for result in results] == ["N001", "N002", "N003"]
    assert [result.status for result in results] == [
        WhisperTranscriptionStatus.SUCCEEDED,
        WhisperTranscriptionStatus.FAILED,
        WhisperTranscriptionStatus.SUCCEEDED,
    ]
    failed = results[1]
    assert failed.transcription is None
    assert failed.error_code == "transcription_error"
    assert failed.error_summary == "Whisper transcription failed"
    assert failed.provenance.detected_language is None
    assert len(model.calls) == 3
    assert "C:" not in repr(failed)
    assert "private" not in repr(failed).lower()


def test_wrapper_transcription_uses_the_existing_json_contract(tmp_path: Path) -> None:
    """Break caught: research provenance fields alter the legacy transcription JSON schema."""
    config = runtime_config(tmp_path)
    audio = tmp_path / "N001.wav"
    audio.write_bytes(b"audio")
    result = transcribe_many(
        [("N001", audio)],
        config=config,
        model_factory=lambda *args, **kwargs: FakeWhisperModel(),
    )[0]
    assert result.transcription is not None
    json_path = tmp_path / "transcription.json"
    text_path = tmp_path / "transcript.txt"

    write_transcription(result.transcription, json_path, text_path)

    assert load_transcription(json_path) == result.transcription
    assert set(json.loads(json_path.read_text(encoding="utf-8"))) == {
        "text",
        "language",
        "language_probability",
        "duration_s",
        "model_size",
        "segments",
    }
    assert text_path.read_text(encoding="utf-8") == "Hello world\n"


def test_historical_legacy_transcription_fixture_still_loads() -> None:
    """Break caught: the new research format makes a prior legacy artifact unreadable."""
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "research"
        / "transcription_legacy_v0.json"
    )

    loaded = load_transcription(fixture)

    assert loaded.text == "Historical synthetic transcript."
    assert loaded.language == "en"
    assert loaded.language_probability == 0.875
    assert loaded.duration_s == 2.5
    assert loaded.model_size == "large-v3"
    assert len(loaded.segments) == 1
    assert loaded.segments[0].text == "Historical synthetic transcript."
    assert loaded.segments[0].words[0].word == "Historical"
    assert loaded.segments[0].words[0].probability == 0.95


def test_versioned_research_artifact_roundtrips_complete_success_result(
    tmp_path: Path,
) -> None:
    """Break caught: durable output drops terminal or provenance/transcript fields."""
    config = runtime_config(tmp_path)
    result = transcribe_many(
        [("N001", tmp_path / "N001.wav")],
        config=config,
        model_factory=lambda *args, **kwargs: FakeWhisperModel(),
    )[0]
    json_path = tmp_path / "research-transcription.v1.json"
    text_path = tmp_path / "transcript.txt"

    whisper_local.write_whisper_transcription_result(result, json_path, text_path)

    assert whisper_local.load_whisper_transcription_result(json_path) == result
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload) == {
        "schema_version",
        "sample_id",
        "status",
        "transcription",
        "provenance",
        "error_code",
        "error_summary",
    }
    assert set(payload["transcription"]) == {
        "text",
        "language",
        "language_probability",
        "duration_s",
        "model_size",
        "segments",
    }
    assert payload["transcription"]["segments"][0]["words"][0]["word"] == "hello"
    assert set(payload["provenance"]) == {
        "model_name",
        "revision",
        "snapshot_sha256",
        "local_files_only",
        "device",
        "compute_type",
        "beam_size",
        "vad_filter",
        "word_timestamps",
        "requested_language",
        "detected_language",
        "language_probability",
        "audio_duration_s",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_s",
    }
    assert text_path.read_text(encoding="utf-8") == "Hello world\n"


def test_versioned_research_artifact_roundtrips_terminal_failure(tmp_path: Path) -> None:
    """Break caught: failed samples cannot be durably reconstructed."""
    config = runtime_config(tmp_path)
    result = transcribe_many(
        [("N001", tmp_path / "N001.wav")],
        config=config,
        model_factory=lambda *args, **kwargs: FakeWhisperModel({"N001.wav"}),
    )[0]
    json_path = tmp_path / "failed.v1.json"
    text_path = tmp_path / "failed.txt"

    whisper_local.write_whisper_transcription_result(result, json_path, text_path)

    loaded = whisper_local.load_whisper_transcription_result(json_path)
    assert loaded == result
    assert loaded.status is WhisperTranscriptionStatus.FAILED
    assert loaded.transcription is None
    assert loaded.error_code == "transcription_error"
    assert text_path.read_text(encoding="utf-8") == ""


def test_versioned_writer_bounds_and_redacts_supplied_terminal_errors(tmp_path: Path) -> None:
    """Break caught: caller-supplied exception details leak into durable metadata."""
    config = runtime_config(tmp_path)
    failed = transcribe_many(
        [("N001", tmp_path / "N001.wav")],
        config=config,
        model_factory=lambda *args, **kwargs: FakeWhisperModel({"N001.wav"}),
    )[0]
    leaked_url = "https://secret.invalid/student"
    leaked_path = r"C:\private\student.wav"
    unsafe = replace(
        failed,
        error_code="NOT SAFE",
        error_summary=f"failure at {leaked_url} from {leaked_path} " + "x" * 400,
    )
    json_path = tmp_path / "redacted.v1.json"

    whisper_local.write_whisper_transcription_result(unsafe, json_path, tmp_path / "out.txt")

    loaded = whisper_local.load_whisper_transcription_result(json_path)
    persisted = json_path.read_text(encoding="utf-8")
    assert loaded.error_code == "transcription_error"
    assert loaded.error_summary is not None
    assert len(loaded.error_summary) <= 240
    assert leaked_url not in persisted
    assert leaked_path not in persisted
    assert "C:" not in persisted


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("whisper_local_files_only", False),
        ("whisper_device", "cpu"),
        ("whisper_compute_type", "int8"),
        ("whisper_beam_size", 1),
        ("whisper_vad", False),
        ("whisper_word_timestamps", False),
    ],
)
def test_runtime_config_rejects_settings_outside_the_frozen_binding(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    """Break caught: a research batch silently drifts from a frozen inference setting."""
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    mapping: dict[str, object] = {
        "whisper_model": MODEL_NAME,
        "whisper_revision": REVISION,
        "whisper_snapshot_sha256": "a" * 64,
        "whisper_local_files_only": True,
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_beam_size": 5,
        "whisper_vad": True,
        "whisper_word_timestamps": True,
    }
    mapping[field] = invalid_value

    with pytest.raises(ValueError, match="frozen Whisper runtime"):
        WhisperRuntimeConfig.from_mapping(
            mapping, validated_snapshot_path=snapshot.resolve()
        )
