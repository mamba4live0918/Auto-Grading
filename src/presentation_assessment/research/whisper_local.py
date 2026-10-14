"""Local-only, batch-reused Whisper transcription for research runs."""

from __future__ import annotations

import hmac
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from presentation_assessment.transcription import (
    TranscriptionResult,
    transcription_from_faster_whisper,
    transcription_from_dict,
)

from .artifacts import atomic_write_json, atomic_write_text
from .cuda_runtime import activate_cuda_runtime
from .preflight import canonical_directory_digest


WHISPER_TRANSCRIPTION_SCHEMA_VERSION = 1
_SAFE_ERROR_CODE = re.compile(r"[a-z0-9_]{1,64}\Z")
_SUMMARY_LIMIT = 240


class WhisperTranscriptionStatus(str, Enum):
    """Terminal states for one requested transcription."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class WhisperRuntimeConfig:
    """Frozen local model identity and inference settings validated before a batch."""

    validated_snapshot_path: Path
    model_name: str
    revision: str
    snapshot_sha256: str
    local_files_only: bool = True
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    language: str | None = None

    def __post_init__(self) -> None:
        snapshot = Path(self.validated_snapshot_path)
        try:
            resolved_snapshot = snapshot.resolve(strict=True)
        except OSError as error:
            raise ValueError("validated Whisper snapshot is unavailable") from error
        object.__setattr__(self, "validated_snapshot_path", resolved_snapshot)
        identity_valid = (
            bool(self.model_name.strip())
            and bool(re.fullmatch(r"[0-9a-f]{40}", self.revision.lower()))
            and bool(re.fullmatch(r"[0-9a-f]{64}", self.snapshot_sha256.lower()))
            and resolved_snapshot.is_dir()
            and resolved_snapshot.name == self.revision
        )
        frozen_settings = (
            self.local_files_only is True
            and self.device == "cuda"
            and self.compute_type == "float16"
            and type(self.beam_size) is int
            and self.beam_size == 5
            and self.vad_filter is True
            and self.word_timestamps is True
            and (self.language is None or bool(self.language.strip()))
        )
        if not identity_valid:
            raise ValueError("validated Whisper snapshot identity is invalid")
        if not frozen_settings:
            raise ValueError("configuration does not match the frozen Whisper runtime")

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], *, validated_snapshot_path: str | Path
    ) -> WhisperRuntimeConfig:
        """Bind the notebook configuration to a preflight-validated local path."""
        language = config.get("whisper_language")
        return cls(
            validated_snapshot_path=Path(validated_snapshot_path),
            model_name=str(config.get("whisper_model", "")),
            revision=str(config.get("whisper_revision", "")),
            snapshot_sha256=str(config.get("whisper_snapshot_sha256", "")),
            local_files_only=config.get("whisper_local_files_only") is True,
            device=str(config.get("whisper_device", "")),
            compute_type=str(config.get("whisper_compute_type", "")),
            beam_size=config.get("whisper_beam_size", 0),
            vad_filter=config.get("whisper_vad") is True,
            word_timestamps=config.get("whisper_word_timestamps") is True,
            language=None if language is None else str(language),
        )


@dataclass(frozen=True)
class WhisperTranscriptionItem:
    """One batch item; its local audio path is never copied into provenance."""

    sample_id: str
    audio_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_path", Path(self.audio_path))


@dataclass(frozen=True)
class WhisperTranscriptionProvenance:
    """Reconstructible model, decoding, language, and timing observations."""

    model_name: str
    revision: str
    snapshot_sha256: str
    local_files_only: bool
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    word_timestamps: bool
    requested_language: str | None
    detected_language: str | None
    language_probability: float | None
    audio_duration_s: float | None
    started_at_utc: str
    completed_at_utc: str
    elapsed_s: float


@dataclass(frozen=True)
class WhisperTranscriptionResult:
    """Terminal research outcome with legacy-compatible transcript content."""

    sample_id: str
    status: WhisperTranscriptionStatus
    transcription: TranscriptionResult | None
    provenance: WhisperTranscriptionProvenance
    error_code: str | None
    error_summary: str | None


class WhisperModel(Protocol):
    def transcribe(self, audio_path: str, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


ModelFactory = Callable[..., WhisperModel]


class LocalWhisperTranscriber:
    """Create one local Whisper model per batch and isolate sample failures."""

    def __init__(
        self, config: WhisperRuntimeConfig, *, model_factory: ModelFactory | None = None
    ) -> None:
        self._config = config
        self._model_factory = model_factory or _default_model_factory
        self._snapshot_is_valid: bool | None = None

    def transcribe_many(
        self,
        items: Iterable[WhisperTranscriptionItem | tuple[str, str | Path]],
    ) -> list[WhisperTranscriptionResult]:
        """Transcribe every item with one model, returning a terminal result for each."""
        return list(self.iter_transcriptions(items))

    def iter_transcriptions(
        self,
        items: Iterable[WhisperTranscriptionItem | tuple[str, str | Path]],
    ) -> Iterator[WhisperTranscriptionResult]:
        """Yield each terminal result as soon as its local inference finishes."""
        self._snapshot_is_valid = None
        batch = [_coerce_item(item) for item in items]
        if not batch:
            return
        if not self._validated_snapshot_matches():
            for item in batch:
                yield self._failed_result(
                    item.sample_id,
                    error_code="snapshot_validation_error",
                    error_summary="Local Whisper snapshot validation failed",
                )
            return
        try:
            model = self._model_factory(
                str(self._config.validated_snapshot_path),
                device=self._config.device,
                compute_type=self._config.compute_type,
                local_files_only=self._config.local_files_only,
            )
        except Exception:
            for item in batch:
                yield self._failed_result(
                    item.sample_id,
                    error_code="model_initialization_error",
                    error_summary="Local Whisper model initialization failed",
                )
            return

        for item in batch:
            started_at = _utc_now()
            started_timer = time.perf_counter()
            try:
                segments, info = model.transcribe(
                    str(item.audio_path),
                    language=self._config.language,
                    beam_size=self._config.beam_size,
                    vad_filter=self._config.vad_filter,
                    word_timestamps=self._config.word_timestamps,
                )
                transcription = transcription_from_faster_whisper(
                    segments, info, model_size=self._config.model_name
                )
            except Exception:
                yield self._failed_result(
                    item.sample_id,
                    started_at=started_at,
                    started_timer=started_timer,
                    error_code="transcription_error",
                    error_summary="Whisper transcription failed",
                )
                continue
            completed_at = _utc_now()
            elapsed_s = _elapsed(started_timer)
            yield WhisperTranscriptionResult(
                sample_id=item.sample_id,
                status=WhisperTranscriptionStatus.SUCCEEDED,
                transcription=transcription,
                provenance=self._provenance(
                    started_at=started_at,
                    completed_at=completed_at,
                    elapsed_s=elapsed_s,
                    transcription=transcription,
                ),
                error_code=None,
                error_summary=None,
            )

    def _validated_snapshot_matches(self) -> bool:
        if self._snapshot_is_valid is None:
            try:
                actual_digest = canonical_directory_digest(
                    self._config.validated_snapshot_path
                )
            except OSError:
                self._snapshot_is_valid = False
            else:
                self._snapshot_is_valid = hmac.compare_digest(
                    actual_digest, self._config.snapshot_sha256.lower()
                )
        return self._snapshot_is_valid

    def _failed_result(
        self,
        sample_id: str,
        *,
        error_code: str,
        error_summary: str,
        started_at: str | None = None,
        started_timer: float | None = None,
    ) -> WhisperTranscriptionResult:
        actual_started_at = started_at or _utc_now()
        actual_started_timer = time.perf_counter() if started_timer is None else started_timer
        return WhisperTranscriptionResult(
            sample_id=sample_id,
            status=WhisperTranscriptionStatus.FAILED,
            transcription=None,
            provenance=self._provenance(
                started_at=actual_started_at,
                completed_at=_utc_now(),
                elapsed_s=_elapsed(actual_started_timer),
                transcription=None,
            ),
            error_code=error_code,
            error_summary=error_summary,
        )

    def _provenance(
        self,
        *,
        started_at: str,
        completed_at: str,
        elapsed_s: float,
        transcription: TranscriptionResult | None,
    ) -> WhisperTranscriptionProvenance:
        return WhisperTranscriptionProvenance(
            model_name=self._config.model_name,
            revision=self._config.revision,
            snapshot_sha256=self._config.snapshot_sha256,
            local_files_only=self._config.local_files_only,
            device=self._config.device,
            compute_type=self._config.compute_type,
            beam_size=self._config.beam_size,
            vad_filter=self._config.vad_filter,
            word_timestamps=self._config.word_timestamps,
            requested_language=self._config.language,
            detected_language=None if transcription is None else transcription.language,
            language_probability=(
                None if transcription is None else transcription.language_probability
            ),
            audio_duration_s=None if transcription is None else transcription.duration_s,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            elapsed_s=elapsed_s,
        )


def transcribe_many(
    items: Iterable[WhisperTranscriptionItem | tuple[str, str | Path]],
    *,
    config: WhisperRuntimeConfig,
    model_factory: ModelFactory | None = None,
) -> list[WhisperTranscriptionResult]:
    """Convenience entry point that scopes one model instance to this batch."""
    return LocalWhisperTranscriber(config, model_factory=model_factory).transcribe_many(items)


def write_whisper_transcription_result(
    result: WhisperTranscriptionResult,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Atomically persist the complete versioned research result and plain text."""
    persisted_result = _sanitized_result(result)
    _validate_result(persisted_result)
    payload = {
        "schema_version": WHISPER_TRANSCRIPTION_SCHEMA_VERSION,
        "sample_id": persisted_result.sample_id,
        "status": persisted_result.status.value,
        "transcription": (
            None
            if persisted_result.transcription is None
            else asdict(persisted_result.transcription)
        ),
        "provenance": asdict(persisted_result.provenance),
        "error_code": persisted_result.error_code,
        "error_summary": persisted_result.error_summary,
    }
    transcript_text = (
        ""
        if persisted_result.transcription is None
        else persisted_result.transcription.text + "\n"
    )
    atomic_write_text(Path(text_path), transcript_text)
    atomic_write_json(Path(json_path), payload)


def load_whisper_transcription_result(path: str | Path) -> WhisperTranscriptionResult:
    """Load a version-1 research result without accepting partial or unsafe metadata."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError
        if set(raw) != {
            "schema_version",
            "sample_id",
            "status",
            "transcription",
            "provenance",
            "error_code",
            "error_summary",
        }:
            raise ValueError
        raw_provenance = raw["provenance"]
        if not isinstance(raw_provenance, dict) or set(raw_provenance) != {
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
        }:
            raise ValueError
        raw_transcription = raw["transcription"]
        if raw_transcription is not None and not isinstance(raw_transcription, dict):
            raise ValueError
        result = WhisperTranscriptionResult(
            sample_id=_required_string(raw["sample_id"]),
            status=WhisperTranscriptionStatus(_required_string(raw["status"])),
            transcription=(
                None
                if raw_transcription is None
                else transcription_from_dict(raw_transcription)
            ),
            provenance=WhisperTranscriptionProvenance(
                model_name=_required_string(raw_provenance["model_name"]),
                revision=_required_string(raw_provenance["revision"]),
                snapshot_sha256=_required_string(raw_provenance["snapshot_sha256"]),
                local_files_only=_required_bool(raw_provenance["local_files_only"]),
                device=_required_string(raw_provenance["device"]),
                compute_type=_required_string(raw_provenance["compute_type"]),
                beam_size=_required_int(raw_provenance["beam_size"]),
                vad_filter=_required_bool(raw_provenance["vad_filter"]),
                word_timestamps=_required_bool(raw_provenance["word_timestamps"]),
                requested_language=_optional_string(raw_provenance["requested_language"]),
                detected_language=_optional_string(raw_provenance["detected_language"]),
                language_probability=_optional_float(
                    raw_provenance["language_probability"]
                ),
                audio_duration_s=_optional_float(raw_provenance["audio_duration_s"]),
                started_at_utc=_required_string(raw_provenance["started_at_utc"]),
                completed_at_utc=_required_string(raw_provenance["completed_at_utc"]),
                elapsed_s=_required_float(raw_provenance["elapsed_s"]),
            ),
            error_code=_optional_string(raw["error_code"]),
            error_summary=_optional_string(raw["error_summary"]),
        )
        _validate_result(result)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid Whisper transcription artifact") from error
    return result


def _coerce_item(
    item: WhisperTranscriptionItem | tuple[str, str | Path],
) -> WhisperTranscriptionItem:
    if isinstance(item, WhisperTranscriptionItem):
        return item
    sample_id, audio_path = item
    return WhisperTranscriptionItem(str(sample_id), Path(audio_path))


def _sanitized_result(result: WhisperTranscriptionResult) -> WhisperTranscriptionResult:
    if result.status is not WhisperTranscriptionStatus.FAILED:
        return result
    error_code = result.error_code or "transcription_error"
    if _SAFE_ERROR_CODE.fullmatch(error_code) is None:
        error_code = "transcription_error"
    return replace(
        result,
        error_code=error_code,
        error_summary=_redacted_summary(result.error_summary or "Whisper transcription failed"),
    )


def _validate_result(result: WhisperTranscriptionResult) -> None:
    if not result.sample_id or re.search(r"[/\\]", result.sample_id):
        raise ValueError("invalid Whisper transcription artifact")
    if result.status is WhisperTranscriptionStatus.SUCCEEDED:
        if (
            result.transcription is None
            or result.error_code is not None
            or result.error_summary is not None
        ):
            raise ValueError("invalid Whisper transcription artifact")
    elif (
        result.transcription is not None
        or result.error_code is None
        or _SAFE_ERROR_CODE.fullmatch(result.error_code) is None
        or result.error_summary is None
        or len(result.error_summary) > _SUMMARY_LIMIT
        or _redacted_summary(result.error_summary) != result.error_summary
    ):
        raise ValueError("invalid Whisper transcription artifact")


def _redacted_summary(value: str) -> str:
    collapsed = " ".join(value.split())
    collapsed = re.sub(r"https?://\S+", "[redacted-url]", collapsed, flags=re.IGNORECASE)
    collapsed = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s]+", "[redacted-path]", collapsed)
    return (collapsed or "Whisper transcription failed")[:_SUMMARY_LIMIT]


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _required_string(value)


def _required_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError
    return value


def _required_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError
    return value


def _required_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _required_float(value)


def _default_model_factory(model_path: str, **kwargs: Any) -> WhisperModel:
    if kwargs.get("device") == "cuda":
        activate_cuda_runtime()
    try:
        from faster_whisper import WhisperModel as FasterWhisperModel
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError("faster_whisper_unavailable") from error
    return FasterWhisperModel(model_path, **kwargs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _elapsed(started_timer: float) -> float:
    return round(max(0.0, time.perf_counter() - started_timer), 6)
