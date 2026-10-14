from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float | None
    end: float | None
    probability: float | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    language_probability: float | None
    duration_s: float | None
    model_size: str
    segments: list[TranscriptSegment]


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    word_timestamps: bool = True,
) -> TranscriptionResult:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Use Python 3.11 and install the real stack "
            'with: python -m pip install -e ".[real]"'
        ) from exc

    resolved_device = _resolve_device(device)
    resolved_compute_type = _resolve_compute_type(compute_type, resolved_device)
    model = WhisperModel(model_size, device=resolved_device, compute_type=resolved_compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        word_timestamps=word_timestamps,
    )

    return transcription_from_faster_whisper(segments_iter, info, model_size=model_size)


def transcription_from_faster_whisper(
    segments_iter: Iterable[Any], info: Any, *, model_size: str
) -> TranscriptionResult:
    """Convert faster-whisper output without changing the persisted JSON contract."""

    segments: list[TranscriptSegment] = []
    text_parts: list[str] = []
    for idx, segment in enumerate(segments_iter):
        segment_text = segment.text.strip()
        text_parts.append(segment_text)
        words = [
            WordTiming(
                word=word.word.strip(),
                start=_round_optional(word.start),
                end=_round_optional(word.end),
                probability=_round_optional(getattr(word, "probability", None)),
            )
            for word in (segment.words or [])
        ]
        segments.append(
            TranscriptSegment(
                id=idx,
                start=round(float(segment.start), 3),
                end=round(float(segment.end), 3),
                text=segment_text,
                avg_logprob=_round_optional(getattr(segment, "avg_logprob", None)),
                no_speech_prob=_round_optional(getattr(segment, "no_speech_prob", None)),
                words=words,
            )
        )

    return TranscriptionResult(
        text=" ".join(part for part in text_parts if part).strip(),
        language=getattr(info, "language", None),
        language_probability=_round_optional(getattr(info, "language_probability", None)),
        duration_s=_round_optional(getattr(info, "duration", None)),
        model_size=model_size,
        segments=segments,
    )


def write_transcription(result: TranscriptionResult, json_path: str | Path, text_path: str | Path) -> None:
    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    text_output.write_text(result.text + "\n", encoding="utf-8")


def load_transcription(path: str | Path) -> TranscriptionResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return transcription_from_dict(raw)


def transcription_from_dict(raw: dict[str, Any]) -> TranscriptionResult:
    segments = [
        TranscriptSegment(
            id=int(segment["id"]),
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]),
            avg_logprob=segment.get("avg_logprob"),
            no_speech_prob=segment.get("no_speech_prob"),
            words=[
                WordTiming(
                    word=str(word["word"]),
                    start=word.get("start"),
                    end=word.get("end"),
                    probability=word.get("probability"),
                )
                for word in segment.get("words", [])
            ],
        )
        for segment in raw.get("segments", [])
    ]
    return TranscriptionResult(
        text=str(raw.get("text", "")),
        language=raw.get("language"),
        language_probability=raw.get("language_probability"),
        duration_s=raw.get("duration_s"),
        model_size=str(raw.get("model_size", "unknown")),
        segments=segments,
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2
    except ImportError:
        return "cpu"
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _round_optional(value: object, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
