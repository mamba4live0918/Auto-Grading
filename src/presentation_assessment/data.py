from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PresentationSample:
    sample_id: str
    video_path: Path
    transcript_path: Path | None
    transcript: str
    human_score: float | None
    speech_delivery_score: float | None
    visual_engagement_score: float | None
    gesture_posture_score: float | None


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _first_float(row: dict[str, str], *columns: str) -> float | None:
    for column in columns:
        value = row.get(column)
        if value is not None and value.strip() != "":
            return float(value)
    return None


def _resolve(base_dir: Path, value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_manifest(path: str | Path) -> list[PresentationSample]:
    manifest_path = Path(path)
    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "data" else manifest_path.parent
    samples: list[PresentationSample] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "video_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            transcript_path = _resolve(base_dir, row.get("transcript_path"))
            transcript = ""
            if transcript_path and transcript_path.exists():
                transcript = transcript_path.read_text(encoding="utf-8").strip()

            video_path = _resolve(base_dir, row.get("video_path"))
            if video_path is None:
                raise ValueError(f"Sample {row.get('sample_id')} has no video_path.")

            samples.append(
                PresentationSample(
                    sample_id=row["sample_id"],
                    video_path=video_path,
                    transcript_path=transcript_path,
                    transcript=transcript,
                    human_score=_optional_float(row.get("human_score")),
                    speech_delivery_score=_first_float(
                        row, "speech_delivery_score", "delivery_score"
                    ),
                    visual_engagement_score=_first_float(
                        row, "visual_engagement_score", "visual_score"
                    ),
                    gesture_posture_score=_first_float(
                        row, "gesture_posture_score", "gesture_score"
                    ),
                )
            )

    return samples
