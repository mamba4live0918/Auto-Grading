from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from presentation_assessment.audio_features import extract_audio_features
from presentation_assessment.config import load_config
from presentation_assessment.data import PresentationSample
from presentation_assessment.media import extract_audio, find_video_files, probe_video
from presentation_assessment.pipeline import AssessmentPipeline
from presentation_assessment.transcription import (
    TranscriptionResult,
    load_transcription,
    transcribe_audio,
    write_transcription,
)
from presentation_assessment.video_features import extract_video_features


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
    transcribe: bool = True,
    transcript_path: str | Path | None = None,
    existing_transcription_json_path: str | Path | None = None,
    sample_fps: float = 1.0,
    max_frames: int = 240,
) -> dict[str, object]:
    video = Path(video_path)
    run_dir = Path(output_dir) / video.stem
    audio_path = run_dir / "audio.wav"
    transcript_txt_path = run_dir / "transcript.txt"
    output_transcription_json_path = run_dir / "transcription.json"
    features_path = run_dir / "features.json"
    result_path = run_dir / "assessment.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    video_info = probe_video(video)
    extract_audio(video, audio_path)

    transcription: TranscriptionResult | None = None
    transcript_text = ""
    if existing_transcription_json_path:
        transcription = load_transcription(existing_transcription_json_path)
        transcript_text = transcription.text
        write_transcription(transcription, output_transcription_json_path, transcript_txt_path)
    elif transcript_path:
        transcript_text = Path(transcript_path).read_text(encoding="utf-8").strip()
        transcript_txt_path.write_text(transcript_text + "\n", encoding="utf-8")
    elif transcribe:
        transcription = transcribe_audio(
            audio_path,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
        )
        transcript_text = transcription.text
        write_transcription(transcription, output_transcription_json_path, transcript_txt_path)

    audio_result = extract_audio_features(audio_path, transcription, transcript_text)
    video_result = extract_video_features(video, sample_fps=sample_fps, max_frames=max_frames)
    feature_payload = {
        "video": asdict(video_info),
        "audio": audio_result.features,
        "video_features": video_result.features,
    }
    features_path.write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sample = PresentationSample(
        sample_id=video.stem,
        video_path=video,
        transcript_path=transcript_txt_path,
        transcript=transcript_text,
        human_score=None,
        speech_delivery_score=audio_result.speech_delivery_score,
        visual_engagement_score=video_result.visual_engagement_score,
        gesture_posture_score=video_result.gesture_posture_score,
    )
    row = AssessmentPipeline(load_config(config_path)).run_sample(sample)
    result_payload = {
        "assessment": row,
        "artifacts": {
            "run_dir": str(run_dir),
            "audio_path": str(audio_path),
            "transcript_path": str(transcript_txt_path),
            "transcription_json_path": (
                str(output_transcription_json_path)
                if output_transcription_json_path.exists()
                else None
            ),
            "features_path": str(features_path),
        },
    }
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_payload


def build_manifest_from_videos(
    video_dir: str | Path,
    output_path: str | Path,
    transcript_dir: str | Path | None = None,
) -> Path:
    videos = find_video_files(video_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    transcript_root = Path(transcript_dir) if transcript_dir else None

    fieldnames = [
        "sample_id",
        "video_path",
        "transcript_path",
        "human_score",
        "speech_delivery_score",
        "visual_engagement_score",
        "gesture_posture_score",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video in videos:
            transcript_path = ""
            if transcript_root:
                candidate = transcript_root / f"{video.stem}.txt"
                if candidate.exists():
                    transcript_path = str(candidate)
            writer.writerow(
                {
                    "sample_id": video.stem,
                    "video_path": str(video),
                    "transcript_path": transcript_path,
                    "human_score": "",
                    "speech_delivery_score": "",
                    "visual_engagement_score": "",
                    "gesture_posture_score": "",
                }
            )
    return output
