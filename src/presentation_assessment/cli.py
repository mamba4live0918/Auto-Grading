from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentation_assessment.config import load_config
from presentation_assessment.data import load_manifest
from presentation_assessment.evaluation import evaluate_prediction_file
from presentation_assessment.pipeline import AssessmentPipeline, write_predictions
from presentation_assessment.video_pipeline import build_manifest_from_videos, process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal presentation assessment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one experiment config")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate prediction CSV")
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--label-column", default="human_score")
    eval_parser.add_argument("--prediction-column", default="final_score")

    ablate_parser = subparsers.add_parser("ablate", help="Run every YAML config in a directory")
    ablate_parser.add_argument("--manifest", required=True)
    ablate_parser.add_argument("--config-dir", required=True)
    ablate_parser.add_argument("--output-dir", required=True)

    video_parser = subparsers.add_parser("process-video", help="Run the real video pipeline")
    video_parser.add_argument("--video", required=True)
    video_parser.add_argument("--output-dir", default="outputs/video_runs")
    video_parser.add_argument("--config", default="configs/full_multimodal.yaml")
    video_parser.add_argument("--model-size", default="large-v3")
    video_parser.add_argument("--language", default=None)
    video_parser.add_argument("--device", default="auto")
    video_parser.add_argument("--compute-type", default="auto")
    video_parser.add_argument("--transcript", default=None)
    video_parser.add_argument("--transcription-json", default=None)
    video_parser.add_argument("--no-transcribe", action="store_true")
    video_parser.add_argument("--sample-fps", type=float, default=1.0)
    video_parser.add_argument("--max-frames", type=int, default=240)

    manifest_parser = subparsers.add_parser(
        "build-manifest", help="Build a manifest CSV from a directory of videos"
    )
    manifest_parser.add_argument("--video-dir", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--transcript-dir", default=None)

    args = parser.parse_args()
    if args.command == "run":
        _run(args.manifest, args.config, args.output)
    elif args.command == "evaluate":
        result = evaluate_prediction_file(args.predictions, args.label_column, args.prediction_column)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    elif args.command == "ablate":
        _ablate(args.manifest, args.config_dir, args.output_dir)
    elif args.command == "process-video":
        result = process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            config_path=args.config,
            model_size=args.model_size,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            transcribe=not args.no_transcribe,
            transcript_path=args.transcript,
            existing_transcription_json_path=args.transcription_json,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "build-manifest":
        output = build_manifest_from_videos(args.video_dir, args.output, args.transcript_dir)
        print(f"Wrote manifest to {output}")


def _run(manifest_path: str, config_path: str, output_path: str) -> None:
    config = load_config(config_path)
    samples = load_manifest(manifest_path)
    rows = AssessmentPipeline(config).run(samples)
    write_predictions(rows, output_path)
    print(f"Wrote {len(rows)} predictions to {output_path}")


def _ablate(manifest_path: str, config_dir: str, output_dir: str) -> None:
    output_root = Path(output_dir)
    config_paths = sorted(Path(config_dir).glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"No YAML configs found in {config_dir}")

    for config_path in config_paths:
        config = load_config(config_path)
        output_path = output_root / f"{config.experiment_name}_predictions.csv"
        _run(manifest_path, str(config_path), str(output_path))
        result = evaluate_prediction_file(output_path)
        print(f"{config.experiment_name}: {json.dumps(result.as_dict(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
