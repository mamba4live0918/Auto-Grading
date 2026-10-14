from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from presentation_assessment.media import VideoInfo, probe_video


@dataclass(frozen=True)
class VideoFeatureResult:
    visual_engagement_score: float
    gesture_posture_score: float
    features: dict[str, float | int | str | None]


def extract_video_features(
    video_path: str | Path,
    sample_fps: float = 1.0,
    max_frames: int = 240,
) -> VideoFeatureResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is not installed. Use Python 3.11 and install the real stack "
            'with: python -m pip install -e ".[real]"'
        ) from exc

    video_info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or video_info.fps or 30.0
    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1
    face_backend, face_detector = _load_face_detector(cv2)

    sampled = 0
    face_detected = 0
    face_area_ratios: list[float] = []
    face_center_offsets: list[float] = []
    motion_values: list[float] = []
    previous_gray_small: np.ndarray | None = None
    frame_index = 0

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        sampled += 1
        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        faces = _detect_faces(cv2, face_backend, face_detector, frame, gray)
        if len(faces) > 0:
            face_detected += 1
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_area_ratios.append((w * h) / float(width * height))
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            face_center_offsets.append(abs(center_x - 0.5) + abs(center_y - 0.42))

        gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if previous_gray_small is not None:
            diff = cv2.absdiff(gray_small, previous_gray_small)
            motion_values.append(float(np.mean(diff)) / 255.0)
        previous_gray_small = gray_small

    cap.release()
    if hasattr(face_detector, "close"):
        face_detector.close()

    face_visible_ratio = face_detected / sampled if sampled else 0.0
    avg_face_area_ratio = float(np.mean(face_area_ratios)) if face_area_ratios else 0.0
    face_center_offset = float(np.mean(face_center_offsets)) if face_center_offsets else None
    motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
    motion_std = float(np.std(motion_values)) if motion_values else 0.0

    features: dict[str, float | int | str | None] = {
        "duration_s": video_info.duration_s,
        "fps": None if fps is None else round(float(fps), 3),
        "width": video_info.width,
        "height": video_info.height,
        "sampled_frames": sampled,
        "face_detected_frames": face_detected,
        "face_visible_ratio": round(face_visible_ratio, 4),
        "avg_face_area_ratio": round(avg_face_area_ratio, 5),
        "face_center_offset": None if face_center_offset is None else round(face_center_offset, 4),
        "motion_mean": round(motion_mean, 5),
        "motion_std": round(motion_std, 5),
        "visual_feature_source": f"{face_backend}_proxy",
        "gesture_feature_source": "opencv_motion_proxy",
    }
    return VideoFeatureResult(
        visual_engagement_score=score_visual_engagement(features),
        gesture_posture_score=score_gesture_posture(features),
        features=features,
    )


def score_visual_engagement(features: dict[str, float | int | str | None]) -> float:
    visible = float(features.get("face_visible_ratio") or 0.0)
    area = float(features.get("avg_face_area_ratio") or 0.0)
    center_offset = features.get("face_center_offset")
    visible_score = min(100.0, visible * 100.0)
    area_score = _target_score(area, target=0.08, tolerance=0.09)
    center_score = 60.0 if center_offset is None else max(0.0, 100.0 * (1.0 - float(center_offset) / 0.75))
    return round(0.55 * visible_score + 0.20 * area_score + 0.25 * center_score, 2)


def score_gesture_posture(features: dict[str, float | int | str | None]) -> float:
    motion_mean = float(features.get("motion_mean") or 0.0)
    motion_std = float(features.get("motion_std") or 0.0)
    motion_score = _target_score(motion_mean, target=0.035, tolerance=0.06)
    variation_score = _target_score(motion_std, target=0.025, tolerance=0.05)
    return round(0.70 * motion_score + 0.30 * variation_score, 2)


def _target_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - abs(value - target) / tolerance))


def _load_face_detector(cv2: object) -> tuple[str, object | None]:
    cascade_candidates = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(__file__).resolve().parents[2] / "models" / "haarcascade_frontalface_default.xml",
        Path.cwd() / "models" / "haarcascade_frontalface_default.xml",
    ]
    for cascade_path in cascade_candidates:
        if not cascade_path.exists():
            continue
        detector = cv2.CascadeClassifier(str(cascade_path))
        if not detector.empty():
            return "opencv_haar", detector

    try:
        import mediapipe as mp
    except ImportError:
        return "no_face_detector", None

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
        return "no_face_detector", None

    return (
        "mediapipe_face_detection",
        mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        ),
    )


def _detect_faces(
    cv2: object,
    backend: str,
    detector: object | None,
    frame: np.ndarray,
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    if detector is None:
        return []
    if backend == "opencv_haar":
        return list(detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5))
    if backend == "mediapipe_face_detection":
        height, width = gray.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        faces: list[tuple[int, int, int, int]] = []
        for detection in results.detections or []:
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * width))
            y = max(0, int(box.ymin * height))
            w = max(0, int(box.width * width))
            h = max(0, int(box.height * height))
            if w > 0 and h > 0:
                faces.append((x, y, w, h))
        return faces
    return []
