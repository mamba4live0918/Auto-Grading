"""Local, blind audio-boundary review server. No score or automatic boundary is served."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import threading
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "human_review_annotations"
STATE_FILE = OUTPUT / "review_state.json"
BOUNDARY_FIELDS = ["clip_id", "annotator", "event_index", "start_seconds", "end_seconds", "label", "uncertain", "notes"]
STATUS_FIELDS = ["clip_id", "annotator", "reviewed", "difficult_to_judge", "notes"]


class AnnotationError(ValueError):
    pass


def ordered_clips(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (0 if row["phase"] == "development" else 1, row["clip_id"]))


def validate_clip_update(payload: dict, duration: float) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("segments", []), list):
        raise AnnotationError("segments must be a list")
    segments = []
    for item in payload.get("segments", []):
        if not isinstance(item, dict):
            raise AnnotationError("segment must be an object")
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnotationError("segment times must be numbers") from exc
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise AnnotationError("segment must fit within clip duration")
        label = item.get("label")
        if label not in ("speech", "uncertain"):
            raise AnnotationError("label must be speech or uncertain")
        notes = item.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 500:
            raise AnnotationError("segment notes must be text up to 500 characters")
        segments.append({
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "label": label,
            "notes": notes.strip(),
        })
    if len(segments) > 300:
        raise AnnotationError("too many segments")
    segments.sort(key=lambda item: item["start_seconds"])
    for previous, current in zip(segments, segments[1:]):
        if current["start_seconds"] < previous["end_seconds"]:
            raise AnnotationError("segments overlap")
    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise AnnotationError("clip notes must be text up to 2000 characters")
    reviewed = payload.get("reviewed", False)
    difficult = payload.get("difficult_to_judge", False)
    no_speech = payload.get("no_speech_confirmed", False)
    if not isinstance(reviewed, bool) or not isinstance(difficult, bool) or not isinstance(no_speech, bool):
        raise AnnotationError("review flags must be true or false")
    if reviewed and not segments and not no_speech:
        raise AnnotationError("confirm no audible speech before reviewing an empty clip")
    if segments and no_speech:
        raise AnnotationError("no-speech confirmation conflicts with segments")
    return {
        "segments": segments,
        "reviewed": reviewed,
        "difficult_to_judge": difficult,
        "no_speech_confirmed": no_speech,
        "notes": notes.strip(),
    }


def _csv_text(fields: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_boundaries_csv(clips: list[dict], state: dict) -> str:
    rows = []
    annotator = state.get("annotator", "student_1")
    for clip in clips:
        record = state.get("clips", {}).get(clip["clip_id"], {})
        if not record.get("reviewed"):
            continue
        for index, segment in enumerate(record.get("segments", []), 1):
            rows.append({
                "clip_id": clip["clip_id"],
                "annotator": annotator,
                "event_index": index,
                "start_seconds": f'{segment["start_seconds"]:.2f}',
                "end_seconds": f'{segment["end_seconds"]:.2f}',
                "label": segment["label"],
                "uncertain": "yes" if segment["label"] == "uncertain" else "no",
                "notes": segment.get("notes", ""),
            })
    return _csv_text(BOUNDARY_FIELDS, rows)


def build_status_csv(clips: list[dict], state: dict) -> str:
    rows = []
    annotator = state.get("annotator", "student_1")
    for clip in clips:
        record = state.get("clips", {}).get(clip["clip_id"], {})
        rows.append({
            "clip_id": clip["clip_id"],
            "annotator": annotator,
            "reviewed": "yes" if record.get("reviewed") else "no",
            "difficult_to_judge": "yes" if record.get("difficult_to_judge") else "no",
            "notes": record.get("notes", ""),
        })
    return _csv_text(STATUS_FIELDS, rows)


def load_manifest() -> tuple[Path, list[dict]]:
    location = ROOT / "output" / "thesis_advisor_pack_20260921" / "private_location.json"
    private_dir = Path(json.loads(location.read_text(encoding="utf-8"))["private_directory"])
    review_dir = private_dir / "human_review"
    with (review_dir / "clip_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        clips = ordered_clips(list(csv.DictReader(file)))
    if len(clips) != 24 or len({row["clip_id"] for row in clips}) != len(clips):
        raise RuntimeError("Expected 24 unique blind review clips")
    for row in clips:
        clip_id = row["clip_id"]
        if not re.fullmatch(r"[DA]\d{2}_[12]", clip_id):
            raise RuntimeError("Unexpected clip ID in manifest")
        audio = review_dir / row["audio_file"]
        if audio.resolve().parent != (review_dir / "audio").resolve() or not audio.is_file():
            raise RuntimeError(f"Missing allowed audio for {clip_id}")
    return review_dir, clips


def read_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"annotator": "student_1", "rules": "", "rules_frozen": False, "clips": {}}


def write_state(state: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def make_handler(review_dir: Path, clips: list[dict]):
    by_id = {row["clip_id"]: row for row in clips}
    state_lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, data: bytes, content_type: str, status: int = 200, filename: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value: dict, status: int = 200):
            self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 100_000:
                raise AnnotationError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise AnnotationError("request must be an object")
            return value

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html", "/app.js", "/styles.css"):
                name = "index.html" if route == "/" else route.lstrip("/")
                content_type = {"index.html": "text/html", "app.js": "text/javascript", "styles.css": "text/css"}[name]
                return self._send((STATIC / name).read_bytes(), content_type + "; charset=utf-8")
            if route == "/api/session":
                with state_lock:
                    state = read_state()
                return self._json({"clips": [{"clip_id": c["clip_id"], "phase": c["phase"], "duration_seconds": float(c["duration_seconds"])} for c in clips], "state": state})
            if route.startswith("/api/audio/"):
                clip_id = unquote(route.removeprefix("/api/audio/"))
                if clip_id not in by_id:
                    return self._json({"error": "unknown clip"}, 404)
                audio = review_dir / by_id[clip_id]["audio_file"]
                return self._send(audio.read_bytes(), "audio/wav")
            if route == "/api/export":
                with state_lock:
                    state = read_state()
                boundaries = build_boundaries_csv(clips, state)
                status = build_status_csv(clips, state)
                OUTPUT.mkdir(parents=True, exist_ok=True)
                (OUTPUT / "boundaries.csv").write_text(boundaries, encoding="utf-8-sig")
                (OUTPUT / "review_status.csv").write_text(status, encoding="utf-8-sig")
                (OUTPUT / "rules.txt").write_text(state.get("rules", ""), encoding="utf-8")
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("boundaries.csv", boundaries)
                    archive.writestr("review_status.csv", status)
                    archive.writestr("rules.txt", state.get("rules", ""))
                return self._send(buffer.getvalue(), "application/zip", filename="human_review_annotations.zip")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                payload = self._body()
                route = urlparse(self.path).path
                with state_lock:
                    state = read_state()
                    if route == "/api/meta":
                        annotator = payload.get("annotator", state["annotator"])
                        rules = payload.get("rules", state["rules"])
                        if not isinstance(annotator, str) or not re.fullmatch(r"[\w-]{1,64}", annotator):
                            raise AnnotationError("annotator must be 1-64 letters, digits, _ or -")
                        if not isinstance(rules, str) or len(rules) > 10000:
                            raise AnnotationError("rules must be text up to 10000 characters")
                        if state["rules_frozen"] and rules != state["rules"]:
                            raise AnnotationError("rules are frozen")
                        state["annotator"] = annotator
                        state["rules"] = rules
                    elif route == "/api/freeze":
                        development = [c["clip_id"] for c in clips if c["phase"] == "development"]
                        if not all(state["clips"].get(clip_id, {}).get("reviewed") for clip_id in development):
                            raise AnnotationError("review all 8 D clips before freezing rules")
                        if not state["rules"].strip():
                            raise AnnotationError("write your listening rules before freezing")
                        state["rules_frozen"] = True
                    elif route.startswith("/api/clip/"):
                        clip_id = unquote(route.removeprefix("/api/clip/"))
                        clip = by_id.get(clip_id)
                        if clip is None:
                            return self._json({"error": "unknown clip"}, 404)
                        if clip["phase"] == "audit" and not state["rules_frozen"]:
                            raise AnnotationError("freeze D rules before A clips")
                        if clip["phase"] == "development" and state["rules_frozen"]:
                            raise AnnotationError("D clips are read-only after freezing")
                        state["clips"][clip_id] = validate_clip_update(payload, float(clip["duration_seconds"]))
                    else:
                        return self._json({"error": "not found"}, 404)
                    write_state(state)
                return self._json({"ok": True, "state": state})
            except (AnnotationError, json.JSONDecodeError) as error:
                return self._json({"error": str(error)}, 400)

    return Handler


def main():
    global OUTPUT, STATE_FILE
    parser = argparse.ArgumentParser(description="Run blind local audio review webpage")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    parser.add_argument("--output-dir", type=Path, help="annotation output directory (mainly for isolated review sessions)")
    args = parser.parse_args()
    if args.output_dir:
        OUTPUT = args.output_dir.resolve()
        STATE_FILE = OUTPUT / "review_state.json"
    review_dir, clips = load_manifest()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(review_dir, clips))
    print(f"Open http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
