from pathlib import Path

from presentation_assessment.video_pipeline import build_manifest_from_videos


def test_build_manifest_from_videos(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"")
    (video_dir / "ignore.txt").write_text("no", encoding="utf-8")
    output = tmp_path / "manifest.csv"

    build_manifest_from_videos(video_dir, output)

    text = output.read_text(encoding="utf-8")
    assert "a.mp4" in text
    assert "speech_delivery_score" in text
