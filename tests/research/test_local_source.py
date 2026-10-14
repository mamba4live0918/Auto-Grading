from __future__ import annotations

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import shutil
import subprocess
import threading

import pytest
import yaml

from presentation_assessment.research.acquisition import AcquisitionStatus
from presentation_assessment.research.local_source import (
    LOCAL_CATALOG_FIELDS,
    LocalAudioAdapter,
    LocalAvailability,
    LocalCatalogRow,
    LocalFileAcquirer,
    LocalSourceCatalog,
    LocalSourceConfig,
    LocalSourceResolver,
    load_local_catalog,
    load_local_source_config,
    local_source_reference_valid,
)
from presentation_assessment.research.source_manifest import (
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


VIDEO_ID = "abcdefghijk"
SOURCE_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"  # privacy-scan: synthetic-fixture


def _write_local_fixture(
    root: Path,
    *,
    row_overrides: dict[str, str] | None = None,
    config_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, bytes]:
    video_root = root / "private-videos"
    video_root.mkdir(parents=True)
    media = b"synthetic mp4 fixture"
    (video_root / f"{VIDEO_ID}.mp4").write_bytes(media)
    row = {
        "local_row_id": "local-r001",
        "source_url": SOURCE_URL,
        "video_id": VIDEO_ID,
        "video_title": "Synthetic local source",
        "total_score": "81.5",
        "duration_seconds": "12.5",
        "local_filename": f"{VIDEO_ID}.mp4",
        "source_sha256": hashlib.sha256(media).hexdigest(),
        "status": "available",
    }
    row.update(row_overrides or {})
    catalog_path = root / "private-catalog.csv"
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=LOCAL_CATALOG_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    catalog_path.write_text(handle.getvalue(), encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "video_root": str(video_root.resolve()),
        "catalog_path": str(catalog_path.resolve()),
        "expected_total_rows": 1,
        "expected_available_rows": 1,
        "expected_unavailable_rows": 0,
        "duration_tolerance_seconds": 0.25,
    }
    payload.update(config_overrides or {})
    config_path = root / "local-source.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path, video_root, media


def _source_row() -> SourceRow:
    return SourceRow(
        source_row_id="local-r001",
        source_page="local_catalog",
        sample_id="N001",
        legacy_sample_id="",
        source_url=SOURCE_URL,
        video_id=VIDEO_ID,
        video_title="Synthetic local source",
        total_score=81.5,
        score_status=ScoreStatus.OBSERVED,
        source_course="",
        verification_status=VerificationStatus.VERIFIED,
        verification_note="",
    )


def test_config_requires_absolute_private_paths_and_exact_schema(tmp_path: Path) -> None:
    """Break caught: runtime silently resolves a relative or undeclared private path."""
    config_path, _, _ = _write_local_fixture(
        tmp_path, config_overrides={"video_root": "relative/videos"}
    )

    with pytest.raises(ValueError, match="video_root must be absolute"):
        load_local_source_config(config_path)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["video_root"] = str((tmp_path / "private-videos").resolve())
    payload["private_note"] = "must not be accepted"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact schema"):
        load_local_source_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_url", "http://www.youtube.com/watch?v=abcdefghijk", "HTTPS YouTube"),
        ("video_id", "short", "11-character"),
        ("total_score", "100.1", "between 0 and 100"),
        ("local_filename", "nested/abcdefghijk.mp4", "basename-only"),
        ("local_filename", "abcdefghijk.webm", r"\.mp4"),
    ],
)
def test_catalog_rejects_invalid_identity_score_and_filename(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    """Break caught: malformed catalog identity or traversal reaches local media access."""
    config_path, _, _ = _write_local_fixture(tmp_path, row_overrides={field: value})
    config = load_local_source_config(config_path)

    with pytest.raises(ValueError, match=message):
        load_local_catalog(config, duration_probe=lambda _: 12.5)


def test_available_catalog_accepts_unique_safe_mp4_basename_not_derived_from_video_id(
    tmp_path: Path,
) -> None:
    """Break caught: a valid read-only corpus is rejected unless files are renamed."""
    local_filename = "approved-corpus-file.mp4"
    config_path, video_root, media = _write_local_fixture(
        tmp_path, row_overrides={"local_filename": local_filename}
    )
    (video_root / f"{VIDEO_ID}.mp4").rename(video_root / local_filename)

    catalog = load_local_catalog(
        load_local_source_config(config_path), duration_probe=lambda _: 12.5
    )
    resolver = LocalSourceResolver(catalog)

    assert catalog.rows[0].local_filename == local_filename
    assert resolver.resolve(
        resolver.locator_for("local-r001"), catalog.rows[0].source_sha256
    ).read_bytes() == media


def test_available_catalog_hashes_and_probes_media_without_persisting_root(
    tmp_path: Path,
) -> None:
    """Break caught: an available row is trusted without validating file bytes and duration."""
    config_path, video_root, _ = _write_local_fixture(tmp_path)
    config = load_local_source_config(config_path)
    probed: list[Path] = []

    catalog = load_local_catalog(
        config, duration_probe=lambda path: probed.append(path) or 12.5
    )

    assert probed == [video_root / f"{VIDEO_ID}.mp4"]
    assert catalog.available_count == 1
    assert catalog.unavailable_count == 0
    assert str(video_root) not in catalog.identity_sha256
    assert catalog.rows[0].local_filename == f"{VIDEO_ID}.mp4"


def test_catalog_rejects_file_hash_or_duration_mismatch(tmp_path: Path) -> None:
    """Break caught: catalog metadata can point at changed media."""
    config_path, _, _ = _write_local_fixture(
        tmp_path, row_overrides={"source_sha256": "0" * 64}
    )
    config = load_local_source_config(config_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_local_catalog(config, duration_probe=lambda _: 12.5)

    config_path, _, _ = _write_local_fixture(
        tmp_path / "duration", row_overrides={"duration_seconds": "9.0"}
    )
    config = load_local_source_config(config_path)
    with pytest.raises(ValueError, match="duration mismatch"):
        load_local_catalog(config, duration_probe=lambda _: 12.5)


def test_available_catalog_requires_strictly_positive_measured_duration(
    tmp_path: Path,
) -> None:
    """Break caught: a zero-duration source is accepted as available media."""
    config_path, _, _ = _write_local_fixture(
        tmp_path, row_overrides={"duration_seconds": "0"}
    )

    with pytest.raises(ValueError, match="greater than zero"):
        load_local_catalog(
            load_local_source_config(config_path), duration_probe=lambda _: 0.0
        )


def test_catalog_rejects_media_mutated_during_probe(tmp_path: Path) -> None:
    """Break caught: bytes can change after the pre-probe hash and still enter the catalog."""
    config_path, video_root, _ = _write_local_fixture(tmp_path)
    config = load_local_source_config(config_path)

    def mutate_during_probe(path: Path) -> float:
        path.write_bytes(b"mutated during catalog probe")
        return 12.5

    with pytest.raises(ValueError, match="changed during probe"):
        load_local_catalog(config, duration_probe=mutate_during_probe)

    assert (video_root / f"{VIDEO_ID}.mp4").read_bytes() == b"mutated during catalog probe"


def test_catalog_rejects_symlink_media_even_when_target_is_inside_root(tmp_path: Path) -> None:
    """Break caught: an accepted media entry can be retargeted through a symbolic link."""
    config_path, video_root, media = _write_local_fixture(tmp_path)
    media_path = video_root / f"{VIDEO_ID}.mp4"
    target = video_root / "stable-target.mp4"
    target.write_bytes(media)
    media_path.unlink()
    try:
        media_path.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this Windows test environment")

    with pytest.raises(ValueError, match="symbolic links"):
        load_local_catalog(
            load_local_source_config(config_path), duration_probe=lambda _: 12.5
        )


def test_unavailable_row_is_explicit_and_does_not_invent_media_fields(tmp_path: Path) -> None:
    """Break caught: an unavailable source acquires fabricated score, video, or file identity."""
    config_path, video_root, _ = _write_local_fixture(
        tmp_path,
        row_overrides={
            "video_title": "Unavailable source",
            "total_score": "",
            "duration_seconds": "",
            "local_filename": "",
            "source_sha256": "",
            "status": "unavailable",
        },
        config_overrides={
            "expected_available_rows": 0,
            "expected_unavailable_rows": 1,
        },
    )
    (video_root / f"{VIDEO_ID}.mp4").unlink()
    catalog = load_local_catalog(
        load_local_source_config(config_path),
        duration_probe=lambda _: pytest.fail("unavailable media must not be probed"),
    )

    row = catalog.rows[0]
    assert catalog.unavailable_count == 1
    assert row.source_url == SOURCE_URL
    assert row.video_id == VIDEO_ID
    assert row.total_score is None
    assert row.local_filename == ""
    assert row.source_sha256 == ""


def test_unavailable_media_preserves_an_authoritative_score(tmp_path: Path) -> None:
    """Break caught: media availability incorrectly erases an independent score."""
    config_path, video_root, _ = _write_local_fixture(
        tmp_path,
        row_overrides={
            "total_score": "73.0",
            "duration_seconds": "",
            "local_filename": "",
            "source_sha256": "",
            "status": "unavailable",
        },
        config_overrides={
            "expected_available_rows": 0,
            "expected_unavailable_rows": 1,
        },
    )
    (video_root / f"{VIDEO_ID}.mp4").unlink()

    catalog = load_local_catalog(
        load_local_source_config(config_path),
        duration_probe=lambda _: pytest.fail("unavailable media must not be probed"),
    )

    assert catalog.rows[0].total_score == 73.0
    assert catalog.rows[0].duration_seconds is None
    assert catalog.rows[0].local_filename == ""
    assert catalog.rows[0].source_sha256 == ""


def test_unavailable_row_requires_verified_youtube_identity(tmp_path: Path) -> None:
    """Break caught: an unavailable row silently drops its authoritative URL/video identity."""
    config_path, video_root, _ = _write_local_fixture(
        tmp_path,
        row_overrides={
            "source_url": "",
            "video_id": "",
            "video_title": "Unavailable source",
            "total_score": "",
            "duration_seconds": "",
            "local_filename": "",
            "source_sha256": "",
            "status": "unavailable",
        },
        config_overrides={
            "expected_available_rows": 0,
            "expected_unavailable_rows": 1,
        },
    )
    (video_root / f"{VIDEO_ID}.mp4").unlink()

    with pytest.raises(ValueError, match="unavailable rows require YouTube identity"):
        load_local_catalog(
            load_local_source_config(config_path), duration_probe=lambda _: 12.5
        )


def test_resolver_contains_paths_and_rechecks_sha256(tmp_path: Path) -> None:
    """Break caught: an opaque locator escapes the root or resolves changed bytes."""
    config_path, video_root, _ = _write_local_fixture(tmp_path)
    catalog = load_local_catalog(
        load_local_source_config(config_path), duration_probe=lambda _: 12.5
    )
    resolver = LocalSourceResolver(catalog)
    locator = resolver.locator_for("local-r001")

    assert str(video_root) not in locator
    assert resolver.resolve(locator, catalog.rows[0].source_sha256).name == f"{VIDEO_ID}.mp4"
    (video_root / f"{VIDEO_ID}.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolver.resolve(locator, catalog.rows[0].source_sha256)
    with pytest.raises(ValueError, match="invalid local source locator"):
        resolver.resolve("local-file:../escape.mp4", catalog.rows[0].source_sha256)


@pytest.mark.parametrize(
    "locator",
    [
        "local-file/local-r999/approved corpus.mp4",
        "other/local-r001/approved corpus.mp4",
        "local-file/../approved corpus.mp4",
        "local-file/./approved corpus.mp4",
        "/local-file/local-r001/approved corpus.mp4",
        "local-file/local-r001/nested/approved corpus.mp4",
        "local-file\\local-r001\\approved corpus.mp4",
    ],
)
def test_local_reference_validator_rejects_wrong_row_and_noncanonical_tokens(
    locator: str,
) -> None:
    """Break caught: a cache locator can change rows or hide traversal lexically."""
    payload = {
        "source_kind": "local_file",
        "source_locator": locator,
        "local_basename": "approved corpus.mp4",
        "source_sha256": "a" * 64,
    }

    assert local_source_reference_valid(
        payload,
        expected_local_row_id="local-r001",
        expected_basename="approved corpus.mp4",
        expected_source_sha256="a" * 64,
    ) is False


def test_local_reference_validator_accepts_exact_approved_lexical_reference() -> None:
    """A safe descriptive basename remains valid without exposing a private path."""
    payload = {
        "source_kind": "local_file",
        "source_locator": "local-file/local-r001/approved corpus.mp4",
        "local_basename": "approved corpus.mp4",
        "source_sha256": "a" * 64,
    }

    assert local_source_reference_valid(
        payload,
        expected_local_row_id="local-r001",
        expected_basename="approved corpus.mp4",
        expected_source_sha256="a" * 64,
    ) is True
    assert local_source_reference_valid(
        payload,
        expected_local_row_id="legacy-r001",
        expected_basename="approved corpus.mp4",
        expected_source_sha256="a" * 64,
    ) is False


@pytest.mark.parametrize(
    "updates",
    [
        {"local_basename": "other approved.mp4"},
        {
            "local_basename": "other approved.mp4",
            "source_locator": "local-file/local-r001/other approved.mp4",
        },
        {"source_sha256": "b" * 64},
    ],
)
def test_local_reference_validator_binds_trusted_catalog_basename_and_hash(
    updates: dict[str, str],
) -> None:
    """Break caught: payload fields can agree with each other but drift from the catalog."""
    payload = {
        "source_kind": "local_file",
        "source_locator": "local-file/local-r001/approved corpus.mp4",
        "local_basename": "approved corpus.mp4",
        "source_sha256": "a" * 64,
        **updates,
    }

    assert local_source_reference_valid(
        payload,
        expected_local_row_id="local-r001",
        expected_basename="approved corpus.mp4",
        expected_source_sha256="a" * 64,
    ) is False


def test_local_acquisition_and_audio_use_opaque_locator_offline(tmp_path: Path) -> None:
    """Break caught: local mode copies media, permits network, or passes a private path to metadata."""
    config_path, video_root, media = _write_local_fixture(tmp_path)
    catalog = load_local_catalog(
        load_local_source_config(config_path), duration_probe=lambda _: 12.5
    )
    resolver = LocalSourceResolver(catalog)
    acquired = LocalFileAcquirer(resolver).acquire(
        _source_row(), tmp_path / "unused-output", allow_network=True
    )

    assert acquired.status is AcquisitionStatus.ACQUIRED
    assert acquired.source_kind == "local_file"
    assert acquired.source_path is None
    assert acquired.local_basename == f"{VIDEO_ID}.mp4"
    assert str(video_root) not in repr(acquired)
    assert not (tmp_path / "unused-output").exists()

    calls: list[tuple[Path, Path]] = []

    def build(source: Path, output: Path) -> dict[str, object]:
        calls.append((source, output))
        assert source.read_bytes() == media
        return {"status": "created"}

    adapter = LocalAudioAdapter(resolver, builder=build)
    result = adapter(
        {
            "source_kind": acquired.source_kind,
            "source_locator": acquired.source_locator,
            "source_sha256": acquired.source_sha256,
        },
        tmp_path / "audio",
    )

    assert result == {"status": "created"}
    assert calls == [(video_root / f"{VIDEO_ID}.mp4", tmp_path / "audio")]


def test_local_acquisition_rejects_stale_source_row_identity(tmp_path: Path) -> None:
    """Break caught: a stale manifest row acquires another catalog row's local file."""
    config_path, _, _ = _write_local_fixture(tmp_path)
    catalog = load_local_catalog(
        load_local_source_config(config_path), duration_probe=lambda _: 12.5
    )
    stale_video_id = "stalevid001"
    stale = SourceRow(
        source_row_id="local-r001",
        source_page="local_catalog",
        sample_id="N001",
        legacy_sample_id="",
        source_url=f"https://www.youtube.com/watch?v={stale_video_id}",  # privacy-scan: synthetic-fixture
        video_id=stale_video_id,
        video_title="Stale local source",
        total_score=81.5,
        score_status=ScoreStatus.OBSERVED,
        source_course="",
        verification_status=VerificationStatus.VERIFIED,
        verification_note="",
    )

    result = LocalFileAcquirer(LocalSourceResolver(catalog)).acquire(
        stale, tmp_path / "unused", allow_network=False
    )

    assert result.status is AcquisitionStatus.FAILED
    assert result.error_code == "local_source_identity_mismatch"


def test_local_audio_rejects_source_drift_and_discards_derived_outputs(
    tmp_path: Path,
) -> None:
    """Break caught: audio artifacts survive when the source mutates during their build."""
    config_path, video_root, _ = _write_local_fixture(tmp_path)
    catalog = load_local_catalog(
        load_local_source_config(config_path), duration_probe=lambda _: 12.5
    )
    resolver = LocalSourceResolver(catalog)
    output_dir = tmp_path / "audio"

    def mutate_during_build(source: Path, output: Path) -> dict[str, object]:
        output.mkdir(parents=True)
        (output / "analysis.wav").write_bytes(b"derived from stale bytes")
        source.write_bytes(b"mutated during audio build")
        return {"status": "created"}

    adapter = LocalAudioAdapter(resolver, builder=mutate_during_build)

    with pytest.raises(ValueError, match="changed during audio build"):
        adapter(
            {
                "source_kind": "local_file",
                "source_locator": resolver.locator_for("local-r001"),
                "source_sha256": catalog.rows[0].source_sha256,
            },
            output_dir,
        )

    assert not (output_dir / "analysis.wav").exists()
    assert (video_root / f"{VIDEO_ID}.mp4").read_bytes() == b"mutated during audio build"


def test_default_catalog_probe_rejects_network_playlist_without_any_request(
    tmp_path: Path,
) -> None:
    """Break caught: FFprobe follows a network URL embedded in a local .mp4 source."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for the default-path protocol test")
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp2t")
            self.end_headers()
            self.wfile.write(b"synthetic network segment")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        video_root = tmp_path / "videos"
        video_root.mkdir()
        port = server.server_address[1]
        media_path = video_root / f"{VIDEO_ID}.mp4"
        media_path.write_text(
            "\n".join(
                (
                    "#EXTM3U",
                    "#EXT-X-VERSION:3",
                    "#EXT-X-TARGETDURATION:1",
                    "#EXTINF:1.0,",
                    f"http://127.0.0.1:{port}/segment.ts",  # privacy-scan: synthetic-fixture
                    "#EXT-X-ENDLIST",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        catalog_path = tmp_path / "catalog.csv"
        _write_catalog_file(
            catalog_path,
            {
                "local_row_id": "local-r001",
                "source_url": SOURCE_URL,
                "video_id": VIDEO_ID,
                "video_title": "Synthetic network playlist",
                "total_score": "81.5",
                "duration_seconds": "",
                "local_filename": media_path.name,
                "source_sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
                "status": "available",
            },
        )
        config = _direct_config(video_root, catalog_path)

        with pytest.raises(Exception) as captured:
            load_local_catalog(config)

        assert getattr(captured.value, "code", "") == "ffprobe_failed"
        assert requests == []
        assert str(port) not in str(captured.value)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_default_catalog_probe_preserves_ordinary_mp4_success(tmp_path: Path) -> None:
    """Break caught: the offline protocol policy also blocks an ordinary local MP4."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the default-path media test")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    media_path = video_root / f"{VIDEO_ID}.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-c:a",
            "aac",
            str(media_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        pytest.skip("local ffmpeg cannot encode the synthetic AAC fixture")
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog_file(
        catalog_path,
        {
            "local_row_id": "local-r001",
            "source_url": SOURCE_URL,
            "video_id": VIDEO_ID,
            "video_title": "Synthetic ordinary MP4",
            "total_score": "81.5",
            "duration_seconds": "",
            "local_filename": media_path.name,
            "source_sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
            "status": "available",
        },
    )

    catalog = load_local_catalog(_direct_config(video_root, catalog_path))

    assert catalog.rows[0].duration_seconds is not None
    assert catalog.rows[0].duration_seconds > 0
    resolver = LocalSourceResolver(catalog)
    audio = LocalAudioAdapter(resolver)(
        {
            "source_kind": "local_file",
            "source_locator": resolver.locator_for("local-r001"),
            "source_sha256": catalog.rows[0].source_sha256,
        },
        tmp_path / "audio",
    )
    assert audio.analysis.status.value == "created"


def test_default_audio_builder_rejects_network_playlist_without_any_request(
    tmp_path: Path,
) -> None:
    """Break caught: default audio conversion follows a network URL inside local media."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the default-path protocol test")
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"synthetic network segment")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        video_root = tmp_path / "videos"
        video_root.mkdir()
        media_path = video_root / f"{VIDEO_ID}.mp4"
        port = server.server_address[1]
        media_path.write_text(
            "\n".join(
                (
                    "#EXTM3U",
                    "#EXT-X-TARGETDURATION:1",
                    "#EXTINF:1.0,",
                    f"http://127.0.0.1:{port}/segment.ts",  # privacy-scan: synthetic-fixture
                    "#EXT-X-ENDLIST",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(media_path.read_bytes()).hexdigest()
        row = LocalCatalogRow(
            local_row_id="local-r001",
            source_url=SOURCE_URL,
            video_id=VIDEO_ID,
            video_title="Synthetic audio network playlist",
            total_score=81.5,
            duration_seconds=1.0,
            local_filename=media_path.name,
            source_sha256=digest,
            status=LocalAvailability.AVAILABLE,
        )
        catalog = LocalSourceCatalog(
            LocalSourceConfig(video_root, tmp_path / "catalog.csv", 1, 1, 0, 0.25),
            (row,),
            "a" * 64,
        )
        resolver = LocalSourceResolver(catalog)

        result = LocalAudioAdapter(resolver)(
            {
                "source_kind": "local_file",
                "source_locator": resolver.locator_for("local-r001"),
                "source_sha256": digest,
            },
            tmp_path / "audio",
        )

        assert result.analysis.status.value == "failed"
        assert result.analysis.error_code == "ffmpeg_conversion_failed"
        assert requests == []
        assert str(port) not in (result.analysis.error_summary or "")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _write_catalog_file(path: Path, row: dict[str, str]) -> None:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=LOCAL_CATALOG_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    path.write_text(handle.getvalue(), encoding="utf-8")


def _direct_config(video_root: Path, catalog_path: Path):
    return LocalSourceConfig(video_root, catalog_path, 1, 1, 0, 0.25)
