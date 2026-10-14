import json
from dataclasses import replace
from pathlib import Path

from presentation_assessment.research.acquisition import AcquisitionStatus, YouTubeAcquirer
from presentation_assessment.research.source_manifest import (
    ScoreStatus,
    SourceRow,
    VerificationStatus,
)


ROW = SourceRow(
    source_row_id="row-1",
    source_page="1",
    sample_id="N001",
    legacy_sample_id="",
    source_url="https://www.youtube.com/watch?v=abcdefghijk",
    video_id="abcdefghijk",
    video_title="Private source title",
    total_score=81.5,
    score_status=ScoreStatus.OBSERVED,
    source_course="",
    verification_status=VerificationStatus.VERIFIED,
    verification_note="",
)


class FakeRunner:
    def __init__(self, *, success: bool = False, error: str | None = None) -> None:
        self.success = success
        self.error = error
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, output_template: Path) -> Path:
        self.calls.append((url, output_template))
        if self.error is not None:
            raise RuntimeError(self.error)
        output = output_template.with_suffix(".webm")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"private media bytes")
        return output


def test_network_disabled_never_calls_runner(tmp_path: Path) -> None:
    runner = FakeRunner(success=True)

    result = YouTubeAcquirer(runner=runner).acquire(ROW, tmp_path, allow_network=False)

    assert result.status is AcquisitionStatus.BLOCKED
    assert runner.calls == []


def test_one_terminal_failure_does_not_raise(tmp_path: Path) -> None:
    result = YouTubeAcquirer(runner=FakeRunner(error="private_video")).acquire(
        ROW, tmp_path, allow_network=True
    )

    assert result.status is AcquisitionStatus.FAILED
    assert result.error_code == "private_video"
    assert result.attempts == 1


def test_terminal_failure_metadata_redacts_urls_and_windows_paths(tmp_path: Path) -> None:
    leaked_url = "https://www.youtube.com/watch?v=abcdefghijk"
    leaked_path = r"C:\private\advisor\source.webm"
    result = YouTubeAcquirer(
        runner=FakeRunner(error=f"private_video at {leaked_url} stored at {leaked_path}")
    ).acquire(ROW, tmp_path, allow_network=True)

    metadata = (tmp_path / result.metadata_path).read_text(encoding="utf-8")

    assert leaked_url not in metadata
    assert leaked_path not in metadata
    assert "C:" not in metadata


def test_transient_network_failure_retries_with_an_injected_sleeper(tmp_path: Path) -> None:
    runner = FakeRunner(error="network_error")
    delays: list[float] = []

    result = YouTubeAcquirer(runner=runner, max_retries=2, sleeper=delays.append).acquire(
        ROW, tmp_path, allow_network=True
    )

    assert result.status is AcquisitionStatus.FAILED
    assert result.attempts == 3
    assert delays == [1.0, 2.0]


def test_acquired_media_and_metadata_are_relative_and_redacted(tmp_path: Path) -> None:
    result = YouTubeAcquirer(runner=FakeRunner(success=True)).acquire(
        ROW, tmp_path, allow_network=True
    )

    assert result.status is AcquisitionStatus.ACQUIRED
    assert result.source_path == "source/N001/abcdefghijk/abcdefghijk.webm"
    assert result.metadata_path == "source/N001/abcdefghijk/metadata.json"
    assert result.source_sha256 is not None
    metadata = (tmp_path / result.metadata_path).read_text(encoding="utf-8")
    assert ROW.source_url not in metadata
    assert str(tmp_path) not in metadata
    assert json.loads(metadata)["source_path"] == result.source_path


def test_matching_cache_metadata_avoids_a_second_download(tmp_path: Path) -> None:
    runner = FakeRunner(success=True)
    acquirer = YouTubeAcquirer(runner=runner)
    first = acquirer.acquire(ROW, tmp_path, allow_network=True)

    second = acquirer.acquire(ROW, tmp_path, allow_network=True)

    assert first.status is AcquisitionStatus.ACQUIRED
    assert second.status is AcquisitionStatus.CACHED
    assert len(runner.calls) == 1


def test_network_disabled_reuses_a_valid_cache_without_mutating_metadata(tmp_path: Path) -> None:
    runner = FakeRunner(success=True)
    acquirer = YouTubeAcquirer(runner=runner)
    first = acquirer.acquire(ROW, tmp_path, allow_network=True)
    metadata_path = tmp_path / first.metadata_path
    original_metadata = metadata_path.read_bytes()

    offline = acquirer.acquire(ROW, tmp_path, allow_network=False)

    assert offline.status is AcquisitionStatus.CACHED
    assert offline.metadata_path == first.metadata_path
    assert metadata_path.read_bytes() == original_metadata
    assert len(runner.calls) == 1


def test_cache_metadata_cannot_redirect_to_a_source_outside_the_output_root(tmp_path: Path) -> None:
    runner = FakeRunner(success=True)
    acquirer = YouTubeAcquirer(runner=runner)
    first = acquirer.acquire(ROW, tmp_path, allow_network=True)
    escaped = tmp_path.parent / "escaped.webm"
    escaped.write_bytes(b"private media bytes")
    metadata_path = tmp_path / first.metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_path"] = "../escaped.webm"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    second = acquirer.acquire(ROW, tmp_path, allow_network=True)

    assert second.status is AcquisitionStatus.ACQUIRED
    assert len(runner.calls) == 2


def test_ineligible_row_cannot_escape_the_output_root_with_its_sample_id(tmp_path: Path) -> None:
    runner = FakeRunner(success=True)
    unsafe_row = replace(ROW, sample_id="../../outside")
    escaped_metadata = tmp_path.parent / "outside" / ROW.video_id / "metadata.json"

    result = YouTubeAcquirer(runner=runner).acquire(unsafe_row, tmp_path, allow_network=True)

    assert result.status is AcquisitionStatus.FAILED
    assert ".." not in result.metadata_path
    assert not escaped_metadata.exists()
    assert runner.calls == []


def test_metadata_write_failure_returns_a_bounded_terminal_result(
    tmp_path: Path, monkeypatch
) -> None:
    runner = FakeRunner(success=True)

    def fail_metadata_write(*args: object, **kwargs: object) -> None:
        raise OSError(r"C:\private\acquisition-metadata.json")

    monkeypatch.setattr(
        "presentation_assessment.research.acquisition.atomic_write_json", fail_metadata_write
    )

    result = YouTubeAcquirer(runner=runner).acquire(ROW, tmp_path, allow_network=True)

    assert result.status is AcquisitionStatus.FAILED
    assert result.error_code == "metadata_write_failed"
    assert result.metadata_path is None
    assert result.source_path == "source/N001/abcdefghijk/abcdefghijk.webm"
    assert result.source_sha256 is not None
    assert "private" not in (result.error_summary or "").lower()
    assert "C:" not in (result.error_summary or "")
