from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

from presentation_assessment.research.privacy import (
    AuthorisationError,
    find_staging_violations,
    load_authorisation,
    require_full_run_authorisation,
)


NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)
FIXTURE = Path("tests/fixtures/research/valid_authorisation.yaml")


def initialise_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)


def stage_text(root: Path, relative_path: str, contents: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    subprocess.run(["git", "add", "--", relative_path], cwd=root, check=True)
    return path


def write_authorisation(path: Path, **overrides: object) -> Path:
    fields: dict[str, object] = {
        "authorised_by": "Research supervisor",
        "source": "Advisor-provided presentation table",
        "purpose": "Local thesis research",
        "data_scope": ["private_advisor_video_scores"],
        "permitted_access": ["researcher"],
        "storage": "Encrypted local storage",
        "retention_until": "2026-12-31",
        "deletion_plan": "Delete private inputs after retention ends",
        "ethics_status": "user_authorised_local_research",
        "acknowledged_at": "2026-08-01T00:00:00+00:00",
    }
    fields.update(overrides)
    path.write_text("\n".join(f"{key}: {value}" for key, value in fields.items()), encoding="utf-8")
    return path


def test_full_run_rejects_missing_authorisation(tmp_path: Path) -> None:
    with pytest.raises(AuthorisationError, match="data_authorisation.yaml"):
        require_full_run_authorisation(tmp_path / "missing.yaml", NOW)


def test_full_run_accepts_current_authorisation() -> None:
    auth = require_full_run_authorisation(FIXTURE, NOW)

    assert auth.ethics_status == "user_authorised_local_research"


def test_authorisation_rejects_missing_required_field(tmp_path: Path) -> None:
    path = write_authorisation(tmp_path / "data_authorisation.yaml", purpose="")

    with pytest.raises(AuthorisationError, match="purpose"):
        load_authorisation(path)


def test_full_run_rejects_expired_retention(tmp_path: Path) -> None:
    path = write_authorisation(tmp_path / "data_authorisation.yaml", retention_until="2026-08-24")

    with pytest.raises(AuthorisationError, match="retention"):
        require_full_run_authorisation(path, NOW)


def test_full_run_rejects_naive_current_time() -> None:
    with pytest.raises(AuthorisationError, match="timezone"):
        require_full_run_authorisation(FIXTURE, datetime(2026, 8, 25))


@pytest.mark.parametrize("ethics_status", ["processing_denied", "unknown"])
def test_full_run_requires_an_affirmative_authorisation_status(
    tmp_path: Path, ethics_status: str
) -> None:
    path = write_authorisation(
        tmp_path / "data_authorisation.yaml",
        ethics_status=ethics_status,
    )

    with pytest.raises(AuthorisationError, match="affirmative"):
        require_full_run_authorisation(path, NOW)


def test_full_run_rejects_an_empty_authorisation_status(tmp_path: Path) -> None:
    path = write_authorisation(tmp_path / "data_authorisation.yaml", ethics_status="")

    with pytest.raises(AuthorisationError):
        require_full_run_authorisation(path, NOW)


@pytest.mark.parametrize("scope", [[], [""], ["unknown"]])
def test_authorisation_rejects_empty_or_unknown_scope(tmp_path: Path, scope: list[str]) -> None:
    path = write_authorisation(tmp_path / "data_authorisation.yaml", data_scope=scope)

    with pytest.raises(AuthorisationError, match="data_scope"):
        load_authorisation(path)


def test_staging_scan_rejects_private_tree(tmp_path: Path) -> None:
    violations = find_staging_violations(tmp_path, [Path("data/private/scores.csv")])

    assert violations[0].code == "private_path"


@pytest.mark.parametrize(
    ("staged_path", "contents", "code"),
    [
        ("data/raw/video.mp4", "", "raw_media"),
        ("artifacts/checkpoints/model.ckpt", "", "checkpoint"),
        ("notebooks/.ipynb_checkpoints/analysis.ipynb", "{}", "checkpoint"),
        ("data/transcripts/run.jsonl", "{}\n", "transcript_jsonl"),
        (  # privacy-scan: synthetic-fixture
            "data/public/notes.txt",
            "contact: private@research.test\n",  # privacy-scan: synthetic-fixture
            "email_address",
        ),
        (  # privacy-scan: synthetic-fixture
            "docs/methods.md",
            "source: https://www.youtube.com/watch?v=redacted\n",  # privacy-scan: synthetic-fixture
            "youtube_url",
        ),
        (
            "docs/embed.md",  # privacy-scan: synthetic-fixture
            "source: https://www.youtube-nocookie.com/embed/redacted\n",
            "youtube_url",
        ),
        (  # privacy-scan: synthetic-fixture
            "data/public/scores.csv",
            "student_id,score\nS01,91\n",  # privacy-scan: synthetic-fixture
            "private_score",  # privacy-scan: synthetic-fixture
        ),
        (  # privacy-scan: synthetic-fixture
            "data/public/advisor_scores.csv",
            "video_id,advisor_score\nS01,91\n",  # privacy-scan: synthetic-fixture
            "private_score",  # privacy-scan: synthetic-fixture
        ),
    ],
)
def test_staging_scan_rejects_sensitive_payloads(
    tmp_path: Path, staged_path: str, contents: str, code: str
) -> None:
    initialise_repository(tmp_path)
    stage_text(tmp_path, staged_path, contents)

    violations = find_staging_violations(tmp_path, [Path(staged_path)])

    assert [violation.code for violation in violations] == [code]


@pytest.mark.parametrize(
    ("contents", "code"),
    [
        (  # privacy-scan: synthetic-fixture
            "contact: alex.rook@research.example\n",
            "email_address",
        ),
        (  # privacy-scan: synthetic-fixture
            "source: https://www.youtube.com/watch?v=private\n",
            "youtube_url",
        ),
        (  # privacy-scan: synthetic-fixture
            "student_id,score\nS01,91\n",  # privacy-scan: synthetic-fixture
            "private_score",  # privacy-scan: synthetic-fixture
        ),
    ],
)
def test_staging_scan_rejects_sensitive_fixture_payloads(
    tmp_path: Path, contents: str, code: str
) -> None:
    initialise_repository(tmp_path)
    stage_text(tmp_path, "tests/fixtures/research/unredacted.yaml", contents)

    violations = find_staging_violations(tmp_path, [Path("tests/fixtures/research/unredacted.yaml")])

    assert [violation.code for violation in violations] == [code]


def test_staging_scan_rejects_paths_outside_repository(tmp_path: Path) -> None:
    violations = find_staging_violations(tmp_path, [Path("../outside.csv")])

    assert violations[0].code == "outside_repository"


def test_staging_scan_rejects_drive_letter_absolute_paths(tmp_path: Path) -> None:
    violations = find_staging_violations(  # privacy-scan: synthetic-fixture
        tmp_path, [Path("C:/private/scores.csv")]
    )

    assert violations[0].code == "absolute_path"


def test_staging_scan_reads_the_staged_blob_not_the_working_tree(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    path = stage_text(
        tmp_path,
        "docs/source.md",
        "contact: person@research.example\n",  # privacy-scan: synthetic-fixture
    )
    path.write_text("redacted content\n", encoding="utf-8")

    violations = find_staging_violations(tmp_path, [Path("docs/source.md")])

    assert [violation.code for violation in violations] == ["email_address"]


def test_staged_blob_cannot_be_redirected_by_worktree_symlink(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    staged = stage_text(
        tmp_path,
        "docs/private.md",
        "contact: person@research.example\n",  # privacy-scan: synthetic-fixture
    )
    staged.unlink()
    target = tmp_path / "clean.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        staged.symlink_to(target)
    except OSError:
        pytest.skip("platform cannot create a test symlink")

    violations = find_staging_violations(tmp_path, [Path("docs/private.md")])

    assert [violation.code for violation in violations] == ["email_address"]


def test_staging_scan_rejects_lexical_parent_before_reading_index(tmp_path: Path) -> None:
    read_paths: list[Path] = []

    def read_staged_blob(root: Path, path: Path) -> bytes:
        read_paths.append(path)
        return b"contact: person@research.example\n"  # privacy-scan: synthetic-fixture

    violations = find_staging_violations(
        tmp_path,
        [Path("docs/../private.md")],
        staged_blob_reader=read_staged_blob,
    )

    assert [violation.code for violation in violations] == ["outside_repository"]
    assert read_paths == []


def test_staging_scan_rejects_root_qualified_path_before_reading_index(
    tmp_path: Path,
) -> None:
    read_paths: list[Path] = []

    def read_staged_blob(root: Path, path: Path) -> bytes:
        read_paths.append(path)
        return b"contact: person@research.example\n"  # privacy-scan: synthetic-fixture

    root_qualified = Path("/docs/private.md")
    assert root_qualified.root

    violations = find_staging_violations(
        tmp_path,
        [root_qualified],
        staged_blob_reader=read_staged_blob,
    )

    assert [violation.code for violation in violations] == ["absolute_path"]
    assert read_paths == []


def test_staging_scan_fails_closed_when_staged_blob_is_unreadable(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    path = tmp_path / "docs" / "unstaged.md"
    path.parent.mkdir(parents=True)
    path.write_text("redacted content\n", encoding="utf-8")

    violations = find_staging_violations(tmp_path, [Path("docs/unstaged.md")])

    assert [violation.code for violation in violations] == ["staged_blob_unreadable"]


@pytest.mark.parametrize(
    "contents",
    [
        "cache: C:\\Research\\local\\records.csv\n",  # privacy-scan: synthetic-fixture
        "cache: \\\\research-server\\share\\records.csv\n",  # privacy-scan: synthetic-fixture
    ],
)
def test_staging_scan_rejects_local_absolute_paths_in_payloads(
    tmp_path: Path, contents: str
) -> None:
    initialise_repository(tmp_path)
    stage_text(tmp_path, "configs/run.yaml", contents)

    violations = find_staging_violations(tmp_path, [Path("configs/run.yaml")])

    assert [violation.code for violation in violations] == ["local_path"]


@pytest.mark.parametrize(
    "relative_path, contents",
    [
        ("data/public/row_scores.csv", "sample_id,total_score\nN001,91\n"),
        (
            "tests/fixtures/research/row_scores.json",
            '[{"sample_id":"N001","total_score":91}]',
        ),
    ],
)
def test_staging_scan_rejects_row_level_score_exports(
    tmp_path: Path, relative_path: str, contents: str
) -> None:
    initialise_repository(tmp_path)
    stage_text(tmp_path, relative_path, contents)

    violations = find_staging_violations(tmp_path, [Path(relative_path)])

    assert [violation.code for violation in violations] == [  # privacy-scan: synthetic-fixture
        "private_score"
    ]


def test_staging_scan_allows_aggregate_charts_redacted_fixtures_and_safe_python_source(
    tmp_path: Path,
) -> None:
    initialise_repository(tmp_path)
    stage_text(
        tmp_path,
        "notebooks/aggregate_chart.ipynb",
        '{"mean_score": 82.4, "sample_count": 15}',
    )
    stage_text(
        tmp_path,
        "tests/fixtures/research/redacted.yaml",
        "contact: redacted@example.invalid\n",
    )
    stage_text(
        tmp_path,
        "src/example.py",
        "CONTACT = 'private@example.invalid'\n",
    )
    violations = find_staging_violations(
        tmp_path,
        [
            Path("notebooks/aggregate_chart.ipynb"),
            Path("tests/fixtures/research/redacted.yaml"),
            Path("src/example.py"),
        ],
    )

    assert violations == []


def test_staging_scan_rejects_private_shaped_payload_in_production_python(
    tmp_path: Path,
) -> None:
    """Break caught: Python source bypasses the staged-blob payload scanner."""
    initialise_repository(tmp_path)
    private_domain = ".".join(("research", "test"))
    private_email = "alex.rook" + "@" + private_domain
    stage_text(
        tmp_path,
        "src/private_payload.py",
        f"CONTACT = {private_email!r}\n",
    )

    violations = find_staging_violations(
        tmp_path, [Path("src/private_payload.py")]
    )

    assert [violation.code for violation in violations] == ["email_address"]


def test_synthetic_fixture_marker_cannot_allowlist_arbitrary_production_python(
    tmp_path: Path,
) -> None:
    """Break caught: a production file self-labels private content as a detector fixture."""
    initialise_repository(tmp_path)
    private_domain = ".".join(("research", "test"))
    private_email = "alex.rook" + "@" + private_domain
    marker = "privacy-scan" + ": synthetic-fixture"
    stage_text(
        tmp_path,
        "src/private_payload.py",
        f"CONTACT = {private_email!r}  # {marker}\n",
    )

    violations = find_staging_violations(
        tmp_path, [Path("src/private_payload.py")]
    )

    assert [violation.code for violation in violations] == ["email_address"]


@pytest.mark.parametrize("payload_kind", ["score", "transcript"])
def test_staging_scan_rejects_structured_private_python_payloads(
    tmp_path: Path, payload_kind: str
) -> None:
    """Break caught: row scores or transcript text hide inside Python literals."""
    initialise_repository(tmp_path)
    if payload_kind == "score":
        sample_key = "sample" + "_id"
        score_key = "total" + "_score"
        contents = f"PAYLOAD = {{{sample_key!r}: 'N417', {score_key!r}: 91.0}}\n"
        expected = "private_score"  # privacy-scan: synthetic-fixture
    else:
        private_text = " ".join(
            ("Confidential", "participant", "speech", "content", "belongs", "offline")
        )
        contents = f"TRANSCRIPT = {private_text!r}\n"
        expected = "transcript_payload"
    stage_text(tmp_path, "src/private_payload.py", contents)

    violations = find_staging_violations(
        tmp_path, [Path("src/private_payload.py")]
    )

    assert [violation.code for violation in violations] == [expected]


def test_private_and_artifact_paths_are_ignored() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    for rule in (
        "data/private/",
        "artifacts/research/",
        "artifacts/models/",
        "notebooks/.ipynb_checkpoints/",
    ):
        assert rule in text


def test_pipeline_config_is_non_secret() -> None:
    text = Path("configs/notebook_pipeline.yaml").read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text  # privacy-scan: synthetic-fixture
    assert "source_manifest: data/private/advisor_video_scores.csv" in text
    assert "whisper_model: Systran/faster-whisper-large-v3" in text
    assert "whisper_revision: edaa852ec7e145841d8ffdb056a99866b5f0a478" in text
    assert "whisper_snapshot_sha256: ae1b67ef9f88f5cec7f51cdf198a163419c5a7529ae85c7ecac18d5cebd98511" in text
    assert "whisper_device: cuda" in text
    assert "whisper_compute_type: float16" in text
    assert "whisper_beam_size: 5" in text
    assert "whisper_vad: true" in text
    assert "whisper_word_timestamps: true" in text
