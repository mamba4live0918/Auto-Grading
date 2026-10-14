from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest

from presentation_assessment.research.canonical import canonical_json_bytes, sha256_bytes
from presentation_assessment.research.protocol import bind_protocol, load_protocol_template
from presentation_assessment.research.qwen import (
    ALLOWED_QWEN_MODEL_ID,
    AffineCalibration,
    CalibrationTrainingPoint,
    FrozenCalibrationFold,
    QwenOutputValidationError,
    QwenScoreInput,
    QwenTranscriptSegment,
    apply_affine_calibration,
    create_local_qwen_session,
    fit_affine_calibration,
    run_local_qwen_score,
    validate_qwen_output,
    verify_calibration_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIGEST = "sha256:" + "a" * 64


class SyntheticLoopbackTransport:
    def __init__(
        self,
        responses: list[dict[str, object]] | None = None,
        *,
        digest: str | None = EXPECTED_DIGEST,
        digests: list[str | None] | None = None,
    ) -> None:
        self.responses = list(responses or [_valid_payload()])
        self.digest = digest
        self.digests = list(digests or [])
        self.probes: list[tuple[str, str, bool]] = []
        self.generations: list[tuple[str, str, str, bool]] = []

    def probe_model(
        self, endpoint: str, model_id: str, *, allow_redirects: bool
    ) -> str | None:
        self.probes.append((endpoint, model_id, allow_redirects))
        return self.digests.pop(0) if self.digests else self.digest

    def generate(
        self,
        endpoint: str,
        model_id: str,
        request_json: str,
        *,
        allow_redirects: bool,
    ) -> dict[str, object]:
        self.generations.append((endpoint, model_id, request_json, allow_redirects))
        return self.responses.pop(0)


def _score_input(
    *, sample_id: str = "sample-001", dataset_id: str = "d" * 64
) -> QwenScoreInput:
    return QwenScoreInput(
        dataset_id=dataset_id,
        scoring_protocol_sha256="c" * 64,
        sample_id=sample_id,
        target_name="total_score",
        scale_min=0.0,
        scale_max=100.0,
        task_context="Score the advisor total score from transcript text only.",
        segments=(
            QwenTranscriptSegment(0.0, 3.0, "Welcome everyone."),
            QwenTranscriptSegment(3.0, 8.0, "Today I explain the method."),
        ),
        rubric_version=None,
    )


def _valid_payload(*, sample_id: str = "sample-001") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "target_name": "total_score",
        "prediction": 72.5,
        "scale_min": 0.0,
        "scale_max": 100.0,
        "evidence": [
            {
                "start_s": 3.0,
                "end_s": 8.0,
                "quote": "Today I explain the method.",
            }
        ],
        "limitations": ["Transcript-only evidence cannot assess visual delivery."],
        "language": "en",
    }


def _session(transport: SyntheticLoopbackTransport):
    result = create_local_qwen_session(
        expected_digest=EXPECTED_DIGEST,
        endpoint="http://127.0.0.1:11434/api/generate",
        transport=transport,
    )
    assert result.availability.status == "ready"
    assert result.session is not None
    return result.session


def _fold():
    split_payload = {
        "schema_version": 1,
        "assignments": [
            {"sample_id": "held-a", "outer_fold": 0},
            {"sample_id": "held-b", "outer_fold": 0},
            {"sample_id": "train-a", "outer_fold": 1},
            {"sample_id": "train-b", "outer_fold": 1},
            {"sample_id": "train-c", "outer_fold": 2},
        ],
    }
    template = load_protocol_template(PROJECT_ROOT / "configs/experiment_protocol.template.yaml")
    protocol = bind_protocol(
        template,
        dataset_id="d" * 64,
        split_sha256=sha256_bytes(canonical_json_bytes(split_payload)),
    )
    return verify_calibration_fold(protocol, split_payload, outer_fold=0)


def test_missing_expected_digest_skips_without_probing_transport() -> None:
    """Break caught: an unspecified identity triggers discovery or a pull."""
    transport = SyntheticLoopbackTransport()
    result = create_local_qwen_session(
        expected_digest=None,
        endpoint="http://127.0.0.1:11434/api/generate",
        transport=transport,
    )
    assert result.availability.status == "not_run"
    assert result.availability.reason == "expected_local_digest_missing"
    assert result.availability.warning is True
    assert result.session is None
    assert transport.probes == []


@pytest.mark.parametrize(
    ("expected", "observed", "reason"),
    [
        ("SHA256:" + "A" * 64, EXPECTED_DIGEST, "expected_local_digest_invalid"),
        (EXPECTED_DIGEST, "sha256:" + "A" * 64, "local_digest_invalid"),
        (EXPECTED_DIGEST, "sha256:" + "b" * 64, "local_digest_mismatch"),
    ],
)
def test_session_factory_rejects_uppercase_or_mismatched_digest(
    expected: str, observed: str, reason: str
) -> None:
    """Break caught: digest normalization silently changes exact model identity."""
    result = create_local_qwen_session(
        expected_digest=expected,
        endpoint="http://127.0.0.1:11434/api/generate",
        transport=SyntheticLoopbackTransport(digest=observed),
    )
    assert result.availability.status == "not_run"
    assert result.availability.reason == reason
    assert result.session is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434/api/generate",
        "http://192.0.2.10:11434/api/generate",
        "http://example.com/api/generate",
        "http://127.0.0.1:11434/api/generate?redirect=http://example.com",
    ],
)
def test_session_factory_accepts_only_structural_loopback_http(endpoint: str) -> None:
    """Break caught: the local transport can target HTTPS, remote hosts, or redirects."""
    transport = SyntheticLoopbackTransport()
    result = create_local_qwen_session(
        expected_digest=EXPECTED_DIGEST,
        endpoint=endpoint,
        transport=transport,
    )
    assert result.availability.status == "not_run"
    assert result.availability.reason == "endpoint_not_loopback_http"
    assert result.session is None
    assert transport.probes == []


def test_scoring_uses_only_factory_embedded_transport_without_redirects() -> None:
    """Break caught: scoring accepts a substitute runner or redirect policy."""
    embedded = SyntheticLoopbackTransport()
    substitute = SyntheticLoopbackTransport()
    session = _session(embedded)
    with pytest.raises(TypeError):
        run_local_qwen_score(_score_input(), session, substitute)  # type: ignore[call-arg]
    result = run_local_qwen_score(_score_input(), session)
    assert result.status == "succeeded"
    assert substitute.generations == []
    assert embedded.probes == [
        ("http://127.0.0.1:11434/api/generate", ALLOWED_QWEN_MODEL_ID, False),
        ("http://127.0.0.1:11434/api/generate", ALLOWED_QWEN_MODEL_ID, False),
        ("http://127.0.0.1:11434/api/generate", ALLOWED_QWEN_MODEL_ID, False),
    ]
    assert embedded.generations[0][3] is False


def test_transport_substitution_after_factory_invalidates_session() -> None:
    """Break caught: replacing the bound transport preserves a usable capability."""
    embedded = SyntheticLoopbackTransport()
    substitute = SyntheticLoopbackTransport()
    session = _session(embedded)
    object.__setattr__(session, "_transport", substitute)
    result = run_local_qwen_score(_score_input(), session)
    assert result.status == "not_run"
    assert result.reason == "session_not_validated"
    assert embedded.generations == []
    assert substitute.generations == []


def test_self_consistent_forged_availability_cannot_score() -> None:
    """Break caught: matching public strings are treated as execution capability."""
    availability = create_local_qwen_session(
        expected_digest=EXPECTED_DIGEST,
        endpoint="http://127.0.0.1:11434/api/generate",
        transport=SyntheticLoopbackTransport(),
    ).availability
    result = run_local_qwen_score(_score_input(), availability)  # type: ignore[arg-type]
    assert result.status == "not_run"
    assert result.reason == "session_not_validated"


def test_malformed_session_missing_sealed_fields_fails_closed() -> None:
    """Break caught: AttributeError from a forged session escapes gate handling."""
    forged = object.__new__(type(_session(SyntheticLoopbackTransport())))

    result = run_local_qwen_score(_score_input(), forged)

    assert result.status == "not_run"
    assert result.reason == "session_not_validated"


@pytest.mark.parametrize(
    ("digests", "generation_count"),
    [
        ([EXPECTED_DIGEST, "sha256:" + "b" * 64], 0),
        ([EXPECTED_DIGEST, EXPECTED_DIGEST, "sha256:" + "b" * 64], 1),
    ],
)
def test_model_swap_before_or_after_generation_fails_closed(
    digests: list[str], generation_count: int
) -> None:
    """Break caught: model identity can change after factory probing."""
    transport = SyntheticLoopbackTransport(digests=digests)
    session = _session(transport)

    result = run_local_qwen_score(_score_input(), session)

    assert result.status == "failed"
    assert result.reason == "model_identity_changed"
    assert len(transport.generations) == generation_count


def test_schema_retry_reattests_before_second_generation() -> None:
    """Break caught: a swapped model serves the schema retry without re-attestation."""
    transport = SyntheticLoopbackTransport(
        [{**_valid_payload(), "prediction": 1000.0}, _valid_payload()],
        digests=[
            EXPECTED_DIGEST,
            EXPECTED_DIGEST,
            EXPECTED_DIGEST,
            "sha256:" + "b" * 64,
        ],
    )
    session = _session(transport)

    result = run_local_qwen_score(_score_input(), session)

    assert result.status == "failed"
    assert result.reason == "model_identity_changed"
    assert result.attempt_count == 2
    assert len(transport.generations) == 1


def test_schema_retry_reuses_same_bound_transport_model_and_request() -> None:
    """Break caught: retry switches transport/model or changes the frozen request."""
    transport = SyntheticLoopbackTransport(
        [{**_valid_payload(), "prediction": 1000.0}, _valid_payload()]
    )
    result = run_local_qwen_score(_score_input(), _session(transport))
    assert result.status == "succeeded"
    assert result.attempt_count == 2
    assert [call[1] for call in transport.generations] == [
        ALLOWED_QWEN_MODEL_ID,
        ALLOWED_QWEN_MODEL_ID,
    ]
    assert transport.generations[0][2] == transport.generations[1][2]


def test_validated_output_keeps_dataset_target_raw_score_and_grounded_evidence() -> None:
    """Break caught: output identity, scale, raw score, or evidence is not enforced."""
    output = validate_qwen_output(_score_input(), _valid_payload())
    assert output.dataset_id == "d" * 64
    assert output.scoring_protocol_sha256 == "c" * 64
    assert output.sample_id == "sample-001"
    assert output.target_name == "total_score"
    assert output.raw_prediction == 72.5
    assert output.calibrated_prediction is None
    assert output.evidence[0].quote == "Today I explain the method."
    assert output.limitations
    assert output.language == "en"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(sample_id="sample-002"), "sample_id"),
        (lambda payload: payload.update(target_name="delivery"), "target_name"),
        (lambda payload: payload.update(scale_max=5.0), "scale"),
        (lambda payload: payload.update(prediction=101.0), "prediction"),
        (lambda payload: payload.update(prediction=math.inf), "prediction"),
        (lambda p: p["evidence"][0].update(start_s=2.5), "timestamps"),  # type: ignore[index,union-attr]
        (lambda p: p["evidence"][0].update(quote="Invented"), "quote"),  # type: ignore[index,union-attr]
        (lambda payload: payload.update(limitations=[]), "limitations"),
        (lambda payload: payload.update(language=""), "language"),
        (lambda payload: payload.update(extra_target=12.0), "fields"),
    ],
)
def test_invalid_outputs_are_rejected(mutation, message: str) -> None:
    """Break caught: malformed or ungrounded JSON reaches evaluation."""
    payload = deepcopy(_valid_payload())
    mutation(payload)
    with pytest.raises(QwenOutputValidationError, match=message):
        validate_qwen_output(_score_input(), payload)


def test_fold_contract_derives_training_and_application_ids_from_bound_split() -> None:
    """Break caught: callers omit or relabel held-out identities."""
    fold = _fold()
    assert fold.dataset_id == "d" * 64
    assert fold.outer_fold == 0
    assert fold.training_sample_ids == ("train-a", "train-b", "train-c")
    assert fold.held_out_sample_ids == ("held-a", "held-b")


def test_affine_calibration_records_fold_identity_hash_and_application_ids() -> None:
    """Break caught: calibration cannot be audited to its exact training fold."""
    calibration = fit_affine_calibration(
        (
            CalibrationTrainingPoint("train-a", 1.0, 10.0),
            CalibrationTrainingPoint("train-b", 2.0, 20.0),
            CalibrationTrainingPoint("train-c", 3.0, 30.0),
        ),
        fold=_fold(),
        session=_session(SyntheticLoopbackTransport()),
        scoring_protocol_sha256="c" * 64,
        target_name="total_score",
        scale_min=0.0,
        scale_max=100.0,
    )
    assert calibration.dataset_id == "d" * 64
    assert calibration.outer_fold == 0
    assert calibration.fold_id == "outer_fold:0"
    assert calibration.fit_scope == "training_fold_only"
    assert calibration.training_sample_ids == ("train-a", "train-b", "train-c")
    assert calibration.application_sample_ids == ("held-a", "held-b")
    assert calibration.held_out_sample_ids == ("held-a", "held-b")
    assert len(calibration.training_sha256) == 64
    assert calibration.target_name == "total_score"
    assert calibration.scale_min == 0.0
    assert calibration.scale_max == 100.0
    assert calibration.model_id == ALLOWED_QWEN_MODEL_ID
    assert calibration.model_digest == EXPECTED_DIGEST


def test_calibration_requires_every_and_only_derived_training_sample() -> None:
    """Break caught: an omitted held-out declaration or leaked row changes the fit set."""
    points = (
        CalibrationTrainingPoint("train-a", 1.0, 10.0),
        CalibrationTrainingPoint("train-b", 2.0, 20.0),
    )
    with pytest.raises(ValueError, match="exact frozen training IDs"):
        fit_affine_calibration(
            points,
            fold=_fold(),
            session=_session(SyntheticLoopbackTransport()),
            scoring_protocol_sha256="c" * 64,
            target_name="total_score",
            scale_min=0.0,
            scale_max=100.0,
        )
    with pytest.raises(ValueError, match="exact frozen training IDs"):
        fit_affine_calibration(
            (*points, CalibrationTrainingPoint("held-a", 3.0, 30.0)),
            fold=_fold(),
            session=_session(SyntheticLoopbackTransport()),
            scoring_protocol_sha256="c" * 64,
            target_name="total_score",
            scale_min=0.0,
            scale_max=100.0,
        )


def test_forged_fold_contract_is_rejected() -> None:
    """Break caught: a caller-created object can claim training-fold provenance."""
    forged = object.__new__(FrozenCalibrationFold)
    object.__setattr__(forged, "dataset_id", "d" * 64)
    object.__setattr__(forged, "outer_fold", 0)
    object.__setattr__(forged, "training_sample_ids", ("train-a", "train-b"))
    object.__setattr__(forged, "held_out_sample_ids", ("held-a",))
    with pytest.raises(ValueError, match="verified frozen fold"):
        fit_affine_calibration(
            (
                CalibrationTrainingPoint("train-a", 1.0, 10.0),
                CalibrationTrainingPoint("train-b", 2.0, 20.0),
            ),
            fold=forged,
            session=_session(SyntheticLoopbackTransport()),
            scoring_protocol_sha256="c" * 64,
            target_name="total_score",
            scale_min=0.0,
            scale_max=100.0,
        )


def test_calibration_application_requires_matching_dataset_and_heldout_sample() -> None:
    """Break caught: calibration is applied across datasets or onto training rows."""
    calibration = fit_affine_calibration(
        (
            CalibrationTrainingPoint("train-a", 1.0, 10.0),
            CalibrationTrainingPoint("train-b", 2.0, 20.0),
            CalibrationTrainingPoint("train-c", 3.0, 30.0),
        ),
        fold=_fold(),
        session=_session(SyntheticLoopbackTransport()),
        scoring_protocol_sha256="c" * 64,
        target_name="total_score",
        scale_min=0.0,
        scale_max=100.0,
    )
    transport = SyntheticLoopbackTransport(
        [{**_valid_payload(sample_id="held-a"), "prediction": 4.0}]
    )
    held_result = run_local_qwen_score(
        _score_input(sample_id="held-a"), _session(transport)
    )
    assert held_result.output is not None
    held = held_result.output
    calibrated = apply_affine_calibration(held, calibration)
    assert calibrated.raw_prediction == 4.0
    assert calibrated.calibrated_prediction == pytest.approx(40.0)
    with pytest.raises(ValueError, match="dataset"):
        apply_affine_calibration(replace(held, dataset_id="e" * 64), calibration)
    with pytest.raises(ValueError, match="held-out"):
        apply_affine_calibration(replace(held, sample_id="train-a"), calibration)


def test_calibration_rejects_target_scale_protocol_and_model_mismatch() -> None:
    """Break caught: a fit is applied under a different scoring construct."""
    session = _session(SyntheticLoopbackTransport())
    calibration = fit_affine_calibration(
        (
            CalibrationTrainingPoint("train-a", 1.0, 10.0),
            CalibrationTrainingPoint("train-b", 2.0, 20.0),
            CalibrationTrainingPoint("train-c", 3.0, 30.0),
        ),
        fold=_fold(),
        session=session,
        scoring_protocol_sha256="c" * 64,
        target_name="total_score",
        scale_min=0.0,
        scale_max=100.0,
    )
    transport = SyntheticLoopbackTransport([_valid_payload(sample_id="held-a")])
    result = run_local_qwen_score(_score_input(sample_id="held-a"), _session(transport))
    assert result.output is not None
    output = result.output

    for changed, message in (
        (replace(output, target_name="delivery"), "target"),
        (replace(output, scale_max=5.0), "scale"),
        (replace(output, scoring_protocol_sha256="e" * 64), "protocol"),
        (replace(output, model_digest="sha256:" + "b" * 64), "model"),
    ):
        with pytest.raises(ValueError, match=message):
            apply_affine_calibration(changed, calibration)


def test_affine_calibration_cannot_be_constructed_replaced_or_tampered() -> None:
    """Break caught: callers forge finite-looking training-fold calibration metadata."""
    calibration = fit_affine_calibration(
        (
            CalibrationTrainingPoint("train-a", 1.0, 10.0),
            CalibrationTrainingPoint("train-b", 2.0, 20.0),
            CalibrationTrainingPoint("train-c", 3.0, 30.0),
        ),
        fold=_fold(),
        session=_session(SyntheticLoopbackTransport()),
        scoring_protocol_sha256="c" * 64,
        target_name="total_score",
        scale_min=0.0,
        scale_max=100.0,
    )
    with pytest.raises(TypeError):
        AffineCalibration()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        replace(calibration, slope=999.0)
    object.__setattr__(calibration, "slope", 999.0)

    transport = SyntheticLoopbackTransport([_valid_payload(sample_id="held-a")])
    result = run_local_qwen_score(_score_input(sample_id="held-a"), _session(transport))
    assert result.output is not None
    with pytest.raises(ValueError, match="verified affine calibration"):
        apply_affine_calibration(result.output, calibration)


def test_affine_fit_and_application_reject_nonfinite_arithmetic() -> None:
    """Break caught: overflow creates an infinite slope or calibrated prediction."""
    common = {
        "fold": _fold(),
        "session": _session(SyntheticLoopbackTransport()),
        "scoring_protocol_sha256": "c" * 64,
        "target_name": "total_score",
        "scale_min": 0.0,
        "scale_max": 1e308,
    }
    with pytest.raises(ValueError, match="parameters must be finite"):
        fit_affine_calibration(
            (
                CalibrationTrainingPoint("train-a", -1e308, 0.0),
                CalibrationTrainingPoint("train-b", 0.0, 1.0),
                CalibrationTrainingPoint("train-c", 1e308, 2.0),
            ),
            **common,
        )
    calibration = fit_affine_calibration(
        (
            CalibrationTrainingPoint("train-a", 0.0, 0.0),
            CalibrationTrainingPoint("train-b", 0.5, 5e307),
            CalibrationTrainingPoint("train-c", 1.0, 1e308),
        ),
        **common,
    )
    transport = SyntheticLoopbackTransport(
        [{**_valid_payload(sample_id="held-a"), "prediction": 2.0, "scale_max": 1e308}]
    )
    large_input = replace(
        _score_input(sample_id="held-a"), scale_max=1e308
    )
    result = run_local_qwen_score(large_input, _session(transport))
    assert result.output is not None
    with pytest.raises(ValueError, match="non-finite"):
        apply_affine_calibration(result.output, calibration)
