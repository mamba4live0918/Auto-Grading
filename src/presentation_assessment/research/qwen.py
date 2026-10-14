"""Sealed local-Qwen execution and frozen-fold calibration contracts."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .canonical import canonical_json_bytes, sha256_bytes
from .protocol import BoundProtocol


ALLOWED_QWEN_MODEL_ID = "qwen3.5:9b-q4_K_M"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DATASET_ID = re.compile(r"[0-9a-f]{64}\Z")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")
_OUTPUT_FIELDS = frozenset(
    {
        "sample_id",
        "target_name",
        "prediction",
        "scale_min",
        "scale_max",
        "evidence",
        "limitations",
        "language",
    }
)
_EVIDENCE_FIELDS = frozenset({"start_s", "end_s", "quote"})
_SESSION_SENTINEL = object()
_SESSION_KEY = secrets.token_bytes(32)
_FOLD_SENTINEL = object()
_FOLD_KEY = secrets.token_bytes(32)
_CALIBRATION_SENTINEL = object()
_CALIBRATION_KEY = secrets.token_bytes(32)


class QwenOutputValidationError(ValueError):
    """A local model response violates the frozen scoring schema."""


class LocalQwenTransport(Protocol):
    """One local transport used for both identity probing and generation."""

    def probe_model(
        self, endpoint: str, model_id: str, *, allow_redirects: bool
    ) -> str | None: ...

    def generate(
        self,
        endpoint: str,
        model_id: str,
        request_json: str,
        *,
        allow_redirects: bool,
    ) -> Mapping[str, object] | str: ...


@dataclass(frozen=True, slots=True)
class QwenAvailability:
    """Public observation only; it is deliberately not an execution capability."""

    status: Literal["ready", "not_run"]
    model_id: str
    expected_digest: str | None
    observed_digest: str | None
    endpoint: str | None
    reason: str | None
    local_only: Literal[True] = True
    network_accessed: Literal[False] = False

    @property
    def warning(self) -> bool:
        return self.status == "not_run"


@dataclass(frozen=True, slots=True, init=False)
class LocalQwenSession:
    """Factory-issued capability binding identity, endpoint, and one transport."""

    model_id: str
    digest: str
    endpoint: str
    _transport: LocalQwenTransport
    _capability: object
    _seal: str

    def __init__(
        self,
        *,
        model_id: str,
        digest: str,
        endpoint: str,
        transport: LocalQwenTransport,
        capability: object,
    ) -> None:
        if capability is not _SESSION_SENTINEL:
            raise TypeError("LocalQwenSession is created only by the local probe factory")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _session_seal(model_id, digest, endpoint, transport))


@dataclass(frozen=True, slots=True)
class QwenSessionResult:
    availability: QwenAvailability
    session: LocalQwenSession | None


@dataclass(frozen=True, slots=True)
class QwenTranscriptSegment:
    start_s: float
    end_s: float
    text: str

    def __post_init__(self) -> None:
        if not _finite_number(self.start_s) or not _finite_number(self.end_s):
            raise ValueError("transcript timestamps must be finite")
        if float(self.start_s) < 0.0 or float(self.start_s) >= float(self.end_s):
            raise ValueError("transcript timestamps must be ordered")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("transcript segment text must be non-empty")


@dataclass(frozen=True, slots=True)
class QwenScoreInput:
    dataset_id: str
    scoring_protocol_sha256: str
    sample_id: str
    target_name: str
    scale_min: float
    scale_max: float
    task_context: str
    segments: tuple[QwenTranscriptSegment, ...]
    rubric_version: str | None

    def __post_init__(self) -> None:
        if _DATASET_ID.fullmatch(self.dataset_id) is None:
            raise ValueError("dataset_id must be a lowercase SHA-256 identity")
        if _DATASET_ID.fullmatch(self.scoring_protocol_sha256) is None:
            raise ValueError("scoring_protocol_sha256 must be lowercase SHA-256")
        for name in ("sample_id", "target_name", "task_context"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not _finite_number(self.scale_min) or not _finite_number(self.scale_max):
            raise ValueError("target scale must be finite")
        if float(self.scale_min) >= float(self.scale_max):
            raise ValueError("target scale must be ordered")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("timestamped transcript segments are required")
        if not all(isinstance(item, QwenTranscriptSegment) for item in self.segments):
            raise ValueError("invalid transcript segment")
        if self.rubric_version is not None and not self.rubric_version.strip():
            raise ValueError("rubric_version cannot be blank")


@dataclass(frozen=True, slots=True)
class QwenEvidence:
    start_s: float
    end_s: float
    quote: str


@dataclass(frozen=True, slots=True)
class QwenScoreOutput:
    dataset_id: str
    scoring_protocol_sha256: str
    sample_id: str
    target_name: str
    raw_prediction: float
    calibrated_prediction: float | None
    scale_min: float
    scale_max: float
    evidence: tuple[QwenEvidence, ...]
    limitations: tuple[str, ...]
    language: str
    model_id: str | None
    model_digest: str | None


@dataclass(frozen=True, slots=True)
class QwenRunResult:
    status: Literal["succeeded", "failed", "not_run"]
    output: QwenScoreOutput | None
    reason: str | None
    attempt_count: int


@dataclass(frozen=True, slots=True)
class CalibrationTrainingPoint:
    sample_id: str
    raw_prediction: float
    human_score: float


@dataclass(frozen=True, slots=True, init=False)
class FrozenCalibrationFold:
    dataset_id: str
    protocol_id: str
    split_sha256: str
    outer_fold: int
    fold_id: str
    training_sample_ids: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]
    _capability: object
    _seal: str

    def __init__(
        self,
        *,
        dataset_id: str,
        protocol_id: str,
        split_sha256: str,
        outer_fold: int,
        fold_id: str,
        training_sample_ids: tuple[str, ...],
        held_out_sample_ids: tuple[str, ...],
        capability: object,
    ) -> None:
        if capability is not _FOLD_SENTINEL:
            raise TypeError("FrozenCalibrationFold requires a verified bound split")
        values = (
            dataset_id,
            protocol_id,
            split_sha256,
            str(outer_fold),
            fold_id,
            *training_sample_ids,
            "--held-out--",
            *held_out_sample_ids,
        )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "split_sha256", split_sha256)
        object.__setattr__(self, "outer_fold", outer_fold)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "training_sample_ids", training_sample_ids)
        object.__setattr__(self, "held_out_sample_ids", held_out_sample_ids)
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _seal(_FOLD_KEY, values))


@dataclass(frozen=True, slots=True, init=False)
class AffineCalibration:
    method: Literal["affine"]
    fit_scope: Literal["training_fold_only"]
    dataset_id: str
    protocol_id: str
    split_sha256: str
    outer_fold: int
    fold_id: str
    slope: float
    intercept: float
    training_sample_ids: tuple[str, ...]
    training_sha256: str
    application_sample_ids: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]
    target_name: str
    scale_min: float
    scale_max: float
    scoring_protocol_sha256: str
    model_id: str
    model_digest: str
    _capability: object
    _seal: str

    def __init__(
        self,
        *,
        dataset_id: str,
        protocol_id: str,
        split_sha256: str,
        outer_fold: int,
        fold_id: str,
        slope: float,
        intercept: float,
        training_sample_ids: tuple[str, ...],
        training_sha256: str,
        application_sample_ids: tuple[str, ...],
        held_out_sample_ids: tuple[str, ...],
        target_name: str,
        scale_min: float,
        scale_max: float,
        scoring_protocol_sha256: str,
        model_id: str,
        model_digest: str,
        capability: object,
    ) -> None:
        if capability is not _CALIBRATION_SENTINEL:
            raise TypeError("AffineCalibration is constructed only by fit")
        fields = {
            "method": "affine",
            "fit_scope": "training_fold_only",
            "dataset_id": dataset_id,
            "protocol_id": protocol_id,
            "split_sha256": split_sha256,
            "outer_fold": outer_fold,
            "fold_id": fold_id,
            "slope": slope,
            "intercept": intercept,
            "training_sample_ids": training_sample_ids,
            "training_sha256": training_sha256,
            "application_sample_ids": application_sample_ids,
            "held_out_sample_ids": held_out_sample_ids,
            "target_name": target_name,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "scoring_protocol_sha256": scoring_protocol_sha256,
            "model_id": model_id,
            "model_digest": model_digest,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _calibration_seal(self))


def create_local_qwen_session(
    *,
    expected_digest: str | None,
    endpoint: str,
    transport: LocalQwenTransport,
) -> QwenSessionResult:
    """Probe one exact model through one loopback transport and seal the session."""
    if expected_digest is None:
        return _session_not_run(None, None, None, "expected_local_digest_missing")
    if _DIGEST.fullmatch(expected_digest) is None:
        return _session_not_run(
            expected_digest, None, None, "expected_local_digest_invalid"
        )
    if not _loopback_http_endpoint(endpoint):
        return _session_not_run(
            expected_digest, None, endpoint, "endpoint_not_loopback_http"
        )
    try:
        observed = transport.probe_model(
            endpoint, ALLOWED_QWEN_MODEL_ID, allow_redirects=False
        )
    except Exception:
        observed = None
    if observed is None:
        return _session_not_run(
            expected_digest, None, endpoint, "local_model_unavailable"
        )
    if not isinstance(observed, str) or _DIGEST.fullmatch(observed) is None:
        return _session_not_run(
            expected_digest, str(observed), endpoint, "local_digest_invalid"
        )
    if not hmac.compare_digest(expected_digest, observed):
        return _session_not_run(
            expected_digest, observed, endpoint, "local_digest_mismatch"
        )
    session = LocalQwenSession(
        model_id=ALLOWED_QWEN_MODEL_ID,
        digest=expected_digest,
        endpoint=endpoint,
        transport=transport,
        capability=_SESSION_SENTINEL,
    )
    availability = QwenAvailability(
        status="ready",
        model_id=ALLOWED_QWEN_MODEL_ID,
        expected_digest=expected_digest,
        observed_digest=observed,
        endpoint=endpoint,
        reason=None,
    )
    return QwenSessionResult(availability=availability, session=session)


def run_local_qwen_score(
    score_input: QwenScoreInput, session: LocalQwenSession
) -> QwenRunResult:
    """Score through the sealed transport, retrying invalid schema exactly once."""
    if not _valid_session(session):
        return QwenRunResult("not_run", None, "session_not_validated", 0)
    request_json = _request_json(score_input)
    for attempt in (1, 2):
        if not _reattest_session(session):
            return QwenRunResult("failed", None, "model_identity_changed", attempt)
        try:
            raw = session._transport.generate(
                session.endpoint,
                session.model_id,
                request_json,
                allow_redirects=False,
            )
        except Exception:
            return QwenRunResult("failed", None, "local_inference_error", attempt)
        if not _reattest_session(session):
            return QwenRunResult("failed", None, "model_identity_changed", attempt)
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(payload, Mapping):
                raise QwenOutputValidationError("output fields are not a mapping")
            output = replace(
                validate_qwen_output(score_input, payload),
                model_id=session.model_id,
                model_digest=session.digest,
            )
        except (json.JSONDecodeError, QwenOutputValidationError, TypeError, ValueError):
            if attempt == 1:
                continue
            return QwenRunResult(
                "failed", None, "invalid_output_after_schema_retry", 2
            )
        return QwenRunResult("succeeded", output, None, attempt)
    raise AssertionError("unreachable")


def validate_qwen_output(
    score_input: QwenScoreInput, payload: Mapping[str, object]
) -> QwenScoreOutput:
    if not isinstance(payload, Mapping) or frozenset(payload) != _OUTPUT_FIELDS:
        raise QwenOutputValidationError("output fields do not match the frozen schema")
    if payload["sample_id"] != score_input.sample_id:
        raise QwenOutputValidationError("sample_id does not match the request")
    if payload["target_name"] != score_input.target_name:
        raise QwenOutputValidationError("target_name does not match the request")
    scale_min = _output_number(payload["scale_min"], "scale")
    scale_max = _output_number(payload["scale_max"], "scale")
    if (scale_min, scale_max) != (
        float(score_input.scale_min),
        float(score_input.scale_max),
    ):
        raise QwenOutputValidationError("scale does not match the request")
    prediction = _output_number(payload["prediction"], "prediction")
    if not scale_min <= prediction <= scale_max:
        raise QwenOutputValidationError("prediction is outside the target scale")
    evidence = _validate_evidence(score_input, payload["evidence"])
    limitations = payload["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item.strip() for item in limitations)
    ):
        raise QwenOutputValidationError("limitations must contain non-empty text")
    language = payload["language"]
    if not isinstance(language, str) or _LANGUAGE.fullmatch(language) is None:
        raise QwenOutputValidationError("language must be a valid non-empty code")
    return QwenScoreOutput(
        dataset_id=score_input.dataset_id,
        scoring_protocol_sha256=score_input.scoring_protocol_sha256,
        sample_id=score_input.sample_id,
        target_name=score_input.target_name,
        raw_prediction=prediction,
        calibrated_prediction=None,
        scale_min=scale_min,
        scale_max=scale_max,
        evidence=evidence,
        limitations=tuple(item.strip() for item in limitations),
        language=language,
        model_id=None,
        model_digest=None,
    )


def verify_calibration_fold(
    protocol: BoundProtocol,
    split_payload: Mapping[str, object],
    *,
    outer_fold: int,
) -> FrozenCalibrationFold:
    """Derive an immutable fold only from a split digest bound by the protocol."""
    if not isinstance(protocol, BoundProtocol):
        raise ValueError("a bound experiment protocol is required")
    if type(outer_fold) is not int or outer_fold < 0:
        raise ValueError("outer_fold must be a non-negative integer")
    actual_split_sha256 = sha256_bytes(canonical_json_bytes(split_payload))
    if not hmac.compare_digest(actual_split_sha256, protocol.split_sha256):
        raise ValueError("split payload does not match the bound protocol")
    raw_assignments = split_payload.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("verified split assignments are required")
    assignments: list[tuple[str, int]] = []
    for item in raw_assignments:
        if not isinstance(item, Mapping):
            raise ValueError("invalid split assignment")
        sample_id = item.get("sample_id")
        fold = item.get("outer_fold")
        if not isinstance(sample_id, str) or not sample_id or type(fold) is not int:
            raise ValueError("invalid split assignment")
        assignments.append((sample_id, fold))
    ids = [sample_id for sample_id, _ in assignments]
    if len(ids) != len(set(ids)):
        raise ValueError("split sample IDs must be unique")
    held_out = tuple(sorted(sample_id for sample_id, fold in assignments if fold == outer_fold))
    training = tuple(sorted(sample_id for sample_id, fold in assignments if fold != outer_fold))
    if not held_out or not training:
        raise ValueError("fold must contain both training and held-out samples")
    return FrozenCalibrationFold(
        dataset_id=protocol.dataset_id,
        protocol_id=protocol.protocol_id,
        split_sha256=protocol.split_sha256,
        outer_fold=outer_fold,
        fold_id=f"outer_fold:{outer_fold}",
        training_sample_ids=training,
        held_out_sample_ids=held_out,
        capability=_FOLD_SENTINEL,
    )


def fit_affine_calibration(
    training_points: Sequence[CalibrationTrainingPoint],
    *,
    fold: FrozenCalibrationFold,
    session: LocalQwenSession,
    scoring_protocol_sha256: str,
    target_name: str,
    scale_min: float,
    scale_max: float,
) -> AffineCalibration:
    """Fit an affine map using every and only row in the verified training fold."""
    if not _valid_fold(fold):
        raise ValueError("a verified frozen fold is required")
    if not _valid_session(session):
        raise ValueError("a validated local scoring session is required")
    if _DATASET_ID.fullmatch(scoring_protocol_sha256) is None:
        raise ValueError("scoring protocol identity must be lowercase SHA-256")
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("target_name must be non-empty")
    if (
        not _finite_number(scale_min)
        or not _finite_number(scale_max)
        or float(scale_min) >= float(scale_max)
    ):
        raise ValueError("calibration target scale must be finite and ordered")
    points = tuple(training_points)
    point_ids = tuple(point.sample_id for point in points)
    if len(point_ids) != len(set(point_ids)) or set(point_ids) != set(
        fold.training_sample_ids
    ):
        raise ValueError("calibration points must match exact frozen training IDs")
    ordered = tuple(sorted(points, key=lambda item: item.sample_id))
    raw = tuple(float(point.raw_prediction) for point in ordered)
    observed = tuple(float(point.human_score) for point in ordered)
    if not all(math.isfinite(value) for value in (*raw, *observed)):
        raise ValueError("calibration values must be finite")
    try:
        raw_mean = sum(raw) / len(raw)
        observed_mean = sum(observed) / len(observed)
        denominator = sum((value - raw_mean) ** 2 for value in raw)
        if denominator <= 0.0:
            raise ValueError("affine calibration requires varying raw predictions")
        slope = sum(
            (x - raw_mean) * (y - observed_mean)
            for x, y in zip(raw, observed, strict=True)
        ) / denominator
        intercept = observed_mean - slope * raw_mean
    except OverflowError as error:
        raise ValueError("affine calibration parameters must be finite") from error
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("affine calibration parameters must be finite")
    training_payload = {
        "dataset_id": fold.dataset_id,
        "protocol_id": fold.protocol_id,
        "split_sha256": fold.split_sha256,
        "outer_fold": fold.outer_fold,
        "points": [
            {
                "sample_id": point.sample_id,
                "raw_prediction": point.raw_prediction,
                "human_score": point.human_score,
            }
            for point in ordered
        ],
        "target_name": target_name,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scoring_protocol_sha256": scoring_protocol_sha256,
        "model_id": session.model_id,
        "model_digest": session.digest,
    }
    return AffineCalibration(
        dataset_id=fold.dataset_id,
        protocol_id=fold.protocol_id,
        split_sha256=fold.split_sha256,
        outer_fold=fold.outer_fold,
        fold_id=f"outer_fold:{fold.outer_fold}",
        slope=slope,
        intercept=intercept,
        training_sample_ids=fold.training_sample_ids,
        training_sha256=sha256_bytes(canonical_json_bytes(training_payload)),
        application_sample_ids=fold.held_out_sample_ids,
        held_out_sample_ids=fold.held_out_sample_ids,
        target_name=target_name,
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        scoring_protocol_sha256=scoring_protocol_sha256,
        model_id=session.model_id,
        model_digest=session.digest,
        capability=_CALIBRATION_SENTINEL,
    )


def apply_affine_calibration(
    output: QwenScoreOutput, calibration: AffineCalibration
) -> QwenScoreOutput:
    if not _valid_calibration(calibration):
        raise ValueError("a verified affine calibration is required")
    if output.dataset_id != calibration.dataset_id:
        raise ValueError("calibration dataset does not match score output")
    if output.target_name != calibration.target_name:
        raise ValueError("calibration target does not match score output")
    if (output.scale_min, output.scale_max) != (
        calibration.scale_min,
        calibration.scale_max,
    ):
        raise ValueError("calibration scale does not match score output")
    if output.scoring_protocol_sha256 != calibration.scoring_protocol_sha256:
        raise ValueError("calibration scoring protocol does not match score output")
    if (output.model_id, output.model_digest) != (
        calibration.model_id,
        calibration.model_digest,
    ):
        raise ValueError("calibration model identity does not match score output")
    if (
        output.sample_id in calibration.training_sample_ids
        or output.sample_id not in calibration.application_sample_ids
    ):
        raise ValueError("calibration applies only to its verified held-out samples")
    prediction = calibration.slope * output.raw_prediction + calibration.intercept
    if not math.isfinite(prediction):
        raise ValueError("calibrated prediction is non-finite")
    bounded = min(output.scale_max, max(output.scale_min, prediction))
    return replace(output, calibrated_prediction=bounded)


def _session_not_run(
    expected: str | None,
    observed: str | None,
    endpoint: str | None,
    reason: str,
) -> QwenSessionResult:
    return QwenSessionResult(
        availability=QwenAvailability(
            status="not_run",
            model_id=ALLOWED_QWEN_MODEL_ID,
            expected_digest=expected,
            observed_digest=observed,
            endpoint=endpoint,
            reason=reason,
        ),
        session=None,
    )


def _loopback_http_endpoint(endpoint: str) -> bool:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
        and bool(parsed.path)
        and port is not None
    )


def _session_seal(
    model_id: str, digest: str, endpoint: str, transport: LocalQwenTransport
) -> str:
    return _seal(_SESSION_KEY, (model_id, digest, endpoint, str(id(transport))))


def _valid_session(value: object) -> bool:
    try:
        return (
            isinstance(value, LocalQwenSession)
            and value._capability is _SESSION_SENTINEL
            and value.model_id == ALLOWED_QWEN_MODEL_ID
            and _DIGEST.fullmatch(value.digest) is not None
            and _loopback_http_endpoint(value.endpoint)
            and hmac.compare_digest(
                value._seal,
                _session_seal(
                    value.model_id, value.digest, value.endpoint, value._transport
                ),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _reattest_session(session: LocalQwenSession) -> bool:
    try:
        observed = session._transport.probe_model(
            session.endpoint, session.model_id, allow_redirects=False
        )
        return (
            isinstance(observed, str)
            and _DIGEST.fullmatch(observed) is not None
            and hmac.compare_digest(observed, session.digest)
        )
    except Exception:
        return False


def _valid_calibration(value: object) -> bool:
    if not isinstance(value, AffineCalibration):
        return False
    try:
        return (
            value._capability is _CALIBRATION_SENTINEL
            and math.isfinite(value.slope)
            and math.isfinite(value.intercept)
            and hmac.compare_digest(value._seal, _calibration_seal(value))
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _calibration_seal(value: AffineCalibration) -> str:
    values = (
        value.method,
        value.fit_scope,
        value.dataset_id,
        value.protocol_id,
        value.split_sha256,
        str(value.outer_fold),
        value.fold_id,
        repr(value.slope),
        repr(value.intercept),
        *value.training_sample_ids,
        value.training_sha256,
        "--application--",
        *value.application_sample_ids,
        "--held-out--",
        *value.held_out_sample_ids,
        value.target_name,
        repr(value.scale_min),
        repr(value.scale_max),
        value.scoring_protocol_sha256,
        value.model_id,
        value.model_digest,
    )
    return _seal(_CALIBRATION_KEY, values)


def _valid_fold(value: object) -> bool:
    if not isinstance(value, FrozenCalibrationFold):
        return False
    try:
        values = (
            value.dataset_id,
            value.protocol_id,
            value.split_sha256,
            str(value.outer_fold),
            value.fold_id,
            *value.training_sample_ids,
            "--held-out--",
            *value.held_out_sample_ids,
        )
        return value._capability is _FOLD_SENTINEL and hmac.compare_digest(
            value._seal, _seal(_FOLD_KEY, values)
        )
    except AttributeError:
        return False


def _seal(key: bytes, values: Sequence[str]) -> str:
    payload = "\0".join(values).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _validate_evidence(
    score_input: QwenScoreInput, raw: object
) -> tuple[QwenEvidence, ...]:
    if not isinstance(raw, list) or not raw:
        raise QwenOutputValidationError("evidence must contain at least one entry")
    validated: list[QwenEvidence] = []
    for item in raw:
        if not isinstance(item, Mapping) or frozenset(item) != _EVIDENCE_FIELDS:
            raise QwenOutputValidationError("evidence fields do not match the schema")
        start_s = _output_number(item["start_s"], "evidence timestamps")
        end_s = _output_number(item["end_s"], "evidence timestamps")
        quote = item["quote"]
        matching = tuple(
            segment
            for segment in score_input.segments
            if float(segment.start_s) == start_s and float(segment.end_s) == end_s
        )
        if not matching:
            raise QwenOutputValidationError(
                "evidence timestamps do not identify an input transcript segment"
            )
        if not isinstance(quote, str) or not quote.strip() or not any(
            quote in segment.text for segment in matching
        ):
            raise QwenOutputValidationError("evidence quote is absent from transcript text")
        validated.append(QwenEvidence(start_s, end_s, quote))
    return tuple(validated)


def _request_json(score_input: QwenScoreInput) -> str:
    payload = {
        "schema_version": 1,
        "dataset_id": score_input.dataset_id,
        "scoring_protocol_sha256": score_input.scoring_protocol_sha256,
        "sample_id": score_input.sample_id,
        "task_context": score_input.task_context,
        "target": {
            "name": score_input.target_name,
            "scale_min": score_input.scale_min,
            "scale_max": score_input.scale_max,
        },
        "rubric_version": score_input.rubric_version,
        "segments": [
            {"start_s": item.start_s, "end_s": item.end_s, "text": item.text}
            for item in score_input.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _output_number(value: object, field_name: str) -> float:
    if not _finite_number(value):
        raise QwenOutputValidationError(f"{field_name} must be finite")
    return float(value)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
