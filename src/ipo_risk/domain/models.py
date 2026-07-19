"""Evidence、RiskFinding、AgentTrace 与 Prediction 的唯一权威定义。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .enums import (
    AgentRole,
    EvidenceType,
    PredictionHorizon,
    PredictionTarget,
    ReviewDecision,
    RiskLevel,
    RiskSeverity,
    ToolCallStatus,
    TraceStatus,
    VerificationStatus,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RiskType = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9_]*$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BoundingBox = tuple[float, float, float, float]


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for auditable default values."""

    return datetime.now(timezone.utc)


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Reject ambiguous local timestamps in trace and contract objects."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


class StrictModel(BaseModel):
    """Shared validation policy for primary and supporting contract objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ContractModel(StrictModel):
    """Versioned base class for the four objects exchanged across modules."""

    schema_version: Literal["1.0"] = "1.0"


class Calculation(StrictModel):
    """Reproducible inputs and formula for a numerical risk conclusion."""

    tool: NonEmptyStr
    formula: NonEmptyStr
    inputs: dict[str, JsonValue] = Field(min_length=1)
    result: JsonValue
    unit: NonEmptyStr
    period: NonEmptyStr


class ToolCall(StrictModel):
    """A deterministic tool invocation recorded inside an AgentTrace."""

    tool_call_id: NonEmptyStr
    tool_name: NonEmptyStr
    started_at: datetime
    ended_at: datetime
    inputs: dict[str, JsonValue]
    output: JsonValue | None = None
    status: ToolCallStatus
    error: NonEmptyStr | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime, info: object) -> datetime:
        field_name = getattr(info, "field_name", "timestamp")
        return ensure_timezone_aware(value, field_name)

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolCall:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be earlier than started_at")
        if self.status is ToolCallStatus.FAILED and self.error is None:
            raise ValueError("a failed tool call must include error")
        return self


class FeatureValue(StrictModel):
    """A model feature with its point-in-time availability and lineage."""

    value: JsonValue
    available_at: date
    source_ref: NonEmptyStr


class Evidence(ContractModel):
    """An immutable fact that can be located again in the source PDF."""

    evidence_id: NonEmptyStr
    company_id: NonEmptyStr
    document_id: NonEmptyStr
    source_file: NonEmptyStr
    source_hash: Sha256
    page: int = Field(ge=1)
    printed_page_label: NonEmptyStr | None = None
    bbox: BoundingBox
    quote: NonEmptyStr
    evidence_type: EvidenceType
    extractor: AgentRole
    confidence: float = Field(ge=0.0, le=1.0)
    available_at: date
    screenshot_path: NonEmptyStr | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_hash", mode="before")
    @classmethod
    def normalize_source_hash(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "created_at")

    @model_validator(mode="after")
    def validate_bbox(self) -> Evidence:
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0:
            raise ValueError("bbox origin must be non-negative")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
        return self


class VerificationCheck(StrictModel):
    """One deterministic check performed while reproducing evidence."""

    name: NonEmptyStr
    passed: bool
    detail: NonEmptyStr


class EvidenceVerification(ContractModel):
    """Auditable result of locating one Evidence object in its source PDF."""

    evidence_id: NonEmptyStr
    passed: bool
    checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    checked_at: datetime = Field(default_factory=utc_now)

    @field_validator("checked_at")
    @classmethod
    def checked_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "checked_at")

    @model_validator(mode="after")
    def outcome_must_match_checks(self) -> EvidenceVerification:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("passed must match the combined check results")
        return self


class FindingVerification(ContractModel):
    """Combined evidence and calculation checks for one RiskFinding."""

    finding_id: NonEmptyStr
    passed: bool
    checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    evidence_verifications: tuple[EvidenceVerification, ...] = ()
    checked_at: datetime = Field(default_factory=utc_now)

    @field_validator("checked_at")
    @classmethod
    def checked_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "checked_at")

    @model_validator(mode="after")
    def outcome_must_match_checks(self) -> FindingVerification:
        combined = all(check.passed for check in self.checks) and all(
            result.passed for result in self.evidence_verifications
        )
        if self.passed != combined:
            raise ValueError("passed must match evidence and finding checks")
        return self


class RiskFinding(ContractModel):
    """A risk claim that references one or more Evidence objects by ID."""

    finding_id: NonEmptyStr
    company_id: NonEmptyStr
    risk_type: RiskType
    title: NonEmptyStr
    severity: RiskSeverity
    claim: NonEmptyStr
    evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    created_by: AgentRole
    calculation: Calculation | None = None
    uncertainty: NonEmptyStr | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "created_at")

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> RiskFinding:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
        return self


class FindingReview(ContractModel):
    """Immutable human decision plus the exact verification snapshot it used."""

    review_id: NonEmptyStr
    run_id: NonEmptyStr
    finding_id: NonEmptyStr
    decision: ReviewDecision
    reviewer: NonEmptyStr
    comment: NonEmptyStr | None = None
    verification: FindingVerification
    resulting_finding: RiskFinding
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "created_at")

    @model_validator(mode="after")
    def validate_review_transition(self) -> FindingReview:
        if self.resulting_finding.finding_id != self.finding_id:
            raise ValueError("resulting finding must keep the reviewed finding_id")
        expected_status = (
            VerificationStatus.REJECTED
            if self.decision is ReviewDecision.REJECTED
            else VerificationStatus.VERIFIED
        )
        if self.resulting_finding.status is not expected_status:
            raise ValueError("resulting finding status does not match review decision")
        if self.decision is not ReviewDecision.CONFIRMED and self.comment is None:
            raise ValueError("rejected and edited reviews require a comment")
        if self.decision is not ReviewDecision.REJECTED and not self.verification.passed:
            raise ValueError("a finding cannot be verified when deterministic checks fail")
        return self


class AgentTrace(ContractModel):
    """An auditable record of one Agent node and its tool calls."""

    trace_id: NonEmptyStr
    run_id: NonEmptyStr
    parent_trace_id: NonEmptyStr | None = None
    agent: AgentRole
    started_at: datetime
    ended_at: datetime
    input_refs: tuple[NonEmptyStr, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    output_refs: tuple[NonEmptyStr, ...] = ()
    status: TraceStatus
    decision: NonEmptyStr
    error: NonEmptyStr | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime, info: object) -> datetime:
        field_name = getattr(info, "field_name", "timestamp")
        return ensure_timezone_aware(value, field_name)

    @model_validator(mode="after")
    def validate_trace(self) -> AgentTrace:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be earlier than started_at")
        if self.status is TraceStatus.FAILED and self.error is None:
            raise ValueError("a failed trace must include error")
        if len(set(self.input_refs)) != len(self.input_refs):
            raise ValueError("input_refs must not contain duplicates")
        if len(set(self.output_refs)) != len(self.output_refs):
            raise ValueError("output_refs must not contain duplicates")
        return self


class Prediction(ContractModel):
    """A point-in-time risk probability for one required trading-day horizon."""

    prediction_id: NonEmptyStr
    company_id: NonEmptyStr
    as_of_date: date
    horizon: PredictionHorizon
    target: PredictionTarget
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    model_version: NonEmptyStr
    feature_snapshot: dict[NonEmptyStr, FeatureValue] = Field(min_length=1)
    explanations: tuple[NonEmptyStr, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "created_at")

    @model_validator(mode="after")
    def forbid_future_features(self) -> Prediction:
        future_features = sorted(
            name
            for name, feature in self.feature_snapshot.items()
            if feature.available_at > self.as_of_date
        )
        if future_features:
            names = ", ".join(future_features)
            raise ValueError(f"features available after as_of_date: {names}")
        return self


__all__ = [
    "AgentTrace",
    "BoundingBox",
    "Calculation",
    "Evidence",
    "EvidenceVerification",
    "FeatureValue",
    "FindingReview",
    "FindingVerification",
    "Prediction",
    "RiskFinding",
    "ToolCall",
    "VerificationCheck",
]
