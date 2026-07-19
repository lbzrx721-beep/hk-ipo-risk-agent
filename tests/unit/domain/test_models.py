from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    Evidence,
    EvidenceType,
    FeatureValue,
    Prediction,
    PredictionHorizon,
    PredictionTarget,
    RiskFinding,
    RiskLevel,
    RiskSeverity,
    TraceStatus,
    VerificationStatus,
)

SOURCE_HASH = "a" * 64
NOW = datetime(2025, 12, 29, 12, 0, tzinfo=timezone.utc)


def make_evidence(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "evidence_id": "E0001",
        "company_id": "03696",
        "document_id": "prospectus_2025-12-18",
        "source_file": "data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        "source_hash": SOURCE_HASH,
        "page": 231,
        "printed_page_label": "222",
        "bbox": (72.0, 146.0, 518.0, 314.0),
        "quote": "招股书原文切片",
        "evidence_type": EvidenceType.TEXT,
        "extractor": AgentRole.FINANCE,
        "confidence": 0.91,
        "available_at": date(2025, 12, 18),
        "created_at": NOW,
    }
    values.update(overrides)
    return Evidence.model_validate(values)


def test_evidence_serializes_with_canonical_field_names() -> None:
    evidence = make_evidence(source_hash=SOURCE_HASH.upper())

    payload = evidence.model_dump(mode="json")

    assert payload["schema_version"] == "1.0"
    assert payload["source_hash"] == SOURCE_HASH
    assert "source_file_hash" not in payload
    assert payload["page"] == 231
    assert payload["printed_page_label"] == "222"
    assert payload["bbox"] == [72.0, 146.0, 518.0, 314.0]


def test_printed_page_label_is_optional_and_never_replaces_physical_page() -> None:
    appendix_evidence = make_evidence(page=501, printed_page_label="I-22")
    no_label_evidence = make_evidence(page=1, printed_page_label=None)

    assert appendix_evidence.page == 501
    assert appendix_evidence.printed_page_label == "I-22"
    assert no_label_evidence.page == 1
    assert no_label_evidence.printed_page_label is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "not-a-sha256"),
        ("page", 0),
        ("bbox", (10.0, 10.0, 5.0, 20.0)),
        ("confidence", 1.01),
    ],
)
def test_evidence_rejects_invalid_source_coordinates_or_confidence(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        make_evidence(**{field: value})


def test_risk_finding_references_evidence_without_copying_it() -> None:
    finding = RiskFinding(
        finding_id="F0001",
        company_id="03696",
        risk_type="cash_burn",
        title="现金覆盖期偏短",
        severity=RiskSeverity.HIGH,
        claim="现有现金覆盖期存在压力。",
        evidence_ids=("E0001", "E0002"),
        status=VerificationStatus.VERIFIED,
        confidence=0.88,
        created_by=AgentRole.FINANCE,
        created_at=NOW,
    )

    payload = finding.model_dump(mode="json")

    assert payload["status"] == "verified"
    assert "verification_status" not in payload
    assert payload["evidence_ids"] == ["E0001", "E0002"]


def test_risk_finding_requires_at_least_one_unique_evidence_id() -> None:
    values = {
        "finding_id": "F0001",
        "company_id": "03696",
        "risk_type": "cash_burn",
        "title": "现金覆盖期偏短",
        "severity": RiskSeverity.HIGH,
        "claim": "现有现金覆盖期存在压力。",
        "status": VerificationStatus.PROPOSED,
        "confidence": 0.88,
        "created_by": AgentRole.FINANCE,
        "created_at": NOW,
    }
    with pytest.raises(ValidationError):
        RiskFinding.model_validate({**values, "evidence_ids": ()})
    with pytest.raises(ValidationError):
        RiskFinding.model_validate(
            {**values, "evidence_ids": ("E0001", "E0001")}
        )


def test_agent_trace_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValidationError):
        AgentTrace(
            trace_id="T0001",
            run_id="RUN_03696_001",
            agent=AgentRole.FINANCE,
            started_at=NOW,
            ended_at=NOW - timedelta(seconds=1),
            status=TraceStatus.SUCCEEDED,
            decision="提交核验",
        )


def test_prediction_forbids_features_from_after_as_of_date() -> None:
    with pytest.raises(ValidationError, match="after as_of_date"):
        Prediction(
            prediction_id="P0001",
            company_id="03696",
            as_of_date=date(2025, 12, 29),
            horizon=PredictionHorizon.DAY_5,
            target=PredictionTarget.SIGNIFICANT_DROP,
            risk_probability=0.67,
            risk_level=RiskLevel.HIGH,
            model_version="baseline_v1",
            feature_snapshot={
                "future_close": FeatureValue(
                    value=29.98,
                    available_at=date(2025, 12, 30),
                    source_ref="market:2025-12-30",
                )
            },
            explanations=("示例解释",),
            created_at=NOW,
        )


def test_prediction_accepts_point_in_time_feature_snapshot() -> None:
    prediction = Prediction(
        prediction_id="P0001",
        company_id="03696",
        as_of_date=date(2025, 12, 29),
        horizon=PredictionHorizon.DAY_5,
        target=PredictionTarget.SIGNIFICANT_DROP,
        risk_probability=0.67,
        risk_level=RiskLevel.HIGH,
        model_version="baseline_v1",
        feature_snapshot={
            "cornerstone_ratio": FeatureValue(
                value=0.3929,
                available_at=date(2025, 12, 29),
                source_ref="company_info:offering.cornerstone_ratio_of_offer",
            )
        },
        explanations=("基石投资者占比为39.29%",),
        created_at=NOW,
    )

    payload = prediction.model_dump(mode="json")

    assert payload["horizon"] == 5
    assert payload["risk_probability"] == 0.67
    assert payload["feature_snapshot"]["cornerstone_ratio"]["available_at"] == "2025-12-29"
