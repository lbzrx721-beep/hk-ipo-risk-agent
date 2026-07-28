from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from ipo_risk.domain import AgentRole, Evidence, EvidenceType, RiskSeverity
from ipo_risk.modeling.features import (
    build_agent_feature_snapshot,
    build_post_listing_metrics,
    build_prelisting_feature_snapshot,
)
from tests.unit.skills.conftest import make_finding


def test_prelisting_snapshot_uses_document_availability_and_has_no_market_labels() -> None:
    report = SimpleNamespace(
        source_document=SimpleNamespace(available_at=date(2025, 12, 18)),
        risk_findings=(make_finding(risk_type="cash_burn"),),
    )
    company_info = {
        "as_of_date": "2025-12-29",
        "offering": {"offer_price_hkd": 24.05, "net_proceeds_hkd": 100.0},
    }

    features = build_prelisting_feature_snapshot(
        report,
        company_info=company_info,
        as_of_date=date(2025, 12, 29),
    )

    assert features["has_cash_burn_finding"].available_at == date(2025, 12, 18)
    assert features["offer_price_hkd"].available_at == date(2025, 12, 29)
    assert not any(name.startswith("first_day_") for name in features)
    assert not any(name.startswith("first_5d_") for name in features)


def test_prelisting_snapshot_rejects_information_available_after_prediction_date() -> None:
    report = SimpleNamespace(
        source_document=SimpleNamespace(available_at=date(2025, 12, 18)),
        risk_findings=(),
    )
    company_info = {
        "as_of_date": "2025-12-30",
        "offering": {"offer_price_hkd": 24.05},
    }

    with pytest.raises(ValueError, match="after as_of_date"):
        build_prelisting_feature_snapshot(
            report,
            company_info=company_info,
            as_of_date=date(2025, 12, 29),
        )


def test_post_listing_metrics_are_built_in_a_separate_entry_point() -> None:
    market_data = {
        "rows": [
            {
                "date": "2025-12-30",
                "high_hkd": 31.0,
                "low_hkd": 28.0,
                "close_hkd": 30.0,
                "change_pct": 10.0,
            }
        ]
    }
    company_info = {
        "as_of_date": "2025-12-29",
        "offering": {"offer_price_hkd": 24.05},
    }

    metrics = build_post_listing_metrics(
        market_data,
        company_info=company_info,
        horizons=(1,),
    )

    assert "first_day_return_pct" in metrics
    assert "first_day_premium_pct" in metrics
    assert "first_1d_close_premium_pct" in metrics


def test_agent_snapshot_supports_dynamic_risk_types_and_traceable_evidence() -> None:
    available_at = date(2025, 12, 18)
    finding = make_finding(
        risk_type="open_revenue_recognition_uncertainty",
        severity=RiskSeverity.CRITICAL,
        confidence=0.91,
        created_by=AgentRole.BUSINESS,
    )
    evidence = Evidence(
        evidence_id="E-001",
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        source_hash="a" * 64,
        page=88,
        bbox=(72.0, 120.0, 520.0, 240.0),
        quote="项目收入确认依赖客户验收，验收时间存在不确定性。",
        evidence_type=EvidenceType.OCR_TEXT,
        extractor=AgentRole.BUSINESS,
        confidence=0.87,
        available_at=available_at,
        created_at=datetime(2025, 12, 18, tzinfo=timezone.utc),
    )
    report = SimpleNamespace(
        run_id="RUN-DYNAMIC",
        source_document=SimpleNamespace(available_at=available_at),
        risk_findings=(finding,),
        evidences=(evidence,),
        agent_traces=(),
    )

    features = build_agent_feature_snapshot(
        report,
        as_of_date=date(2025, 12, 29),
    )

    assert features["agent_finding_count"].value == 1
    assert features["agent_ocr_evidence_ratio"].value == 1.0
    assert features["agent_role_business_finding_count"].value == 1
    assert features["agent_dynamic_risk_finding_count"].value == 1
    assert features["agent_canonical_risk_finding_count"].value == 0
    assert "agent_has_risk_revenue_recognition_uncertainty" not in features
