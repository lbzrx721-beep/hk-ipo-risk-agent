from __future__ import annotations

from datetime import date, datetime, timezone

from ipo_risk.domain import AgentRole, Evidence, EvidenceType, RiskSeverity
from ipo_risk.modeling.risk_semantics import (
    build_risk_semantic_feature_snapshot,
    classify_risk_finding,
    risk_semantic_catalog,
)
from tests.unit.skills.conftest import make_finding


def _evidence(quote: str = "相关招股书原文") -> Evidence:
    return Evidence(
        evidence_id="E-001",
        company_id="03696",
        document_id="prospectus",
        source_file="data/raw/prospectus/03696/招股书.pdf",
        source_hash="a" * 64,
        page=88,
        bbox=(72.0, 120.0, 520.0, 240.0),
        quote=quote,
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.97,
        available_at=date(2024, 1, 1),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_dynamic_finding_maps_to_stable_subtype_from_title() -> None:
    finding = make_finding(
        risk_type="open_123456789abc",
        claim="公司收入依赖少数客户，未来订单存在不确定性。",
        created_by=AgentRole.BUSINESS,
    ).model_copy(update={"title": "前五大客户收入占比达到75%，客户集中度较高"})

    result = classify_risk_finding(finding, evidence_by_id={"E-001": _evidence()})

    assert result.primary_family == "operational"
    assert result.primary_subtype == "customer_concentration"
    assert result.method == "title_keyword"
    assert result.mapping_confidence == 0.90
    assert {
        item["subtype"] for item in risk_semantic_catalog()
    } >= {"customer_concentration", "cash_burn_liquidity", "redemption_vam"}


def test_canonical_risk_type_takes_priority_over_noisy_text() -> None:
    finding = make_finding(
        risk_type="redemption_or_vam",
        claim="文字同时提到客户与供应商。",
        created_by=AgentRole.LEGAL,
    )

    result = classify_risk_finding(finding, evidence_by_id={"E-001": _evidence()})

    assert result.primary_subtype == "redemption_vam"
    assert result.method == "canonical_risk_type"
    assert result.mapping_confidence == 1.0


def test_unmatched_finding_is_retained_in_role_fallback() -> None:
    finding = make_finding(
        risk_type="open_abcdef123456",
        claim="完全无法归入词典的新型风险。",
        created_by=AgentRole.LEGAL,
    )

    result = classify_risk_finding(finding, evidence_by_id={"E-001": _evidence()})

    assert result.primary_family == "legal_compliance"
    assert result.primary_subtype == "legal_other"
    assert result.method == "agent_role_fallback"


def test_company_aggregation_builds_fixed_family_and_subtype_features() -> None:
    customer = make_finding(
        finding_id="F-001",
        risk_type="open_123456789abc",
        severity=RiskSeverity.HIGH,
        claim="最大客户及五大客户收入占比较高。",
        created_by=AgentRole.BUSINESS,
    ).model_copy(update={"title": "客户集中度显著上升"})
    redemption = make_finding(
        finding_id="F-002",
        risk_type="redemption_or_vam",
        severity=RiskSeverity.CRITICAL,
        claim="投资者享有赎回及回购权。",
        created_by=AgentRole.LEGAL,
    )

    features, classifications = build_risk_semantic_feature_snapshot(
        (customer, redemption),
        (_evidence(),),
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-1",
    )

    assert len(classifications) == 2
    assert features["agent_semantic_specific_classified_share"].value == 1.0
    assert features["agent_semantic_family_operational_count"].value == 1
    assert features["agent_semantic_family_legal_compliance_count"].value == 1
    assert features["agent_semantic_has_customer_concentration"].value == 1
    assert features["agent_semantic_risk_customer_concentration_max_severity"].value == 3
    assert features["agent_semantic_has_redemption_vam"].value == 1
    assert features["agent_semantic_risk_redemption_vam_max_severity"].value == 4
    assert features["agent_semantic_has_supplier_concentration"].value == 0
    assert features["agent_semantic_supported_finding_count"].value == 2
    assert features["agent_semantic_supported_finding_share"].value == 1.0
    assert features[
        "agent_semantic_risk_customer_concentration_supported_count"
    ].value == 1
