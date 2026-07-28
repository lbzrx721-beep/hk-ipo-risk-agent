"""Skills 测试的共享 fixture 工厂。"""

from __future__ import annotations

from datetime import datetime, timezone

from ipo_risk.domain import (
    AgentRole,
    Calculation,
    RiskFinding,
    RiskSeverity,
    VerificationStatus,
)


def make_finding(
    *,
    finding_id: str = "F-001",
    risk_type: str = "cash_burn",
    severity: RiskSeverity = RiskSeverity.HIGH,
    confidence: float = 0.85,
    claim: str = "测试风险声明",
    calculation: Calculation | None = None,
    created_by: AgentRole = AgentRole.FINANCE,
) -> RiskFinding:
    """构造一个最小合法的 RiskFinding 用于测试。"""
    return RiskFinding(
        finding_id=finding_id,
        company_id="03696",
        risk_type=risk_type,
        title=f"测试-{risk_type}",
        severity=severity,
        claim=claim,
        evidence_ids=("E-001",),
        status=VerificationStatus.VERIFIED,
        confidence=confidence,
        created_by=created_by,
        calculation=calculation,
        created_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


def make_calculation(
    *,
    result: float = 12.5,
    tool: str = "calculate_percentage",
    formula: str = "a / b * 100",
) -> Calculation:
    """构造一个最小合法的 Calculation 用于测试。"""
    return Calculation(
        tool=tool,
        formula=formula,
        inputs={"a": 25.0, "b": 200.0},
        result=result,
        unit="percent",
        period="2025-12-29",
        source_refs={"a": "src:a", "b": "src:b"},
    )
