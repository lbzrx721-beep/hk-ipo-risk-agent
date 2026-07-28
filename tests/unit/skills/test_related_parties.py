"""related_parties skill 单元测试。"""

from __future__ import annotations

from datetime import date

from ipo_risk.domain import RiskSeverity
from ipo_risk.skills.related_parties import extract_related_party_features

from .conftest import make_calculation, make_finding


class TestExtractRelatedPartyFeatures:
    def test_empty_findings_returns_empty(self) -> None:
        assert extract_related_party_features([]) == {}

    def test_basic_rpt_finding(self) -> None:
        finding = make_finding(
            risk_type="related_party_transaction",
            severity=RiskSeverity.HIGH,
            confidence=0.82,
        )
        features = extract_related_party_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_related_party_finding"].value is True
        assert features["related_party_severity_score"].value == 3
        assert features["related_party_confidence"].value == 0.82
        assert features["related_party_finding_count"].value == 1
        assert features["related_party_max_severity_score"].value == 3
        assert features["related_party_has_calculation"].value is False
        assert features["related_party_has_critical"].value is False
        assert features["has_related_party_finding"].available_at == date(2025, 12, 29)

    def test_risk_score_calculation(self) -> None:
        finding = make_finding(
            risk_type="related_party_transaction",
            severity=RiskSeverity.HIGH,  # score=3
            confidence=0.8,
        )
        features = extract_related_party_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        # 3 * 0.8 = 2.4
        assert features["related_party_risk_score"].value == 2.4

    def test_critical_severity(self) -> None:
        finding = make_finding(
            risk_type="related_party_transaction",
            severity=RiskSeverity.CRITICAL,
            confidence=0.95,
        )
        features = extract_related_party_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["related_party_has_critical"].value is True
        assert features["related_party_severity_score"].value == 4

    def test_multiple_findings(self) -> None:
        f1 = make_finding(
            finding_id="F-001",
            risk_type="related_party_transaction",
            severity=RiskSeverity.MEDIUM,
            confidence=0.6,
        )
        f2 = make_finding(
            finding_id="F-002",
            risk_type="related_party_transaction",
            severity=RiskSeverity.HIGH,
            confidence=0.9,
        )
        features = extract_related_party_features(
            [f1, f2], source_available_at=date(2025, 12, 29)
        )

        assert features["related_party_finding_count"].value == 2
        assert features["related_party_max_severity_score"].value == 3
        # best confidence is f2 (0.9), severity HIGH=3
        assert features["related_party_confidence"].value == 0.9

    def test_with_calculation(self) -> None:
        calc = make_calculation(result=5.5)
        finding = make_finding(
            risk_type="related_party_transaction",
            calculation=calc,
        )
        features = extract_related_party_features(
            [finding], source_available_at=date(2025, 12, 29)
        )
        assert features["related_party_has_calculation"].value is True

    def test_unrelated_findings_ignored(self) -> None:
        finding = make_finding(risk_type="cash_burn")
        features = extract_related_party_features([finding])
        assert features == {}
