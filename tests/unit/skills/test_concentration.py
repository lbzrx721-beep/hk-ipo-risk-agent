"""concentration skill 单元测试。"""

from __future__ import annotations

from datetime import date

from ipo_risk.domain import RiskSeverity
from ipo_risk.skills.concentration import extract_concentration_features

from .conftest import make_finding


class TestExtractConcentrationFeatures:
    def test_empty_findings_returns_empty(self) -> None:
        assert extract_concentration_features([]) == {}

    def test_customer_concentration(self) -> None:
        finding = make_finding(
            risk_type="customer_concentration",
            severity=RiskSeverity.HIGH,
            confidence=0.88,
        )
        features = extract_concentration_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_customer_concentration"].value is True
        assert features["customer_concentration_severity_score"].value == 3
        assert features["customer_concentration_confidence"].value == 0.88
        assert features["customer_concentration_finding_count"].value == 1
        assert features["concentration_risk_count"].value == 1
        assert features["concentration_max_severity_score"].value == 3

    def test_supplier_concentration(self) -> None:
        finding = make_finding(
            risk_type="supplier_concentration",
            severity=RiskSeverity.MEDIUM,
            confidence=0.75,
        )
        features = extract_concentration_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_supplier_concentration"].value is True
        assert features["supplier_concentration_severity_score"].value == 2

    def test_single_product_dependency(self) -> None:
        finding = make_finding(
            risk_type="single_product_dependency",
            severity=RiskSeverity.CRITICAL,
            confidence=0.92,
        )
        features = extract_concentration_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_single_product_dependency"].value is True
        assert features["single_product_dependency_severity_score"].value == 4

    def test_multiple_concentration_types_aggregate(self) -> None:
        f1 = make_finding(
            finding_id="F-001",
            risk_type="customer_concentration",
            severity=RiskSeverity.HIGH,
        )
        f2 = make_finding(
            finding_id="F-002",
            risk_type="supplier_concentration",
            severity=RiskSeverity.MEDIUM,
        )
        f3 = make_finding(
            finding_id="F-003",
            risk_type="single_product_dependency",
            severity=RiskSeverity.LOW,
        )
        features = extract_concentration_features(
            [f1, f2, f3], source_available_at=date(2025, 12, 29)
        )

        assert features["concentration_risk_count"].value == 3
        assert features["concentration_max_severity_score"].value == 3
        # 3 + 2 + 1 = 6
        assert features["concentration_total_severity_score"].value == 6

    def test_unrelated_findings_ignored(self) -> None:
        finding = make_finding(risk_type="cash_burn")
        features = extract_concentration_features([finding])
        assert features == {}

    def test_available_at_uses_finding_date(self) -> None:
        finding = make_finding(risk_type="customer_concentration")
        features = extract_concentration_features(
            [finding], source_available_at=date(2025, 12, 29)
        )
        assert features["has_customer_concentration"].available_at == date(2025, 12, 29)

    def test_total_severity_counts_each_risk_type_once(self) -> None:
        findings = [
            make_finding(
                finding_id=f"F-{index}",
                risk_type="customer_concentration",
                severity=RiskSeverity.CRITICAL,
            )
            for index in range(1, 6)
        ]

        features = extract_concentration_features(
            findings, source_available_at=date(2025, 12, 29)
        )

        assert features["concentration_total_severity_score"].value == 4
