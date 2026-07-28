"""cash_burn skill 单元测试。"""

from __future__ import annotations

from datetime import date

from ipo_risk.domain import RiskSeverity
from ipo_risk.skills.cash_burn import extract_cash_burn_features

from .conftest import make_calculation, make_finding


class TestExtractCashBurnFeatures:
    def test_empty_findings_returns_empty(self) -> None:
        assert extract_cash_burn_features([]) == {}

    def test_cash_burn_finding_produces_core_features(self) -> None:
        finding = make_finding(
            risk_type="cash_burn",
            severity=RiskSeverity.HIGH,
            confidence=0.9,
        )
        features = extract_cash_burn_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_cash_burn_finding"].value is True
        assert features["cash_burn_severity_score"].value == 3  # HIGH=3
        assert features["cash_burn_confidence"].value == 0.9
        assert features["cash_burn_finding_count"].value == 1
        assert features["cash_burn_has_calculation"].value is False
        assert features["has_cash_burn_finding"].available_at == date(2025, 12, 29)

    def test_persistent_loss_finding(self) -> None:
        finding = make_finding(
            risk_type="persistent_loss",
            severity=RiskSeverity.CRITICAL,
            confidence=0.95,
        )
        features = extract_cash_burn_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["has_persistent_loss_finding"].value is True
        assert features["persistent_loss_severity_score"].value == 4
        assert features["persistent_loss_confidence"].value == 0.95
        assert "has_cash_burn_finding" not in features

    def test_calculation_result_extracted(self) -> None:
        calc = make_calculation(result=18.5)
        finding = make_finding(
            risk_type="cash_burn",
            calculation=calc,
        )
        features = extract_cash_burn_features(
            [finding], source_available_at=date(2025, 12, 29)
        )

        assert features["cash_burn_has_calculation"].value is True
        assert features["cash_burn_calculation_result"].value == 18.5

    def test_multiple_findings_picks_highest_confidence(self) -> None:
        f1 = make_finding(
            finding_id="F-001",
            risk_type="cash_burn",
            confidence=0.7,
            severity=RiskSeverity.MEDIUM,
        )
        f2 = make_finding(
            finding_id="F-002",
            risk_type="cash_burn",
            confidence=0.95,
            severity=RiskSeverity.HIGH,
        )
        features = extract_cash_burn_features(
            [f1, f2], source_available_at=date(2025, 12, 29)
        )

        assert features["cash_burn_confidence"].value == 0.95
        assert features["cash_burn_severity_score"].value == 3
        assert features["cash_burn_finding_count"].value == 2

    def test_company_info_proceeds(self) -> None:
        company_info = {
            "as_of_date": "2025-12-29",
            "offering": {
                "net_proceeds_hkd": 2025800000,
                "gross_proceeds_hkd": 2277300000,
            },
        }
        features = extract_cash_burn_features([], company_info=company_info)

        assert features["net_proceeds_hkd"].value == 2025800000.0
        assert features["gross_proceeds_hkd"].value == 2277300000.0
        assert features["net_proceeds_hkd"].available_at == date(2025, 12, 29)

    def test_missing_company_info_fields_graceful(self) -> None:
        company_info = {"as_of_date": "2025-12-29", "offering": {}}
        features = extract_cash_burn_features([], company_info=company_info)
        assert "net_proceeds_hkd" not in features

    def test_invalid_company_info_graceful(self) -> None:
        features = extract_cash_burn_features(
            [], company_info={"offering": "not_a_dict"}
        )
        assert features == {}

    def test_unrelated_findings_ignored(self) -> None:
        finding = make_finding(risk_type="litigation_or_penalty")
        features = extract_cash_burn_features([finding])
        assert features == {}
