"""valuation skill 单元测试。"""

from __future__ import annotations

from datetime import date

import pytest

from ipo_risk.skills.valuation import extract_valuation_features


COMPANY_INFO = {
    "as_of_date": "2025-12-29",
    "offering": {
        "offer_price_hkd": 24.05,
        "listing_market_cap_hkd": 13405914925,
        "gross_proceeds_hkd": 2277300000,
        "net_proceeds_hkd": 2025800000,
        "estimated_listing_expenses_hkd": 251500000,
        "post_offering_shares_before_overallotment": 557418500,
    },
}


def _make_market_data(rows: list[dict]) -> dict:
    return {"rows": rows}


def _row(
    date_str: str,
    *,
    high_hkd: float = 32.0,
    low_hkd: float = 28.0,
    close_hkd: float = 31.0,
) -> dict:
    return {
        "date": date_str,
        "open_hkd": 30.0,
        "high_hkd": high_hkd,
        "low_hkd": low_hkd,
        "close_hkd": close_hkd,
        "volume_shares": 1000000,
        "turnover_hkd": 30000000,
        "amplitude_pct": 10.0,
        "change_pct": 3.0,
        "change_hkd": 1.0,
        "turnover_rate_pct": 0.5,
    }


class TestExtractValuationFeatures:
    def test_window_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_days"):
            extract_valuation_features(COMPANY_INFO, window_days=0)

    def test_basic_offering_features(self) -> None:
        features = extract_valuation_features(COMPANY_INFO)

        assert features["offer_price_hkd"].value == 24.05
        assert features["listing_market_cap_hkd"].value == 13405914925.0
        assert features["gross_proceeds_hkd"].value == 2277300000.0
        assert features["net_proceeds_hkd"].value == 2025800000.0
        assert features["offer_price_hkd"].available_at == date(2025, 12, 29)

    def test_expense_ratio(self) -> None:
        features = extract_valuation_features(COMPANY_INFO)

        # 251500000 / 2277300000 ≈ 0.11044
        ratio = features["listing_expense_ratio"].value
        assert 0.11 < ratio < 0.111

    def test_net_proceeds_ratio(self) -> None:
        features = extract_valuation_features(COMPANY_INFO)

        # 2025800000 / 2277300000 ≈ 0.8896
        ratio = features["net_proceeds_ratio"].value
        assert 0.889 < ratio < 0.890

    def test_market_cap_to_net_proceeds(self) -> None:
        features = extract_valuation_features(COMPANY_INFO)

        # 13405914925 / 2025800000 ≈ 6.617
        ratio = features["market_cap_to_net_proceeds"].value
        assert 6.6 < ratio < 6.7

    def test_first_day_premium(self) -> None:
        market_data = _make_market_data([
            _row("2025-12-30", close_hkd=29.98),
        ])
        features = extract_valuation_features(
            COMPANY_INFO, market_data=market_data,
        )

        # (29.98/24.05 - 1) * 100 ≈ 24.66%
        premium = features["first_day_premium_pct"].value
        assert 24.0 < premium < 25.0
        assert features["first_day_premium_pct"].available_at == date(2025, 12, 30)

    def test_incomplete_window_keeps_first_day_but_not_five_day_metrics(self) -> None:
        market_data = _make_market_data([_row("2025-12-30", close_hkd=29.98)])

        features = extract_valuation_features(
            COMPANY_INFO,
            market_data=market_data,
            window_days=5,
        )

        assert "first_day_premium_pct" in features
        assert not any(name.startswith("first_5d_") for name in features)

    def test_anomalous_date_invalidates_the_original_window(self) -> None:
        market_data = _make_market_data([
            _row("2025-12-30", close_hkd=29.0),
            _row("2025-12-31", close_hkd=1.0),
            _row("2026-01-02", close_hkd=31.0),
        ])
        market_data["data_quality"] = {"anomaly_dates": ["2025-12-31"]}

        features = extract_valuation_features(
            COMPANY_INFO,
            market_data=market_data,
            window_days=2,
        )

        assert not any(name.startswith("first_2d_") for name in features)

    def test_window_max_premium_and_min_discount(self) -> None:
        market_data = _make_market_data([
            _row("2025-12-30", high_hkd=35.7, low_hkd=29.98, close_hkd=29.98),
            _row("2025-12-31", high_hkd=38.68, low_hkd=30.48, close_hkd=37.14),
        ])
        features = extract_valuation_features(
            COMPANY_INFO, market_data=market_data, window_days=2,
        )

        # max premium = (38.68/24.05 - 1)*100 ≈ 60.83%
        assert features["first_2d_max_premium_pct"].value > 60
        # min discount = (29.98/24.05 - 1)*100 ≈ 24.66% (still premium)
        assert features["first_2d_min_discount_pct"].value > 0

    def test_no_market_data_no_premium_features(self) -> None:
        features = extract_valuation_features(COMPANY_INFO)
        assert "first_day_premium_pct" not in features

    def test_missing_as_of_date_returns_empty(self) -> None:
        features = extract_valuation_features({"offering": {}})
        assert features == {}

    def test_missing_offering_fields_graceful(self) -> None:
        info = {"as_of_date": "2025-12-29", "offering": {}}
        features = extract_valuation_features(info)
        assert features == {}

    def test_window_end_close_premium(self) -> None:
        market_data = _make_market_data([
            _row("2025-12-30", close_hkd=30.0),
            _row("2025-12-31", close_hkd=35.0),
            _row("2026-01-02", close_hkd=32.0),
        ])
        features = extract_valuation_features(
            COMPANY_INFO, market_data=market_data, window_days=3,
        )

        # (32/24.05 - 1)*100 ≈ 33.06%
        premium = features["first_3d_close_premium_pct"].value
        assert 33.0 < premium < 33.5
