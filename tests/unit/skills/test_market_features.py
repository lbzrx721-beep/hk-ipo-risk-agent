"""market_features skill 单元测试。"""

from __future__ import annotations

from datetime import date

import pytest

from ipo_risk.skills.market_features import extract_market_features


def _make_market_data(rows: list[dict]) -> dict:
    return {"schema_version": 2, "company_id": "03696", "rows": rows}


def _row(
    date_str: str,
    *,
    open_hkd: float = 30.0,
    high_hkd: float = 32.0,
    low_hkd: float = 28.0,
    close_hkd: float = 31.0,
    volume_shares: float = 1000000,
    turnover_hkd: float = 30000000,
    amplitude_pct: float = 10.0,
    change_pct: float = 3.0,
    turnover_rate_pct: float = 0.5,
) -> dict:
    return {
        "date": date_str,
        "open_hkd": open_hkd,
        "high_hkd": high_hkd,
        "low_hkd": low_hkd,
        "close_hkd": close_hkd,
        "volume_shares": volume_shares,
        "turnover_hkd": turnover_hkd,
        "amplitude_pct": amplitude_pct,
        "change_pct": change_pct,
        "change_hkd": close_hkd - open_hkd,
        "turnover_rate_pct": turnover_rate_pct,
    }


COMPANY_INFO = {
    "as_of_date": "2025-12-29",
    "offering": {
        "offer_price_hkd": 24.05,
        "listing_market_cap_hkd": 13405914925,
        "cornerstone_ratio_of_offer": 0.3929,
        "overallocated_shares": 14203500,
        "offer_shares": 94690500,
    },
}


class TestExtractMarketFeatures:
    def test_window_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_days"):
            extract_market_features(_make_market_data([_row("2025-12-30")]), window_days=0)

    def test_empty_rows_returns_empty(self) -> None:
        assert extract_market_features({"rows": []}) == {}

    def test_missing_rows_key_returns_empty(self) -> None:
        assert extract_market_features({}) == {}

    def test_first_day_features(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", change_pct=24.66, turnover_hkd=1.2e9,
                 turnover_rate_pct=6.77, amplitude_pct=23.78,
                 volume_shares=37735345),
        ])
        features = extract_market_features(data)

        assert features["first_day_return_pct"].value == 24.66
        assert features["first_day_turnover_hkd"].value == 1.2e9
        assert features["first_day_turnover_rate_pct"].value == 6.77
        assert features["first_day_amplitude_pct"].value == 23.78
        assert features["first_day_volume_shares"].value == 37735345.0
        assert features["first_day_return_pct"].available_at == date(2025, 12, 30)

    def test_window_cumulative_return(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", close_hkd=30.0),
            _row("2025-12-31", close_hkd=33.0),
            _row("2026-01-02", close_hkd=36.0),
        ])
        features = extract_market_features(data, window_days=3)

        # (36/30 - 1) * 100 = 20%
        assert features["first_3d_cumulative_return_pct"].value == 20.0

    def test_window_volatility(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", change_pct=10.0),
            _row("2025-12-31", change_pct=-5.0),
            _row("2026-01-02", change_pct=5.0),
        ])
        features = extract_market_features(data, window_days=3)

        assert "first_3d_volatility_pct" in features
        # mean=3.33, var=((10-3.33)^2+(-5-3.33)^2+(5-3.33)^2)/2 ≈ 58.33, std≈7.64
        vol = features["first_3d_volatility_pct"].value
        assert 7.0 < vol < 8.5

    def test_broke_issue_price(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", low_hkd=22.0, high_hkd=35.0, close_hkd=30.0),
            _row("2025-12-31", low_hkd=25.0, high_hkd=32.0, close_hkd=28.0),
        ])
        features = extract_market_features(
            data, company_info=COMPANY_INFO, window_days=2,
        )

        # offer_price=24.05, min low=22.0 < 24.05 → broke
        assert features["first_2d_broke_issue_price"].value is True
        # max_discount = (22/24.05 - 1)*100 ≈ -8.52%
        assert features["first_2d_max_discount_pct"].value < 0

    def test_no_broke_issue_price(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", low_hkd=25.0, high_hkd=35.0, close_hkd=30.0),
        ])
        features = extract_market_features(
            data, company_info=COMPANY_INFO, window_days=1,
        )
        assert features["first_1d_broke_issue_price"].value is False

    def test_offering_features_from_company_info(self) -> None:
        data = _make_market_data([_row("2025-12-30")])
        features = extract_market_features(data, company_info=COMPANY_INFO)

        assert features["cornerstone_ratio_of_offer"].value == 0.3929
        assert features["listing_market_cap_hkd"].value == 13405914925.0
        assert "overallotment_ratio" in features

    def test_window_max_drawdown(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", close_hkd=30.0),
            _row("2025-12-31", close_hkd=40.0),
            _row("2026-01-02", close_hkd=25.0),
            _row("2026-01-05", close_hkd=35.0),
        ])
        features = extract_market_features(data, window_days=4)

        # peak=40, trough=25, dd=(25-40)/40*100=-37.5%
        assert features["first_4d_max_drawdown_pct"].value == -37.5

    def test_single_row_no_window_features(self) -> None:
        data = _make_market_data([_row("2025-12-30")])
        features = extract_market_features(data)

        assert "first_day_return_pct" in features
        assert "first_5d_cumulative_return_pct" not in features

    def test_incomplete_window_does_not_publish_a_five_day_metric(self) -> None:
        data = _make_market_data([_row("2025-12-30", low_hkd=20.0)])

        features = extract_market_features(
            data,
            company_info=COMPANY_INFO,
            window_days=5,
        )

        assert not any(name.startswith("first_5d_") for name in features)

    def test_anomalous_date_invalidates_the_original_window(self) -> None:
        data = _make_market_data([
            _row("2025-12-30", close_hkd=30.0),
            _row("2025-12-31", close_hkd=1.0),
            _row("2026-01-02", close_hkd=33.0),
        ])
        data["data_quality"] = {"anomaly_dates": ["2025-12-31"]}

        features = extract_market_features(data, window_days=2)

        assert not any(name.startswith("first_2d_") for name in features)
