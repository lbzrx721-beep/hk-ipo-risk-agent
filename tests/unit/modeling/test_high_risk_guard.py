from __future__ import annotations

from pathlib import Path

from ipo_risk.modeling.high_risk_guard import (
    evaluate_high_risk_guard,
    load_high_risk_guard_policy,
)


def _policy() -> dict:
    return {
        "policy_version": "test",
        "watchlist_threshold": 0.6,
        "minimum_available_features": 4,
        "features": {
            "offering_offer_price_mid_hkd": {
                "direction": "down",
                "quantiles": [1.0, 5.0, 10.0],
            },
            "offering_market_cap_mid_hkd": {
                "direction": "down",
                "quantiles": [100.0, 500.0, 1000.0],
            },
            "fundamental_revenue_latest_native": {
                "direction": "down",
                "quantiles": [10.0, 50.0, 100.0],
            },
            "fundamental_ending_cash_latest_native": {
                "direction": "down",
                "quantiles": [5.0, 25.0, 50.0],
            },
            "customer_top5_ratio_pct": {
                "direction": "up",
                "quantiles": [10.0, 50.0, 90.0],
            },
            "offering_share_ratio_pct": {
                "direction": "up",
                "quantiles": [5.0, 15.0, 25.0],
            },
        },
    }


def test_guard_marks_multi_factor_high_risk_shape_for_review() -> None:
    result = evaluate_high_risk_guard(
        {
            "offering_offer_price_mid_hkd": 1.2,
            "offering_market_cap_mid_hkd": 120.0,
            "fundamental_revenue_latest_native": 12.0,
            "fundamental_ending_cash_latest_native": 6.0,
            "customer_top5_ratio_pct": 88.0,
            "offering_share_ratio_pct": 24.0,
        },
        _policy(),
    )

    assert result["watchlist"] is True
    assert result["score"] > result["threshold"]
    assert result["available_feature_count"] == 6
    assert 1 <= len(result["reasons"]) <= 3


def test_guard_does_not_treat_missing_values_as_risk() -> None:
    result = evaluate_high_risk_guard(
        {
            "offering_offer_price_mid_hkd": 1.2,
            "customer_top5_ratio_pct": 88.0,
            "offering_share_ratio_pct": 24.0,
        },
        _policy(),
    )

    assert result["watchlist"] is False
    assert result["available_feature_count"] == 3


def test_project_policy_is_valid() -> None:
    path = Path(__file__).parents[3] / "configs" / "high_risk_recall_guard.json"
    policy = load_high_risk_guard_policy(path)

    assert policy["policy_version"] == "stable_prelisting_guard_20260826_v1"
    assert len(policy["features"]) == 6
