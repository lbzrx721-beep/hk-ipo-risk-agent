from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_risk.modeling.stable_agent_model import (
    _apply_batch_alert_cap,
    _select_business_threshold,
    _time_decay_sample_weight,
    add_derived_prediction_features,
    train_stable_agent_model,
)
from ipo_risk.modeling.predict import load_model_bundle, predict_feature_rows


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows: list[dict[str, object]] = []
    number = 1
    for year in (2020, 2021, 2022, 2023, 2024, 2025):
        for index in range(30):
            target = int(index % 3 == 0)
            month = index // 15 + 5
            day = index % 15 + 1
            rows.append(
                {
                    "company_id": f"{number:05d}",
                    "ticker": f"{number:05d}.HK",
                    "company_name": f"公司{number}",
                    "prospectus_date": f"{year}-{month:02d}-{day:02d}",
                    "run_id": f"RUN-{number}",
                    "target_significant_drop_5d": target,
                    "base_signal": target * 10 if index % 2 == 0 else index / 100,
                    "base_noise": (number * 7) % 13,
                    "agent_signal": target * 8 if index % 2 == 1 else index / 100,
                    "agent_noise": (number * 11) % 17,
                    "agent_finding_count": 2,
                    "agent_unique_risk_type_count": 1,
                    "agent_severity_low_count": 0,
                    "agent_severity_medium_count": 1,
                    "agent_severity_high_count": 1,
                    "agent_severity_critical_count": 0,
                    "agent_role_finance_finding_count": 2 if target else 0,
                    "agent_role_legal_finding_count": 0 if target else 2,
                    "agent_role_business_finding_count": 0,
                    "agent_role_market_finding_count": 0,
                    "agent_uncertainty_count": 0,
                    "agent_failed_trace_count": 0,
                }
            )
            number += 1
    matrix = tmp_path / "matrix.csv"
    pd.DataFrame(rows).to_csv(matrix, index=False)
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps({"feature_columns": ["base_signal", "base_noise"]}), encoding="utf-8"
    )
    agent = tmp_path / "agent.json"
    agent.write_text(
        json.dumps(
            {
                "agent_feature_columns": [
                    "agent_signal",
                    "agent_noise",
                    "agent_failed_trace_count",
                ]
            }
        ),
        encoding="utf-8",
    )
    return matrix, base, agent


def test_stable_model_excludes_2024_from_selection_and_test(tmp_path: Path) -> None:
    matrix, base, agent = _fixture(tmp_path)
    result = train_stable_agent_model(
        matrix,
        base_summary_path=base,
        agent_summary_path=agent,
        output_dir=tmp_path / "models",
        base_feature_limit=1,
        agent_feature_limit=1,
        permutation_repeats=1,
        base_feature_whitelist=("base_signal", "base_noise"),
        agent_feature_whitelist=("agent_signal", "agent_noise"),
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.feature_manifest_path.read_text(encoding="utf-8"))
    assert metrics["cohort_counts"] == {
        "development_2020_2023": 120,
        "test_2024": 30,
        "external_2025_plus": 30,
    }
    assert metrics["leakage_controls"]["feature_selection_excludes_2024"] is True
    assert metrics["leakage_controls"]["threshold_selection_excludes_2024"] is True
    assert manifest["selected_base_count"] == 1
    assert manifest["selected_agent_count"] == 1
    assert manifest["selected_base_features"] == manifest["selected_base_only_features"]
    assert manifest["business_base_whitelist"] == ["base_signal", "base_noise"]
    assert manifest["agent_semantic_whitelist"] == ["agent_signal", "agent_noise"]
    assert "agent_failed_trace_count" in manifest["excluded_low_information_agent_features"]
    assert result.model_path.is_file()
    assert result.base_model_path.is_file()
    assert metrics["test_2024_base"]["sample_count"] == 30
    assert "agent_incremental_value_2024" in metrics
    predictions = pd.read_csv(result.predictions_path)
    assert len(predictions) == 30
    assert set(pd.to_datetime(predictions["prospectus_date"]).dt.year) == {2024}
    bundle = load_model_bundle(result.model_path)
    source = pd.read_csv(matrix)
    test_rows = source[pd.to_datetime(source["prospectus_date"]).dt.year == 2024]
    reloaded = predict_feature_rows(bundle, test_rows)
    assert len(reloaded) == 30
    assert reloaded["predicted_significant_drop_5d"].mean() <= 0.40
    predictions = pd.read_csv(result.predictions_path)
    np.testing.assert_allclose(
        reloaded["predicted_probability"].to_numpy(),
        predictions["ensemble_probability"].to_numpy(),
    )
    assert np.array_equal(
        reloaded["predicted_significant_drop_5d"].to_numpy(),
        predictions["predicted_significant_drop_5d"].to_numpy(),
    )


def test_business_threshold_prioritizes_recall_with_alert_cap() -> None:
    target = pd.Series([1, 1, 0, 0, 0, 0]).to_numpy()
    probability = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1, 0.0]).to_numpy()
    threshold, policy = _select_business_threshold(
        target,
        probability,
        min_recall=1.0,
        max_alert_fraction=0.5,
    )
    predicted = probability >= threshold
    assert predicted.mean() <= 0.5
    assert int((predicted & (target == 1)).sum()) == 2
    assert policy["policy"] == "minimum_recall_met_then_maximize_precision"


def test_batch_alert_cap_is_strict_with_tied_probabilities() -> None:
    probability = pd.Series([0.9] * 10).to_numpy()
    predicted = _apply_batch_alert_cap(
        probability,
        threshold=0.5,
        max_alert_fraction=0.4,
    )
    assert predicted.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_batch_policy_never_promotes_rows_below_threshold() -> None:
    probability = pd.Series([value / 100 for value in range(30)]).to_numpy()
    predicted = _apply_batch_alert_cap(
        probability,
        threshold=0.20,
        max_alert_fraction=0.4,
        target_alert_fraction=0.35,
    )
    assert int(predicted.sum()) == 10
    assert predicted.mean() <= 0.4
    assert (probability[predicted.astype(bool)] >= 0.20).all()

    none = _apply_batch_alert_cap(
        probability,
        threshold=0.99,
        max_alert_fraction=0.4,
        target_alert_fraction=0.35,
    )
    assert int(none.sum()) == 0


def test_single_company_uses_historical_threshold_without_batch_cap() -> None:
    probability = pd.Series([0.9]).to_numpy()
    predicted = _apply_batch_alert_cap(
        probability,
        threshold=0.5,
        max_alert_fraction=0.4,
        target_alert_fraction=0.35,
    )
    assert predicted.tolist() == [1]


def test_prediction_derives_agent_semantic_ratios_from_raw_counts(tmp_path: Path) -> None:
    matrix, base, agent = _fixture(tmp_path)
    result = train_stable_agent_model(
        matrix,
        base_summary_path=base,
        agent_summary_path=agent,
        output_dir=tmp_path / "derived-model",
        base_feature_limit=1,
        agent_feature_limit=1,
        permutation_repeats=1,
        base_feature_whitelist=("base_signal", "base_noise"),
        agent_feature_whitelist=("agent_role_finance_share",),
    )
    bundle = load_model_bundle(result.model_path)
    source = pd.read_csv(matrix)
    test_rows = source[pd.to_datetime(source["prospectus_date"]).dt.year == 2024]
    assert "agent_role_finance_share" not in test_rows
    predicted = predict_feature_rows(bundle, test_rows)
    assert len(predicted) == 30


def test_engineered_features_use_only_existing_predictors() -> None:
    frame = pd.DataFrame(
        {
            "fundamental_revenue_latest_hkd_mn": [100.0, 0.0],
            "offering_market_cap_mid_hkd": [500.0, 200.0],
            "offering_net_proceeds_hkd": [50.0, 40.0],
            "fundamental_operating_cash_flow_latest_hkd_mn": [-20.0, 5.0],
            "fundamental_ending_cash_latest_native": [30.0, 10.0],
            "fundamental_fx_to_hkd": [1.0, 1.0],
            "fundamental_revenue_cagr_3y_pct": [25.0, np.nan],
            "offering_share_ratio_pct": [10.0, np.nan],
            "market_volatility_20d_pct": [2.0, 3.0],
            "agent_finding_count": [2, 1],
            "agent_unique_risk_type_count": [2, 1],
            "agent_severity_low_count": [0, 0],
            "agent_severity_medium_count": [1, 1],
            "agent_severity_high_count": [1, 0],
            "agent_severity_critical_count": [0, 0],
            "agent_role_finance_finding_count": [1, 0],
            "agent_role_legal_finding_count": [0, 0],
            "agent_role_business_finding_count": [1, 1],
            "agent_role_market_finding_count": [0, 0],
            "agent_uncertainty_count": [0, 0],
            "agent_semantic_family_financial_share": [0.5, 0.0],
            "agent_semantic_family_market_liquidity_share": [0.0, 0.0],
            "agent_semantic_family_governance_share": [0.0, 0.0],
            "agent_semantic_has_related_party_transaction": [0, 0],
            "agent_semantic_has_free_float_liquidity": [0, 0],
            "agent_semantic_has_cash_burn_liquidity": [1, 0],
            "agent_semantic_supported_finding_count": [2, 1],
            "agent_semantic_distinct_subtype_count": [2, 1],
        }
    )
    result = add_derived_prediction_features(frame)
    assert result.loc[0, "engineered_price_to_sales"] == 5.0
    assert result.loc[0, "engineered_fundraising_to_revenue"] == 0.5
    assert result.loc[0, "engineered_operating_cash_flow_margin"] == -0.2
    assert result.loc[0, "engineered_cash_replenishment_ratio"] == 2.5
    assert result.loc[0, "engineered_cash_runway_months"] == 18.0
    assert pd.isna(result.loc[1, "engineered_cash_runway_months"])
    assert pd.isna(result.loc[1, "engineered_price_to_sales"])
    assert result.loc[1, "missing_revenue_cagr_3y"] == 1


def test_time_decay_weights_prioritize_recent_years() -> None:
    rows = pd.DataFrame({"modeling_year": [2020, 2021, 2022, 2023]})
    np.testing.assert_allclose(
        _time_decay_sample_weight(rows),
        np.array([0.50, 0.65, 0.80, 1.00]),
    )
