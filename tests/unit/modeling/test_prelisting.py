from __future__ import annotations

from datetime import date

from ipo_risk.domain import FeatureValue
from ipo_risk.modeling.prelisting import (
    ModelFeatureContract,
    _merge_supplemental,
    load_model_feature_contract,
    project_model_inputs,
)


def test_rag_supplement_only_fills_missing_values() -> None:
    available_at = date(2025, 1, 1)
    deterministic = FeatureValue(
        value=188.4,
        available_at=available_at,
        source_ref="table:T-1",
    )
    rag_value = FeatureValue(
        value=200.0,
        available_at=available_at,
        source_ref="rag:C-1",
    )
    rag_only = FeatureValue(
        value=25.0,
        available_at=available_at,
        source_ref="rag:C-2",
    )
    features = {"offering_net_proceeds_hkd": deterministic}
    warnings: list[str] = []

    _merge_supplemental(
        features,
        {
            "offering_net_proceeds_hkd": rag_value,
            "customer_top5_ratio_pct": rag_only,
        },
        warnings,
    )

    assert features["offering_net_proceeds_hkd"] is deterministic
    assert features["customer_top5_ratio_pct"] is rag_only
    assert warnings == ["rag_conflict_kept_deterministic:offering_net_proceeds_hkd"]


def test_candidate_pool_can_be_wider_than_frozen_model_contract() -> None:
    contract = ModelFeatureContract.from_payload(
        {
            "model_name": "test",
            "minimum_feature_completeness": 0.5,
            "review_below_completeness": 0.75,
            "features": [
                {"name": "offering_net_proceeds_hkd", "group": "offering"},
                {"name": "agent_finding_count", "group": "agent"},
            ],
        }
    )
    available_at = date(2025, 1, 1)
    candidates = {
        "agent_finding_count": FeatureValue(
            value=6,
            available_at=available_at,
            source_ref="analysis:run",
        ),
        "offering_net_proceeds_hkd": FeatureValue(
            value=188.4,
            available_at=available_at,
            source_ref="chunks:C-1",
        ),
        "fundamental_revenue_cagr_3y_pct": FeatureValue(
            value=20.0,
            available_at=available_at,
            source_ref="table:T-1",
        ),
    }

    projection = project_model_inputs(
        candidates,
        contract,
        prediction_as_of=date(2025, 1, 10),
    )

    assert list(projection.values) == [
        "offering_net_proceeds_hkd",
        "agent_finding_count",
    ]
    assert projection.status == "ready"
    assert projection.completeness == 1.0


def test_model_contract_rejects_future_information() -> None:
    contract = ModelFeatureContract.from_payload(
        {
            "model_name": "test",
            "features": [
                {
                    "name": "market_volatility_20d_pct",
                    "group": "market",
                    "required": True,
                }
            ],
        }
    )
    projection = project_model_inputs(
        {
            "market_volatility_20d_pct": FeatureValue(
                value=18.0,
                available_at=date(2025, 1, 11),
                source_ref="market",
            )
        },
        contract,
        prediction_as_of=date(2025, 1, 10),
    )

    assert projection.status == "blocked"
    assert projection.values["market_volatility_20d_pct"] is None
    assert "market_volatility_20d_pct" in projection.missing_features


def test_target_columns_are_forbidden_in_model_contract() -> None:
    try:
        ModelFeatureContract.from_payload(
            {
                "model_name": "test",
                "features": ["target_significant_drop_5d"],
            }
        )
    except ValueError as exc:
        assert "target columns are forbidden" in str(exc)
    else:
        raise AssertionError("target feature must be rejected")


def test_contract_blocks_unit_and_formula_drift() -> None:
    contract = ModelFeatureContract.from_payload(
        {
            "model_name": "test",
            "prediction_as_of": "prospectus_date",
            "features": [
                {
                    "name": "offering_net_proceeds_hkd",
                    "group": "offering",
                    "unit": "HKD",
                    "formula_version": "structured_fundamentals_v1",
                }
            ],
        }
    )
    projection = project_model_inputs(
        {"offering_net_proceeds_hkd": 188_400_000.0},
        contract,
        feature_units={"offering_net_proceeds_hkd": "HKD_million"},
        feature_formula_versions={
            "offering_net_proceeds_hkd": "structured_fundamentals_v0"
        },
    )

    assert projection.status == "blocked"
    assert any("元数据不一致" in issue for issue in projection.issues)


def test_listing_date_contract_blocks_when_listing_date_is_missing() -> None:
    contract = ModelFeatureContract.from_payload(
        {
            "model_name": "test",
            "prediction_as_of": "listing_date",
            "features": [{"name": "market_volatility_20d_pct", "group": "market"}],
        }
    )

    projection = project_model_inputs(
        {"market_volatility_20d_pct": 12.0},
        contract,
    )

    assert projection.status == "blocked"
    assert any("缺少上市日期" in issue for issue in projection.issues)


def test_sidecar_contract_must_match_model_name_and_feature_order(tmp_path) -> None:
    sidecar = tmp_path / "feature_contract.json"
    sidecar.write_text(
        """
        {
          "model_name": "wrong-model",
          "prediction_as_of": "prospectus_date",
          "features": [{"name": "b"}, {"name": "a"}]
        }
        """,
        encoding="utf-8",
    )

    try:
        load_model_feature_contract(
            {"model_name": "expected-model", "features": ["a", "b"]},
            sidecar_path=sidecar,
        )
    except ValueError as exc:
        assert "model_name mismatch" in str(exc)
    else:
        raise AssertionError("mismatched sidecar must be rejected")
