from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_risk.modeling.predict import predict_feature_rows


class _ProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = self._probabilities[: len(frame)]
        return np.column_stack([1.0 - positive, positive])


class _NumericDtypeProbabilityModel(_ProbabilityModel):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
        return super().predict_proba(frame)


def test_selective_policy_marks_low_high_and_uncertain() -> None:
    model = _ProbabilityModel([0.20, 0.50, 0.80])
    bundle = {
        "features": ["x"],
        "threshold": 0.50,
        "weights": {"lgb": 1.0, "hgb": 0.0, "log": 0.0},
        "lgb_models": [model],
        "hgb_models": [model],
        "log_models": [model],
        "selective_policy": {
            "low_probability_max": 0.30,
            "high_probability_min": 0.70,
        },
    }

    result = predict_feature_rows(bundle, pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    assert result["predicted_decision"].tolist() == [
        "low_risk",
        "uncertain",
        "high_risk",
    ]
    assert result["prediction_requires_review"].tolist() == [False, True, False]


def test_blend3_rf_uses_separate_feature_sets_and_component_weights() -> None:
    base = _ProbabilityModel([0.20])
    rf = _ProbabilityModel([0.80])
    bundle = {
        "model_name": "enhanced_blend4_rf_v3",
        "features": ["base_x", "market_x"],
        "base_features": ["base_x"],
        "rf_features": ["base_x", "market_x"],
        "threshold": 0.50,
        "weights": {"lgb": 1.0, "hgb": 0.0, "log": 0.0},
        "component_weights": {"blend3": 0.40, "rf": 0.60},
        "lgb_models": [base],
        "hgb_models": [base],
        "log_models": [base],
        "rf_models": [rf],
    }

    result = predict_feature_rows(
        bundle,
        pd.DataFrame({"base_x": [1.0], "market_x": [2.0]}),
    )

    assert np.isclose(result["predicted_probability"].iloc[0], 0.56)
    assert result["predicted_significant_drop_5d"].tolist() == [1]


def test_runtime_single_row_missing_numeric_feature_keeps_numeric_dtype() -> None:
    model = _NumericDtypeProbabilityModel([0.20])
    feature = "fundamental_revenue_cagr_3y_pct"
    bundle = {
        "features": [feature],
        "threshold": 0.50,
        "weights": {"lgb": 1.0, "hgb": 0.0, "log": 0.0},
        "lgb_models": [model],
        "hgb_models": [model],
        "log_models": [model],
    }

    result = predict_feature_rows(bundle, pd.DataFrame([{feature: None}]))

    assert result[feature].isna().all()


def test_legacy_base_ensemble_averages_model_probabilities() -> None:
    bundle = {
        "feature_columns": ["base_x"],
        "threshold": 0.50,
        "models": [
            _ProbabilityModel([0.20]),
            _ProbabilityModel([0.60]),
        ],
    }

    result = predict_feature_rows(bundle, pd.DataFrame({"base_x": [1.0]}))

    assert np.isclose(result["predicted_probability"].iloc[0], 0.40)
    assert result["predicted_significant_drop_5d"].tolist() == [0]


def test_multi_seed_bundle_can_apply_percentile_score() -> None:
    bundle = {
        "features": ["x"],
        "models": [
            _ProbabilityModel([0.30]),
            _ProbabilityModel([0.70]),
        ],
        "threshold": 0.50,
        "score_mode": "percentile",
        "reference_probabilities": [
            np.array([0.10, 0.20, 0.30, 0.40]),
            np.array([0.50, 0.60, 0.70, 0.80]),
        ],
    }

    result = predict_feature_rows(bundle, pd.DataFrame({"x": [1.0]}))

    assert np.isclose(result["predicted_probability"].iloc[0], 0.75)
    assert result["predicted_significant_drop_5d"].tolist() == [1]
