"""保存模型的加载与批量预测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"invalid model bundle: {path}")
    is_single = isinstance(bundle.get("pipeline"), Pipeline)
    lightgbm_models = bundle.get("lightgbm_pipelines")
    is_multi_seed = (
        isinstance(lightgbm_models, list)
        and bool(lightgbm_models)
        and all(isinstance(model, Pipeline) for model in lightgbm_models)
    )
    is_ensemble = isinstance(bundle.get("logistic_pipeline"), Pipeline) and (
        isinstance(bundle.get("lightgbm_pipeline"), Pipeline) or is_multi_seed
    )
    base_models = bundle.get("models")
    is_base_ensemble = (
        isinstance(base_models, list)
        and bool(base_models)
        and all(isinstance(model, Pipeline) for model in base_models)
        and isinstance(bundle.get("features"), list)
        and isinstance(bundle.get("threshold"), (int, float))
    )
    is_blend3 = (
        isinstance(bundle.get("lgb_models"), list)
        and isinstance(bundle.get("hgb_models"), list)
        and isinstance(bundle.get("log_models"), list)
        and bool(bundle.get("lgb_models"))
        and isinstance(bundle.get("weights"), dict)
        and isinstance(bundle.get("features"), list)
        and isinstance(bundle.get("threshold"), (int, float))
    )
    is_blend3_rf = (
        is_blend3
        and isinstance(bundle.get("rf_models"), list)
        and bool(bundle.get("rf_models"))
        and isinstance(bundle.get("base_features"), list)
        and isinstance(bundle.get("rf_features"), list)
        and isinstance(bundle.get("component_weights"), dict)
    )
    is_supported_blend = is_blend3_rf if "rf_models" in bundle else is_blend3
    if not (is_single or is_ensemble or is_base_ensemble or is_supported_blend):
        raise ValueError(f"invalid model bundle: {path}")
    return bundle


def predict_feature_rows(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """使用冻结的特征列和验证集阈值输出概率及方向性预警。"""
    feature_columns = bundle.get("feature_columns") or bundle.get("features")
    threshold = bundle.get("threshold")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("model bundle does not contain feature_columns")
    if not isinstance(threshold, (int, float)):
        raise ValueError("model bundle does not contain a threshold")
    frame = _prepare_derived_features(frame, required_features=set(feature_columns))
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame missing features: {', '.join(missing)}")
    frame = frame.copy()
    frame[feature_columns] = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if bundle.get("rf_models") is not None:
        probabilities = _predict_blend3_rf(bundle, frame)
    elif bundle.get("lgb_models") is not None:
        probabilities = _predict_blend3(bundle, frame, feature_columns)
    elif isinstance(bundle.get("models"), list):
        probabilities = np.mean(
            [model.predict_proba(frame[feature_columns])[:, 1] for model in bundle["models"]],
            axis=0,
        )
    elif isinstance(bundle.get("pipeline"), Pipeline):
        probabilities = bundle["pipeline"].predict_proba(frame[feature_columns])[:, 1]
    else:
        logistic = bundle.get("logistic_pipeline")
        lightgbm = bundle.get("lightgbm_pipeline")
        lightgbm_models = bundle.get("lightgbm_pipelines")
        lightgbm_weight = bundle.get("lightgbm_weight")
        logistic_weight = bundle.get("logistic_weight")
        valid_multi_seed = (
            isinstance(lightgbm_models, list)
            and bool(lightgbm_models)
            and all(isinstance(model, Pipeline) for model in lightgbm_models)
        )
        if not isinstance(logistic, Pipeline) or not (
            isinstance(lightgbm, Pipeline) or valid_multi_seed
        ):
            raise ValueError("model bundle does not contain valid ensemble pipelines")
        if not isinstance(lightgbm_weight, (int, float)) or not isinstance(
            logistic_weight, (int, float)
        ):
            raise ValueError("model bundle does not contain valid ensemble weights")
        if valid_multi_seed:
            lightgbm_probability = np.mean(
                [model.predict_proba(frame[feature_columns])[:, 1] for model in lightgbm_models],
                axis=0,
            )
        else:
            lightgbm_probability = lightgbm.predict_proba(frame[feature_columns])[:, 1]
        probabilities = (
            float(lightgbm_weight) * lightgbm_probability
            + float(logistic_weight) * logistic.predict_proba(frame[feature_columns])[:, 1]
        )
    result = frame.copy()
    result["predicted_probability"] = probabilities
    predicted = (probabilities >= float(threshold)).astype(int)
    max_alert_fraction = bundle.get("max_alert_fraction")
    if isinstance(max_alert_fraction, (int, float)) and len(probabilities) >= 10:
        cap = int(np.floor(len(probabilities) * float(max_alert_fraction)))
        target_alert_fraction = bundle.get("target_alert_fraction")
        if isinstance(target_alert_fraction, (int, float)) and len(probabilities) >= 10:
            target = min(
                cap,
                int(np.floor(len(probabilities) * float(target_alert_fraction))),
            )
            above_threshold = np.flatnonzero(predicted)
            eligible = above_threshold[np.argsort(-probabilities[above_threshold], kind="stable")][
                :target
            ]
        else:
            eligible = np.flatnonzero(predicted)
        predicted[:] = 0
        if cap > 0 and len(eligible):
            ranked = eligible[np.argsort(-probabilities[eligible], kind="stable")]
            predicted[ranked[:cap]] = 1
    result["predicted_significant_drop_5d"] = predicted
    selective_policy = bundle.get("selective_policy")
    if isinstance(selective_policy, dict):
        low = selective_policy.get("low_probability_max")
        high = selective_policy.get("high_probability_min")
        if (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and 0.0 <= float(low) < float(high) <= 1.0
        ):
            decisions = np.full(len(probabilities), "uncertain", dtype=object)
            decisions[probabilities <= float(low)] = "low_risk"
            decisions[probabilities >= float(high)] = "high_risk"
            result["predicted_decision"] = decisions
            result["prediction_requires_review"] = decisions == "uncertain"
    return result


def _predict_blend3(
    bundle: dict[str, Any], frame: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    """enhanced_blend3_v2 格式：每 seed 三模型加权平均，再跨 seed 平均。"""
    lgb_models = bundle["lgb_models"]
    hgb_models = bundle["hgb_models"]
    log_models = bundle["log_models"]
    weights = bundle["weights"]
    if not (
        isinstance(weights, dict)
        and {"lgb", "hgb", "log"}.issubset(weights)
        and len(lgb_models) == len(hgb_models) == len(log_models)
    ):
        raise ValueError("model bundle does not contain valid blend3 weights/models")
    per_seed: list[np.ndarray] = []
    for i in range(len(lgb_models)):
        per_seed.append(
            float(weights["lgb"]) * lgb_models[i].predict_proba(frame[feature_columns])[:, 1]
            + float(weights["hgb"]) * hgb_models[i].predict_proba(frame[feature_columns])[:, 1]
            + float(weights["log"]) * log_models[i].predict_proba(frame[feature_columns])[:, 1]
        )
    return np.mean(per_seed, axis=0)


def _predict_blend3_rf(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """v3 格式：融合原 blend3 与带市场宽度特征的随机森林。"""
    base_features = bundle.get("base_features")
    rf_features = bundle.get("rf_features")
    rf_models = bundle.get("rf_models")
    component_weights = bundle.get("component_weights")
    if not (
        isinstance(base_features, list)
        and base_features
        and isinstance(rf_features, list)
        and rf_features
        and isinstance(rf_models, list)
        and rf_models
        and isinstance(component_weights, dict)
        and {"blend3", "rf"}.issubset(component_weights)
    ):
        raise ValueError("model bundle does not contain valid blend3-rf components")
    base_probability = _predict_blend3(bundle, frame, base_features)
    rf_probability = np.mean(
        [model.predict_proba(frame[rf_features])[:, 1] for model in rf_models],
        axis=0,
    )
    return (
        float(component_weights["blend3"]) * base_probability
        + float(component_weights["rf"]) * rf_probability
    )


def _prepare_derived_features(frame: pd.DataFrame, *, required_features: set[str]) -> pd.DataFrame:
    from ipo_risk.modeling.agent_precision import (
        AGENT_PRECISION_FEATURE_NAMES,
        reconcile_precision_agent_features,
    )

    if required_features.intersection(AGENT_PRECISION_FEATURE_NAMES):
        frame = reconcile_precision_agent_features(frame)
    derived_agent_features = {
        "agent_severity_weighted_mean",
        "agent_high_or_critical_share",
        "agent_unique_risk_type_share",
        "agent_role_finance_share",
        "agent_role_legal_share",
        "agent_role_business_share",
        "agent_role_market_share",
        "agent_uncertainty_share",
    }
    engineered_prefixes = ("engineered_", "missing_", "agent_interaction_")
    needs_engineered = any(feature.startswith(engineered_prefixes) for feature in required_features)
    required_derived = derived_agent_features.intersection(required_features)
    derived_values_ready = required_derived.issubset(frame.columns) and all(
        pd.to_numeric(frame[name], errors="coerce").notna().any() for name in required_derived
    )
    if not needs_engineered and (not required_derived or derived_values_ready):
        return frame
    needs_agent_derived = bool(required_derived) or any(
        feature.startswith("agent_interaction_") for feature in required_features
    )
    if needs_agent_derived and "agent_finding_count" not in frame:
        return frame
    from ipo_risk.modeling.stable_agent_model import (
        add_derived_prediction_features,
    )

    return add_derived_prediction_features(frame)


__all__ = [
    "load_model_bundle",
    "predict_feature_rows",
    "_predict_blend3",
    "_predict_blend3_rf",
]
