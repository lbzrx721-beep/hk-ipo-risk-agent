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
    is_multi_seed = isinstance(lightgbm_models, list) and bool(lightgbm_models) and all(
        isinstance(model, Pipeline) for model in lightgbm_models
    )
    is_ensemble = isinstance(bundle.get("logistic_pipeline"), Pipeline) and (
        isinstance(bundle.get("lightgbm_pipeline"), Pipeline) or is_multi_seed
    )
    if not (is_single or is_ensemble):
        raise ValueError(f"invalid model bundle: {path}")
    return bundle


def predict_feature_rows(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """使用冻结的特征列和验证集阈值输出概率及方向性预警。"""
    feature_columns = bundle.get("feature_columns")
    threshold = bundle.get("threshold")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("model bundle does not contain feature_columns")
    if not isinstance(threshold, (int, float)):
        raise ValueError("model bundle does not contain a threshold")
    frame = _prepare_derived_features(frame, required_features=set(feature_columns))
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame missing features: {', '.join(missing)}")

    pipeline = bundle.get("pipeline")
    if isinstance(pipeline, Pipeline):
        probabilities = pipeline.predict_proba(frame[feature_columns])[:, 1]
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
                [
                    model.predict_proba(frame[feature_columns])[:, 1]
                    for model in lightgbm_models
                ],
                axis=0,
            )
        else:
            lightgbm_probability = lightgbm.predict_proba(frame[feature_columns])[:, 1]
        probabilities = (
            float(lightgbm_weight)
            * lightgbm_probability
            + float(logistic_weight)
            * logistic.predict_proba(frame[feature_columns])[:, 1]
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
            eligible = above_threshold[
                np.argsort(-probabilities[above_threshold], kind="stable")
            ][:target]
        else:
            eligible = np.flatnonzero(predicted)
        predicted[:] = 0
        if cap > 0 and len(eligible):
            ranked = eligible[np.argsort(-probabilities[eligible], kind="stable")]
            predicted[ranked[:cap]] = 1
    result["predicted_significant_drop_5d"] = predicted
    return result


def _prepare_derived_features(
    frame: pd.DataFrame, *, required_features: set[str]
) -> pd.DataFrame:
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
    needs_engineered = any(
        feature.startswith(engineered_prefixes) for feature in required_features
    )
    required_derived = derived_agent_features.intersection(required_features)
    if (
        not needs_engineered
        and (not required_derived or required_derived.issubset(frame.columns))
    ):
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


__all__ = ["load_model_bundle", "predict_feature_rows"]
