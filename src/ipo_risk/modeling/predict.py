"""保存模型的加载与批量预测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("pipeline"), Pipeline):
        raise ValueError(f"invalid model bundle: {path}")
    return bundle


def predict_feature_rows(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """使用冻结的特征列和验证集阈值输出概率及方向性预警。"""
    pipeline = bundle.get("pipeline")
    feature_columns = bundle.get("feature_columns")
    threshold = bundle.get("threshold")
    if not isinstance(pipeline, Pipeline):
        raise ValueError("model bundle does not contain a sklearn pipeline")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("model bundle does not contain feature_columns")
    if not isinstance(threshold, (int, float)):
        raise ValueError("model bundle does not contain a threshold")
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame missing features: {', '.join(missing)}")

    probabilities = pipeline.predict_proba(frame[feature_columns])[:, 1]
    result = frame.copy()
    result["predicted_probability"] = probabilities
    result["predicted_significant_drop_5d"] = (
        result["predicted_probability"] >= float(threshold)
    ).astype(int)
    return result


__all__ = ["load_model_bundle", "predict_feature_rows"]
