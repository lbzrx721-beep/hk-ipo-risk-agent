"""可解释线性模型的全局系数和逐样本贡献。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def explain_logistic_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    *,
    top_n: int = 5,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """返回全局系数和每个样本对下跌概率方向贡献最大的特征。"""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("pipeline preprocessor must be a ColumnTransformer")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("pipeline classifier must be LogisticRegression")

    names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    transformed = np.asarray(preprocessor.transform(frame), dtype=float)
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    global_rows = [
        {"feature": name, "coefficient": float(coefficient)}
        for name, coefficient in sorted(
            zip(names, coefficients, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    ]
    local_rows: list[list[dict[str, Any]]] = []
    for transformed_row in transformed:
        contributions = transformed_row * coefficients
        ranked = sorted(
            zip(names, contributions, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]
        local_rows.append(
            [
                {
                    "feature": name,
                    "contribution": float(value),
                    "direction": "提高下跌风险" if value > 0 else "降低下跌风险",
                }
                for name, value in ranked
                if value != 0
            ]
        )
    return global_rows, local_rows


def _clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


__all__ = ["explain_logistic_pipeline"]
