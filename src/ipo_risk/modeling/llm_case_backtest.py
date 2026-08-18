"""LLM历史案例RAG对照回测的纯数据逻辑。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)


BACKTEST_ARMS = ("without_case_rag", "with_case_rag")


def select_balanced_pilot(
    frame: pd.DataFrame,
    *,
    split: str,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """从指定时间分组中确定性地抽取同时包含正负样本的小规模试验集。"""
    required = {
        "company_id",
        "company_name",
        "prospectus_date",
        "split",
        "target_significant_drop_5d",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature matrix is missing columns: {sorted(missing)}")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")

    candidates = frame.loc[frame["split"].astype(str) == split].copy()
    candidates["_target"] = pd.to_numeric(candidates["target_significant_drop_5d"], errors="coerce")
    candidates = candidates.loc[candidates["_target"].isin((0, 1))]
    candidates["_target"] = candidates["_target"].astype(int)
    if len(candidates) < sample_size:
        raise ValueError(
            f"split {split!r} has only {len(candidates)} labeled rows, below {sample_size}"
        )
    counts = candidates["_target"].value_counts()
    if not {0, 1}.issubset(counts.index):
        raise ValueError(f"split {split!r} does not contain both target classes")

    positive_count = max(
        1,
        min(int(round(sample_size * counts[1] / len(candidates))), sample_size - 1),
    )
    negative_count = sample_size - positive_count
    positive_count = min(positive_count, int(counts[1]))
    negative_count = min(negative_count, int(counts[0]))
    while positive_count + negative_count < sample_size:
        if positive_count < int(counts[1]):
            positive_count += 1
        elif negative_count < int(counts[0]):
            negative_count += 1
        else:
            break

    selected = pd.concat(
        (
            candidates.loc[candidates["_target"] == 1].sample(
                n=positive_count, random_state=random_state
            ),
            candidates.loc[candidates["_target"] == 0].sample(
                n=negative_count, random_state=random_state
            ),
        ),
        ignore_index=True,
    )
    selected["_prospectus_date"] = pd.to_datetime(selected["prospectus_date"], errors="coerce")
    return (
        selected.sort_values(["_prospectus_date", "company_id"])
        .drop(columns=["_target", "_prospectus_date"])
        .reset_index(drop=True)
    )


def compute_backtest_metrics(
    observations: Sequence[Mapping[str, Any]],
    *,
    alert_threshold: int | None = None,
) -> dict[str, dict[str, Any]]:
    """按实验组计算分类、概率、耗时及Token指标。"""
    metrics: dict[str, dict[str, Any]] = {}
    for arm in BACKTEST_ARMS:
        rows = [item for item in observations if item.get("arm") == arm and not item.get("error")]
        if not rows:
            metrics[arm] = {"completed": 0}
            continue
        y_true = np.asarray([int(item["actual_significant_drop_5d"]) for item in rows])
        y_pred = np.asarray(
            [
                int(float(item["risk_score"]) >= alert_threshold)
                if alert_threshold is not None
                else int(item["predicted_significant_drop_5d"])
                for item in rows
            ]
        )
        probabilities = np.asarray([float(item["risk_score"]) / 100.0 for item in rows])
        usage = [item.get("usage") or {} for item in rows]
        estimated_costs = [item.get("estimated_cost_rmb") for item in usage]
        known_costs: list[float] = []
        costs_known = True
        for value in estimated_costs:
            if isinstance(value, int | float):
                known_costs.append(float(value))
            else:
                costs_known = False
        metrics[arm] = {
            "completed": len(rows),
            "positive_count": int(y_true.sum()),
            "predicted_positive_count": int(y_pred.sum()),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "pr_auc": (
                round(float(average_precision_score(y_true, probabilities)), 6)
                if len(np.unique(y_true)) > 1
                else None
            ),
            "brier_score": round(float(brier_score_loss(y_true, probabilities)), 6),
            "elapsed_seconds": round(
                sum(float(item.get("elapsed_seconds") or 0.0) for item in rows), 3
            ),
            "model_calls": sum(int(item.get("calls") or 0) for item in usage),
            "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
            "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
            "cached_input_tokens": sum(int(item.get("cached_input_tokens") or 0) for item in usage),
            "estimated_cost_rmb": (
                round(sum(known_costs), 6) if estimated_costs and costs_known else None
            ),
        }
    return metrics


def comparison_delta(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    """返回案例RAG相对无案例RAG的指标差值。"""
    baseline = metrics.get("without_case_rag", {})
    enhanced = metrics.get("with_case_rag", {})
    result: dict[str, float | None] = {}
    for name in ("accuracy", "precision", "recall", "f1", "pr_auc", "brier_score"):
        before = baseline.get(name)
        after = enhanced.get(name)
        result[name] = (
            round(float(after) - float(before), 6)
            if before is not None and after is not None
            else None
        )
    return result


__all__ = [
    "BACKTEST_ARMS",
    "comparison_delta",
    "compute_backtest_metrics",
    "select_balanced_pilot",
]
