"""评估树模型主通道与单调评分召回保护网。

阈值仅在时间顺序开发折上选择；2025年12月只用于验收。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score

from evaluate_monotonic_risk_score import (
    FEATURE_SETS,
    _calibrate_against_history,
)
from optimize_high_risk_recall_model import (
    CORE_STABLE_FEATURES,
    RECENT_IPO_FEATURE,
    TARGET,
    _build_random_forest,
    _fit_final_models,
    _fit_predict,
    _load_combined_frame,
    _metrics,
    _predict_models,
)


TREE_FEATURES = (*CORE_STABLE_FEATURES, RECENT_IPO_FEATURE)
MONOTONIC_FEATURES = FEATURE_SETS["compact_stable"]


def _folds(frame: pd.DataFrame, holdout_start: pd.Timestamp):
    return (
        (
            frame.loc[frame["modeling_year"].between(2020, 2023)],
            frame.loc[frame["modeling_year"].eq(2024)],
            "2024",
        ),
        (
            frame.loc[frame["modeling_year"].between(2020, 2024)],
            frame.loc[
                frame["prospectus_date"].between(
                    "2025-01-01", "2025-07-01", inclusive="left"
                )
            ],
            "2025_h1",
        ),
        (
            frame.loc[frame["prospectus_date"].lt("2025-07-01")],
            frame.loc[
                frame["prospectus_date"].between(
                    "2025-07-01", holdout_start, inclusive="left"
                )
            ],
            "2025_h2_before_holdout",
        ),
    )


def _policy_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "alert_fraction": float(predicted.mean()),
        "alert_count": int(predicted.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def main(
    project_root: Path = Path.cwd(),
    output_dir: Path = Path(
        "artifacts/experiments/high_risk_recall_guard_20260826"
    ),
    holdout_start: str = "2025-12-01",
) -> None:
    root = project_root.resolve()
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    frame = _load_combined_frame(root)
    holdout_timestamp = pd.Timestamp(holdout_start)

    fold_payloads: list[dict[str, Any]] = []
    for train, validation, name in _folds(frame, holdout_timestamp):
        tree_score = _fit_predict(
            train,
            validation,
            TREE_FEATURES,
            _build_random_forest,
            weighting="none",
            score_mode="percentile",
        )
        monotonic_score = _calibrate_against_history(
            train,
            validation,
            MONOTONIC_FEATURES,
            weighted=False,
        )
        fold_payloads.append(
            {
                "name": name,
                "labels": validation[TARGET].astype(int).to_numpy(),
                "tree_score": tree_score,
                "monotonic_score": monotonic_score,
            }
        )

    policies: list[dict[str, Any]] = []
    for tree_threshold in np.arange(0.45, 0.851, 0.025):
        for guard_threshold in np.arange(0.45, 0.901, 0.025):
            fold_metrics: list[dict[str, Any]] = []
            all_labels: list[np.ndarray] = []
            all_predicted: list[np.ndarray] = []
            for payload in fold_payloads:
                predicted = (
                    (payload["tree_score"] >= tree_threshold)
                    | (payload["monotonic_score"] >= guard_threshold)
                ).astype(int)
                fold_metrics.append(
                    {
                        "fold": payload["name"],
                        **_policy_metrics(payload["labels"], predicted),
                    }
                )
                all_labels.append(payload["labels"])
                all_predicted.append(predicted)
            aggregate = _policy_metrics(
                np.concatenate(all_labels), np.concatenate(all_predicted)
            )
            if aggregate["alert_fraction"] > 0.70 or any(
                row["alert_fraction"] > 0.70 for row in fold_metrics
            ):
                continue
            policies.append(
                {
                    "tree_threshold": float(tree_threshold),
                    "guard_threshold": float(guard_threshold),
                    "aggregate": aggregate,
                    "worst_fold_recall": min(row["recall"] for row in fold_metrics),
                    "folds": fold_metrics,
                }
            )
    if not policies:
        raise ValueError("没有召回保护策略满足70%的预警容量上限")
    best = max(
        policies,
        key=lambda item: (
            item["worst_fold_recall"],
            item["aggregate"]["recall"],
            item["aggregate"]["f2"],
            item["aggregate"]["precision"],
            -item["aggregate"]["alert_fraction"],
        ),
    )

    train = frame.loc[frame["prospectus_date"].lt(holdout_timestamp)]
    holdout = frame.loc[frame["prospectus_date"].ge(holdout_timestamp)]
    tree_models = _fit_final_models(
        train,
        TREE_FEATURES,
        _build_random_forest,
        weighting="none",
        score_mode="percentile",
        as_of=holdout_timestamp,
    )
    tree_score = _predict_models(tree_models, holdout, TREE_FEATURES)
    monotonic_score = _calibrate_against_history(
        train,
        holdout,
        MONOTONIC_FEATURES,
        weighted=False,
    )
    predicted = (
        (tree_score >= best["tree_threshold"])
        | (monotonic_score >= best["guard_threshold"])
    ).astype(int)
    labels = holdout[TARGET].astype(int).to_numpy()
    holdout_policy = _policy_metrics(labels, predicted)
    # 连续分数用于衡量排序质量；分类结果由双门槛策略给出。
    combined_score = np.maximum(tree_score, monotonic_score)
    holdout_ranking = _metrics(labels, combined_score, 0.5)

    predictions = holdout[
        ["company_id", "company_name", "prospectus_date", TARGET]
    ].copy()
    predictions["tree_percentile"] = tree_score
    predictions["monotonic_percentile"] = monotonic_score
    predictions["predicted_significant_drop_5d"] = predicted
    predictions.to_csv(
        destination / "predictions_holdout.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "schema_version": 1,
        "target": TARGET,
        "holdout_start": holdout_timestamp.date().isoformat(),
        "test_used_for_policy_selection": False,
        "tree_features": list(TREE_FEATURES),
        "monotonic_features": list(MONOTONIC_FEATURES),
        "selected_policy": best,
        "holdout_policy_metrics": holdout_policy,
        "holdout_ranking_metrics": {
            "roc_auc": holdout_ranking["roc_auc"],
            "pr_auc": holdout_ranking["pr_auc"],
        },
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    typer.run(main)
