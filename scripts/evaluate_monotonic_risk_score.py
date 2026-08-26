"""评估跨期稳定、方向受约束的五日显著下跌风险评分。

该实验不覆盖线上模型。每个验证阶段只使用此前样本建立参考分布，
避免小样本树模型把稳定风险方向学反。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from optimize_high_risk_recall_model import (
    TARGET,
    _choose_recall_threshold,
    _current_model_holdout_metrics,
    _load_combined_frame,
    _metrics,
    _rank_capacity_metrics,
)


RISK_DIRECTIONS = {
    "offering_offer_price_low_hkd": "down",
    "offering_offer_price_high_hkd": "down",
    "offering_offer_price_mid_hkd": "down",
    "offering_market_cap_low_hkd": "down",
    "offering_market_cap_high_hkd": "down",
    "offering_market_cap_mid_hkd": "down",
    "fundamental_revenue_latest_native": "down",
    "fundamental_ending_cash_latest_native": "down",
    "customer_top5_ratio_pct": "up",
    "offering_share_ratio_pct": "up",
    "market_return_5d_pct": "down",
}

FEATURE_SETS = {
    "compact_stable": (
        "offering_offer_price_mid_hkd",
        "offering_market_cap_mid_hkd",
        "fundamental_revenue_latest_native",
        "fundamental_ending_cash_latest_native",
        "customer_top5_ratio_pct",
        "offering_share_ratio_pct",
    ),
    "compact_stable_plus_market": (
        "offering_offer_price_mid_hkd",
        "offering_market_cap_mid_hkd",
        "fundamental_revenue_latest_native",
        "fundamental_ending_cash_latest_native",
        "customer_top5_ratio_pct",
        "offering_share_ratio_pct",
        "market_return_5d_pct",
    ),
    "all_stable_plus_market": tuple(RISK_DIRECTIONS),
}


def _feature_weight(train: pd.DataFrame, feature: str, weighted: bool) -> float:
    if not weighted:
        return 1.0
    values = pd.to_numeric(train[feature], errors="coerce")
    valid = values.notna()
    labels = train.loc[valid, TARGET].astype(int)
    if valid.sum() < 30 or labels.nunique() < 2 or values.loc[valid].nunique() < 2:
        return 0.25
    auc = float(roc_auc_score(labels, values.loc[valid]))
    # 仅用训练期估算强度，方向由跨期业务审查冻结。
    return max(0.25, 2.0 * abs(auc - 0.5))


def _score(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    features: tuple[str, ...],
    *,
    weighted: bool,
) -> np.ndarray:
    total = np.zeros(len(rows), dtype=float)
    total_weight = 0.0
    for feature in features:
        reference = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy()
        values = pd.to_numeric(rows[feature], errors="coerce").to_numpy()
        weight = _feature_weight(train, feature, weighted)
        if not len(reference):
            risk = np.full(len(rows), 0.5, dtype=float)
        else:
            ordered = np.sort(reference)
            percentile = np.searchsorted(ordered, values, side="right") / len(ordered)
            risk = percentile if RISK_DIRECTIONS[feature] == "up" else 1.0 - percentile
            risk[np.isnan(values)] = 0.5
        total += risk * weight
        total_weight += weight
    return total / total_weight


def _calibrate_against_history(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    features: tuple[str, ...],
    *,
    weighted: bool,
) -> np.ndarray:
    reference_score = np.sort(_score(train, train, features, weighted=weighted))
    row_score = _score(train, rows, features, weighted=weighted)
    return np.searchsorted(reference_score, row_score, side="right") / len(
        reference_score
    )


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


def main(
    project_root: Path = Path.cwd(),
    output_dir: Path = Path(
        "artifacts/experiments/monotonic_high_risk_score_20260826"
    ),
    holdout_start: str = "2025-12-01",
) -> None:
    root = project_root.resolve()
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    frame = _load_combined_frame(root)
    holdout_timestamp = pd.Timestamp(holdout_start)

    candidates: list[dict[str, Any]] = []
    for feature_set, features in FEATURE_SETS.items():
        for weighted in (False, True):
            for calibrated in (False, True):
                probabilities: list[np.ndarray] = []
                labels_list: list[np.ndarray] = []
                fold_metrics: list[dict[str, Any]] = []
                for train, validation, name in _folds(frame, holdout_timestamp):
                    if calibrated:
                        probability = _calibrate_against_history(
                            train, validation, features, weighted=weighted
                        )
                    else:
                        probability = _score(
                            train, validation, features, weighted=weighted
                        )
                    labels = validation[TARGET].astype(int).to_numpy()
                    probabilities.append(probability)
                    labels_list.append(labels)
                    fold_metrics.append(
                        {
                            "fold": name,
                            "sample_count": int(len(labels)),
                            "positive_count": int(labels.sum()),
                            "roc_auc": float(roc_auc_score(labels, probability)),
                            "pr_auc": float(
                                average_precision_score(labels, probability)
                            ),
                            "rank_recall_at_40pct": _rank_capacity_metrics(
                                labels, probability, 0.40
                            )["recall"],
                        }
                    )
                dev_probability = np.concatenate(probabilities)
                dev_labels = np.concatenate(labels_list)
                threshold = _choose_recall_threshold(dev_labels, dev_probability)
                metrics = _metrics(dev_labels, dev_probability, threshold)
                metrics["rank_recall_at_40pct"] = _rank_capacity_metrics(
                    dev_labels, dev_probability, 0.40
                )["recall"]
                metrics["worst_fold_pr_auc"] = min(
                    row["pr_auc"] for row in fold_metrics
                )
                candidates.append(
                    {
                        "feature_set": feature_set,
                        "weighted": weighted,
                        "calibrated": calibrated,
                        "features": features,
                        "threshold": threshold,
                        "metrics": metrics,
                        "folds": fold_metrics,
                    }
                )

    eligible = [
        item for item in candidates if item["metrics"]["alert_fraction"] <= 0.60
    ]
    best = max(
        eligible,
        key=lambda item: (
            (item["metrics"]["pr_auc"] + item["metrics"]["worst_fold_pr_auc"])
            / 2.0,
            item["metrics"]["rank_recall_at_40pct"],
            item["metrics"]["f2"],
        ),
    )

    train = frame.loc[frame["prospectus_date"].lt(holdout_timestamp)]
    holdout = frame.loc[frame["prospectus_date"].ge(holdout_timestamp)]
    if best["calibrated"]:
        holdout_probability = _calibrate_against_history(
            train,
            holdout,
            best["features"],
            weighted=best["weighted"],
        )
    else:
        holdout_probability = _score(
            train,
            holdout,
            best["features"],
            weighted=best["weighted"],
        )
    holdout_labels = holdout[TARGET].astype(int).to_numpy()
    holdout_metrics = _metrics(
        holdout_labels, holdout_probability, best["threshold"]
    )
    holdout_rank = {
        str(fraction): _rank_capacity_metrics(
            holdout_labels, holdout_probability, fraction
        )
        for fraction in (0.20, 0.30, 0.40, 0.50, 0.60)
    }
    current = _current_model_holdout_metrics(root, holdout)

    predictions = holdout[
        ["company_id", "company_name", "prospectus_date", TARGET]
    ].copy()
    predictions["risk_score"] = holdout_probability
    predictions["predicted_significant_drop_5d"] = (
        holdout_probability >= best["threshold"]
    ).astype(int)
    predictions.to_csv(
        destination / "predictions_holdout.csv", index=False, encoding="utf-8-sig"
    )

    candidate_rows = []
    for item in candidates:
        candidate_rows.append(
            {
                "feature_set": item["feature_set"],
                "weighted": item["weighted"],
                "calibrated": item["calibrated"],
                "feature_count": len(item["features"]),
                **item["metrics"],
            }
        )
    pd.DataFrame(candidate_rows).to_csv(
        destination / "candidate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "schema_version": 1,
        "target": TARGET,
        "holdout_start": holdout_timestamp.date().isoformat(),
        "test_used_for_candidate_selection": False,
        "selected": best,
        "holdout": holdout_metrics,
        "holdout_rank_capacity": holdout_rank,
        "current_model_holdout": current,
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    typer.run(main)
