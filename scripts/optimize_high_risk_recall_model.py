"""优化上市后5日显著下跌风险的识别能力。

本脚本不覆盖线上模型。它使用严格的时间顺序：

* 2020-2023 -> 验证 2024；
* 2020-2024 -> 验证 2025 年前三季度；
* 2020-2024 + 留出日前的 2025 样本 -> 测试留出日之后的新样本。

业务目标以少漏报高风险公司为先，阈值选择使用 F2，并要求开发期召回率
不低于 70%。2025 年新增信号只允许在留出期之前发现，留出期仅用于最终评估。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "target_significant_drop_5d"
DEFAULT_HOLDOUT_START = pd.Timestamp("2025-12-01")

CORE_STABLE_FEATURES = (
    "offering_offer_price_low_hkd",
    "offering_offer_price_high_hkd",
    "offering_offer_price_mid_hkd",
    "offering_market_cap_low_hkd",
    "offering_market_cap_high_hkd",
    "offering_market_cap_mid_hkd",
    "fundamental_revenue_latest_native",
    "fundamental_ending_cash_latest_native",
    "customer_top5_ratio_pct",
    "offering_share_ratio_pct",
)
RECENT_IPO_FEATURE = "recent_ipo_1d_mean_return_90d_pct"
ISSUANCE_SEVERITY_FEATURE = (
    "agent_semantic_family_issuance_structure_max_severity"
)

ALLOWED_PREFIXES = (
    "agent_",
    "offering_",
    "fundamental_",
    "market_",
    "recent_ipo_",
    "document_",
    "calendar_",
    "engineered_",
    "missing_",
    "data_quality_",
)


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    model_family: str
    training_window: str
    weighting: str
    score_mode: str
    features: tuple[str, ...]
    dev_probability: np.ndarray
    dev_labels: np.ndarray
    threshold: float
    dev_metrics: dict[str, Any]


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"company_id": str}, low_memory=False)
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    frame["prospectus_date"] = pd.to_datetime(
        frame["prospectus_date"], errors="coerce"
    )
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame


def _load_combined_frame(root: Path) -> pd.DataFrame:
    historical = _read_frame(
        root
        / "data"
        / "processed"
        / "modeling"
        / "agent_augmented_feature_matrix_v5_compatible.csv"
    )
    recent_paths = (
        root
        / "artifacts"
        / "experiments"
        / "agent_2025_30_full_20260825T1700"
        / "agent_augmented_feature_matrix.csv",
        root
        / "artifacts"
        / "experiments"
        / "agent_2025_remaining_deepseek_v4_flash_20260825T203114"
        / "agent_augmented_feature_matrix.csv",
    )
    recent = pd.concat([_read_frame(path) for path in recent_paths], ignore_index=True)
    recent = recent.drop_duplicates("company_id", keep="last")
    historical = historical.loc[historical["prospectus_date"].dt.year.le(2024)]
    combined = pd.concat([historical, recent], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["prospectus_date", TARGET]).copy()
    combined["modeling_year"] = combined["prospectus_date"].dt.year.astype(int)
    combined[TARGET] = combined[TARGET].astype(int)
    return combined.sort_values(["prospectus_date", "company_id"]).reset_index(drop=True)


def _numeric_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, list(features)].copy()
    for feature in features:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result


def _single_feature_auc(rows: pd.DataFrame, feature: str) -> dict[str, float] | None:
    values = pd.to_numeric(rows[feature], errors="coerce")
    labels = rows[TARGET]
    valid = values.notna() & labels.notna()
    if valid.sum() < 30 or values.loc[valid].nunique() < 2:
        return None
    present_labels = labels.loc[valid].astype(int)
    if present_labels.nunique() < 2:
        return None
    raw_auc = float(roc_auc_score(present_labels, values.loc[valid]))
    return {
        "auc_raw": raw_auc,
        "auc_strength": max(raw_auc, 1.0 - raw_auc),
        "missing_pct": float(1.0 - valid.mean()),
        "n": int(valid.sum()),
    }


def _screen_new_2025_signals(
    frame: pd.DataFrame,
    *,
    excluded: set[str],
    holdout_start: pd.Timestamp,
    limit: int = 8,
) -> tuple[list[str], list[dict[str, Any]]]:
    """只用留出期之前的 2025 标签发现新信号。"""
    early_2025 = frame.loc[
        frame["prospectus_date"].between(
            "2025-01-01", holdout_start, inclusive="left"
        )
    ]
    historical = frame.loc[frame["modeling_year"].between(2020, 2024)]
    rows: list[dict[str, Any]] = []
    for feature in frame.columns:
        if feature in excluded or feature.startswith("target_"):
            continue
        if not feature.startswith(ALLOWED_PREFIXES):
            continue
        early_result = _single_feature_auc(early_2025, feature)
        history_result = _single_feature_auc(historical, feature)
        if early_result is None or history_result is None:
            continue
        if early_result["missing_pct"] > 0.35 or history_result["missing_pct"] > 0.50:
            continue
        if early_result["auc_strength"] < 0.60:
            continue
        rows.append(
            {
                "feature": feature,
                "early_2025_auc_raw": early_result["auc_raw"],
                "early_2025_auc_strength": early_result["auc_strength"],
                "early_2025_missing_pct": early_result["missing_pct"],
                "history_auc_raw": history_result["auc_raw"],
                "history_auc_strength": history_result["auc_strength"],
                "history_missing_pct": history_result["missing_pct"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["early_2025_auc_strength"],
            -row["early_2025_missing_pct"],
            row["history_auc_strength"],
        ),
        reverse=True,
    )
    return [row["feature"] for row in ranked[:limit]], ranked


def _build_logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_lightgbm(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=220,
                    learning_rate=0.025,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=30,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                    verbosity=-1,
                    deterministic=True,
                    force_row_wise=True,
                ),
            ),
        ]
    )


def _build_random_forest(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=4,
                    min_samples_leaf=12,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[[int], Pipeline]] = {
    "logistic": _build_logistic,
    "lightgbm": _build_lightgbm,
    "random_forest": _build_random_forest,
}


def _sample_weights(
    train: pd.DataFrame,
    *,
    scheme: str,
    as_of: pd.Timestamp,
) -> np.ndarray | None:
    if scheme == "none":
        return None
    half_life_days = {"half_life_2y": 730.0, "half_life_1y": 365.0}[scheme]
    age_days = (as_of - train["prospectus_date"]).dt.days.clip(lower=0)
    # 防止早年样本完全失去作用，同时让近期制度与市场环境获得更高权重。
    return np.maximum(0.20, np.power(0.5, age_days.to_numpy() / half_life_days))


def _percentile_score(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if not len(ordered):
        return np.asarray(values, dtype=float)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    builder: Callable[[int], Pipeline],
    *,
    weighting: str,
    score_mode: str,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    weights = _sample_weights(
        train,
        scheme=weighting,
        as_of=validation["prospectus_date"].min(),
    )
    train_matrix = _numeric_frame(train, features)
    validation_matrix = _numeric_frame(validation, features)
    for seed in (42, 137, 271):
        model = builder(seed)
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        model.fit(train_matrix, train[TARGET].astype(int), **fit_kwargs)
        validation_probability = model.predict_proba(validation_matrix)[:, 1]
        if score_mode == "percentile":
            reference_probability = model.predict_proba(train_matrix)[:, 1]
            validation_probability = _percentile_score(
                reference_probability, validation_probability
            )
        probabilities.append(validation_probability)
    return np.mean(probabilities, axis=0)


def _choose_recall_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_recall: float = 0.70,
    maximum_alert_fraction: float = 0.60,
) -> float:
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in np.arange(0.10, 0.901, 0.005):
        predicted = probability >= threshold
        recall = recall_score(labels, predicted, zero_division=0)
        alert_fraction = float(predicted.mean())
        if recall < minimum_recall or alert_fraction > maximum_alert_fraction:
            continue
        precision = precision_score(labels, predicted, zero_division=0)
        f2 = fbeta_score(labels, predicted, beta=2, zero_division=0)
        candidates.append((float(f2), float(precision), -alert_fraction, float(threshold)))
    if not candidates:
        # 无法达到目标召回时仍严格遵守预警容量，不能靠“全部报警”制造高召回。
        for threshold in np.arange(0.05, 0.951, 0.005):
            predicted = probability >= threshold
            if float(predicted.mean()) > maximum_alert_fraction:
                continue
            f2 = fbeta_score(labels, predicted, beta=2, zero_division=0)
            precision = precision_score(labels, predicted, zero_division=0)
            recall = recall_score(labels, predicted, zero_division=0)
            candidates.append(
                (
                    float(f2),
                    float(recall),
                    float(precision),
                    float(threshold),
                )
            )
    if not candidates:
        raise ValueError("无法在预警比例上限内找到可用分类阈值")
    return max(candidates)[3]


def _metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "alert_count": int(predicted.sum()),
        "alert_fraction": float(predicted.mean()),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _rank_capacity_metrics(
    labels: np.ndarray, probability: np.ndarray, alert_fraction: float
) -> dict[str, Any]:
    alert_count = max(1, int(np.ceil(len(labels) * alert_fraction)))
    selected = np.argsort(-probability)[:alert_count]
    true_positive = int(labels[selected].sum())
    return {
        "alert_fraction": float(alert_fraction),
        "alert_count": alert_count,
        "true_positive": true_positive,
        "false_positive": int(alert_count - true_positive),
        "false_negative": int(labels.sum() - true_positive),
        "recall": float(true_positive / labels.sum()) if labels.sum() else 0.0,
        "precision": float(true_positive / alert_count),
    }


def _development_predictions(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    builder: Callable[[int], Pipeline],
    *,
    holdout_start: pd.Timestamp,
    training_window: str,
    weighting: str,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    folds = (
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
            f"2025_h2_before_{holdout_start.date().isoformat()}",
        ),
    )
    all_probability: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for train, validation, name in folds:
        if training_window == "recent_3y":
            cutoff = validation["prospectus_date"].min() - pd.DateOffset(years=3)
            train = train.loc[train["prospectus_date"].ge(cutoff)]
        probability = _fit_predict(
            train,
            validation,
            features,
            builder,
            weighting=weighting,
            score_mode=score_mode,
        )
        labels = validation[TARGET].astype(int).to_numpy()
        all_probability.append(probability)
        all_labels.append(labels)
        details.append(
            {
                "fold": name,
                "sample_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "roc_auc": float(roc_auc_score(labels, probability)),
                "pr_auc": float(average_precision_score(labels, probability)),
                "rank_recall_at_40pct": _rank_capacity_metrics(
                    labels, probability, 0.40
                )["recall"],
            }
        )
    return np.concatenate(all_probability), np.concatenate(all_labels), details


def _fit_final_models(
    train: pd.DataFrame,
    features: tuple[str, ...],
    builder: Callable[[int], Pipeline],
    *,
    weighting: str,
    score_mode: str,
    as_of: pd.Timestamp,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    train_matrix = _numeric_frame(train, features)
    weights = _sample_weights(train, scheme=weighting, as_of=as_of)
    for seed in (42, 137, 271):
        model = builder(seed)
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        model.fit(train_matrix, train[TARGET].astype(int), **fit_kwargs)
        reference = None
        if score_mode == "percentile":
            reference = np.sort(model.predict_proba(train_matrix)[:, 1])
        models.append({"model": model, "reference_probability": reference})
    return models


def _predict_models(
    models: list[dict[str, Any]], rows: pd.DataFrame, features: tuple[str, ...]
) -> np.ndarray:
    matrix = _numeric_frame(rows, features)
    probabilities: list[np.ndarray] = []
    for payload in models:
        probability = payload["model"].predict_proba(matrix)[:, 1]
        reference = payload["reference_probability"]
        if reference is not None:
            probability = _percentile_score(reference, probability)
        probabilities.append(probability)
    return np.mean(probabilities, axis=0)


def _current_model_holdout_metrics(root: Path, holdout: pd.DataFrame) -> dict[str, Any] | None:
    path = (
        root
        / "artifacts"
        / "experiments"
        / "agent_2025_full_validation_20260826"
        / "predictions.csv"
    )
    if not path.exists():
        return None
    current = pd.read_csv(path, dtype={"company_id": str})
    current["company_id"] = current["company_id"].astype(str).str.zfill(5)
    joined = holdout[["company_id", TARGET]].merge(
        current[["company_id", "probability", "threshold"]],
        on="company_id",
        how="inner",
    )
    if joined.empty:
        return None
    return _metrics(
        joined[TARGET].astype(int).to_numpy(),
        joined["probability"].to_numpy(),
        float(joined["threshold"].iloc[0]),
    )


def main(
    project_root: Path = Path.cwd(),
    output_dir: Path = Path(
        "artifacts/experiments/high_risk_recall_rolling_20260826"
    ),
    holdout_start: str = "2025-12-01",
) -> None:
    root = project_root.resolve()
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    frame = _load_combined_frame(root)
    holdout_timestamp = pd.Timestamp(holdout_start)

    missing_core = [feature for feature in CORE_STABLE_FEATURES if feature not in frame]
    if missing_core:
        raise ValueError(f"核心稳定特征缺失：{missing_core}")

    excluded = {
        *CORE_STABLE_FEATURES,
        RECENT_IPO_FEATURE,
        ISSUANCE_SEVERITY_FEATURE,
    }
    screened_new, screening_rows = _screen_new_2025_signals(
        frame,
        excluded=excluded,
        holdout_start=holdout_timestamp,
        limit=8,
    )
    pd.DataFrame(screening_rows).to_csv(
        destination / "early_2025_signal_screening.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 新信号在 2025 年内方向快速反转，本轮只使用跨期方向较稳定的核心字段，
    # 并单独检验“近期 IPO 情绪”是否能在滚动验证中带来增益。
    feature_sets = {
        "core_stable": tuple(CORE_STABLE_FEATURES),
        "core_plus_recent_ipo": (*CORE_STABLE_FEATURES, RECENT_IPO_FEATURE),
    }
    training_configs = (
        ("full_history", "none", "raw"),
        ("full_history", "none", "percentile"),
        ("full_history", "half_life_1y", "percentile"),
        ("recent_3y", "none", "percentile"),
        ("recent_3y", "half_life_1y", "percentile"),
    )

    candidates: list[Candidate] = []
    candidate_rows: list[dict[str, Any]] = []
    for feature_set, features in feature_sets.items():
        unique_features = tuple(dict.fromkeys(features))
        for training_window, weighting, score_mode in training_configs:
            for model_family in ("lightgbm", "random_forest"):
                builder = MODEL_BUILDERS[model_family]
                probability, labels, fold_metrics = _development_predictions(
                    frame,
                    unique_features,
                    builder,
                    holdout_start=holdout_timestamp,
                    training_window=training_window,
                    weighting=weighting,
                    score_mode=score_mode,
                )
                threshold = _choose_recall_threshold(labels, probability)
                metrics = _metrics(labels, probability, threshold)
                metrics["rank_recall_at_40pct"] = _rank_capacity_metrics(
                    labels, probability, 0.40
                )["recall"]
                metrics["worst_fold_pr_auc"] = min(
                    fold["pr_auc"] for fold in fold_metrics
                )
                metrics["folds"] = fold_metrics
                candidate = Candidate(
                    feature_set=feature_set,
                    model_family=model_family,
                    training_window=training_window,
                    weighting=weighting,
                    score_mode=score_mode,
                    features=unique_features,
                    dev_probability=probability,
                    dev_labels=labels,
                    threshold=threshold,
                    dev_metrics=metrics,
                )
                candidates.append(candidate)
                candidate_rows.append(
                    {
                        "feature_set": feature_set,
                        "model_family": model_family,
                        "training_window": training_window,
                        "weighting": weighting,
                        "score_mode": score_mode,
                        "feature_count": len(unique_features),
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "folds"
                        },
                    }
                )
                print(
                    f"开发评估：{feature_set}/{model_family}/"
                    f"{training_window}/{weighting}/{score_mode} "
                    f"recall={metrics['recall']:.3f} "
                    f"precision={metrics['precision']:.3f} "
                    f"PR-AUC={metrics['pr_auc']:.3f}",
                    flush=True,
                )

    # 先保证每个时间段都不是偶然有效，再看总体排序与有限预警容量下的召回。
    # 最终 12 月样本不参与候选选择。
    eligible_candidates = [
        item
        for item in candidates
        if item.dev_metrics["alert_fraction"] <= 0.60
    ]
    if not eligible_candidates:
        raise ValueError("没有候选模型满足开发期预警比例不超过60%的硬门禁")
    best = max(
        eligible_candidates,
        key=lambda item: (
            (
                item.dev_metrics["worst_fold_pr_auc"]
                + item.dev_metrics["pr_auc"]
            )
            / 2.0,
            item.dev_metrics["rank_recall_at_40pct"],
            item.dev_metrics["f2"],
            item.dev_metrics["recall"],
        ),
    )

    train_final = frame.loc[frame["prospectus_date"].lt(holdout_timestamp)]
    if best.training_window == "recent_3y":
        train_final = train_final.loc[
            train_final["prospectus_date"].ge(
                holdout_timestamp - pd.DateOffset(years=3)
            )
        ]
    holdout = frame.loc[frame["prospectus_date"].ge(holdout_timestamp)]
    final_models = _fit_final_models(
        train_final,
        best.features,
        MODEL_BUILDERS[best.model_family],
        weighting=best.weighting,
        score_mode=best.score_mode,
        as_of=holdout_timestamp,
    )
    holdout_probability = _predict_models(final_models, holdout, best.features)
    holdout_labels = holdout[TARGET].astype(int).to_numpy()
    holdout_metrics = _metrics(holdout_labels, holdout_probability, best.threshold)
    holdout_rank_metrics = {
        str(fraction): _rank_capacity_metrics(
            holdout_labels, holdout_probability, fraction
        )
        for fraction in (0.20, 0.30, 0.40, 0.50, 0.60)
    }
    current_metrics = _current_model_holdout_metrics(root, holdout)

    predictions = holdout[
        ["company_id", "company_name", "prospectus_date", TARGET]
    ].copy()
    predictions["probability"] = holdout_probability
    predictions["predicted_significant_drop_5d"] = (
        holdout_probability >= best.threshold
    ).astype(int)
    predictions["correct"] = (
        predictions["predicted_significant_drop_5d"] == predictions[TARGET]
    ).astype(int)
    predictions.to_csv(
        destination / "predictions_holdout.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(candidate_rows).sort_values(
        ["pr_auc", "f2", "recall"], ascending=False
    ).to_csv(destination / "candidate_metrics.csv", index=False, encoding="utf-8-sig")

    issuance_rows: list[dict[str, Any]] = []
    for period, rows in (
        ("2020-2023", frame.loc[frame["modeling_year"].between(2020, 2023)]),
        ("2024", frame.loc[frame["modeling_year"].eq(2024)]),
        (
            "2025_before_holdout",
            frame.loc[
                frame["prospectus_date"].between(
                    "2025-01-01", holdout_timestamp, inclusive="left"
                )
            ],
        ),
        ("2025_holdout", holdout),
    ):
        for label in (0, 1):
            values = pd.to_numeric(
                rows.loc[rows[TARGET].eq(label), ISSUANCE_SEVERITY_FEATURE],
                errors="coerce",
            )
            issuance_rows.append(
                {
                    "period": period,
                    "label": label,
                    "count": int(values.notna().sum()),
                    "mean_severity": float(values.mean()),
                    "median_severity": float(values.median()),
                }
            )
    pd.DataFrame(issuance_rows).to_csv(
        destination / "issuance_severity_sanity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "schema_version": 1,
        "target": TARGET,
        "objective": "prioritize_significant_drop_recall_with_controlled_alert_volume",
        "holdout_policy": {
            "development_end_exclusive": holdout_timestamp.date().isoformat(),
            "test_start": holdout_timestamp.date().isoformat(),
            "test_used_for_candidate_selection": False,
        },
        "cohort_counts": {
            "all": int(len(frame)),
            "development": int(len(train_final)),
            "holdout": int(len(holdout)),
            "holdout_positive": int(holdout[TARGET].sum()),
        },
        "core_stable_features": list(CORE_STABLE_FEATURES),
        "recent_ipo_feature_treated_as_new_signal": RECENT_IPO_FEATURE,
        "issuance_severity_policy": {
            "feature": ISSUANCE_SEVERITY_FEATURE,
            "mapping_verified": "low=1, medium=2, high=3, critical=4, absent=0",
            "included_in_model": False,
            "reason": "direction is consistently counter-intuitive and likely confounded",
        },
        "screened_new_2025_features": screened_new,
        "selected": {
            "feature_set": best.feature_set,
            "model_family": best.model_family,
            "training_window": best.training_window,
            "weighting": best.weighting,
            "score_mode": best.score_mode,
            "features": list(best.features),
            "threshold": best.threshold,
            "development_metrics": best.dev_metrics,
        },
        "holdout": holdout_metrics,
        "holdout_rank_capacity": holdout_rank_metrics,
        "current_model_holdout": current_metrics,
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": "high_risk_recall_2025q4_experiment",
            "model_family": best.model_family,
            "training_window": best.training_window,
            "weighting": best.weighting,
            "score_mode": best.score_mode,
            "features": list(best.features),
            "models": final_models,
            "threshold": best.threshold,
            "development_metrics": best.dev_metrics,
            "holdout_metrics": holdout_metrics,
            "holdout_rank_capacity": holdout_rank_metrics,
        },
        destination / "model.joblib",
    )

    print("\n最终时间外评估", flush=True)
    print(json.dumps(holdout_metrics, ensure_ascii=False, indent=2), flush=True)
    if current_metrics is not None:
        print("\n当前模型同一批样本", flush=True)
        print(json.dumps(current_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录：{destination}", flush=True)


if __name__ == "__main__":
    typer.run(main)
