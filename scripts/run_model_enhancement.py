"""阶段B：结构化模型增强实验（干净特征集 + 滚动特征选择）。

流程（防泄漏边界不变）：
1. 滚动特征稳定性排名：2022/2023 折 LightGBM permutation importance（只用 2020-2023）
2. 取稳定特征 top-N
3. 模型族 × 校准器对比（lightgbm / histgb / blend × none / isotonic / platt）
4. 2024 仅作最终独立评估

数据约束：当前财务金额字段存在单位/币种质量问题，只使用单位自洽的
发行、市场、近期IPO环境与可靠比率字段。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.stable_agent_model import (
    TIME_DECAY_WEIGHTS,
    _ensemble_seeds,
    _select_business_threshold,
)
from ipo_risk.modeling.train import _classification_metrics
from ipo_risk.storage.files import write_json

ROLLING_VALIDATION_YEARS = (2022, 2023)
DEV_YEARS = (2020, 2021, 2022, 2023)
TEST_YEAR = 2024

CLEAN_BASE_FEATURES = (
    "offering_offer_price_mid_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "offering_share_count",
    "engineered_net_proceeds_to_market_cap",
    "market_return_5d_pct",
    "market_return_20d_pct",
    "market_return_60d_pct",
    "market_volatility_20d_pct",
    "market_volatility_60d_pct",
    "market_advance_ratio_20d",
    "market_median_daily_return_20d_pct",
    "market_amplitude_20d_pct",
    "market_turnover_trend_20d",
    "recent_ipo_5d_drop_rate_180d",
    "recent_ipo_5d_mean_return_180d_pct",
    "recent_ipo_count_30d",
    "recent_ipo_count_90d",
    "recent_ipo_1d_mean_return_90d_pct",
    "recent_ipo_5d_observed_count_180d",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_gross_margin_latest_pct",
    "fundamental_gross_margin_change_3y_pct_points",
    "fundamental_operating_cash_flow_to_revenue_latest_pct",
    "fundamental_revenue_year_count",
    "missing_offering_share_ratio",
    "missing_revenue_cagr_3y",
    "missing_gross_margin_latest",
    "missing_operating_cash_flow_to_revenue",
)

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=25,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.0,
    n_jobs=1,
    verbosity=-1,
)

HISTGB_PARAMS = dict(
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=25,
    l2_regularization=1.0,
    max_iter=300,
)


def _sample_weight(rows: pd.DataFrame) -> np.ndarray:
    years = pd.to_numeric(rows["modeling_year"], errors="coerce")
    decay = years.map(TIME_DECAY_WEIGHTS).fillna(1.0).to_numpy(dtype=float)
    y = rows[PRIMARY_TARGET].astype(int).to_numpy()
    positive = int(y.sum())
    negative = int(len(y) - positive)
    balance = np.where(
        y == 1,
        len(y) / (2 * positive) if positive else 1.0,
        len(y) / (2 * negative) if negative else 1.0,
    )
    return decay * balance


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["engineered_net_proceeds_to_market_cap"] = pd.to_numeric(
        result["offering_net_proceeds_hkd"], errors="coerce"
    ).div(
        pd.to_numeric(result["offering_market_cap_mid_hkd"], errors="coerce").where(
            lambda s: s > 0
        )
    )
    result["missing_offering_share_ratio"] = (
        pd.to_numeric(result["offering_share_ratio_pct"], errors="coerce").isna().astype(int)
    )
    result["missing_revenue_cagr_3y"] = (
        pd.to_numeric(result["fundamental_revenue_cagr_3y_pct"], errors="coerce")
        .isna()
        .astype(int)
    )
    result["missing_gross_margin_latest"] = (
        pd.to_numeric(result["fundamental_gross_margin_latest_pct"], errors="coerce")
        .isna()
        .astype(int)
    )
    result["missing_operating_cash_flow_to_revenue"] = (
        pd.to_numeric(
            result["fundamental_operating_cash_flow_to_revenue_latest_pct"], errors="coerce"
        )
        .isna()
        .astype(int)
    )
    return result


def _fit_lgb(
    train: pd.DataFrame,
    features: list[str],
    *,
    n_estimators: int,
    seed: int,
    weight: np.ndarray,
):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        **LGB_PARAMS,
        n_estimators=n_estimators,
        class_weight=None,
        random_state=seed,
    )
    model.fit(
        train[features],
        train[PRIMARY_TARGET].astype(int),
        sample_weight=weight,
    )
    return model


def _fit_lgb_with_early_stopping(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    weight: np.ndarray,
) -> tuple[Any, int]:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        **LGB_PARAMS,
        n_estimators=2000,
        class_weight=None,
        random_state=seed,
    )
    model.fit(
        train[features],
        train[PRIMARY_TARGET].astype(int),
        sample_weight=weight,
        eval_set=[(validation[features], validation[PRIMARY_TARGET].astype(int))],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model, int(model.best_iteration_ or 0)


def _fit_histgb(
    train: pd.DataFrame,
    features: list[str],
    *,
    weight: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        **HISTGB_PARAMS,
        random_state=seed,
    )
    model.fit(train[features], train[PRIMARY_TARGET].astype(int), sample_weight=weight)
    return model


def _rolling_feature_ranking(
    frame: pd.DataFrame,
    features: list[str],
    *,
    seed_count: int = 3,
    permutation_repeats: int = 3,
) -> list[str]:
    """2022/2023 滚动 LightGBM permutation importance，返回稳定排名（只用 2020-2023）。"""
    per_feature: dict[str, list[float]] = {feature: [] for feature in features}
    fold_scores: dict[str, int] = {feature: 0 for feature in features}
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train)
        year_means: dict[str, list[float]] = {feature: [] for feature in features}
        for seed in _ensemble_seeds(42, seed_count):
            model, _ = _fit_lgb_with_early_stopping(
                train, validation, features, seed=seed, weight=weight
            )
            result = permutation_importance(
                model,
                validation[features],
                validation[PRIMARY_TARGET].astype(int),
                scoring="average_precision",
                n_repeats=permutation_repeats,
                random_state=seed,
                n_jobs=1,
            )
            for index, feature in enumerate(features):
                year_means[feature].append(float(result.importances_mean[index]))
        for feature in features:
            mean = float(np.mean(year_means[feature]))
            per_feature[feature].append(mean)
            if mean > 0:
                fold_scores[feature] += 1
    ranked = sorted(
        features,
        key=lambda feature: (
            fold_scores[feature],
            float(np.mean(per_feature[feature])),
            feature,
        ),
        reverse=True,
    )
    return ranked


def _family_probability(
    family: str,
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    weight: np.ndarray,
    n_estimators: int | None,
) -> np.ndarray:
    if family in ("lightgbm", "blend_lgb"):
        model = _fit_lgb(
            train, features, n_estimators=n_estimators or 500, seed=seed, weight=weight
        )
        return model.predict_proba(predict_on[features])[:, 1]
    if family in ("histgb", "blend_hgb"):
        model = _fit_histgb(train, features, weight=weight, seed=seed)
        return model.predict_proba(predict_on[features])[:, 1]
    raise ValueError(f"unknown family: {family}")


def _rolling_oof_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    *,
    model_family: str,
    seed_count: int,
    use_time_decay: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """滚动折外概率；blend 表示 lightgbm 与 histgb 概率平均。"""
    oof_prob: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train) if use_time_decay else None
        families = (
            ("lightgbm", "histgb") if model_family == "blend" else (model_family,)
        )
        fold_probs: list[np.ndarray] = []
        best_iterations: list[int] = []
        for seed in _ensemble_seeds(42, seed_count):
            component_probs: list[np.ndarray] = []
            for fam in families:
                if fam == "lightgbm":
                    model, best_iteration = _fit_lgb_with_early_stopping(
                        train, validation, features, seed=seed, weight=weight
                    )
                    best_iterations.append(best_iteration)
                else:
                    model = _fit_histgb(train, features, weight=weight, seed=seed)
                component_probs.append(model.predict_proba(validation[features])[:, 1])
            fold_probs.append(np.mean(component_probs, axis=0))
        probability = np.mean(fold_probs, axis=0)
        oof_prob.append(probability)
        oof_y.append(validation[PRIMARY_TARGET].astype(int).to_numpy())
        y_true = validation[PRIMARY_TARGET].astype(int).to_numpy()
        fold_metrics.append(
            {
                "validation_year": validation_year,
                "sample_count": int(len(validation)),
                "positive_count": int(y_true.sum()),
                "pr_auc": float(average_precision_score(y_true, probability)),
                "roc_auc": float(roc_auc_score(y_true, probability)),
                "lightgbm_best_iteration": (
                    int(round(float(np.median(best_iterations or [0]))))
                    if best_iterations
                    else None
                ),
            }
        )
    return np.concatenate(oof_prob), np.concatenate(oof_y), fold_metrics


def _fit_final_models(
    dev: pd.DataFrame,
    features: list[str],
    *,
    model_family: str,
    seed_count: int,
    n_estimators: int | None,
    use_time_decay: bool,
) -> list[Any]:
    """最终 2020-2023 全量模型；blend 返回 (lgb_models, hgb_models) 嵌套列表。"""
    weight = _sample_weight(dev) if use_time_decay else None
    families = ("lightgbm", "histgb") if model_family == "blend" else (model_family,)
    models: list[Any] = []
    for seed in _ensemble_seeds(42, seed_count):
        seed_models: list[Any] = []
        for fam in families:
            if fam == "lightgbm":
                seed_models.append(
                    _fit_lgb(
                        dev, features, n_estimators=n_estimators or 500, seed=seed, weight=weight
                    )
                )
            else:
                seed_models.append(_fit_histgb(dev, features, weight=weight, seed=seed))
        models.append(seed_models)
    return models


def _predict_ensemble_models(
    models: list[Any], rows: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """models 为 [seed][component] 结构；blend 时逐 seed 平均两族再跨 seed 平均。"""
    per_seed: list[np.ndarray] = []
    for seed_models in models:
        probs = [
            model.predict_proba(rows[features])[:, 1] for model in seed_models
        ]
        per_seed.append(np.mean(probs, axis=0))
    return np.mean(per_seed, axis=0)


def _calibrator(name: str):
    if name == "none":
        return None
    if name == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    if name == "platt":
        return LogisticRegression(C=1e3, max_iter=2000)
    raise ValueError(f"unknown calibrator: {name}")


def _apply_calibrator(
    calibrator: Any, probability: np.ndarray, y_true: np.ndarray | None
) -> np.ndarray:
    if calibrator is None:
        return probability
    if y_true is not None:
        calibrator.fit(
            (
                probability.reshape(-1, 1)
                if isinstance(calibrator, LogisticRegression)
                else probability
            ),
            y_true,
        )
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(probability.reshape(-1, 1))[:, 1]
    return calibrator.predict(probability)


def run_experiment(
    project_root: Path,
    *,
    output_dir: Path,
    model_families: tuple[str, ...],
    calibrators: tuple[str, ...],
    feature_sizes: tuple[int, ...] = (16,),
    seed_count: int = 5,
    use_time_decay: bool = True,
) -> Path:
    root = project_root.resolve()
    matrix_path = root / "data" / "processed" / "modeling" / "prelisting_feature_matrix.csv"
    frame = pd.read_csv(matrix_path, dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    validate_primary_target(frame)
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("prospectus_date contains invalid values")
    frame["modeling_year"] = dates.dt.year.astype(int)
    frame = _prepare_features(frame)

    candidates = [f for f in CLEAN_BASE_FEATURES if f in frame.columns]
    missing = sorted(set(CLEAN_BASE_FEATURES).difference(candidates))
    if missing:
        raise ValueError(f"clean base features missing: {missing}")

    ranked = _rolling_feature_ranking(frame, candidates, seed_count=min(3, seed_count))

    dev = frame.loc[frame["modeling_year"].isin(DEV_YEARS)].copy()
    test = frame.loc[frame["modeling_year"] == TEST_YEAR].copy()
    if dev.empty or test.empty:
        raise ValueError("development or test cohort is empty")

    results: dict[str, Any] = {}
    for size in feature_sizes:
        features = ranked[:size]
        for family in model_families:
            oof_raw, oof_y, fold_metrics = _rolling_oof_probabilities(
                frame, features, model_family=family, seed_count=seed_count,
                use_time_decay=use_time_decay,
            )
            for calibrator_name in calibrators:
                cal = _calibrator(calibrator_name)
                oof_prob = _apply_calibrator(cal, oof_raw, oof_y)
                threshold, threshold_policy = _select_business_threshold(
                    oof_y, oof_prob, min_recall=0.70, max_alert_fraction=0.40
                )
                rolling_metrics = _classification_metrics(
                    oof_y, oof_prob, threshold=threshold, top_k_fraction=0.20
                )
                rolling_metrics["accuracy"] = (
                    int(rolling_metrics["confusion_matrix"]["true_positive"])
                    + int(rolling_metrics["confusion_matrix"]["true_negative"])
                ) / len(oof_y)

                n_estimators = None
                if family in ("lightgbm", "blend"):
                    iterations = [
                        int(item["lightgbm_best_iteration"])
                        for item in fold_metrics
                        if item.get("lightgbm_best_iteration")
                    ]
                    n_estimators = max(50, int(round(float(np.median(iterations or [150])))))
                models = _fit_final_models(
                    dev, features, model_family=family, seed_count=seed_count,
                    n_estimators=n_estimators, use_time_decay=use_time_decay,
                )
                test_raw = _predict_ensemble_models(models, test, features)
                test_prob = _apply_calibrator(cal, test_raw, None)
                test_metrics = _classification_metrics(
                    test[PRIMARY_TARGET].astype(int).to_numpy(),
                    test_prob,
                    threshold=threshold,
                    top_k_fraction=0.20,
                )
                test_metrics["accuracy"] = (
                    int(test_metrics["confusion_matrix"]["true_positive"])
                    + int(test_metrics["confusion_matrix"]["true_negative"])
                ) / len(test)

                key = f"top{size}_{family}+{calibrator_name}"
                results[key] = {
                    "feature_size": size,
                    "model_family": family,
                    "calibrator": calibrator_name,
                    "features": features,
                    "rolling": {
                        "fold_metrics": fold_metrics,
                        "pr_auc": rolling_metrics["pr_auc"],
                        "roc_auc": rolling_metrics["roc_auc"],
                        "brier_score": rolling_metrics["brier_score"],
                        "threshold": threshold,
                        "threshold_policy": threshold_policy,
                        "sample_count": len(oof_y),
                        "positive_count": int(oof_y.sum()),
                    },
                    "test_2024": {
                        "sample_count": int(len(test)),
                        "positive_count": int(test[PRIMARY_TARGET].astype(int).sum()),
                        "positive_rate": float(test[PRIMARY_TARGET].astype(int).mean()),
                        "pr_auc": test_metrics["pr_auc"],
                        "roc_auc": test_metrics["roc_auc"],
                        "brier_score": test_metrics["brier_score"],
                        "log_loss": test_metrics["log_loss"],
                        "precision": test_metrics["precision"],
                        "recall": test_metrics["recall"],
                        "f1": test_metrics["f1"],
                        "f2": test_metrics["f2"],
                        "accuracy": test_metrics["accuracy"],
                        "alert_count": test_metrics["confusion_matrix"]["true_positive"]
                        + test_metrics["confusion_matrix"]["false_positive"],
                        "top_k": test_metrics["top_k"],
                        "calibration_bins": test_metrics["calibration_bins"],
                    },
                    "leakage_controls": {
                        "feature_ranking_uses_2020_2023_only": True,
                        "calibrator_fit_on_rolling_oof_only": calibrator_name != "none",
                        "threshold_selected_on_rolling_oof_only": True,
                        "test_2024_used_for_fitting": False,
                    },
                }
                print(
                    f"[{key}] rolling PR-AUC={rolling_metrics['pr_auc']:.4f} | "
                    f"2024 PR-AUC={test_metrics['pr_auc']:.4f} ROC={test_metrics['roc_auc']:.4f} "
                    f"Brier={test_metrics['brier_score']:.4f}",
                    flush=True,
                )

    summary = {
        "schema_version": 1,
        "experiment": "model_enhancement_v2_clean_features_rolling_selection",
        "target": PRIMARY_TARGET,
        "feature_ranking": ranked,
        "cohort_counts": {
            "development_2020_2023": int(len(dev)),
            "test_2024": int(len(test)),
        },
        "configuration": {
            "model_families": list(model_families),
            "calibrators": list(calibrators),
            "feature_sizes": list(feature_sizes),
            "seed_count": seed_count,
            "use_time_decay": use_time_decay,
            "time_decay_weights": TIME_DECAY_WEIGHTS,
            "min_recall": 0.70,
            "max_alert_fraction": 0.40,
        },
        "data_notes": {
            "feature_scope": "clean unit-consistent features only; native financial amounts "
            "excluded due to known unit/currency quality issues",
            "currency_normalization": "blocked by unresolved currencies (~140 companies)",
        },
        "results": results,
        "baseline_reference_v5_business_base_2024": {
            "pr_auc": 0.4700,
            "roc_auc": 0.605,
            "note": "旧版 hkd_mn 财务字段口径，与本实验特征集不同，仅作参考",
        },
    }
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    write_json(summary, metrics_path)
    print(f"\n结果已写入：{metrics_path}", flush=True)
    return metrics_path


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), "--project-root"),
    output_dir: Path = typer.Option(
        Path("artifacts/models/model_enhancement_v2"), "--output-dir"
    ),
    families: str = typer.Option(
        "lightgbm,histgb,blend", help="模型族：lightgbm,histgb,blend"
    ),
    calibrators: str = typer.Option(
        "none,isotonic,platt", help="校准器：none,isotonic,platt"
    ),
    feature_sizes: str = typer.Option("16", help="特征数量，逗号分隔"),
    seed_count: int = typer.Option(5, min=1),
) -> None:
    run_experiment(
        project_root,
        output_dir=output_dir,
        model_families=tuple(families.split(",")),
        calibrators=tuple(calibrators.split(",")),
        feature_sizes=tuple(int(v) for v in feature_sizes.split(",")),
        seed_count=seed_count,
    )


if __name__ == "__main__":
    app()
