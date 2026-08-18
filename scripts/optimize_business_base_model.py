"""只使用上市前基础特征，滚动验证并优化第5日显著下跌模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.stable_agent_model import (
    _apply_batch_alert_cap,
    _classification_metrics_with_predictions,
    _select_business_threshold,
    _time_decay_sample_weight,
    add_derived_prediction_features,
)
from ipo_risk.modeling.train import _classification_metrics
from ipo_risk.storage.files import write_json


app = typer.Typer(add_completion=False)
VALIDATION_YEARS = (2022, 2023)
SEARCH_SEEDS = (42, 137, 271)
FINAL_SEEDS = (42, 137, 271, 389, 521)

PARAMETER_PROFILES: dict[str, dict[str, Any]] = {
    "current_balanced": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
    },
    "shallow_regularized": {
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 25,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "class_weight": "balanced",
    },
    "strong_regularized": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 35,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "class_weight": "balanced",
    },
    "current_unweighted": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "class_weight": None,
    },
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    validate_primary_target(frame)
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("feature matrix contains invalid prospectus_date")
    frame = frame.copy()
    frame["modeling_year"] = dates.dt.year.astype(int)
    return add_derived_prediction_features(frame)


def _load_ranked_base_features(path: Path, frame: pd.DataFrame) -> list[str]:
    stability = pd.read_csv(path)
    base = stability.loc[stability["feature_group"].eq("base")].copy()
    base["group_rank"] = pd.to_numeric(base["group_rank"], errors="coerce")
    base = base.sort_values("group_rank")
    ranked = [str(value) for value in base["feature"] if str(value) in frame.columns]
    if len(ranked) < 16:
        raise ValueError("base feature stability file contains fewer than 16 usable features")
    return ranked


def _feature_sets(ranked: list[str]) -> dict[str, list[str]]:
    sets = {f"ranked_{size}": ranked[:size] for size in (8, 9, 10, 11, 12, 13, 15, 16)}

    current = ranked[:8]
    sets["curated_10"] = [
        *current,
        "offering_share_ratio_pct",
        "market_return_20d_pct",
    ]
    sets["curated_11_cashflow"] = [
        *sets["curated_10"],
        "engineered_operating_cash_flow_margin",
    ]
    sets["curated_12_sentiment"] = [
        *sets["curated_11_cashflow"],
        "recent_ipo_5d_mean_return_180d_pct",
    ]
    # 尚未进入旧白名单，但均可在上市前由官方行情或公司资料确定。
    for suffix, feature in (
        ("market_breadth", "market_advance_ratio_20d"),
        ("market_median_return", "market_median_daily_return_20d_pct"),
        ("market_amplitude", "market_amplitude_20d_pct"),
        ("market_turnover", "market_turnover_trend_20d"),
        ("recent_ipo_day1", "recent_ipo_1d_mean_return_90d_pct"),
        ("recent_ipo_observations", "recent_ipo_5d_observed_count_180d"),
        ("offering_share_count", "offering_share_count"),
        ("industry", "industry_primary"),
    ):
        sets[f"plus_{suffix}_9"] = [*current, feature]
    sets["market_regime_11"] = [
        *current,
        "market_advance_ratio_20d",
        "market_median_daily_return_20d_pct",
        "market_amplitude_20d_pct",
    ]
    sets["recent_ipo_context_11"] = [
        *current,
        "recent_ipo_1d_mean_return_90d_pct",
        "recent_ipo_5d_observed_count_180d",
        "recent_ipo_count_90d",
    ]
    # 保序去重，防止稳定性排名变化后同一字段出现两次。
    return {name: list(dict.fromkeys(features)) for name, features in sets.items()}


def _build_preprocessor(rows: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [
        feature for feature in features if not pd.api.types.is_numeric_dtype(rows[feature])
    ]
    numeric = [feature for feature in features if feature not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.set_output(transform="pandas")
    return preprocessor


def _fit_model(
    train: pd.DataFrame,
    features: list[str],
    profile: dict[str, Any],
    *,
    seed: int,
    n_estimators: int,
    validation: pd.DataFrame | None = None,
) -> tuple[Pipeline, int]:
    preprocessor = _build_preprocessor(train, features)
    train_x = preprocessor.fit_transform(train[features])
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        **profile,
    )
    fit_kwargs: dict[str, Any] = {
        "sample_weight": _time_decay_sample_weight(train),
    }
    if validation is not None:
        validation_x = preprocessor.transform(validation[features])
        fit_kwargs.update(
            {
                "eval_set": [(validation_x, validation[PRIMARY_TARGET].astype(int))],
                "eval_metric": "binary_logloss",
                "callbacks": [lgb.early_stopping(50, verbose=False)],
            }
        )
    classifier.fit(train_x, train[PRIMARY_TARGET].astype(int), **fit_kwargs)
    best_iteration = int(classifier.best_iteration_ or n_estimators)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)]), best_iteration


def _rolling_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    profile: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    all_iterations: list[int] = []
    for year in VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(2020, year - 1)]
        validation = frame.loc[frame["modeling_year"].eq(year)]
        seed_probabilities: list[np.ndarray] = []
        seed_iterations: list[int] = []
        for seed in SEARCH_SEEDS:
            model, best_iteration = _fit_model(
                train,
                features,
                profile,
                seed=seed,
                n_estimators=1000,
                validation=validation,
            )
            seed_probabilities.append(model.predict_proba(validation[features])[:, 1])
            seed_iterations.append(best_iteration)
            all_iterations.append(best_iteration)
        probability = np.mean(seed_probabilities, axis=0)
        fold = validation[["company_id", PRIMARY_TARGET]].copy()
        fold["validation_year"] = year
        fold["probability"] = probability
        predictions.append(fold)
        fold_rows.append(
            {
                "validation_year": year,
                "sample_count": int(len(validation)),
                "positive_count": int(validation[PRIMARY_TARGET].sum()),
                "pr_auc": float(
                    average_precision_score(validation[PRIMARY_TARGET].astype(int), probability)
                ),
                "best_iteration": int(round(float(np.median(seed_iterations)))),
            }
        )
    return (
        pd.concat(predictions, ignore_index=True),
        fold_rows,
        max(20, int(round(float(np.median(all_iterations or [150]))))),
    )


def _search(frame: pd.DataFrame, feature_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_set, features in feature_sets.items():
        for profile_name, profile in PARAMETER_PROFILES.items():
            oof, folds, final_iterations = _rolling_evaluate(frame, features, profile)
            probability = oof["probability"].to_numpy()
            y_true = oof[PRIMARY_TARGET].astype(int).to_numpy()
            threshold, policy = _select_business_threshold(
                y_true,
                probability,
                min_recall=0.70,
                max_alert_fraction=0.40,
            )
            metrics = _classification_metrics(
                y_true,
                probability,
                threshold=threshold,
                top_k_fraction=0.20,
            )
            confusion = metrics["confusion_matrix"]
            accuracy = (confusion["true_positive"] + confusion["true_negative"]) / len(oof)
            rows.append(
                {
                    "feature_set": feature_set,
                    "parameter_profile": profile_name,
                    "feature_count": len(features),
                    "features": features,
                    "rolling_pr_auc": float(metrics["pr_auc"]),
                    "minimum_fold_pr_auc": min(float(item["pr_auc"]) for item in folds),
                    "rolling_precision": float(metrics["precision"]),
                    "rolling_recall": float(metrics["recall"]),
                    "rolling_f1": float(metrics["f1"]),
                    "rolling_accuracy": float(accuracy),
                    "threshold": float(threshold),
                    "threshold_policy": policy,
                    "folds": folds,
                    "final_iterations": final_iterations,
                }
            )
            typer.echo(
                f"{feature_set:24s} · {profile_name:20s} · "
                f"PR-AUC={metrics['pr_auc']:.4f} · min-fold={rows[-1]['minimum_fold_pr_auc']:.4f}"
            )
    return rows


def _fit_final_ensemble(
    development: pd.DataFrame,
    features: list[str],
    profile: dict[str, Any],
    n_estimators: int,
) -> list[Pipeline]:
    return [
        _fit_model(
            development,
            features,
            profile,
            seed=seed,
            n_estimators=n_estimators,
        )[0]
        for seed in FINAL_SEEDS
    ]


def _test_metrics(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, list[Pipeline]]:
    development = frame.loc[frame["modeling_year"].between(2020, 2023)].copy()
    test = frame.loc[frame["modeling_year"].eq(2024)].copy()
    features = list(candidate["features"])
    models = _fit_final_ensemble(
        development,
        features,
        PARAMETER_PROFILES[str(candidate["parameter_profile"])],
        int(candidate["final_iterations"]),
    )
    probability = np.mean(
        [model.predict_proba(test[features])[:, 1] for model in models],
        axis=0,
    )
    policy = candidate["threshold_policy"]
    prediction = _apply_batch_alert_cap(
        probability,
        threshold=float(candidate["threshold"]),
        max_alert_fraction=0.40,
        target_alert_fraction=float(policy.get("selected_alert_fraction", 0.0)),
    )
    metrics = _classification_metrics_with_predictions(
        test[PRIMARY_TARGET].astype(int).to_numpy(),
        probability,
        prediction,
        threshold=float(candidate["threshold"]),
        top_k_fraction=0.20,
    )
    identity = [
        column
        for column in ("company_id", "ticker", "company_name", "prospectus_date")
        if column in test
    ]
    predictions = test[identity + [PRIMARY_TARGET]].copy()
    predictions["probability"] = probability
    predictions["prediction"] = prediction
    return metrics, predictions, models


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    output_dir: Path | None = typer.Option(None, help="实验输出目录"),
) -> None:
    root = project_root.resolve()
    modeling = root / "data" / "processed" / "modeling"
    stable = root / "artifacts" / "models" / "day5_stable_agent_v5_strict_ablation"
    destination = (output_dir or root / "artifacts" / "models" / "day5_business_base_v1").resolve()
    destination.mkdir(parents=True, exist_ok=True)

    frame = _load_frame(modeling / "agent_augmented_feature_matrix.csv")
    ranked = _load_ranked_base_features(stable / "feature_stability.csv", frame)
    feature_sets = _feature_sets(ranked)
    search = _search(frame, feature_sets)
    ranking_best = max(
        search,
        key=lambda row: (
            float(row["rolling_pr_auc"]),
            float(row["minimum_fold_pr_auc"]),
            -int(row["feature_count"]),
        ),
    )
    baseline = next(
        row
        for row in search
        if row["feature_set"] == "ranked_8"
        and row["parameter_profile"] == "current_balanced"
    )
    # 用户当前目标是提升准确率，但不能靠牺牲排序能力“全部判安全”。
    # 因此先要求滚动 PR-AUC 与最差年份均不低于当前基线 0.005 以上，
    # 再在合格模型中选择滚动准确率最高者。
    accuracy_candidates = [
        row
        for row in search
        if float(row["rolling_pr_auc"]) >= float(baseline["rolling_pr_auc"]) - 0.005
        and float(row["minimum_fold_pr_auc"])
        >= float(baseline["minimum_fold_pr_auc"]) - 0.005
    ]
    best = max(
        accuracy_candidates,
        key=lambda row: (
            float(row["rolling_accuracy"]),
            float(row["rolling_f1"]),
            float(row["rolling_pr_auc"]),
            -int(row["feature_count"]),
        ),
    )

    best_metrics, predictions, best_models = _test_metrics(frame, best)
    baseline_metrics, _, _ = _test_metrics(frame, baseline)
    ranking_best_metrics, _, _ = _test_metrics(frame, ranking_best)
    search_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"features", "threshold_policy", "folds"}
            }
            for row in search
        ]
    ).sort_values(["rolling_pr_auc", "minimum_fold_pr_auc"], ascending=False)
    search_frame.to_csv(destination / "rolling_search.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(destination / "predictions_2024.csv", index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "models": best_models,
            "features": best["features"],
            "threshold": best["threshold"],
            "threshold_policy": best["threshold_policy"],
            "parameter_profile": best["parameter_profile"],
            "final_iterations": best["final_iterations"],
        },
        destination / "business_base_ensemble.joblib",
    )
    payload = {
        "selection_rule": (
            "仅使用2022和2023滚动验证选型：先约束PR-AUC和最差年份不低于当前基线0.005，"
            "再最大化准确率；2024只做最终对照。"
        ),
        "target": PRIMARY_TARGET,
        "selected": best,
        "ranking_best": ranking_best,
        "rolling_candidate_count": len(search),
        "test_2024": best_metrics,
        "ranking_best_test_2024": ranking_best_metrics,
        "baseline": baseline,
        "baseline_test_2024": baseline_metrics,
        "test_2024_delta": {
            metric: float(best_metrics[metric]) - float(baseline_metrics[metric])
            for metric in ("pr_auc", "precision", "recall", "f1", "accuracy", "brier_score")
        },
    }
    write_json(payload, destination / "metrics.json")
    typer.echo("\n滚动验证选型完成（2024未参与选择）")
    typer.echo(
        f"准确率约束下最佳：{best['feature_set']} / {best['parameter_profile']} / "
        f"{best['feature_count']}维 / rolling PR-AUC={best['rolling_pr_auc']:.4f}"
    )
    typer.echo(
        f"2024：PR-AUC={best_metrics['pr_auc']:.4f}，准确率={best_metrics['accuracy']:.4f}，"
        f"召回率={best_metrics['recall']:.4f}，预警准确率={best_metrics['precision']:.4f}"
    )
    typer.echo(f"结果：{destination / 'metrics.json'}")


if __name__ == "__main__":
    app()
