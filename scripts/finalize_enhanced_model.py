"""固化增强模型 v2：298 候选池（含行业频率）稳定排名 + 三模型融合（lgb_b/hgb_a/equal）。

流程（严格防泄漏）：
1. 滚动特征稳定性排名（2022/2023 折 permutation，3 次聚合降方差）→ top16
2. 滚动 OOF 三模型融合概率 → 阈值选择（OOF）
3. 2020-2023 全量训练最终 blend3（5 seeds）
4. 2024 仅作独立评估
输出：model.joblib / predictions_2024.csv / metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target  # noqa: E402
from ipo_risk.modeling.stable_agent_model import (  # noqa: E402
    AGENT_SEMANTIC_FEATURES,
    BUSINESS_BASE_FEATURES,
    TIME_DECAY_WEIGHTS,
    _ensemble_seeds,
    _select_business_threshold,
)
from ipo_risk.modeling.train import _classification_metrics  # noqa: E402
from ipo_risk.storage.files import write_json  # noqa: E402

# 预测目标列：默认五日主任务；main 按 --horizon 切换 1/20/60
TARGET: str = PRIMARY_TARGET

ROLLING_VALIDATION_YEARS = (2022, 2023)
DEV_YEARS = (2020, 2021, 2022, 2023)
TEST_YEAR = 2024
FEATURE_SIZE = 16

LGB_CFG = dict(
    learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.2, reg_lambda=2.0,
)
# 排名用标准 LightGBM 配置（通用重要性估计），建模用保守 LGB_CFG
LGB_RANK_CFG = dict(
    learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=25,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.2, reg_lambda=1.0,
)
HGB_CFG = dict(
    learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0,
)
WEIGHTS = {"lgb": 1 / 3, "hgb": 1 / 3, "log": 1 / 3}


def _fit_lgb_rank(train, validation, features, *, seed, weight):
    """排名专用 LightGBM（标准配置，确定性）。"""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary", **LGB_RANK_CFG, n_estimators=2000,
        class_weight=None, random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_row_wise=True,
    )
    model.fit(
        train[features], train[TARGET].astype(int), sample_weight=weight,
        eval_set=[(validation[features], validation[TARGET].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _sample_weight(rows: pd.DataFrame) -> np.ndarray:
    years = pd.to_numeric(rows["modeling_year"], errors="coerce")
    decay = years.map(TIME_DECAY_WEIGHTS).fillna(1.0).to_numpy(dtype=float)
    y = rows[TARGET].astype(int).to_numpy()
    positive = int(y.sum())
    negative = int(len(y) - positive)
    balance = np.where(
        y == 1,
        len(y) / (2 * positive) if positive else 1.0,
        len(y) / (2 * negative) if negative else 1.0,
    )
    return decay * balance


def _fit_lgb(train, features, *, n_estimators, seed, weight):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary", **LGB_CFG, n_estimators=n_estimators,
        class_weight=None, random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_row_wise=True,
    )
    model.fit(train[features], train[TARGET].astype(int), sample_weight=weight)
    return model


def _fit_lgb_with_early_stopping(train, validation, features, *, seed, weight):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary", **LGB_CFG, n_estimators=2000,
        class_weight=None, random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_row_wise=True,
    )
    model.fit(
        train[features], train[TARGET].astype(int), sample_weight=weight,
        eval_set=[(validation[features], validation[TARGET].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_ or 0)


def _fit_hgb(train, features, *, seed, weight):
    model = HistGradientBoostingClassifier(**HGB_CFG, max_iter=300, random_state=seed)
    model.fit(train[features], train[TARGET].astype(int), sample_weight=weight)
    return model


def _fit_logistic(train, features, *, seed, weight):
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(train[f])]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight=None, max_iter=3000, random_state=seed)),
        ]
    )
    pipeline.fit(train[features], train[TARGET].astype(int), classifier__sample_weight=weight)
    return pipeline


def _rolling_feature_ranking(frame, features, *, seed_count=3, permutation_repeats=2):
    per_feature: dict[str, list[float]] = {f: [] for f in features}
    fold_scores: dict[str, int] = {f: 0 for f in features}
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train)
        year_means: dict[str, list[float]] = {f: [] for f in features}
        for seed in _ensemble_seeds(42, seed_count):
            model = _fit_lgb_rank(
                train, validation, features, seed=seed, weight=weight
            )
            result = permutation_importance(
                model, validation[features], validation[TARGET].astype(int),
                scoring="average_precision", n_repeats=permutation_repeats,
                random_state=seed, n_jobs=1,
            )
            for index, feature in enumerate(features):
                year_means[feature].append(float(result.importances_mean[index]))
        for feature in features:
            mean = float(np.mean(year_means[feature]))
            per_feature[feature].append(mean)
            if mean > 0:
                fold_scores[feature] += 1
    return sorted(
        features,
        key=lambda feature: (fold_scores[feature], float(np.mean(per_feature[feature])), feature),
        reverse=True,
    )


def _aggregate_ranking(frame, features, *, runs=3):
    """多次滚动排名取平均排名，降低小样本下特征选择方差。"""
    ranks = []
    for r in range(runs):
        repeats = 2 + (r % 2)
        ranked = _rolling_feature_ranking(frame, features, permutation_repeats=repeats)
        ranks.append({feature: float(i) for i, feature in enumerate(ranked)})
    avg = {
        feature: float(np.mean([item.get(feature, float(len(features))) for item in ranks]))
        for feature in features
    }
    return sorted(features, key=lambda feature: (avg[feature], feature))


def main(project_root: Path, horizon: int = 5) -> None:
    global TARGET
    if horizon == 5:
        TARGET = PRIMARY_TARGET
    elif horizon in (1, 20, 60):
        TARGET = f"target_significant_drop_{horizon}d"
    else:
        raise ValueError(f"unsupported horizon: {horizon}")
    root = project_root.resolve()
    matrix_path = (
        root / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix_multihorizon.csv"
    )
    frame = pd.read_csv(matrix_path, dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    validate_primary_target(frame)
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("prospectus_date contains invalid values")
    frame["modeling_year"] = dates.dt.year.astype(int)

    candidates = [
        f for f in (*BUSINESS_BASE_FEATURES, *AGENT_SEMANTIC_FEATURES) if f in frame.columns
    ]
    # 行业频率编码（只用开发集算频率，防泄漏），加入候选池
    dev_rows = frame.loc[frame["modeling_year"].isin(DEV_YEARS)]
    freq = dev_rows["industry_primary"].fillna("unknown").value_counts(normalize=True)
    frame["industry_freq"] = frame["industry_primary"].fillna("unknown").map(freq).fillna(0.0)
    candidates.append("industry_freq")

    print("稳定滚动特征排名中（3 次聚合）…", flush=True)
    ranked = _aggregate_ranking(frame, candidates, runs=3)
    features = ranked[:FEATURE_SIZE]

    oof_prob_list, oof_y_list = [], []
    fold_metrics, best_iters = [], []
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train)
        per_seed = []
        for seed in _ensemble_seeds(42, 5):
            lgb_model, best = _fit_lgb_with_early_stopping(
                train, validation, features, seed=seed, weight=weight
            )
            best_iters.append(best)
            hgb_model = _fit_hgb(train, features, seed=seed, weight=weight)
            log_model = _fit_logistic(train, features, seed=seed, weight=weight)
            per_seed.append(
                np.mean(
                    [
                        lgb_model.predict_proba(validation[features])[:, 1],
                        hgb_model.predict_proba(validation[features])[:, 1],
                        log_model.predict_proba(validation[features])[:, 1],
                    ],
                    axis=0,
                )
            )
        probability = np.mean(per_seed, axis=0)
        oof_prob_list.append(probability)
        oof_y_list.append(validation[TARGET].astype(int).to_numpy())
        y_true = validation[TARGET].astype(int).to_numpy()
        fold_metrics.append(
            {
                "validation_year": validation_year,
                "sample_count": int(len(validation)),
                "positive_count": int(y_true.sum()),
                "pr_auc": float(average_precision_score(y_true, probability)),
                "roc_auc": float(roc_auc_score(y_true, probability)),
            }
        )
    oof_prob = np.concatenate(oof_prob_list)
    oof_y = np.concatenate(oof_y_list)
    threshold, threshold_policy = _select_business_threshold(
        oof_y, oof_prob, min_recall=0.70, max_alert_fraction=0.40
    )
    rolling_pr = float(average_precision_score(oof_y, oof_prob))
    rolling_roc = float(roc_auc_score(oof_y, oof_prob))

    dev = frame.loc[frame["modeling_year"].isin(DEV_YEARS)]
    test = frame.loc[frame["modeling_year"] == TEST_YEAR]
    dev_weight = _sample_weight(dev)
    n_estimators = max(50, int(round(float(np.median(best_iters or [150])))))
    seeds = _ensemble_seeds(42, 5)
    lgb_models = [_fit_lgb(dev, features, n_estimators=n_estimators, seed=s, weight=dev_weight) for s in seeds]
    hgb_models = [_fit_hgb(dev, features, seed=s, weight=dev_weight) for s in seeds]
    log_models = [_fit_logistic(dev, features, seed=s, weight=dev_weight) for s in seeds]

    def predict_blend3(rows: pd.DataFrame) -> np.ndarray:
        per_seed = []
        for i in range(len(seeds)):
            per_seed.append(
                np.mean(
                    [
                        lgb_models[i].predict_proba(rows[features])[:, 1],
                        hgb_models[i].predict_proba(rows[features])[:, 1],
                        log_models[i].predict_proba(rows[features])[:, 1],
                    ],
                    axis=0,
                )
            )
        return np.mean(per_seed, axis=0)

    test_raw = predict_blend3(test)
    test_metrics = _classification_metrics(
        test[TARGET].astype(int).to_numpy(), test_raw, threshold=threshold, top_k_fraction=0.20
    )
    test_metrics["accuracy"] = (
        int(test_metrics["confusion_matrix"]["true_positive"])
        + int(test_metrics["confusion_matrix"]["true_negative"])
    ) / len(test)

    destination = root / "artifacts" / "models" / f"enhanced_blend3_v2_h{horizon}"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": "enhanced_blend3_v2",
            "features": features,
            "feature_count": len(features),
            "feature_ranking": ranked,
            "threshold": float(threshold),
            "threshold_policy": threshold_policy,
            "weights": WEIGHTS,
            "lgb_cfg": LGB_CFG,
            "hgb_cfg": HGB_CFG,
            "n_estimators": n_estimators,
            "seeds": seeds,
            "lgb_models": lgb_models,
            "hgb_models": hgb_models,
            "log_models": log_models,
            "blend": "mean(lightgbm, histgb, logistic)",
        },
        destination / "model.joblib",
    )
    identity = [c for c in ("company_id", "ticker", "company_name", "prospectus_date", "run_id") if c in test]
    predictions = test[identity].copy()
    predictions[TARGET] = test[TARGET].astype(int).to_numpy()
    predictions["blend3_probability"] = test_raw
    predictions["predicted_significant_drop_5d"] = (test_raw >= threshold).astype(int)
    predictions.to_csv(destination / "predictions_2024.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "schema_version": 1,
        "model_name": "enhanced_blend3_v2",
        "target": TARGET,
        "cohort_counts": {"development_2020_2023": int(len(dev)), "test_2024": int(len(test))},
        "features": features,
        "feature_ranking": ranked,
        "rolling": {
            "fold_metrics": fold_metrics,
            "pr_auc": rolling_pr,
            "roc_auc": rolling_roc,
            "threshold": float(threshold),
            "threshold_policy": threshold_policy,
        },
        "test_2024": {
            "pr_auc": test_metrics["pr_auc"],
            "roc_auc": test_metrics["roc_auc"],
            "brier_score": test_metrics["brier_score"],
            "recall": test_metrics["recall"],
            "precision": test_metrics["precision"],
            "accuracy": test_metrics["accuracy"],
            "f2": test_metrics["f2"],
            "top_k": test_metrics["top_k"],
            "calibration_bins": test_metrics["calibration_bins"],
            "threshold": float(threshold),
        },
        "baseline": {
            "v5_agent_pr_auc": 0.4725,
            "v5_agent_roc_auc": 0.5717,
            "enhanced_v1_pr_auc": 0.5792,
            "enhanced_v1_roc_auc": 0.6467,
        },
        "leakage_controls": {
            "feature_ranking_uses_2020_2023_only": True,
            "industry_freq_encoded_on_dev_only": True,
            "threshold_selected_on_rolling_oof_only": True,
            "test_2024_used_for_fitting": False,
        },
    }
    write_json(metrics, destination / "metrics.json")
    print(f"模型已固化: {destination / 'model.joblib'}")
    print(f"2024 评估: PR-AUC={test_metrics['pr_auc']:.4f} ROC={test_metrics['roc_auc']:.4f} "
          f"Brier={test_metrics['brier_score']:.4f} recall={test_metrics['recall']:.4f}")
    print(f"rolling: PR-AUC={rolling_pr:.4f} ROC={rolling_roc:.4f}")
    print(f"对比: v5 PR 0.4725→{test_metrics['pr_auc']:.4f} | v1 PR 0.5792→{test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def cli(
        project_root: Path = typer.Option(Path.cwd(), "--project-root"),
        horizon: int = typer.Option(5, "--horizon", min=1, max=60),
    ) -> None:
        main(project_root, horizon=horizon)

    app()
