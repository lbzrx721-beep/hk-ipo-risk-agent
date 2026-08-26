"""ML 优化 v5：超参搜索 + 融合策略对比（滚动折选型，2024 最终验证）。

协议（防泄漏）：
- 特征：滚动稳定性排名 top16（只用 2020-2023）
- 选型：2022/2023 滚动折 PR-AUC（平均 + 最差折）为唯一选型标准
- 2024 仅最终评估，不参与任何超参/权重/阈值选择

搜索空间：
- LightGBM 3 组超参 × HistGB 2 组超参 = 6 组模型配置
- 融合 2 种：均权(equal) / 滚动OOF约束权重搜索(oof_search)
- 共 12 个配置
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.stable_agent_model import (
    AGENT_SEMANTIC_FEATURES,
    BUSINESS_BASE_FEATURES,
    TIME_DECAY_WEIGHTS,
    _ensemble_seeds,
    _select_business_threshold,
)
from ipo_risk.modeling.train import _classification_metrics
from ipo_risk.storage.files import write_json

ROLLING_VALIDATION_YEARS = (2022, 2023)
DEV_YEARS = (2020, 2021, 2022, 2023)
TEST_YEAR = 2024
FEATURE_SIZE = 16

LGB_CONFIGS = {
    "lgb_a": dict(
        learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=25,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=1.0,
    ),
    "lgb_b": dict(
        learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=2.0,
    ),
    "lgb_c": dict(
        learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=15,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.5,
    ),
}

HGB_CONFIGS = {
    "hgb_a": dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0),
    "hgb_b": dict(learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=2.0),
}

FUSION_MODES = ("equal", "oof_search")


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


def _fit_lgb(
    train: pd.DataFrame, features: list[str], *, cfg: dict[str, Any], n_estimators: int,
    seed: int, weight: np.ndarray,
):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        **cfg,
        n_estimators=n_estimators,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(train[features], train[PRIMARY_TARGET].astype(int), sample_weight=weight)
    return model


def _fit_lgb_with_early_stopping(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str], *,
    cfg: dict[str, Any], seed: int, weight: np.ndarray,
) -> tuple[Any, int]:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        **cfg,
        n_estimators=2000,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(
        train[features],
        train[PRIMARY_TARGET].astype(int),
        sample_weight=weight,
        eval_set=[(validation[features], validation[PRIMARY_TARGET].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_ or 0)


def _fit_hgb(
    train: pd.DataFrame, features: list[str], *, cfg: dict[str, Any], seed: int, weight: np.ndarray,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**cfg, max_iter=300, random_state=seed)
    model.fit(train[features], train[PRIMARY_TARGET].astype(int), sample_weight=weight)
    return model


def _fit_logistic(
    train: pd.DataFrame, features: list[str], *, seed: int, weight: np.ndarray,
) -> Any:
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
    pipeline.fit(
        train[features],
        train[PRIMARY_TARGET].astype(int),
        classifier__sample_weight=weight,
    )
    return pipeline


def _rolling_feature_ranking(
    frame: pd.DataFrame, features: list[str], *, seed_count: int = 3,
    permutation_repeats: int = 2, lgb_cfg: dict[str, Any] | None = None,
) -> list[str]:
    per_feature: dict[str, list[float]] = {f: [] for f in features}
    fold_scores: dict[str, int] = {f: 0 for f in features}
    ranking_cfg = lgb_cfg or LGB_CONFIGS["lgb_a"]
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train)
        year_means: dict[str, list[float]] = {f: [] for f in features}
        for seed in _ensemble_seeds(42, seed_count):
            model, _ = _fit_lgb_with_early_stopping(
                train, validation, features, cfg=ranking_cfg, seed=seed, weight=weight
            )
            result = permutation_importance(
                model, validation[features], validation[PRIMARY_TARGET].astype(int),
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


def _rolling_component_oof(
    frame: pd.DataFrame, features: list[str], *, lgb_cfg: dict[str, Any], hgb_cfg: dict[str, Any],
    seed_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[int]]:
    """滚动折外，分别保存 lgb/hgb/log 概率与标签。"""
    lgb_oof: list[np.ndarray] = []
    hgb_oof: list[np.ndarray] = []
    log_oof: list[np.ndarray] = []
    y_oof: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    all_best: list[int] = []
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = _sample_weight(train)
        lgb_p, hgb_p, log_p = [], [], []
        best_iters: list[int] = []
        for seed in _ensemble_seeds(42, seed_count):
            lgb_model, best = _fit_lgb_with_early_stopping(
                train, validation, features, cfg=lgb_cfg, seed=seed, weight=weight
            )
            best_iters.append(best)
            hgb_model = _fit_hgb(train, features, cfg=hgb_cfg, seed=seed, weight=weight)
            log_model = _fit_logistic(train, features, seed=seed, weight=weight)
            lgb_p.append(lgb_model.predict_proba(validation[features])[:, 1])
            hgb_p.append(hgb_model.predict_proba(validation[features])[:, 1])
            log_p.append(log_model.predict_proba(validation[features])[:, 1])
        lgb_oof.append(np.mean(lgb_p, axis=0))
        hgb_oof.append(np.mean(hgb_p, axis=0))
        log_oof.append(np.mean(log_p, axis=0))
        y_oof.append(validation[PRIMARY_TARGET].astype(int).to_numpy())
        all_best.extend(best_iters)
        y_true = validation[PRIMARY_TARGET].astype(int).to_numpy()
        fold_metrics.append(
            {
                "validation_year": validation_year,
                "sample_count": int(len(validation)),
                "positive_count": int(y_true.sum()),
                "lgb_pr_auc": float(average_precision_score(y_true, np.mean(lgb_p, axis=0))),
                "hgb_pr_auc": float(average_precision_score(y_true, np.mean(hgb_p, axis=0))),
                "log_pr_auc": float(average_precision_score(y_true, np.mean(log_p, axis=0))),
            }
        )
    return (
        np.concatenate(lgb_oof), np.concatenate(hgb_oof), np.concatenate(log_oof),
        np.concatenate(y_oof), fold_metrics, all_best,
    )


def _search_fusion_weights(
    lgb_p: np.ndarray, hgb_p: np.ndarray, log_p: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """滚动折上搜索三模型权重（各 ≥ 0.2，步长 0.2，和=1），返回权重与融合概率。"""
    best_w: dict[str, float] | None = None
    best_score = -1.0
    best_prob: np.ndarray | None = None
    for wl in np.arange(0.2, 0.7, 0.2):
        for wh in np.arange(0.2, 0.7, 0.2):
            wl, wh = round(float(wl), 1), round(float(wh), 1)
            wlog = round(1.0 - wl - wh, 1)
            if wlog < 0.19:
                continue
            prob = wl * lgb_p + wh * hgb_p + wlog * log_p
            score = float(average_precision_score(y, prob))
            if score > best_score:
                best_score = score
                best_w = {"lgb": wl, "hgb": wh, "log": wlog}
                best_prob = prob
    assert best_w is not None and best_prob is not None
    return best_prob, best_w


def _fit_final_models(
    dev: pd.DataFrame, features: list[str], *, lgb_cfg: dict[str, Any], hgb_cfg: dict[str, Any],
    seed_count: int, n_estimators: int,
) -> dict[str, list[Any]]:
    weight = _sample_weight(dev)
    seeds = _ensemble_seeds(42, seed_count)
    return {
        "lgb": [_fit_lgb(dev, features, cfg=lgb_cfg, n_estimators=n_estimators, seed=s, weight=weight) for s in seeds],
        "hgb": [_fit_hgb(dev, features, cfg=hgb_cfg, seed=s, weight=weight) for s in seeds],
        "log": [_fit_logistic(dev, features, seed=s, weight=weight) for s in seeds],
    }


def _predict_final(
    models: dict[str, list[Any]], rows: pd.DataFrame, features: list[str],
    weights: dict[str, float], seed_count: int,
) -> np.ndarray:
    per_seed = []
    for i in range(seed_count):
        per_seed.append(
            weights["lgb"] * models["lgb"][i].predict_proba(rows[features])[:, 1]
            + weights["hgb"] * models["hgb"][i].predict_proba(rows[features])[:, 1]
            + weights["log"] * models["log"][i].predict_proba(rows[features])[:, 1]
        )
    return np.mean(per_seed, axis=0)


def _aggregate_ranking(
    frame: pd.DataFrame, features: list[str], *, runs: int = 3,
    lgb_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """多次滚动排名取平均排名，降低小样本下特征选择方差。"""
    ranks: list[dict[str, float]] = []
    for r in range(runs):
        repeats = 2 + (r % 2)
        ranked = _rolling_feature_ranking(
            frame, features, seed_count=3, permutation_repeats=repeats, lgb_cfg=lgb_cfg
        )
        ranks.append({feature: float(i) for i, feature in enumerate(ranked)})
    avg: dict[str, float] = {}
    for feature in features:
        values = [item.get(feature, float(len(features))) for item in ranks]
        avg[feature] = float(np.mean(values))
    return sorted(features, key=lambda feature: (avg[feature], feature))


def run_search(
    project_root: Path,
    *,
    output_dir: Path,
    seed_count: int = 5,
    feature_sizes: tuple[int, ...] = (16,),
    lgb_name: str = "lgb_a",
    hgb_name: str = "hgb_a",
    fusion: str = "equal",
    include_industry: bool = False,
    ranking_runs: int = 3,
) -> Path:
    root = project_root.resolve()
    matrix_path = root / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix_v5_compatible.csv"
    frame = pd.read_csv(matrix_path, dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    validate_primary_target(frame)
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    frame["modeling_year"] = dates.dt.year.astype(int)

    candidates = [
        f for f in (*BUSINESS_BASE_FEATURES, *AGENT_SEMANTIC_FEATURES) if f in frame.columns
    ]
    if include_industry and "industry_primary" in frame.columns:
        # 行业频率编码：只用 2020-2023（开发集）计算频率，2024 映射同一编码，防泄漏
        dev_rows = frame.loc[frame["modeling_year"].isin(DEV_YEARS)]
        freq = dev_rows["industry_primary"].fillna("unknown").value_counts(normalize=True)
        frame["industry_freq"] = (
            frame["industry_primary"].fillna("unknown").map(freq).fillna(0.0)
        )
        candidates.append("industry_freq")

    lgb_cfg = LGB_CONFIGS[lgb_name]
    hgb_cfg = HGB_CONFIGS[hgb_name]
    ranked = _aggregate_ranking(frame, candidates, runs=ranking_runs, lgb_cfg=lgb_cfg)

    dev = frame.loc[frame["modeling_year"].isin(DEV_YEARS)]
    test = frame.loc[frame["modeling_year"] == TEST_YEAR]
    y_test = test[PRIMARY_TARGET].astype(int).to_numpy()

    rows_out: list[dict[str, Any]] = []
    for size in feature_sizes:
        features = ranked[:size]
        lgb_oof, hgb_oof, log_oof, y_oof, fold_metrics, best_iters = _rolling_component_oof(
            frame, features, lgb_cfg=lgb_cfg, hgb_cfg=hgb_cfg, seed_count=seed_count
        )
        n_estimators = max(50, int(round(float(np.median(best_iters or [150])))))
        if fusion == "equal":
            weights = {"lgb": 1 / 3, "hgb": 1 / 3, "log": 1 / 3}
            oof_prob = (lgb_oof + hgb_oof + log_oof) / 3
            fusion_weights = dict(weights)
        else:
            oof_prob, fusion_weights = _search_fusion_weights(lgb_oof, hgb_oof, log_oof, y_oof)
            weights = fusion_weights
        threshold, policy = _select_business_threshold(
            y_oof, oof_prob, min_recall=0.70, max_alert_fraction=0.40
        )
        rolling_pr = float(average_precision_score(y_oof, oof_prob))
        rolling_roc = float(roc_auc_score(y_oof, oof_prob))

        models = _fit_final_models(
            dev, features, lgb_cfg=lgb_cfg, hgb_cfg=hgb_cfg,
            seed_count=seed_count, n_estimators=n_estimators,
        )
        test_prob = _predict_final(models, test, features, weights, seed_count)
        test_metrics = _classification_metrics(
            y_test, test_prob, threshold=threshold, top_k_fraction=0.20
        )
        rows_out.append(
            {
                "feature_size": size,
                "lgb_cfg": lgb_name,
                "hgb_cfg": hgb_name,
                "fusion": fusion,
                "include_industry": include_industry,
                "weights": fusion_weights,
                "rolling_pr_auc": round(rolling_pr, 4),
                "rolling_roc_auc": round(rolling_roc, 4),
                "test_2024_pr_auc": round(float(test_metrics["pr_auc"]), 4),
                "test_2024_roc_auc": round(float(test_metrics["roc_auc"]), 4),
                "test_2024_brier": round(float(test_metrics["brier_score"]), 4),
                "test_2024_recall": round(float(test_metrics["recall"]), 4),
                "test_2024_precision": round(float(test_metrics["precision"]), 4),
                "n_estimators": n_estimators,
                "threshold": round(float(threshold), 4),
                "fold_metrics": fold_metrics,
            }
        )
        print(
            f"[size={size}{'+industry' if include_industry else ''}] rolling PR={rolling_pr:.4f} | "
            f"2024 PR={test_metrics['pr_auc']:.4f} ROC={test_metrics['roc_auc']:.4f} "
            f"Brier={test_metrics['brier_score']:.4f}",
            flush=True,
        )

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment": "ml_optimization_v5_feature_size_scan",
        "lgb_cfg": lgb_name,
        "hgb_cfg": hgb_name,
        "fusion": fusion,
        "feature_sizes": list(feature_sizes),
        "include_industry": include_industry,
        "features_by_size": {str(size): ranked[:size] for size in feature_sizes},
        "cohort_counts": {"development": int(len(dev)), "test_2024": int(len(test))},
        "results": rows_out,
        "note": "选型标准为 rolling_pr_auc；2024 仅最终评估不参与选择",
    }
    metrics_path = destination / "search_results.json"
    write_json(summary, metrics_path)
    print(f"\n完成 {len(rows_out)} 个配置，结果：{metrics_path}", flush=True)
    return metrics_path


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), "--project-root"),
    output_dir: Path = typer.Option(Path("artifacts/models/ml_optimization_v5"), "--output-dir"),
    seed_count: int = typer.Option(5, min=1),
    feature_sizes: str = typer.Option("16", help="特征数，逗号分隔"),
    lgb_name: str = typer.Option("lgb_a"),
    hgb_name: str = typer.Option("hgb_a"),
    fusion: str = typer.Option("equal"),
    include_industry: bool = typer.Option(False),
    ranking_runs: int = typer.Option(3, min=1),
) -> None:
    run_search(
        project_root,
        output_dir=output_dir,
        seed_count=seed_count,
        feature_sizes=tuple(int(v) for v in feature_sizes.split(",")),
        lgb_name=lgb_name,
        hgb_name=hgb_name,
        fusion=fusion,
        include_industry=include_industry,
        ranking_runs=ranking_runs,
    )


if __name__ == "__main__":
    app()
