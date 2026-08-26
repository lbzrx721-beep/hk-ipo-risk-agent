"""固化增强模型 v4：三模型融合 + 市场宽度随机森林。

市场特征严格锚定上市日并排除上市日及之后数据。使用 2020-2023 年训练及
滚动 OOF 选择阈值，2024 年不参与拟合。
由于研发阶段已查看过 2024 结果，该年度仅作为开发回测，不宣称为最终盲测。
旧 v2/v3 模型不会被覆盖。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_enhanced_model as v2_finalizer  # noqa: E402
from ipo_risk.modeling.prelisting import ModelFeatureContract  # noqa: E402
from ipo_risk.modeling.stable_agent_model import (  # noqa: E402
    add_derived_prediction_features,
)
from ipo_risk.storage.files import write_json  # noqa: E402


TARGET = "target_significant_drop_5d"
DEV_YEARS = (2020, 2021, 2022, 2023)
OOF_YEARS = (2021, 2022, 2023)
TEST_YEAR = 2024
RF_EXTRA_FEATURES = ("market_advance_ratio_20d", "market_volatility_20d_pct")
RF_SEEDS = (42, 137, 271)
RF_CONFIG = {
    "n_estimators": 700,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
}
MODEL_NAME = "enhanced_blend4_rf_v4_listing"


def _balanced_sample_weight(rows: pd.DataFrame) -> np.ndarray:
    y = rows[TARGET].astype(int).to_numpy()
    positive = max(1, int(y.sum()))
    negative = max(1, int(len(y) - positive))
    return np.where(y == 1, negative / positive, 1.0)


def _fit_rf(rows: pd.DataFrame, features: list[str], *, seed: int) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                RandomForestClassifier(
                    **RF_CONFIG,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(
        rows[features],
        rows[TARGET].astype(int),
        model__sample_weight=_balanced_sample_weight(rows),
    )
    return model


def _predict_rf(models: list[Pipeline], rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.mean(
        [model.predict_proba(rows[features])[:, 1] for model in models],
        axis=0,
    )


def _fit_predict_base(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    features = list(bundle["features"])
    weight = v2_finalizer._sample_weight(train)
    per_seed: list[np.ndarray] = []
    for seed in bundle["seeds"]:
        lgb_model = v2_finalizer._fit_lgb(
            train,
            features,
            n_estimators=int(bundle["n_estimators"]),
            seed=int(seed),
            weight=weight,
        )
        hgb_model = v2_finalizer._fit_hgb(
            train, features, seed=int(seed), weight=weight
        )
        log_model = v2_finalizer._fit_logistic(
            train, features, seed=int(seed), weight=weight
        )
        per_seed.append(
            (
                float(bundle["weights"]["lgb"])
                * lgb_model.predict_proba(validation[features])[:, 1]
                + float(bundle["weights"]["hgb"])
                * hgb_model.predict_proba(validation[features])[:, 1]
                + float(bundle["weights"]["log"])
                * log_model.predict_proba(validation[features])[:, 1]
            )
        )
    return np.mean(per_seed, axis=0)


def _predict_saved_base(bundle: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    per_seed: list[np.ndarray] = []
    for index in range(len(bundle["lgb_models"])):
        per_seed.append(
            float(bundle["weights"]["lgb"])
            * bundle["lgb_models"][index].predict_proba(rows[features])[:, 1]
            + float(bundle["weights"]["hgb"])
            * bundle["hgb_models"][index].predict_proba(rows[features])[:, 1]
            + float(bundle["weights"]["log"])
            * bundle["log_models"][index].predict_proba(rows[features])[:, 1]
        )
    return np.mean(per_seed, axis=0)


def _fit_base_models(
    rows: pd.DataFrame,
    bundle: dict[str, Any],
) -> dict[str, list[Any]]:
    features = list(bundle["features"])
    weight = v2_finalizer._sample_weight(rows)
    lgb_models: list[Any] = []
    hgb_models: list[Any] = []
    log_models: list[Any] = []
    for seed in bundle["seeds"]:
        lgb_models.append(
            v2_finalizer._fit_lgb(
                rows,
                features,
                n_estimators=int(bundle["n_estimators"]),
                seed=int(seed),
                weight=weight,
            )
        )
        hgb_models.append(
            v2_finalizer._fit_hgb(rows, features, seed=int(seed), weight=weight)
        )
        log_models.append(
            v2_finalizer._fit_logistic(rows, features, seed=int(seed), weight=weight)
        )
    return {
        "lgb_models": lgb_models,
        "hgb_models": hgb_models,
        "log_models": log_models,
    }


def _choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for threshold in np.arange(0.20, 0.801, 0.005):
        predicted = (probability >= threshold).astype(int)
        recall = float(recall_score(y, predicted, zero_division=0))
        if recall < 0.45:
            continue
        candidates.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y, predicted)),
                "precision": float(precision_score(y, predicted, zero_division=0)),
                "recall": recall,
                "f1": float(f1_score(y, predicted, zero_division=0)),
            }
        )
    if not candidates:
        raise RuntimeError("OOF 概率无法找到召回率不低于 45% 的阈值。")
    best = max(candidates, key=lambda row: (row["accuracy"], row["f1"], row["precision"]))
    return best["threshold"], best


def _choose_component_weights(
    y: np.ndarray,
    base_probability: np.ndarray,
    rf_probability: np.ndarray,
) -> tuple[dict[str, float], float, dict[str, float]]:
    candidates: list[tuple[dict[str, float], float, dict[str, float], float]] = []
    for rf_weight in np.arange(0.0, 1.001, 0.1):
        weights = {"blend3": float(1.0 - rf_weight), "rf": float(rf_weight)}
        probability = (
            weights["blend3"] * base_probability + weights["rf"] * rf_probability
        )
        threshold, policy = _choose_threshold(y, probability)
        candidates.append(
            (
                weights,
                threshold,
                policy,
                float(average_precision_score(y, probability)),
            )
        )
    weights, threshold, policy, pr_auc = max(
        candidates,
        key=lambda item: (
            item[2]["accuracy"],
            item[2]["f1"],
            item[2]["precision"],
            item[3],
            -item[0]["rf"],
        ),
    )
    return weights, threshold, {**policy, "pr_auc": pr_auc}


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def main(project_root: Path = Path.cwd()) -> None:
    root = project_root.resolve()
    source = root / "artifacts" / "models" / "enhanced_blend3_v2_h5"
    destination = root / "artifacts" / "models" / MODEL_NAME
    matrix_path = (
        root
        / "data"
        / "processed"
        / "modeling"
        / "agent_augmented_feature_matrix_multihorizon.csv"
    )
    frame = pd.read_csv(matrix_path, dtype={"company_id": str}, low_memory=False)
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    frame = frame.copy()
    frame["modeling_year"] = pd.to_datetime(
        frame["prospectus_date"], errors="raise"
    ).dt.year.astype(int)
    frame = add_derived_prediction_features(frame)

    base_bundle = joblib.load(source / "model.joblib")
    base_features = list(base_bundle["features"])
    rf_features = list(dict.fromkeys([*base_features, *RF_EXTRA_FEATURES]))
    missing = [name for name in rf_features if name not in frame]
    if missing:
        raise ValueError(f"训练矩阵缺少 v3 特征：{', '.join(missing)}")

    oof_y: list[np.ndarray] = []
    oof_base_probability: list[np.ndarray] = []
    oof_rf_probability: list[np.ndarray] = []
    for year in OOF_YEARS:
        train = frame.loc[frame["modeling_year"].between(DEV_YEARS[0], year - 1)]
        validation = frame.loc[frame["modeling_year"] == year]
        base_probability = _fit_predict_base(train, validation, base_bundle)
        rf_models = [_fit_rf(train, rf_features, seed=seed) for seed in RF_SEEDS]
        rf_probability = _predict_rf(rf_models, validation, rf_features)
        labels = validation[TARGET].astype(int).to_numpy()
        oof_y.append(labels)
        oof_base_probability.append(base_probability)
        oof_rf_probability.append(rf_probability)
        print(f"滚动验证完成：{year}（{len(validation)} 家）", flush=True)

    labels_oof = np.concatenate(oof_y)
    base_probability_oof = np.concatenate(oof_base_probability)
    rf_probability_oof = np.concatenate(oof_rf_probability)
    component_weights, threshold, threshold_policy = _choose_component_weights(
        labels_oof,
        base_probability_oof,
        rf_probability_oof,
    )
    probability_oof = (
        component_weights["blend3"] * base_probability_oof
        + component_weights["rf"] * rf_probability_oof
    )
    fold_metrics: list[dict[str, Any]] = []
    for year, labels, base_probability, rf_probability in zip(
        OOF_YEARS,
        oof_y,
        oof_base_probability,
        oof_rf_probability,
        strict=True,
    ):
        probability = (
            component_weights["blend3"] * base_probability
            + component_weights["rf"] * rf_probability
        )
        fold_metrics.append(
            {
                "validation_year": year,
                "sample_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "pr_auc": float(average_precision_score(labels, probability)),
                "roc_auc": float(roc_auc_score(labels, probability)),
            }
        )
    rolling_metrics = _metrics(labels_oof, probability_oof, threshold)

    development = frame.loc[frame["modeling_year"].isin(DEV_YEARS)]
    test = frame.loc[frame["modeling_year"] == TEST_YEAR]
    final_base_models = _fit_base_models(development, base_bundle)
    final_rf_models = [_fit_rf(development, rf_features, seed=seed) for seed in RF_SEEDS]
    refreshed_base_bundle = dict(base_bundle)
    refreshed_base_bundle.update(final_base_models)
    base_test_probability = _predict_saved_base(refreshed_base_bundle, test)
    rf_test_probability = _predict_rf(final_rf_models, test, rf_features)
    test_probability = (
        component_weights["blend3"] * base_test_probability
        + component_weights["rf"] * rf_test_probability
    )
    test_labels = test[TARGET].astype(int).to_numpy()
    test_metrics = _metrics(test_labels, test_probability, threshold)

    model_bundle = dict(refreshed_base_bundle)
    model_bundle.update(
        {
            "schema_version": 2,
            "model_name": MODEL_NAME,
            "model_family": "blend3_rf",
            "prediction_as_of": "listing_date",
            "features": rf_features,
            "feature_count": len(rf_features),
            "feature_ranking": [
                *base_bundle.get("feature_ranking", base_features),
                *[name for name in RF_EXTRA_FEATURES if name not in base_features],
            ],
            "base_features": base_features,
            "rf_features": rf_features,
            "rf_models": final_rf_models,
            "rf_seeds": list(RF_SEEDS),
            "rf_config": RF_CONFIG,
            "component_weights": component_weights,
            "threshold": float(threshold),
            "threshold_policy": {
                "policy": "maximize_oof_accuracy_with_minimum_recall",
                "minimum_recall": 0.45,
                **threshold_policy,
            },
            "blend": (
                f"{component_weights['blend3']:.2f} * blend3 + "
                f"{component_weights['rf']:.2f} * balanced_random_forest"
            ),
        }
    )
    feature_contract = ModelFeatureContract.from_bundle(model_bundle).to_payload()
    model_bundle["feature_contract"] = feature_contract
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, destination / "model.joblib")
    write_json(feature_contract, destination / "feature_contract.json")

    identity = [
        name
        for name in ("company_id", "ticker", "company_name", "prospectus_date", "run_id")
        if name in test
    ]
    predictions = test[identity].copy()
    predictions[TARGET] = test_labels
    predictions["blend3_probability"] = base_test_probability
    predictions["rf_probability"] = rf_test_probability
    predictions["blend4_probability"] = test_probability
    predictions["predicted_significant_drop_5d"] = (
        test_probability >= threshold
    ).astype(int)
    predictions.to_csv(
        destination / "predictions_2024.csv", index=False, encoding="utf-8-sig"
    )

    prior_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    metrics = {
        "schema_version": 2,
        "model_name": MODEL_NAME,
        "target": TARGET,
        "cohort_counts": {
            "development_2020_2023": int(len(development)),
            "rolling_oof_2021_2023": int(len(labels_oof)),
            "test_2024": int(len(test)),
        },
        "features": rf_features,
        "base_features": base_features,
        "rf_extra_features": list(RF_EXTRA_FEATURES),
        "component_weights": component_weights,
        "rolling": {"fold_metrics": fold_metrics, **rolling_metrics},
        "test_2024": test_metrics,
        "comparison_to_v2": {
            "rolling_pr_auc_v2": 0.3690682838745587,
            "rolling_accuracy_v2": 0.6551724137931034,
            "rolling_recall_v2": 0.5087719298245614,
            "test_2024_pr_auc_v2": prior_metrics["test_2024"]["pr_auc"],
            "test_2024_accuracy_v2": prior_metrics["test_2024"]["accuracy"],
            "test_2024_recall_v2": prior_metrics["test_2024"]["recall"],
        },
        "leakage_controls": {
            "prediction_as_of": "listing_date",
            "market_features_use_trade_date_strictly_before_listing": True,
            "model_training_uses_2020_2023_only": True,
            "threshold_selected_on_rolling_oof_2021_2023_only": True,
            "test_2024_used_for_model_fitting": False,
            "test_2024_used_for_threshold_selection": False,
            "test_2024_observed_during_candidate_research": True,
            "test_2024_is_final_blind_test": False,
            "target_columns_excluded_from_features": True,
        },
    }
    write_json(metrics, destination / "metrics.json")
    print(f"v4 模型已固化：{destination / 'model.joblib'}", flush=True)
    print(f"OOF选择融合权重：{component_weights}", flush=True)
    print(
        "2024："
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"PR-AUC={test_metrics['pr_auc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
