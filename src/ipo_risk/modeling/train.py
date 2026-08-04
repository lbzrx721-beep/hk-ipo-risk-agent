"""五日显著下跌的可复现时间切分基线。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ipo_risk.modeling.dataset import IDENTITY_COLUMNS, TARGET_COLUMNS
from ipo_risk.modeling.explain import explain_logistic_pipeline
from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.split import SPLIT_ORDER, validate_time_ordered_splits
from ipo_risk.storage.files import read_json, write_json


@dataclass(frozen=True)
class TrainingResult:
    """训练、预测和评测产物。"""

    model_path: Path
    metrics_path: Path
    predictions_path: Path
    explanations_path: Path
    selected_threshold: float
    primary_variant: str


def train_day5_baselines(
    feature_matrix_path: Path,
    *,
    output_dir: Path,
    summary_path: Path | None = None,
    random_state: int = 42,
    top_k_fraction: float = 0.2,
    max_alert_fraction: float = 0.5,
) -> TrainingResult:
    """训练招股书结构、市场环境与融合三组逻辑回归基线。"""
    if not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if not 0 < max_alert_fraction < 1:
        raise ValueError("max_alert_fraction must be in (0, 1)")
    matrix_path = feature_matrix_path.resolve()
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(matrix_path, dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    validate_primary_target(frame)
    boundaries = validate_time_ordered_splits(frame)

    feature_columns = _feature_columns(frame, summary_path=summary_path)
    variants = _feature_variants(feature_columns)
    all_metrics: dict[str, Any] = {}
    primary_bundle: dict[str, Any] | None = None
    primary_pipeline: Pipeline | None = None
    primary_features: list[str] = []
    primary_threshold = 0.5
    primary_predictions: np.ndarray | None = None

    for variant, columns in variants.items():
        pipeline = _build_pipeline(frame.loc[frame["split"] == "train"], columns, random_state)
        train_rows = frame.loc[frame["split"] == "train"]
        validation_rows = frame.loc[frame["split"] == "validation"]
        pipeline.fit(train_rows[columns], train_rows[PRIMARY_TARGET].astype(int))
        validation_probability = pipeline.predict_proba(
            validation_rows[columns]
        )[:, 1]
        threshold = _select_threshold(
            validation_rows[PRIMARY_TARGET].astype(int).to_numpy(),
            validation_probability,
            max_alert_fraction=max_alert_fraction,
        )
        variant_metrics: dict[str, Any] = {
            "feature_count": len(columns),
            "feature_columns": columns,
            "selected_threshold": threshold,
            "threshold_selected_on": "validation_f2_with_alert_cap",
            "splits": {},
        }
        full_probability = pipeline.predict_proba(frame[columns])[:, 1]
        for split in SPLIT_ORDER:
            mask = frame["split"] == split
            variant_metrics["splits"][split] = _classification_metrics(
                frame.loc[mask, PRIMARY_TARGET].astype(int).to_numpy(),
                full_probability[mask.to_numpy()],
                threshold=threshold,
                top_k_fraction=top_k_fraction,
            )
        validation_metrics = variant_metrics["splits"]["validation"]
        validation_prevalence = float(validation_metrics["positive_rate"])
        validation_pr_auc = float(validation_metrics["pr_auc"])
        validation_top_k_precision = float(
            validation_metrics["top_k"]["precision"]
        )
        variant_metrics["business_quality_gate"] = {
            "passed": (
                validation_pr_auc > validation_prevalence
                and validation_top_k_precision > validation_prevalence
            ),
            "validation_pr_auc_above_base_rate": (
                validation_pr_auc > validation_prevalence
            ),
            "validation_top_k_precision_above_base_rate": (
                validation_top_k_precision > validation_prevalence
            ),
            "note": "未通过时仅可视为研究基线，不应作为正式业务预警模型。",
        }
        all_metrics[variant] = variant_metrics

        bundle = {
            "schema_version": 1,
            "model_name": f"day5_significant_drop_{variant}_logistic_v1",
            "pipeline": pipeline,
            "feature_columns": columns,
            "threshold": threshold,
            "threshold_selected_on": "validation_f2_with_alert_cap",
            "prediction_as_of": "prospectus_date",
            "target": PRIMARY_TARGET,
            "random_state": random_state,
        }
        joblib.dump(bundle, destination / f"{variant}_model.joblib")
        if variant == "fused":
            primary_bundle = bundle
            primary_pipeline = pipeline
            primary_features = columns
            primary_threshold = threshold
            primary_predictions = full_probability

    if primary_bundle is None or primary_pipeline is None or primary_predictions is None:
        raise RuntimeError("fused baseline was not trained")

    global_explanation, local_explanations = explain_logistic_pipeline(
        primary_pipeline,
        frame[primary_features],
    )
    predictions = frame[
        [
            *IDENTITY_COLUMNS,
            *TARGET_COLUMNS,
        ]
    ].copy()
    predictions["predicted_probability"] = primary_predictions
    predictions["predicted_significant_drop_5d"] = (
        primary_predictions >= primary_threshold
    ).astype(int)
    predictions["risk_rank_within_split"] = predictions.groupby("split")[
        "predicted_probability"
    ].rank(ascending=False, method="first").astype(int)
    predictions["top_drivers_json"] = [
        json.dumps(rows, ensure_ascii=False) for rows in local_explanations
    ]
    predictions_path = destination / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    model_path = destination / "fused_model.joblib"
    joblib.dump(primary_bundle, model_path)
    explanations_path = destination / "explanations.json"
    write_json(
        {
            "schema_version": 1,
            "model": primary_bundle["model_name"],
            "global_coefficients": global_explanation,
            "local_explanations_file": predictions_path.name,
        },
        explanations_path,
    )
    metrics_path = destination / "metrics.json"
    write_json(
        {
            "schema_version": 1,
            "target": PRIMARY_TARGET,
            "primary_variant": "fused",
            "split_strategy": "time_ordered",
            "split_boundaries": boundaries,
            "threshold_policy": "validation_f2_max_with_alert_cap",
            "max_validation_alert_fraction": max_alert_fraction,
            "top_k_fraction": top_k_fraction,
            "variants": all_metrics,
            "leakage_controls": {
                "preprocessing_fit_on_train_only": True,
                "model_fit_on_train_only": True,
                "threshold_selected_on_validation_only": True,
                "test_used_for_fitting_or_threshold": False,
            },
        },
        metrics_path,
    )
    return TrainingResult(
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        explanations_path=explanations_path,
        selected_threshold=primary_threshold,
        primary_variant="fused",
    )


def _feature_columns(
    frame: pd.DataFrame,
    *,
    summary_path: Path | None,
) -> list[str]:
    if summary_path is not None and summary_path.is_file():
        summary = read_json(summary_path)
        columns = summary.get("feature_columns") if isinstance(summary, dict) else None
        if isinstance(columns, list) and columns:
            missing = sorted(set(columns).difference(frame.columns))
            if missing:
                raise ValueError(f"feature summary references missing columns: {missing}")
            return [str(column) for column in columns]
    excluded = set(IDENTITY_COLUMNS).union(TARGET_COLUMNS)
    return [column for column in frame.columns if column not in excluded]


def _feature_variants(feature_columns: list[str]) -> dict[str, list[str]]:
    market = [
        column
        for column in feature_columns
        if column.startswith(("market_", "recent_ipo_"))
    ]
    prospectus = [column for column in feature_columns if column not in market]
    if not prospectus:
        raise ValueError("prospectus feature group is empty")
    if not market:
        raise ValueError("market feature group is empty")
    return {
        "prospectus_only": prospectus,
        "market_only": market,
        "fused": feature_columns,
    }


def _build_pipeline(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> Pipeline:
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    numeric = [column for column in feature_columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _select_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    max_alert_fraction: float,
) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.asarray(probability, dtype=float),
            ]
        )
    )
    best_threshold = 0.5
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for candidate in candidates:
        predicted = (probability >= candidate).astype(int)
        if predicted.mean() > max_alert_fraction or predicted.sum() == 0:
            continue
        score = (
            fbeta_score(y_true, predicted, beta=2, zero_division=0),
            recall_score(y_true, predicted, zero_division=0),
            precision_score(y_true, predicted, zero_division=0),
            -abs(float(candidate) - 0.5),
        )
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    if best_score[0] < 0:
        raise ValueError("no threshold satisfies max_alert_fraction")
    return best_threshold


def _classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    top_k_fraction: float,
) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    sample_count = len(y_true)
    positive_count = int(y_true.sum())
    k = max(1, int(math.ceil(sample_count * top_k_fraction)))
    top_indices = np.argsort(-probability)[:k]
    top_hits = int(y_true[top_indices].sum())
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "positive_rate": positive_count / sample_count if sample_count else None,
        "pr_auc": (
            float(average_precision_score(y_true, probability))
            if positive_count
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probability))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "top_k": {
            "fraction": top_k_fraction,
            "sample_count": k,
            "hit_count": top_hits,
            "precision": top_hits / k,
            "recall": top_hits / positive_count if positive_count else None,
        },
        "calibration_bins": _calibration_bins(y_true, probability),
    }
    return result


def _calibration_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    bin_count: int = 5,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assigned = np.minimum(np.digitize(probability, edges[1:-1]), bin_count - 1)
    rows: list[dict[str, Any]] = []
    for index in range(bin_count):
        mask = assigned == index
        if not mask.any():
            continue
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "sample_count": int(mask.sum()),
                "mean_predicted_probability": float(probability[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
            }
        )
    return rows


__all__ = ["TrainingResult", "train_day5_baselines"]
