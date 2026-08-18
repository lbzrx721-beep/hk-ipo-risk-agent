"""基础特征与 Agent 增强特征的无泄漏时间回放比较。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.train import (
    _build_pipeline,
    _classification_metrics,
    _select_threshold,
)
from ipo_risk.storage.files import read_json, write_json


AUXILIARY_TARGET = "target_broke_issue_price_1d"
DEVELOPMENT_SPLITS = ("train", "validation", "test")
EXCLUDED_LOW_INFORMATION_AGENT_FEATURES = frozenset(
    {
        "agent_calculation_count",
        "agent_failed_trace_count",
        "agent_has_risk_pipeline_uncertainty",
        "agent_risk_pipeline_uncertainty_count",
        "agent_risk_pipeline_uncertainty_evidence_count",
        "agent_risk_pipeline_uncertainty_max_confidence",
        "agent_risk_pipeline_uncertainty_max_severity",
        "agent_risk_commercialization_uncertainty_max_confidence",
        "agent_risk_data_privacy_compliance_max_confidence",
        "agent_risk_single_product_dependency_max_confidence",
    }
)


@dataclass(frozen=True)
class AgentComparisonResult:
    """两组模型的训练及比较产物。"""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_manifest_path: Path
    feature_importance_path: Path
    base_model_path: Path
    agent_model_path: Path
    lightgbm_base_model_path: Path
    lightgbm_agent_model_path: Path


def train_agent_feature_comparison(
    feature_matrix_path: Path,
    *,
    base_summary_path: Path,
    agent_summary_path: Path,
    output_dir: Path,
    random_state: int = 42,
    top_k_fraction: float = 0.2,
    max_alert_fraction: float = 0.5,
) -> AgentComparisonResult:
    """在同一公司样本上比较 64 维基础特征和 Agent 增强特征。

    时间边界固定为：2020—2022 训练、2023 验证、2024 测试；2025 年及以后
    仅输出外部案例预测，不参与拟合、阈值选择或模型比较。
    """
    if not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if not 0 < max_alert_fraction < 1:
        raise ValueError("max_alert_fraction must be in (0, 1)")

    frame = pd.read_csv(feature_matrix_path.resolve(), dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    if frame["company_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["company_id"].duplicated(), "company_id"].unique())
        raise ValueError(f"feature matrix contains duplicate companies: {duplicates}")
    validate_primary_target(frame)

    base_features = _summary_feature_columns(
        base_summary_path,
        key="feature_columns",
        frame=frame,
    )
    raw_agent_features = _summary_feature_columns(
        agent_summary_path,
        key="agent_feature_columns",
        frame=frame,
    )
    agent_features = [
        column
        for column in raw_agent_features
        if column not in EXCLUDED_LOW_INFORMATION_AGENT_FEATURES
    ]
    _validate_feature_contract(base_features, agent_features)

    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    if dates.isna().any():
        companies = frame.loc[dates.isna(), "company_id"].tolist()
        raise ValueError(f"invalid prospectus_date for companies: {companies}")
    frame = frame.copy()
    frame["modeling_split"] = dates.dt.year.map(_split_for_year)
    development = frame.loc[frame["modeling_split"].isin(DEVELOPMENT_SPLITS)].copy()
    external = frame.loc[frame["modeling_split"] == "external"].copy()
    development["split"] = development["modeling_split"]

    _validate_development_cohort(development)
    variants = {
        "base_64": base_features,
        "agent_augmented": [*base_features, *agent_features],
    }
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    predictions = development[_prediction_identity_columns(development)].copy()
    predictions[PRIMARY_TARGET] = development[PRIMARY_TARGET].astype(int)
    if AUXILIARY_TARGET in development:
        predictions[AUXILIARY_TARGET] = pd.to_numeric(
            development[AUXILIARY_TARGET], errors="coerce"
        )

    external_predictions = external[_prediction_identity_columns(external)].copy()
    if not external.empty and PRIMARY_TARGET in external:
        external_predictions[PRIMARY_TARGET] = pd.to_numeric(
            external[PRIMARY_TARGET], errors="coerce"
        )
    if not external.empty and AUXILIARY_TARGET in external:
        external_predictions[AUXILIARY_TARGET] = pd.to_numeric(
            external[AUXILIARY_TARGET], errors="coerce"
        )

    metrics: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    importance_rows: list[dict[str, Any]] = []
    train_rows = development.loc[development["split"] == "train"]
    validation_rows = development.loc[development["split"] == "validation"]
    for model_family in ("logistic", "lightgbm"):
        for feature_variant, feature_columns in variants.items():
            variant = f"{model_family}_{feature_variant}"
            if model_family == "logistic":
                pipeline = _build_pipeline(train_rows, feature_columns, random_state)
                pipeline.fit(
                    train_rows[feature_columns],
                    train_rows[PRIMARY_TARGET].astype(int),
                )
                best_iteration: int | None = None
            else:
                pipeline, best_iteration = _fit_lightgbm_pipeline(
                    train_rows,
                    validation_rows,
                    feature_columns,
                    random_state=random_state,
                )

            validation_probability = pipeline.predict_proba(validation_rows[feature_columns])[:, 1]
            threshold, threshold_policy = _select_comparison_threshold(
                validation_rows[PRIMARY_TARGET].astype(int).to_numpy(),
                validation_probability,
                max_alert_fraction=max_alert_fraction,
            )
            full_probability = pipeline.predict_proba(development[feature_columns])[:, 1]
            variant_metrics: dict[str, Any] = {
                "model_family": model_family,
                "feature_variant": feature_variant,
                "feature_count": len(feature_columns),
                "selected_threshold": threshold,
                "threshold_selected_on": threshold_policy,
                "best_iteration": best_iteration,
                "splits": {},
            }
            for split in DEVELOPMENT_SPLITS:
                mask = development["split"] == split
                y_true = development.loc[mask, PRIMARY_TARGET].astype(int).to_numpy()
                probability = full_probability[mask.to_numpy()]
                split_metrics = _classification_metrics(
                    y_true,
                    probability,
                    threshold=threshold,
                    top_k_fraction=top_k_fraction,
                )
                if AUXILIARY_TARGET in development:
                    split_metrics["auxiliary_first_day_break"] = _auxiliary_metrics(
                        development.loc[mask, AUXILIARY_TARGET],
                        probability,
                        threshold,
                    )
                variant_metrics["splits"][split] = split_metrics

            validation_metrics = variant_metrics["splits"]["validation"]
            base_rate = float(validation_metrics["positive_rate"])
            variant_metrics["business_quality_gate"] = {
                "passed": bool(
                    float(validation_metrics["pr_auc"]) > base_rate
                    and float(validation_metrics["top_k"]["precision"]) > base_rate
                ),
                "note": "未通过时仅作为研究结果，不进入正式预警。",
            }
            metrics[variant] = variant_metrics

            probability_column = f"predicted_probability_{variant}"
            prediction_column = f"predicted_significant_drop_5d_{variant}"
            predictions[probability_column] = full_probability
            predictions[prediction_column] = (full_probability >= threshold).astype(int)
            predictions[f"risk_rank_within_split_{variant}"] = (
                predictions.assign(_probability=full_probability)
                .groupby("split")["_probability"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
            if not external.empty:
                probability = pipeline.predict_proba(external[feature_columns])[:, 1]
                external_predictions[probability_column] = probability
                external_predictions[prediction_column] = (probability >= threshold).astype(int)

            if model_family == "lightgbm":
                importance_rows.extend(
                    _lightgbm_feature_importance(
                        pipeline,
                        validation_rows,
                        feature_columns,
                        variant=variant,
                        random_state=random_state,
                    )
                )

            model_path = destination / f"{variant}_model.joblib"
            joblib.dump(
                {
                    "schema_version": 1,
                    "model_name": f"day5_significant_drop_{variant}_v1",
                    "model_family": model_family,
                    "pipeline": pipeline,
                    "feature_columns": feature_columns,
                    "threshold": threshold,
                    "target": PRIMARY_TARGET,
                    "prediction_as_of": "prospectus_date",
                    "training_years": [2020, 2021, 2022],
                    "validation_year": 2023,
                    "test_year": 2024,
                    "random_state": random_state,
                    "best_iteration": best_iteration,
                },
                model_path,
            )
            model_paths[variant] = model_path

    feature_importance_path = destination / "lightgbm_feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(
        feature_importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_path = destination / "predictions_2020_2024.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    external_path: Path | None = None
    if not external_predictions.empty:
        external_path = destination / "external_cases_2025_plus.csv"
        external_predictions.to_csv(external_path, index=False, encoding="utf-8-sig")

    feature_manifest_path = destination / "feature_manifest.json"
    write_json(
        {
            "schema_version": 1,
            "base_feature_count": len(base_features),
            "raw_agent_feature_count": len(raw_agent_features),
            "agent_feature_count": len(agent_features),
            "agent_augmented_feature_count": len(base_features) + len(agent_features),
            "base_feature_columns": base_features,
            "agent_feature_columns": agent_features,
            "excluded_agent_feature_columns": sorted(
                set(raw_agent_features).intersection(EXCLUDED_LOW_INFORMATION_AGENT_FEATURES)
            ),
            "agent_feature_exclusion_policy": (
                "固定排除在2020—2022训练样本中恒定、极低方差或仅反映Agent运行状态的字段；"
                "原始数据仍保留。"
            ),
            "blocked_feature_prefixes": ["target_", "label_"],
            "target": PRIMARY_TARGET,
            "auxiliary_observation_only": AUXILIARY_TARGET,
        },
        feature_manifest_path,
    )
    metrics_path = destination / "metrics.json"
    write_json(
        {
            "schema_version": 1,
            "target": PRIMARY_TARGET,
            "auxiliary_observation_only": AUXILIARY_TARGET,
            "split_strategy": {
                "train": "prospectus year 2020-2022",
                "validation": "prospectus year 2023",
                "test": "prospectus year 2024",
                "external_cases": "prospectus year 2025+",
            },
            "cohort_counts": {
                split: int((development["split"] == split).sum()) for split in DEVELOPMENT_SPLITS
            },
            "external_case_count": int(len(external)),
            "variants": metrics,
            "comparison": _all_model_comparisons(metrics),
            "leakage_controls": {
                "explicit_feature_manifests_only": True,
                "all_target_prefixed_columns_blocked": True,
                "all_label_prefixed_columns_blocked": True,
                "preprocessing_fit_on_train_only": True,
                "model_fit_on_train_only": True,
                "threshold_selected_on_validation_only": True,
                "test_used_for_fitting_or_threshold": False,
                "external_cases_used_for_selection": False,
            },
            "external_predictions_path": str(external_path) if external_path else None,
        },
        metrics_path,
    )
    return AgentComparisonResult(
        output_dir=destination,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        feature_manifest_path=feature_manifest_path,
        feature_importance_path=feature_importance_path,
        base_model_path=model_paths["logistic_base_64"],
        agent_model_path=model_paths["logistic_agent_augmented"],
        lightgbm_base_model_path=model_paths["lightgbm_base_64"],
        lightgbm_agent_model_path=model_paths["lightgbm_agent_augmented"],
    )


def _fit_lightgbm_pipeline(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[Pipeline, int | None]:
    """拟合保守参数的 LightGBM，并仅使用验证集进行早停。"""
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(train_rows[column])
    ]
    numeric = [column for column in feature_columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        # 保留 NaN，让 LightGBM 自己学习缺失值应进入哪一侧分支。
        transformers.append(("numeric", "passthrough", numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
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
    preprocessor.set_output(transform="pandas")
    train_x = preprocessor.fit_transform(train_rows[feature_columns])
    validation_x = preprocessor.transform(validation_rows[feature_columns])
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    classifier.fit(
        train_x,
        train_rows[PRIMARY_TARGET].astype(int),
        sample_weight=sample_weight,
        eval_set=[
            (
                validation_x,
                validation_rows[PRIMARY_TARGET].astype(int),
            )
        ],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline, classifier.best_iteration_


def _select_comparison_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    max_alert_fraction: float,
) -> tuple[float, str]:
    try:
        return (
            _select_threshold(
                y_true,
                probability,
                max_alert_fraction=max_alert_fraction,
            ),
            "2023_validation_f2_with_alert_cap",
        )
    except ValueError as exc:
        if "no threshold satisfies" not in str(exc):
            raise
        # 概率全部并列时无法只选择其中一部分；宁可不预警，也不突破业务告警上限。
        return (
            float(np.nextafter(np.max(probability), np.inf)),
            "2023_validation_no_alert_fallback_for_tied_probabilities",
        )


def _lightgbm_feature_importance(
    pipeline: Pipeline,
    validation_rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    variant: str,
    random_state: int,
) -> list[dict[str, Any]]:
    """输出原始145维字段级的验证集置换重要性和树增益。"""
    y_true = validation_rows[PRIMARY_TARGET].astype(int)
    permutation = permutation_importance(
        pipeline,
        validation_rows[feature_columns],
        y_true,
        scoring="average_precision",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    gain_by_feature, split_by_feature = _aggregate_native_importance(
        pipeline,
        feature_columns,
    )
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_columns):
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "gain": gain_by_feature.get(feature, 0.0),
                "split_count": split_by_feature.get(feature, 0),
                "validation_permutation_mean": float(permutation.importances_mean[index]),
                "validation_permutation_std": float(permutation.importances_std[index]),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["validation_permutation_mean"]),
            float(row["gain"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _aggregate_native_importance(
    pipeline: Pipeline,
    feature_columns: list[str],
) -> tuple[dict[str, float], dict[str, int]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier: lgb.LGBMClassifier = pipeline.named_steps["classifier"]
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    gains = classifier.booster_.feature_importance(importance_type="gain")
    splits = classifier.booster_.feature_importance(importance_type="split")
    gain_by_feature = {feature: 0.0 for feature in feature_columns}
    split_by_feature = {feature: 0 for feature in feature_columns}
    for transformed, gain, split_count in zip(transformed_names, gains, splits, strict=True):
        original = _original_feature_name(transformed, feature_columns)
        gain_by_feature[original] += float(gain)
        split_by_feature[original] += int(split_count)
    return gain_by_feature, split_by_feature


def _original_feature_name(transformed: str, feature_columns: list[str]) -> str:
    suffix = transformed.split("__", maxsplit=1)[-1]
    if suffix in feature_columns:
        return suffix
    matches = [feature for feature in feature_columns if suffix.startswith(f"{feature}_")]
    if not matches:
        raise ValueError(f"cannot map transformed feature to source column: {transformed}")
    return max(matches, key=len)


def _summary_feature_columns(path: Path, *, key: str, frame: pd.DataFrame) -> list[str]:
    payload = read_json(path.resolve())
    columns = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"feature summary missing non-empty {key}: {path}")
    normalized = [str(column) for column in columns]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"feature summary contains duplicate columns: {path}")
    missing = sorted(set(normalized).difference(frame.columns))
    if missing:
        raise ValueError(f"feature summary references missing columns: {missing}")
    return normalized


def _validate_feature_contract(base: list[str], agent: list[str]) -> None:
    overlap = sorted(set(base).intersection(agent))
    if overlap:
        raise ValueError(f"base and Agent feature manifests overlap: {overlap}")
    unsafe = sorted(
        column for column in [*base, *agent] if column.startswith(("target_", "label_"))
    )
    if unsafe:
        raise ValueError(f"post-listing label leakage blocked: {unsafe}")
    invalid_agent = sorted(column for column in agent if not column.startswith("agent_"))
    if invalid_agent:
        raise ValueError(f"Agent feature columns must start with agent_: {invalid_agent}")


def _split_for_year(year: int) -> str:
    if 2020 <= year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    if year == 2024:
        return "test"
    if year >= 2025:
        return "external"
    return "excluded"


def _validate_development_cohort(frame: pd.DataFrame) -> None:
    for split in DEVELOPMENT_SPLITS:
        rows = frame.loc[frame["split"] == split]
        if rows.empty:
            raise ValueError(f"time split is empty: {split}")
    train_values = set(frame.loc[frame["split"] == "train", PRIMARY_TARGET].astype(int))
    if train_values != {0, 1}:
        raise ValueError("training split must contain both target classes")


def _prediction_identity_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "company_id",
        "ticker",
        "company_name",
        "prospectus_date",
        "as_of_date",
        "split",
        "modeling_split",
        "run_id",
    ]
    return [column for column in preferred if column in frame]


def _auxiliary_metrics(
    raw_target: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    target = pd.to_numeric(raw_target, errors="coerce").to_numpy(dtype=float)
    observed = ~np.isnan(target)
    alerts = probability >= threshold
    alert_observed = observed & alerts
    non_alert_observed = observed & ~alerts
    return {
        "sample_count": int(observed.sum()),
        "overall_break_rate": float(target[observed].mean()) if observed.any() else None,
        "alert_group_break_rate": (
            float(target[alert_observed].mean()) if alert_observed.any() else None
        ),
        "non_alert_group_break_rate": (
            float(target[non_alert_observed].mean()) if non_alert_observed.any() else None
        ),
        "note": "仅作事后观察，不参与拟合或阈值选择。",
    }


def _all_model_comparisons(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "agent_feature_delta_within_logistic": (
            "logistic_agent_augmented",
            "logistic_base_64",
        ),
        "agent_feature_delta_within_lightgbm": (
            "lightgbm_agent_augmented",
            "lightgbm_base_64",
        ),
        "lightgbm_delta_on_base_features": (
            "lightgbm_base_64",
            "logistic_base_64",
        ),
        "lightgbm_delta_on_agent_features": (
            "lightgbm_agent_augmented",
            "logistic_agent_augmented",
        ),
    }
    result: dict[str, Any] = {}
    for comparison, (left_name, right_name) in comparisons.items():
        result[comparison] = {}
        for split in ("validation", "test"):
            left = metrics[left_name]["splits"][split]
            right = metrics[right_name]["splits"][split]
            result[comparison][split] = {
                "pr_auc_delta": _metric_delta(left, right, "pr_auc"),
                "roc_auc_delta": _metric_delta(left, right, "roc_auc"),
                "recall_delta": _metric_delta(left, right, "recall"),
                "f2_delta": _metric_delta(left, right, "f2"),
                "top_k_precision_delta": (
                    float(left["top_k"]["precision"]) - float(right["top_k"]["precision"])
                ),
                "brier_score_reduction": _metric_delta(right, left, "brier_score"),
            }
    return result


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key]) - float(right[key])


__all__ = [
    "AgentComparisonResult",
    "EXCLUDED_LOW_INFORMATION_AGENT_FEATURES",
    "train_agent_feature_comparison",
]
