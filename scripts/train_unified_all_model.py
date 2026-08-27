"""使用2020—2025全部有效公司训练统一五日显著下跌预测模型。

模型最终拟合使用全部带标签样本；阈值与回测指标来自严格按年份向前验证：
2020—2023预测2024，2020—2024预测2025。最终产物同时包含基础/市场
字段和Agent语义字段，并冻结特征契约供线上预测复用。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from optimize_high_risk_recall_model import (  # noqa: E402
    TARGET,
    _build_lightgbm,
    _choose_recall_threshold,
    _fit_predict,
    _load_combined_frame,
    _metrics,
    _numeric_frame,
    _sample_weights,
)
from run_stable_regression_experiment import _repair_currency_and_hkd_mn  # noqa: E402
from ipo_risk.modeling.predict import (  # noqa: E402
    load_model_bundle,
    predict_feature_rows,
)
from ipo_risk.modeling.prelisting import (  # noqa: E402
    ModelFeatureContract,
    ModelFeatureField,
    feature_formula_version,
    feature_group,
    feature_unit,
    load_model_feature_contract,
)
from ipo_risk.modeling.stable_agent_model import (  # noqa: E402
    add_derived_prediction_features,
)


MODEL_NAME = "unified_all_2020_2025_v1"
SEEDS = (42, 137, 271, 389, 521)
MINIMUM_IMPORTANT_FEATURE_COVERAGE = 0.20

# 重要性优先的上市前发行、财务、集中度和市场字段。缺失值交给模型填补，
# 同时保留缺失标记；不因覆盖率不足而舍弃有明确业务含义的信号。
BASE_FEATURES = (
    "offering_offer_price_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "fundamental_revenue_latest_hkd_mn",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_gross_margin_latest_pct",
    "fundamental_operating_cash_flow_latest_hkd_mn",
    "customer_top5_ratio_pct",
    "supplier_top5_ratio_pct",
    "market_return_20d_pct",
    "market_volatility_20d_pct",
    "recent_ipo_5d_drop_rate_180d",
    "recent_ipo_5d_mean_return_180d_pct",
    "engineered_price_to_sales",
    "engineered_fundraising_to_revenue",
    "engineered_net_proceeds_to_market_cap",
    "engineered_operating_cash_flow_margin",
    "missing_revenue_latest",
    "missing_revenue_cagr_3y",
    "missing_operating_cash_flow_latest",
    "missing_offering_share_ratio",
    "missing_net_proceeds",
)

# 严格时间回测中方向相对稳定、且具备清晰业务语义的Agent风险字段。
AGENT_FEATURES = (
    "agent_semantic_risk_customer_concentration_supported_count",
    "agent_semantic_risk_minority_shareholder_count",
    "agent_semantic_risk_minority_shareholder_max_severity",
    "agent_semantic_risk_foreign_exchange_max_severity",
    "agent_semantic_has_minority_shareholder",
    "agent_semantic_risk_capital_commitment_supported_count",
    "agent_semantic_risk_offer_structure_max_severity",
    "agent_semantic_risk_capital_commitment_count",
    "agent_semantic_risk_customer_concentration_count",
    "agent_semantic_risk_channel_dependency_count",
    "agent_semantic_risk_capital_commitment_max_severity",
    "agent_semantic_risk_channel_dependency_max_severity",
    "agent_semantic_risk_preipo_investor_rights_supported_count",
    "agent_semantic_has_foreign_exchange",
    "agent_semantic_risk_channel_dependency_supported_count",
    "agent_semantic_risk_offer_structure_count",
)

FEATURES = (*BASE_FEATURES, *AGENT_FEATURES)


def _prepare_frame(root: Path) -> pd.DataFrame:
    frame = _load_combined_frame(root)
    frame = _repair_currency_and_hkd_mn(frame, root)
    frame = add_derived_prediction_features(frame)
    frame = frame.dropna(subset=[TARGET, "prospectus_date"]).copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    if frame["company_id"].duplicated().any():
        duplicates = frame.loc[frame["company_id"].duplicated(), "company_id"].tolist()
        raise ValueError(f"训练矩阵存在重复公司：{duplicates[:10]}")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("目标标签必须为0/1")
    missing_columns = sorted(set(FEATURES).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"训练矩阵缺少冻结字段：{', '.join(missing_columns)}")
    return frame.sort_values("prospectus_date").reset_index(drop=True)


def _feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce")
        coverage = float(values.notna().mean())
        rows.append(
            {
                "feature": name,
                "group": "agent" if name in AGENT_FEATURES else feature_group(name),
                "coverage": coverage,
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "nonzero_fraction": float(values.fillna(0).ne(0).mean()),
                "unit": feature_unit(name),
                "formula_version": feature_formula_version(name),
            }
        )
        if coverage < MINIMUM_IMPORTANT_FEATURE_COVERAGE:
            raise ValueError(
                f"重要字段{name}覆盖率{coverage:.1%}低于"
                f"{MINIMUM_IMPORTANT_FEATURE_COVERAGE:.0%}"
            )
        if values.nunique(dropna=True) < 2:
            raise ValueError(f"字段{name}没有可学习的变化")
    return pd.DataFrame(rows)


def _forward_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], float]:
    folds = (
        (
            frame.loc[frame["modeling_year"].between(2020, 2023)],
            frame.loc[frame["modeling_year"].eq(2024)],
            "train_2020_2023_validate_2024",
        ),
        (
            frame.loc[frame["modeling_year"].between(2020, 2024)],
            frame.loc[frame["modeling_year"].eq(2025)],
            "train_2020_2024_validate_2025",
        ),
    )
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for train, validation, fold_name in folds:
        if train.empty or validation.empty:
            raise ValueError(f"时间验证折为空：{fold_name}")
        probability = _fit_predict(
            train,
            validation,
            FEATURES,
            _build_lightgbm,
            weighting="half_life_2y",
            score_mode="percentile",
        )
        labels = validation[TARGET].astype(int).to_numpy()
        fold_metrics.append(
            {
                "fold": fold_name,
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "positive_count": int(labels.sum()),
            }
        )
        identity = [
            column
            for column in ("company_id", "company_name", "prospectus_date")
            if column in validation
        ]
        part = validation[identity].copy()
        part["fold"] = fold_name
        part[TARGET] = labels
        part["predicted_probability"] = probability
        predictions.append(part)

    output = pd.concat(predictions, ignore_index=True)
    labels = output[TARGET].astype(int).to_numpy()
    probability = output["predicted_probability"].to_numpy(dtype=float)
    threshold = _choose_recall_threshold(labels, probability)
    output["predicted_significant_drop_5d"] = (probability >= threshold).astype(int)
    overall = _metrics(labels, probability, threshold)
    for item in fold_metrics:
        rows = output.loc[output["fold"].eq(item["fold"])]
        item["metrics_at_global_threshold"] = _metrics(
            rows[TARGET].astype(int).to_numpy(),
            rows["predicted_probability"].to_numpy(dtype=float),
            threshold,
        )
    overall["folds"] = fold_metrics
    overall["evaluation_note"] = (
        "按年份向前验证；每家公司预测时只使用更早年份，最终全量模型未参与这些预测。"
    )
    return output, overall, threshold


def _fit_all(frame: pd.DataFrame) -> tuple[list[Any], list[np.ndarray]]:
    matrix = _numeric_frame(frame, FEATURES)
    as_of = frame["prospectus_date"].max() + pd.Timedelta(days=1)
    weights = _sample_weights(frame, scheme="half_life_2y", as_of=as_of)
    models: list[Any] = []
    references: list[np.ndarray] = []
    for seed in SEEDS:
        model = _build_lightgbm(seed)
        model.fit(matrix, frame[TARGET].astype(int), model__sample_weight=weights)
        models.append(model)
        references.append(np.sort(model.predict_proba(matrix)[:, 1]))
    return models, references


def _feature_contract() -> ModelFeatureContract:
    return ModelFeatureContract(
        model_name=MODEL_NAME,
        fields=tuple(
            ModelFeatureField(
                name=name,
                group="agent" if name in AGENT_FEATURES else feature_group(name),
                unit=feature_unit(name),
                required=False,
                missing_policy="model_impute",
                formula_version=feature_formula_version(name),
            )
            for name in FEATURES
        ),
        prediction_as_of="listing_date",
        minimum_feature_completeness=0.60,
        review_below_completeness=0.80,
        review_group_completeness=(("agent", 1.00), ("market", 1.00)),
        required_group_completeness=(("market", 1.00),),
    )


def main(
    output_dir: Path = Path("artifacts/models/unified_all_2020_2025_v1"),
) -> None:
    destination = output_dir if output_dir.is_absolute() else ROOT / output_dir
    destination.mkdir(parents=True, exist_ok=True)

    print("1/4 装配2020—2025全部有效训练数据", flush=True)
    frame = _prepare_frame(ROOT)
    coverage = _feature_coverage(frame)
    coverage.to_csv(destination / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    print(
        f"样本{len(frame)}家，正例{int(frame[TARGET].sum())}家，特征{len(FEATURES)}个",
        flush=True,
    )

    print("2/4 执行严格按年份向前回测并冻结阈值", flush=True)
    oof, validation_metrics, threshold = _forward_validation(frame)
    oof.to_csv(destination / "forward_validation_predictions.csv", index=False, encoding="utf-8-sig")

    print("3/4 使用全部公司拟合最终统一模型", flush=True)
    models, references = _fit_all(frame)
    contract = _feature_contract()
    bundle = {
        "schema_version": 2,
        "model_name": MODEL_NAME,
        "target": TARGET,
        "prediction_as_of": "listing_date",
        "features": list(FEATURES),
        "base_features": list(BASE_FEATURES),
        "agent_features": list(AGENT_FEATURES),
        "feature_count": len(FEATURES),
        "models": models,
        "seeds": list(SEEDS),
        "score_mode": "percentile",
        "reference_probabilities": references,
        "threshold": float(threshold),
        "training_policy": "all_labeled_companies_2020_2025_with_2y_half_life_weights",
        "training_sample_count": int(len(frame)),
        "training_year_counts": {
            str(year): int(count)
            for year, count in frame["modeling_year"].value_counts().sort_index().items()
        },
        "forward_validation_metrics": validation_metrics,
        "feature_contract": contract.to_payload(),
    }
    model_path = destination / "model.joblib"
    joblib.dump(bundle, model_path)
    (destination / "feature_contract.json").write_text(
        json.dumps(contract.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "model_path": str(model_path.relative_to(ROOT)),
        "target": TARGET,
        "final_training_uses_all_samples": True,
        "sample_count": int(len(frame)),
        "positive_count": int(frame[TARGET].sum()),
        "feature_count": len(FEATURES),
        "base_feature_count": len(BASE_FEATURES),
        "agent_feature_count": len(AGENT_FEATURES),
        "feature_selection_policy": "business_importance_with_model_imputation",
        "minimum_important_feature_coverage": MINIMUM_IMPORTANT_FEATURE_COVERAGE,
        "threshold": float(threshold),
        "forward_validation_metrics": validation_metrics,
        "warning": (
            "最终模型已使用全部2020—2025标签，不能再用同一批数据报告独立测试准确率；"
            "summary中的指标来自最终拟合前的按年份向前回测。"
            "混合币种且尚未标准化的期末现金字段未纳入模型。"
        ),
    }
    (destination / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("4/4 重新加载模型与契约进行端到端自检", flush=True)
    loaded = load_model_bundle(model_path)
    loaded_contract = load_model_feature_contract(
        loaded, sidecar_path=destination / "feature_contract.json"
    )
    if loaded_contract.fingerprint != contract.fingerprint:
        raise ValueError("模型契约自检失败")
    checked = predict_feature_rows(loaded, frame.head(5))
    if checked["predicted_probability"].isna().any():
        raise ValueError("模型预测自检产生空概率")

    metrics = validation_metrics
    print(
        "完成："
        f"向前回测准确率{metrics['accuracy']:.1%}，"
        f"召回率{metrics['recall']:.1%}，"
        f"PR-AUC {metrics['pr_auc']:.3f}；"
        f"最终模型已使用全部{len(frame)}家公司。",
        flush=True,
    )
    print(f"模型：{model_path}", flush=True)


if __name__ == "__main__":
    typer.run(main)
