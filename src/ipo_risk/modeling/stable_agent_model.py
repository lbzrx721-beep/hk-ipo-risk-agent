"""基于滚动年份验证的稳定 Agent 特征模型。"""

from __future__ import annotations

import math
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
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ipo_risk.modeling.agent_comparison import (
    EXCLUDED_LOW_INFORMATION_AGENT_FEATURES,
    _aggregate_native_importance,
    _fit_lightgbm_pipeline,
    _summary_feature_columns,
    _validate_feature_contract,
)
from ipo_risk.modeling.agent_precision import AGENT_PRECISION_FEATURE_NAMES
from ipo_risk.modeling.labels import PRIMARY_TARGET, validate_primary_target
from ipo_risk.modeling.risk_semantics import semantic_model_feature_names
from ipo_risk.modeling.train import _build_pipeline, _classification_metrics
from ipo_risk.storage.files import write_json


ROLLING_VALIDATION_YEARS = (2022, 2023)

# 业务白名单只保留上市前可获得、含义清楚且可用于新股预测的字段。
# 不纳入文件大小、页数、年份等容易形成伪规律的技术或身份字段；
# 同一发行概念只保留一个主口径，避免同时放入募资上限、下限和中间值。
RAW_BUSINESS_BASE_FEATURES = (
    "offering_offer_price_mid_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "fundamental_revenue_latest_hkd_mn",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_gross_margin_latest_pct",
    "fundamental_operating_cash_flow_latest_hkd_mn",
    "fundamental_operating_cash_flow_to_revenue_latest_pct",
    "pipeline_stage_score",
    "market_return_5d_pct",
    "market_return_20d_pct",
    "market_return_60d_pct",
    "market_volatility_20d_pct",
    "market_volatility_60d_pct",
    "recent_ipo_5d_drop_rate_180d",
    "recent_ipo_5d_mean_return_180d_pct",
    "recent_ipo_count_30d",
)

# 仅由上市前已有字段确定性计算，不新增外部数据，也不接触任何标签。
ENGINEERED_BASE_FEATURES = (
    "engineered_log_revenue_hkd_mn",
    "engineered_log_market_cap_hkd_mn",
    "engineered_log_net_proceeds_hkd_mn",
    "engineered_price_to_sales",
    "engineered_fundraising_to_revenue",
    "engineered_net_proceeds_to_market_cap",
    "engineered_operating_cash_flow_margin",
    "engineered_cash_replenishment_ratio",
    "engineered_cash_runway_months",
    "engineered_growth_adjusted_price_to_sales",
    "engineered_zero_or_negative_revenue",
    "engineered_negative_operating_cash_flow",
    "missing_revenue_cagr_3y",
    "missing_offering_share_ratio",
    "missing_revenue_latest",
    "missing_operating_cash_flow_latest",
    "missing_net_proceeds",
)

BUSINESS_BASE_FEATURES = (*RAW_BUSINESS_BASE_FEATURES, *ENGINEERED_BASE_FEATURES)

AGENT_INTERACTION_FEATURES = (
    "agent_interaction_financial_risk_negative_cash_flow",
    "agent_interaction_market_liquidity_volatility_20d",
    "agent_interaction_related_party_governance",
    "agent_interaction_free_float_low_issue_ratio",
    "agent_interaction_cash_burn_fundraising_intensity",
    "agent_interaction_supported_high_severity_share",
    "agent_interaction_semantic_risk_density",
)

TIME_DECAY_WEIGHTS = {
    2020: 0.50,
    2021: 0.65,
    2022: 0.80,
    2023: 1.00,
}
DEFAULT_ENSEMBLE_SEEDS = (42, 137, 271, 389, 521)

# Agent 白名单表达“发现了什么风险”，而不是模型调用次数、失败次数、
# 证据数量或置信度等分析过程统计。
AGENT_SEMANTIC_FEATURES = (
    "agent_finding_count",
    "agent_max_severity_score",
    "agent_severity_weighted_mean",
    "agent_high_or_critical_share",
    "agent_role_finance_share",
    "agent_role_legal_share",
    "agent_role_business_share",
    "agent_role_market_share",
    "agent_uncertainty_share",
    *AGENT_PRECISION_FEATURE_NAMES,
    *semantic_model_feature_names(),
    *AGENT_INTERACTION_FEATURES,
)


@dataclass(frozen=True)
class StableAgentModelResult:
    """稳定特征模型的落盘结果。"""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_stability_path: Path
    feature_manifest_path: Path
    model_path: Path
    base_model_path: Path


def train_stable_agent_model(
    feature_matrix_path: Path,
    *,
    base_summary_path: Path,
    agent_summary_path: Path,
    output_dir: Path,
    random_state: int = 42,
    base_feature_limit: int = 30,
    agent_feature_limit: int = 20,
    min_recall: float = 0.70,
    max_alert_fraction: float = 0.40,
    top_k_fraction: float = 0.20,
    permutation_repeats: int = 5,
    ensemble_seed_count: int = 1,
    stability_seed_count: int = 1,
    use_time_decay: bool = True,
    base_feature_whitelist: tuple[str, ...] | None = None,
    agent_feature_whitelist: tuple[str, ...] | None = None,
) -> StableAgentModelResult:
    """从业务白名单滚动筛选稳定特征，并完成 Agent 增强消融实验。"""
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1]")
    if not 0 < max_alert_fraction < 1:
        raise ValueError("max_alert_fraction must be in (0, 1)")
    if base_feature_limit < 1 or agent_feature_limit < 1:
        raise ValueError("feature limits must be positive")
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    if ensemble_seed_count < 1 or stability_seed_count < 1:
        raise ValueError("seed counts must be positive")

    frame = pd.read_csv(feature_matrix_path.resolve(), dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    if frame["company_id"].duplicated().any():
        raise ValueError("feature matrix contains duplicate companies")
    validate_primary_target(frame)
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("feature matrix contains invalid prospectus_date")
    frame = frame.copy()
    frame["modeling_year"] = dates.dt.year.astype(int)
    frame = add_derived_prediction_features(frame)

    raw_base_features = _summary_feature_columns(
        base_summary_path, key="feature_columns", frame=frame
    )
    raw_base_features.extend(
        feature
        for feature in ENGINEERED_BASE_FEATURES
        if feature in frame and feature not in raw_base_features
    )
    declared_agent_features = _summary_feature_columns(
        agent_summary_path, key="agent_feature_columns", frame=frame
    )
    usable_agent_features = [
        feature
        for feature in declared_agent_features
        if feature not in EXCLUDED_LOW_INFORMATION_AGENT_FEATURES
    ]
    usable_agent_features.extend(
        feature
        for feature in AGENT_SEMANTIC_FEATURES
        if feature in frame and feature not in usable_agent_features
    )
    base_features = _apply_feature_whitelist(
        raw_base_features,
        base_feature_whitelist or BUSINESS_BASE_FEATURES,
        group="business base",
    )
    agent_features = _apply_feature_whitelist(
        usable_agent_features,
        agent_feature_whitelist or AGENT_SEMANTIC_FEATURES,
        group="Agent semantic",
    )
    _validate_feature_contract(base_features, agent_features)

    selection_rows = frame.loc[frame["modeling_year"].between(2020, 2023)]
    eligible_base = _eligible_features(selection_rows, base_features)
    eligible_agent = _eligible_features(selection_rows, agent_features)
    eligible_features = [*eligible_base, *eligible_agent]
    if not eligible_base or not eligible_agent:
        raise ValueError("eligible base or Agent feature group is empty")

    base_stability_rows, fold_definitions = _rolling_feature_stability(
        frame,
        eligible_base,
        random_state=random_state,
        permutation_repeats=permutation_repeats,
        seed_count=stability_seed_count,
        use_time_decay=use_time_decay,
    )
    joint_stability_rows, _ = _rolling_feature_stability(
        frame,
        eligible_features,
        random_state=random_state,
        permutation_repeats=permutation_repeats,
        seed_count=stability_seed_count,
        use_time_decay=use_time_decay,
    )
    base_stability = pd.DataFrame(base_stability_rows)
    base_stability["feature_group"] = "base"
    joint_stability = pd.DataFrame(joint_stability_rows)
    agent_stability = joint_stability.loc[
        joint_stability["feature"].isin(set(eligible_agent))
    ].copy()
    agent_stability["feature_group"] = "agent"
    stability = pd.concat([base_stability, agent_stability], ignore_index=True)
    stability = stability.sort_values(
        [
            "feature_group",
            "positive_permutation_fold_count",
            "permutation_mean",
            "gain_share_mean",
            "feature",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    stability["group_rank"] = stability.groupby("feature_group").cumcount() + 1
    core_stable_mask = (
        stability["positive_permutation_fold_count"] == len(ROLLING_VALIDATION_YEARS)
    ) & (stability["permutation_min"] > 0)
    conditional_stable_mask = (stability["positive_permutation_fold_count"] >= 1) & (
        stability["permutation_mean"] > 0
    )
    stability["stability_tier"] = np.select(
        [core_stable_mask, conditional_stable_mask],
        ["core", "conditional"],
        default="excluded",
    )
    eligible_stable_mask = core_stable_mask | conditional_stable_mask
    ranked_base = stability.loc[
        eligible_stable_mask & (stability["feature_group"] == "base"), "feature"
    ].tolist()
    ranked_agent = stability.loc[
        eligible_stable_mask & (stability["feature_group"] == "agent"), "feature"
    ].tolist()
    if not ranked_base or not ranked_agent:
        raise ValueError("no cross-year stable base or Agent features")
    base_subset = _select_base_feature_subset(
        frame,
        ranked_base,
        base_feature_limit=base_feature_limit,
        random_state=random_state,
        top_k_fraction=top_k_fraction,
        seed_count=ensemble_seed_count,
        use_time_decay=use_time_decay,
    )
    selected_base_only = ranked_base[: int(base_subset["base_feature_count"])]
    subset = _select_feature_subset(
        frame,
        selected_base_only,
        ranked_agent,
        agent_feature_limit=agent_feature_limit,
        random_state=random_state,
        top_k_fraction=top_k_fraction,
        seed_count=ensemble_seed_count,
        use_time_decay=use_time_decay,
    )
    selected_base_count = len(selected_base_only)
    selected_agent_count = int(subset["agent_feature_count"])
    stability["selected"] = (
        (stability["feature_group"] == "base") & (stability["group_rank"] <= selected_base_count)
    ) | (
        (stability["feature_group"] == "agent") & (stability["group_rank"] <= selected_agent_count)
    )
    selected_base = stability.loc[
        stability["selected"] & (stability["feature_group"] == "base"), "feature"
    ].tolist()
    selected_agent = stability.loc[
        stability["selected"] & (stability["feature_group"] == "agent"), "feature"
    ].tolist()
    selected_features = [*selected_base, *selected_agent]
    stability["selected_for_base_model"] = stability["feature"].isin(selected_base_only)

    oof = subset["oof_predictions"]
    fold_model_metrics = subset["fold_metrics"]
    blend_weight = float(subset["lightgbm_weight"])
    blend_search = subset["blend_search"]
    oof_probability = (
        blend_weight * oof["lightgbm_probability"].to_numpy()
        + (1.0 - blend_weight) * oof["logistic_probability"].to_numpy()
    )
    oof_threshold, threshold_policy = _select_business_threshold(
        oof[PRIMARY_TARGET].astype(int).to_numpy(),
        oof_probability,
        min_recall=min_recall,
        max_alert_fraction=max_alert_fraction,
    )
    rolling_metrics = _classification_metrics(
        oof[PRIMARY_TARGET].astype(int).to_numpy(),
        oof_probability,
        threshold=oof_threshold,
        top_k_fraction=top_k_fraction,
    )
    rolling_metrics["accuracy"] = _accuracy_from_metrics(rolling_metrics)

    base_oof = base_subset["oof_predictions"]
    base_fold_metrics = base_subset["fold_metrics"]
    base_blend_weight = float(base_subset["lightgbm_weight"])
    base_blend_search = base_subset["blend_search"]
    base_oof_probability = (
        base_blend_weight * base_oof["lightgbm_probability"].to_numpy()
        + (1.0 - base_blend_weight) * base_oof["logistic_probability"].to_numpy()
    )
    base_oof_threshold, base_threshold_policy = _select_business_threshold(
        base_oof[PRIMARY_TARGET].astype(int).to_numpy(),
        base_oof_probability,
        min_recall=min_recall,
        max_alert_fraction=max_alert_fraction,
    )
    base_rolling_metrics = _classification_metrics(
        base_oof[PRIMARY_TARGET].astype(int).to_numpy(),
        base_oof_probability,
        threshold=base_oof_threshold,
        top_k_fraction=top_k_fraction,
    )
    base_rolling_metrics["accuracy"] = _accuracy_from_metrics(base_rolling_metrics)

    development = frame.loc[frame["modeling_year"].between(2020, 2023)].copy()
    test = frame.loc[frame["modeling_year"] == 2024].copy()
    external = frame.loc[frame["modeling_year"] >= 2025].copy()
    if test.empty:
        raise ValueError("2024 test cohort is empty")
    if set(development[PRIMARY_TARGET].astype(int)) != {0, 1}:
        raise ValueError("development cohort must contain both target classes")

    logistic = _build_pipeline(development, selected_features, random_state)
    development_weight = _time_decay_sample_weight(development) if use_time_decay else None
    logistic.fit(
        development[selected_features],
        development[PRIMARY_TARGET].astype(int),
        **(
            {"classifier__sample_weight": development_weight}
            if development_weight is not None
            else {}
        ),
    )
    best_iterations = [
        int(item["lightgbm_best_iteration"])
        for item in fold_model_metrics
        if item.get("lightgbm_best_iteration")
    ]
    final_iterations = max(20, int(round(float(np.median(best_iterations or [150])))))
    final_seeds = _ensemble_seeds(random_state, ensemble_seed_count)
    lightgbm_models = [
        _fit_lightgbm_fixed(
            development,
            selected_features,
            random_state=seed,
            n_estimators=final_iterations,
            sample_weight=development_weight,
        )
        for seed in final_seeds
    ]
    base_logistic = _build_pipeline(development, selected_base_only, random_state)
    base_logistic.fit(
        development[selected_base_only],
        development[PRIMARY_TARGET].astype(int),
        **(
            {"classifier__sample_weight": development_weight}
            if development_weight is not None
            else {}
        ),
    )
    base_best_iterations = [
        int(item["lightgbm_best_iteration"])
        for item in base_fold_metrics
        if item.get("lightgbm_best_iteration")
    ]
    base_final_iterations = max(20, int(round(float(np.median(base_best_iterations or [150])))))
    base_lightgbm_models = [
        _fit_lightgbm_fixed(
            development,
            selected_base_only,
            random_state=seed,
            n_estimators=base_final_iterations,
            sample_weight=development_weight,
        )
        for seed in final_seeds
    ]

    threshold = oof_threshold
    threshold_policy["deployment_probability_threshold"] = threshold
    threshold_policy["batch_alert_cap"] = max_alert_fraction
    threshold_policy["deployment_rule"] = (
        "先按历史折外阈值判断；若批量预警超过上限，仅保留概率最高的额度内公司。"
    )

    test_logistic = logistic.predict_proba(test[selected_features])[:, 1]
    test_lightgbm = np.mean(
        [model.predict_proba(test[selected_features])[:, 1] for model in lightgbm_models],
        axis=0,
    )
    test_probability = blend_weight * test_lightgbm + (1.0 - blend_weight) * test_logistic
    test_prediction = _apply_batch_alert_cap(
        test_probability,
        threshold=threshold,
        max_alert_fraction=max_alert_fraction,
        target_alert_fraction=float(threshold_policy.get("selected_alert_fraction", 0.0)),
    )
    test_metrics = _classification_metrics_with_predictions(
        test[PRIMARY_TARGET].astype(int).to_numpy(),
        test_probability,
        test_prediction,
        threshold=threshold,
        top_k_fraction=top_k_fraction,
    )
    test_base_logistic = base_logistic.predict_proba(test[selected_base_only])[:, 1]
    test_base_lightgbm = np.mean(
        [model.predict_proba(test[selected_base_only])[:, 1] for model in base_lightgbm_models],
        axis=0,
    )
    test_base_probability = (
        base_blend_weight * test_base_lightgbm + (1.0 - base_blend_weight) * test_base_logistic
    )
    test_base_prediction = _apply_batch_alert_cap(
        test_base_probability,
        threshold=base_oof_threshold,
        max_alert_fraction=max_alert_fraction,
        target_alert_fraction=float(base_threshold_policy.get("selected_alert_fraction", 0.0)),
    )
    test_base_metrics = _classification_metrics_with_predictions(
        test[PRIMARY_TARGET].astype(int).to_numpy(),
        test_base_probability,
        test_base_prediction,
        threshold=base_oof_threshold,
        top_k_fraction=top_k_fraction,
    )
    deployment_recommendation = _deployment_recommendation(
        augmented=test_metrics,
        base=test_base_metrics,
    )

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    feature_stability_path = destination / "feature_stability.csv"
    stability.to_csv(feature_stability_path, index=False, encoding="utf-8-sig")

    identity = [
        column
        for column in ("company_id", "ticker", "company_name", "prospectus_date", "run_id")
        if column in test
    ]
    predictions = test[identity].copy()
    predictions[PRIMARY_TARGET] = test[PRIMARY_TARGET].astype(int).to_numpy()
    predictions["logistic_probability"] = test_logistic
    predictions["lightgbm_probability"] = test_lightgbm
    predictions["ensemble_probability"] = test_probability
    predictions["predicted_significant_drop_5d"] = test_prediction
    predictions["base_probability"] = test_base_probability
    predictions["base_predicted_significant_drop_5d"] = test_base_prediction
    predictions_path = destination / "predictions_2024.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    feature_manifest_path = destination / "feature_manifest.json"
    write_json(
        {
            "schema_version": 1,
            "selection_data_years": [2020, 2021, 2022, 2023],
            "rolling_validation_years": list(ROLLING_VALIDATION_YEARS),
            "declared_base_candidate_count": len(raw_base_features),
            "declared_agent_candidate_count": len(declared_agent_features),
            "base_candidate_count": len(base_features),
            "agent_candidate_count": len(agent_features),
            "business_base_whitelist": base_features,
            "agent_semantic_whitelist": agent_features,
            "base_feature_policy": (
                "仅使用发行、财务、上市前市场和同期IPO情绪；排除文件大小、页数、"
                "年份和同一概念的上下限重复字段。"
            ),
            "agent_feature_policy": (
                "仅使用风险类型、严重程度、角色分布和不确定性等语义特征；"
                "并纳入经公式、报告期和有效状态校验的现金续航、赎回严苛度、"
                "核心管线阶段及关联销售收入占比；排除证据数量、置信度和调用状态等过程统计。"
            ),
            "eligible_base_count": len(eligible_base),
            "eligible_agent_count": len(eligible_agent),
            "selected_base_count": len(selected_base),
            "selected_agent_count": len(selected_agent),
            "selected_feature_count": len(selected_features),
            "selected_base_features": selected_base,
            "selected_agent_features": selected_agent,
            "selected_features": selected_features,
            "selected_base_only_count": len(selected_base_only),
            "selected_base_only_features": selected_base_only,
            "stability_policy": {
                "core": "两个滚动验证年份的Permutation重要性均严格大于0",
                "conditional": "至少一个年份为正且两年Permutation重要性均值大于0",
                "excluded": "不进入候选子集",
            },
            "feature_size_search": subset["feature_size_search"],
            "base_feature_size_search": base_subset["feature_size_search"],
            "excluded_low_information_agent_features": sorted(
                set(declared_agent_features).intersection(EXCLUDED_LOW_INFORMATION_AGENT_FEATURES)
            ),
            "excluded_non_business_base_features": sorted(
                set(raw_base_features).difference(base_features)
            ),
            "excluded_non_semantic_agent_features": sorted(
                set(declared_agent_features).difference(agent_features)
            ),
            "blocked_feature_prefixes": ["target_", "label_"],
        },
        feature_manifest_path,
    )

    metrics_path = destination / "metrics.json"
    write_json(
        {
            "schema_version": 1,
            "target": PRIMARY_TARGET,
            "strategy": {
                "feature_selection": "rolling validation stability on 2022 and 2023",
                "final_training": "2020-2023",
                "independent_test": 2024,
                "external_only": "2025+",
                "minimum_recall_requested": min_recall,
                "maximum_alert_fraction": max_alert_fraction,
                "time_decay_weights": TIME_DECAY_WEIGHTS if use_time_decay else None,
                "lightgbm_ensemble_seeds": final_seeds,
            },
            "cohort_counts": {
                "development_2020_2023": int(len(development)),
                "test_2024": int(len(test)),
                "external_2025_plus": int(len(external)),
            },
            "rolling_folds": fold_definitions,
            "rolling_model_metrics": fold_model_metrics,
            "base_rolling_model_metrics": base_fold_metrics,
            "blend_weight_lightgbm": blend_weight,
            "blend_weight_logistic": 1.0 - blend_weight,
            "blend_search": blend_search,
            "base_blend_weight_lightgbm": base_blend_weight,
            "base_blend_weight_logistic": 1.0 - base_blend_weight,
            "base_blend_search": base_blend_search,
            "selected_threshold": threshold,
            "threshold_policy": threshold_policy,
            "base_selected_threshold": base_oof_threshold,
            "base_threshold_policy": base_threshold_policy,
            "historical_tuning_performance": rolling_metrics,
            "base_historical_tuning_performance": base_rolling_metrics,
            "historical_tuning_warning": (
                "2022/2023同时参与早停、特征和阈值选择，该指标仅描述调参表现，"
                "不是无偏泛化估计；2024才是独立测试。"
            ),
            "test_2024": test_metrics,
            "test_2024_base": test_base_metrics,
            "agent_incremental_value_2024": _metric_deltas(test_metrics, test_base_metrics),
            "deployment_recommendation": deployment_recommendation,
            "final_lightgbm_iterations": final_iterations,
            "base_final_lightgbm_iterations": base_final_iterations,
            "final_lightgbm_model_count": len(lightgbm_models),
            "leakage_controls": {
                "feature_selection_excludes_2024": True,
                "blend_selection_excludes_2024": True,
                "threshold_selection_excludes_2024": True,
                "all_target_and_label_features_blocked": True,
                "2025_used_for_training": False,
            },
        },
        metrics_path,
    )

    model_path = destination / "stable_agent_ensemble.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": "day5_stable_agent_ensemble_v2",
            "target": PRIMARY_TARGET,
            "feature_columns": selected_features,
            "logistic_pipeline": logistic,
            "lightgbm_pipelines": lightgbm_models,
            "lightgbm_weight": blend_weight,
            "logistic_weight": 1.0 - blend_weight,
            "threshold": threshold,
            "max_alert_fraction": max_alert_fraction,
            "target_alert_fraction": float(threshold_policy.get("selected_alert_fraction", 0.0)),
            "training_years": [2020, 2021, 2022, 2023],
            "test_year": 2024,
            "random_state": random_state,
            "ensemble_seeds": final_seeds,
            "time_decay_weights": TIME_DECAY_WEIGHTS if use_time_decay else None,
        },
        model_path,
    )
    base_model_path = destination / "stable_business_base_ensemble.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": "day5_stable_business_base_ensemble_v2",
            "target": PRIMARY_TARGET,
            "feature_columns": selected_base_only,
            "logistic_pipeline": base_logistic,
            "lightgbm_pipelines": base_lightgbm_models,
            "lightgbm_weight": base_blend_weight,
            "logistic_weight": 1.0 - base_blend_weight,
            "threshold": base_oof_threshold,
            "max_alert_fraction": max_alert_fraction,
            "target_alert_fraction": float(
                base_threshold_policy.get("selected_alert_fraction", 0.0)
            ),
            "training_years": [2020, 2021, 2022, 2023],
            "test_year": 2024,
            "random_state": random_state,
            "ensemble_seeds": final_seeds,
            "time_decay_weights": TIME_DECAY_WEIGHTS if use_time_decay else None,
        },
        base_model_path,
    )
    return StableAgentModelResult(
        output_dir=destination,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        feature_stability_path=feature_stability_path,
        feature_manifest_path=feature_manifest_path,
        model_path=model_path,
        base_model_path=base_model_path,
    )


def _apply_feature_whitelist(
    declared: list[str],
    whitelist: tuple[str, ...],
    *,
    group: str,
) -> list[str]:
    """按固定业务白名单取特征，并尽早暴露字段契约漂移。"""
    missing = [feature for feature in whitelist if feature not in declared]
    if missing:
        raise ValueError(f"{group} whitelist references undeclared features: {missing}")
    if len(whitelist) != len(set(whitelist)):
        raise ValueError(f"{group} whitelist contains duplicate features")
    return list(whitelist)


def add_derived_agent_semantic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """把规模相关计数转换为可跨公司的风险结构比例。"""
    required = {
        "agent_finding_count",
        "agent_unique_risk_type_count",
        "agent_severity_low_count",
        "agent_severity_medium_count",
        "agent_severity_high_count",
        "agent_severity_critical_count",
        "agent_role_finance_finding_count",
        "agent_role_legal_finding_count",
        "agent_role_business_finding_count",
        "agent_role_market_finding_count",
        "agent_uncertainty_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cannot derive Agent semantic features; missing: {missing}")
    result = frame.copy()
    finding_count = pd.to_numeric(result["agent_finding_count"], errors="coerce").fillna(0)
    safe_count = finding_count.replace(0, np.nan)
    high = pd.to_numeric(result["agent_severity_high_count"], errors="coerce").fillna(0)
    critical = pd.to_numeric(result["agent_severity_critical_count"], errors="coerce").fillna(0)
    medium = pd.to_numeric(result["agent_severity_medium_count"], errors="coerce").fillna(0)
    low = pd.to_numeric(result["agent_severity_low_count"], errors="coerce").fillna(0)
    result["agent_severity_weighted_mean"] = (
        (low + 2 * medium + 3 * high + 4 * critical).div(safe_count).fillna(0)
    )
    result["agent_high_or_critical_share"] = (high + critical).div(safe_count).fillna(0)
    result["agent_unique_risk_type_share"] = (
        pd.to_numeric(result["agent_unique_risk_type_count"], errors="coerce")
        .fillna(0)
        .div(safe_count)
        .fillna(0)
    )
    for role in ("finance", "legal", "business", "market"):
        result[f"agent_role_{role}_share"] = (
            pd.to_numeric(result[f"agent_role_{role}_finding_count"], errors="coerce")
            .fillna(0)
            .div(safe_count)
            .fillna(0)
        )
    result["agent_uncertainty_share"] = (
        pd.to_numeric(result["agent_uncertainty_count"], errors="coerce")
        .fillna(0)
        .div(safe_count)
        .fillna(0)
    )
    return result


def add_derived_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """基于现有上市前字段构造可解释比率、缺失标志及跨域交互。"""
    agent_ratio_sources = {
        "agent_finding_count",
        "agent_unique_risk_type_count",
        "agent_severity_low_count",
        "agent_severity_medium_count",
        "agent_severity_high_count",
        "agent_severity_critical_count",
        "agent_role_finance_finding_count",
        "agent_role_legal_finding_count",
        "agent_role_business_finding_count",
        "agent_role_market_finding_count",
        "agent_uncertainty_count",
    }
    result = (
        add_derived_agent_semantic_features(frame)
        if agent_ratio_sources.issubset(frame.columns)
        else frame.copy()
    )

    def numeric(name: str) -> pd.Series:
        if name not in result:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[name], errors="coerce")

    def safe_positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        valid = denominator.gt(0) & numerator.notna()
        values = pd.Series(np.nan, index=result.index, dtype=float)
        values.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
        return values.replace([np.inf, -np.inf], np.nan)

    revenue = numeric("fundamental_revenue_latest_hkd_mn")
    market_cap = numeric("offering_market_cap_mid_hkd")
    net_proceeds = numeric("offering_net_proceeds_hkd")
    operating_cash_flow = numeric("fundamental_operating_cash_flow_latest_hkd_mn")
    ending_cash_native_mn = numeric("fundamental_ending_cash_latest_native")
    fx_to_hkd = numeric("fundamental_fx_to_hkd")
    revenue_cagr = numeric("fundamental_revenue_cagr_3y_pct")
    share_ratio = numeric("offering_share_ratio_pct")

    for name, values in {
        "revenue_hkd_mn": revenue,
        "market_cap_hkd_mn": market_cap,
        "net_proceeds_hkd_mn": net_proceeds,
    }.items():
        result[f"engineered_log_{name}"] = np.log1p(values.clip(lower=0))
    result["engineered_price_to_sales"] = safe_positive_ratio(market_cap, revenue)
    result["engineered_fundraising_to_revenue"] = safe_positive_ratio(net_proceeds, revenue)
    result["engineered_net_proceeds_to_market_cap"] = safe_positive_ratio(net_proceeds, market_cap)
    result["engineered_operating_cash_flow_margin"] = (
        operating_cash_flow.div(revenue.where(revenue.abs().gt(1e-9)))
        .replace([np.inf, -np.inf], np.nan)
        .clip(-10, 10)
    )
    cash_outflow = (-operating_cash_flow).where(operating_cash_flow.lt(0))
    result["engineered_cash_replenishment_ratio"] = safe_positive_ratio(
        net_proceeds, cash_outflow
    ).clip(upper=100)
    ending_cash_hkd_mn = ending_cash_native_mn.mul(fx_to_hkd)
    runway_valid = ending_cash_hkd_mn.ge(0) & operating_cash_flow.lt(0)
    result["engineered_cash_runway_months"] = np.nan
    result.loc[runway_valid, "engineered_cash_runway_months"] = (
        ending_cash_hkd_mn.loc[runway_valid]
        .div(operating_cash_flow.loc[runway_valid].abs())
        .mul(12.0)
        .clip(upper=1_200)
    )
    positive_growth = revenue_cagr.where(revenue_cagr.gt(0))
    result["engineered_growth_adjusted_price_to_sales"] = (
        result["engineered_price_to_sales"]
        .div(positive_growth)
        .replace([np.inf, -np.inf], np.nan)
        .clip(upper=100)
    )
    result["engineered_zero_or_negative_revenue"] = revenue.le(0).fillna(False).astype(int)
    result["engineered_negative_operating_cash_flow"] = (
        operating_cash_flow.lt(0).fillna(False).astype(int)
    )
    for source, target in (
        (revenue_cagr, "missing_revenue_cagr_3y"),
        (share_ratio, "missing_offering_share_ratio"),
        (revenue, "missing_revenue_latest"),
        (operating_cash_flow, "missing_operating_cash_flow_latest"),
        (net_proceeds, "missing_net_proceeds"),
    ):
        result[target] = source.isna().astype(int)

    financial_share = numeric("agent_semantic_family_financial_share").fillna(0)
    market_share = numeric("agent_semantic_family_market_liquidity_share").fillna(0)
    governance_share = numeric("agent_semantic_family_governance_share").fillna(0)
    related_party = numeric("agent_semantic_has_related_party_transaction").fillna(0)
    free_float = numeric("agent_semantic_has_free_float_liquidity").fillna(0)
    cash_burn = numeric("agent_semantic_has_cash_burn_liquidity").fillna(0)
    supported = numeric("agent_semantic_supported_finding_count").fillna(0)
    finding_count = numeric("agent_finding_count").fillna(0)
    high_share = numeric("agent_high_or_critical_share").fillna(0)
    distinct = numeric("agent_semantic_distinct_subtype_count").fillna(0)
    volatility_20d = numeric("market_volatility_20d_pct").fillna(0)
    result["agent_interaction_financial_risk_negative_cash_flow"] = (
        financial_share * result["engineered_negative_operating_cash_flow"]
    )
    result["agent_interaction_market_liquidity_volatility_20d"] = market_share * volatility_20d
    result["agent_interaction_related_party_governance"] = related_party * governance_share
    result["agent_interaction_free_float_low_issue_ratio"] = free_float * (
        1.0 - share_ratio.fillna(0).clip(0, 100).div(100)
    )
    result["agent_interaction_cash_burn_fundraising_intensity"] = cash_burn * result[
        "engineered_fundraising_to_revenue"
    ].fillna(0).clip(0, 100)
    result["agent_interaction_supported_high_severity_share"] = (
        supported.div(finding_count.replace(0, np.nan)).fillna(0) * high_share
    )
    result["agent_interaction_semantic_risk_density"] = distinct.div(
        finding_count.replace(0, np.nan)
    ).fillna(0)
    return result


def _time_decay_sample_weight(rows: pd.DataFrame) -> np.ndarray:
    """近期年份权重更高；未知年份默认不低于最近训练年份。"""
    years = pd.to_numeric(rows["modeling_year"], errors="coerce")
    return years.map(TIME_DECAY_WEIGHTS).fillna(1.0).to_numpy(dtype=float)


def _metric_deltas(augmented: dict[str, Any], base: dict[str, Any]) -> dict[str, float | None]:
    """正数表示加入 Agent 语义风险后优于纯基础模型。"""
    deltas: dict[str, float | None] = {}
    for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "accuracy"):
        left = augmented.get(metric)
        right = base.get(metric)
        deltas[f"{metric}_delta"] = (
            float(left) - float(right) if left is not None and right is not None else None
        )
    left_brier = augmented.get("brier_score")
    right_brier = base.get("brier_score")
    deltas["brier_score_reduction"] = (
        float(right_brier) - float(left_brier)
        if left_brier is not None and right_brier is not None
        else None
    )
    return deltas


def _deployment_recommendation(
    *,
    augmented: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    """在独立测试上只给研究建议，不反向参与训练或阈值选择。"""
    augmented_recall = float(augmented["recall"])
    base_recall = float(base["recall"])
    augmented_pr_auc = float(augmented["pr_auc"])
    base_pr_auc = float(base["pr_auc"])
    agent_improves_both = augmented_recall >= base_recall and augmented_pr_auc > base_pr_auc
    return {
        "recommended_variant": ("agent_augmented" if agent_improves_both else "business_base"),
        "agent_augmented_ready_to_replace_base": agent_improves_both,
        "rule": "Agent增强必须不降低召回率且提高PR-AUC，才可替代基础模型。",
        "note": "2024仅用于独立评测和版本选择说明，不参与本次模型拟合或阈值选择。",
    }


def _eligible_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    result: list[str] = []
    for feature in features:
        series = frame[feature]
        if float(series.isna().mean()) > 0.95:
            continue
        if series.dropna().nunique() <= 1:
            continue
        result.append(feature)
    return result


def _rolling_feature_stability(
    frame: pd.DataFrame,
    features: list[str],
    *,
    random_state: int,
    permutation_repeats: int,
    seed_count: int,
    use_time_decay: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_feature: dict[str, list[dict[str, float]]] = {feature: [] for feature in features}
    folds: list[dict[str, Any]] = []
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(2020, validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        if train.empty or validation.empty:
            raise ValueError(f"rolling fold is empty for validation year {validation_year}")
        fold_permutations = {feature: [] for feature in features}
        fold_gains = {feature: [] for feature in features}
        best_iterations: list[int] = []
        for seed in _ensemble_seeds(random_state, seed_count):
            pipeline, best_iteration = _fit_lightgbm_pipeline(
                train,
                validation,
                features,
                random_state=seed,
                sample_weight=(_time_decay_sample_weight(train) if use_time_decay else None),
            )
            if best_iteration:
                best_iterations.append(int(best_iteration))
            permutation = permutation_importance(
                pipeline,
                validation[features],
                validation[PRIMARY_TARGET].astype(int),
                scoring="average_precision",
                n_repeats=permutation_repeats,
                random_state=seed,
                n_jobs=1,
            )
            gains, _ = _aggregate_native_importance(pipeline, features)
            total_gain = max(sum(gains.values()), 1e-12)
            for index, feature in enumerate(features):
                fold_permutations[feature].append(float(permutation.importances_mean[index]))
                fold_gains[feature].append(float(gains.get(feature, 0.0) / total_gain))
        for feature in features:
            per_feature[feature].append(
                {
                    "permutation": float(np.mean(fold_permutations[feature])),
                    "gain_share": float(np.mean(fold_gains[feature])),
                }
            )
        folds.append(
            {
                "train_years": list(range(2020, validation_year)),
                "validation_year": validation_year,
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "lightgbm_best_iteration": int(round(float(np.median(best_iterations or [0])))),
                "stability_seed_count": seed_count,
            }
        )
    rows: list[dict[str, Any]] = []
    for feature, values in per_feature.items():
        permutations = [item["permutation"] for item in values]
        gain_shares = [item["gain_share"] for item in values]
        rows.append(
            {
                "feature": feature,
                "positive_permutation_fold_count": sum(value > 0 for value in permutations),
                "permutation_mean": float(np.mean(permutations)),
                "permutation_min": float(np.min(permutations)),
                "gain_share_mean": float(np.mean(gain_shares)),
                **{
                    f"permutation_{year}": permutations[index]
                    for index, year in enumerate(ROLLING_VALIDATION_YEARS)
                },
            }
        )
    return rows, folds


def _rolling_model_predictions(
    frame: pd.DataFrame,
    features: list[str],
    *,
    random_state: int,
    top_k_fraction: float,
    seed_count: int,
    use_time_decay: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for validation_year in ROLLING_VALIDATION_YEARS:
        train = frame.loc[frame["modeling_year"].between(2020, validation_year - 1)]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        logistic = _build_pipeline(train, features, random_state)
        logistic_weight = _time_decay_sample_weight(train) if use_time_decay else None
        logistic.fit(
            train[features],
            train[PRIMARY_TARGET].astype(int),
            **(
                {"classifier__sample_weight": logistic_weight}
                if logistic_weight is not None
                else {}
            ),
        )
        logistic_probability = logistic.predict_proba(validation[features])[:, 1]
        lightgbm_probabilities: list[np.ndarray] = []
        best_iterations: list[int] = []
        for seed in _ensemble_seeds(random_state, seed_count):
            lightgbm, best_iteration = _fit_lightgbm_pipeline(
                train,
                validation,
                features,
                random_state=seed,
                sample_weight=logistic_weight,
            )
            lightgbm_probabilities.append(lightgbm.predict_proba(validation[features])[:, 1])
            if best_iteration:
                best_iterations.append(int(best_iteration))
        lightgbm_probability = np.mean(lightgbm_probabilities, axis=0)
        fold = validation[["company_id", PRIMARY_TARGET]].copy()
        fold["validation_year"] = validation_year
        fold["logistic_probability"] = logistic_probability
        fold["lightgbm_probability"] = lightgbm_probability
        predictions.append(fold)
        fold_metrics.append(
            {
                "validation_year": validation_year,
                "sample_count": int(len(validation)),
                "positive_count": int(validation[PRIMARY_TARGET].astype(int).sum()),
                "logistic_pr_auc": float(
                    average_precision_score(
                        validation[PRIMARY_TARGET].astype(int), logistic_probability
                    )
                ),
                "lightgbm_pr_auc": float(
                    average_precision_score(
                        validation[PRIMARY_TARGET].astype(int), lightgbm_probability
                    )
                ),
                "lightgbm_best_iteration": int(round(float(np.median(best_iterations or [0])))),
                "lightgbm_seed_count": seed_count,
                "top_k_fraction": top_k_fraction,
            }
        )
    return pd.concat(predictions, ignore_index=True), fold_metrics


def _select_feature_subset(
    frame: pd.DataFrame,
    fixed_base_features: list[str],
    ranked_agent: list[str],
    *,
    agent_feature_limit: int,
    random_state: int,
    top_k_fraction: float,
    seed_count: int,
    use_time_decay: bool,
) -> dict[str, Any]:
    """冻结基础模型特征，只用历史滚动折外结果选择 Agent 特征数量。"""
    agent_max = min(agent_feature_limit, len(ranked_agent))
    agent_sizes = sorted({min(agent_max, value) for value in (1, 2, 3, 5, agent_max)})
    agent_sizes = [value for value in agent_sizes if value > 0]
    candidates: list[dict[str, Any]] = []
    for agent_count in agent_sizes:
        features = [*fixed_base_features, *ranked_agent[:agent_count]]
        oof, fold_metrics = _rolling_model_predictions(
            frame,
            features,
            random_state=random_state,
            top_k_fraction=top_k_fraction,
            seed_count=seed_count,
            use_time_decay=use_time_decay,
        )
        lightgbm_weight, blend_search = _select_blend_weight(oof)
        probability = (
            lightgbm_weight * oof["lightgbm_probability"].to_numpy()
            + (1.0 - lightgbm_weight) * oof["logistic_probability"].to_numpy()
        )
        pr_auc = float(average_precision_score(oof[PRIMARY_TARGET].astype(int), probability))
        fold_pr_auc = [float(item["lightgbm_pr_auc"]) for item in fold_metrics]
        candidates.append(
            {
                "base_feature_count": len(fixed_base_features),
                "agent_feature_count": agent_count,
                "feature_count": len(features),
                "rolling_pr_auc": pr_auc,
                "minimum_fold_pr_auc": min(fold_pr_auc),
                "lightgbm_weight": lightgbm_weight,
                "oof_predictions": oof,
                "fold_metrics": fold_metrics,
                "blend_search": blend_search,
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            float(item["rolling_pr_auc"]),
            float(item["minimum_fold_pr_auc"]),
            -int(item["feature_count"]),
        ),
    )
    best = dict(best)
    best["feature_size_search"] = [
        {
            key: value
            for key, value in candidate.items()
            if key
            not in {
                "oof_predictions",
                "fold_metrics",
                "blend_search",
            }
        }
        for candidate in candidates
    ]
    return best


def _select_base_feature_subset(
    frame: pd.DataFrame,
    ranked_base: list[str],
    *,
    base_feature_limit: int,
    random_state: int,
    top_k_fraction: float,
    seed_count: int,
    use_time_decay: bool,
) -> dict[str, Any]:
    """使用相同滚动折和目标函数独立选择纯基础模型的特征数量。"""
    base_max = min(base_feature_limit, len(ranked_base))
    base_sizes = sorted({min(base_max, value) for value in (3, 5, 8, 15, base_max)})
    candidates: list[dict[str, Any]] = []
    for base_count in (value for value in base_sizes if value > 0):
        features = ranked_base[:base_count]
        oof, fold_metrics = _rolling_model_predictions(
            frame,
            features,
            random_state=random_state,
            top_k_fraction=top_k_fraction,
            seed_count=seed_count,
            use_time_decay=use_time_decay,
        )
        lightgbm_weight, blend_search = _select_blend_weight(oof)
        probability = (
            lightgbm_weight * oof["lightgbm_probability"].to_numpy()
            + (1.0 - lightgbm_weight) * oof["logistic_probability"].to_numpy()
        )
        candidates.append(
            {
                "base_feature_count": base_count,
                "feature_count": base_count,
                "rolling_pr_auc": float(
                    average_precision_score(oof[PRIMARY_TARGET].astype(int), probability)
                ),
                "minimum_fold_pr_auc": min(float(item["lightgbm_pr_auc"]) for item in fold_metrics),
                "lightgbm_weight": lightgbm_weight,
                "oof_predictions": oof,
                "fold_metrics": fold_metrics,
                "blend_search": blend_search,
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            float(item["rolling_pr_auc"]),
            float(item["minimum_fold_pr_auc"]),
            -int(item["feature_count"]),
        ),
    )
    result = dict(best)
    result["feature_size_search"] = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"oof_predictions", "fold_metrics", "blend_search"}
        }
        for candidate in candidates
    ]
    return result


def _select_blend_weight(oof: pd.DataFrame) -> tuple[float, list[dict[str, float]]]:
    y_true = oof[PRIMARY_TARGET].astype(int).to_numpy()
    logistic = oof["logistic_probability"].to_numpy()
    lightgbm = oof["lightgbm_probability"].to_numpy()
    rows: list[dict[str, float]] = []
    for weight in np.linspace(0.0, 1.0, 11):
        probability = weight * lightgbm + (1.0 - weight) * logistic
        rows.append(
            {
                "lightgbm_weight": float(weight),
                "pr_auc": float(average_precision_score(y_true, probability)),
                "brier_score": float(brier_score_loss(y_true, probability)),
            }
        )
    best = max(rows, key=lambda row: (row["pr_auc"], -row["brier_score"]))
    return float(best["lightgbm_weight"]), rows


def _select_business_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    min_recall: float,
    max_alert_fraction: float,
) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), np.asarray(probability)]))
    evaluated: list[dict[str, float]] = []
    for threshold in candidates:
        predicted = probability >= threshold
        alert_fraction = float(predicted.mean())
        if not predicted.any() or alert_fraction > max_alert_fraction + 1e-12:
            continue
        true_positive = int((predicted & (y_true == 1)).sum())
        recall = true_positive / int((y_true == 1).sum())
        precision = true_positive / int(predicted.sum())
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        evaluated.append(
            {
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": float(precision),
                "f2": float(f2),
                "alert_fraction": alert_fraction,
            }
        )
    if not evaluated:
        threshold = float(np.nextafter(np.max(probability), np.inf))
        return threshold, {
            "policy": "no_alert_fallback",
            "minimum_recall_requested": min_recall,
            "maximum_alert_fraction": max_alert_fraction,
        }
    feasible = [row for row in evaluated if row["recall"] >= min_recall]
    if feasible:
        best = max(feasible, key=lambda row: (row["precision"], row["f2"], -row["alert_fraction"]))
        policy = "minimum_recall_met_then_maximize_precision"
    else:
        best = max(evaluated, key=lambda row: (row["recall"], row["precision"], row["f2"]))
        policy = "recall_target_infeasible_under_alert_cap"
    return float(best["threshold"]), {
        "policy": policy,
        "minimum_recall_requested": min_recall,
        "maximum_alert_fraction": max_alert_fraction,
        "selected_recall": best["recall"],
        "selected_precision": best["precision"],
        "selected_f2": best["f2"],
        "selected_alert_fraction": best["alert_fraction"],
    }


def _fit_lightgbm_fixed(
    rows: pd.DataFrame,
    features: list[str],
    *,
    random_state: int,
    n_estimators: int,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
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
    transformed = preprocessor.fit_transform(rows[features])
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
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
        transformed,
        rows[PRIMARY_TARGET].astype(int),
        sample_weight=sample_weight,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _ensemble_seeds(random_state: int, count: int) -> list[int]:
    seeds = [int(random_state)]
    seeds.extend(seed for seed in DEFAULT_ENSEMBLE_SEEDS if seed != random_state)
    while len(seeds) < count:
        seeds.append(int(random_state + 104729 * len(seeds)))
    return seeds[:count]


def _accuracy_from_metrics(metrics: dict[str, Any]) -> float:
    confusion = metrics["confusion_matrix"]
    correct = int(confusion["true_positive"]) + int(confusion["true_negative"])
    total = sum(int(value) for value in confusion.values())
    return correct / total if total else math.nan


def _apply_batch_alert_cap(
    probability: np.ndarray,
    *,
    threshold: float,
    max_alert_fraction: float,
    target_alert_fraction: float | None = None,
) -> np.ndarray:
    """先执行历史阈值，再对批量任务施加严格预警上限。"""
    probability = np.asarray(probability, dtype=float)
    eligible = np.flatnonzero(probability >= threshold)
    if len(probability) < 10:
        predicted = np.zeros(len(probability), dtype=int)
        predicted[eligible] = 1
        return predicted
    cap = int(math.floor(len(probability) * max_alert_fraction))
    predicted = np.zeros(len(probability), dtype=int)
    if cap <= 0:
        return predicted
    if target_alert_fraction is not None and len(probability) >= 10:
        target = min(
            cap,
            int(math.floor(len(probability) * max(0.0, target_alert_fraction))),
        )
        if target <= 0:
            return predicted
        if not len(eligible):
            return predicted
        ranked = eligible[np.argsort(-probability[eligible], kind="stable")]
        predicted[ranked[:target]] = 1
        return predicted
    if not len(eligible):
        return predicted
    ranked = eligible[np.argsort(-probability[eligible], kind="stable")]
    predicted[ranked[:cap]] = 1
    return predicted


def _classification_metrics_with_predictions(
    y_true: np.ndarray,
    probability: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold: float,
    top_k_fraction: float,
) -> dict[str, Any]:
    """沿用公共概率指标，并用受控批量预警覆盖阈值分类指标。"""
    metrics = _classification_metrics(
        y_true,
        probability,
        threshold=threshold,
        top_k_fraction=top_k_fraction,
    )
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    true_positive = int(((predicted == 1) & (y_true == 1)).sum())
    true_negative = int(((predicted == 0) & (y_true == 0)).sum())
    false_positive = int(((predicted == 1) & (y_true == 0)).sum())
    false_negative = int(((predicted == 0) & (y_true == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if predicted.sum() else 0.0
    recall = true_positive / int(y_true.sum()) if y_true.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    metrics.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "accuracy": (true_positive + true_negative) / len(y_true),
            "alert_count": int(predicted.sum()),
            "alert_fraction": float(predicted.mean()),
            "confusion_matrix": {
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            },
        }
    )
    return metrics


__all__ = [
    "ROLLING_VALIDATION_YEARS",
    "StableAgentModelResult",
    "train_stable_agent_model",
]
