"""决策 Agent 完成后的五日显著下跌运行时预测。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ipo_risk.modeling.high_risk_guard import (
    evaluate_high_risk_guard,
    load_high_risk_guard_policy,
)
from ipo_risk.modeling.market_regime import (
    MarketRegimeResult,
    adjust_probability,
    detect_market_regime,
)
from ipo_risk.modeling.dataset import (
    DEFAULT_MARKET_WINDOWS,
    build_listing_pre_market_feature_snapshot,
    build_recent_ipo_feature_snapshot,
)
from ipo_risk.modeling.prelisting import (
    load_model_feature_contract,
    project_model_inputs,
)
from ipo_risk.modeling.predict import load_model_bundle, predict_feature_rows
from ipo_risk.storage.files import read_json, write_json


DEFAULT_MODEL_PATH = Path(
    "artifacts/models/unified_h5/model.joblib"
)
DEFAULT_RECALL_GUARD_POLICY_PATH = Path("configs/high_risk_recall_guard.json")
_FEATURE_LABELS = {
    "fundamental_revenue_latest_hkd_mn": "最近一期收入",
    "engineered_operating_cash_flow_margin": "经营现金流率",
    "agent_semantic_family_financial_supported_count": "有证据支持的财务风险",
    "agent_role_finance_share": "财务风险占比",
    "agent_role_business_share": "业务风险占比",
    "engineered_price_to_sales": "发行市销率",
    "offering_share_ratio_pct": "发行比例",
    "engineered_log_market_cap_hkd_mn": "发行市值规模",
    "agent_semantic_risk_labor_compliance_supported_count": "有证据支持的用工合规风险",
    "engineered_log_net_proceeds_hkd_mn": "净募资规模",
    "agent_semantic_risk_use_of_proceeds_count": "募集资金用途风险",
    "agent_severity_weighted_mean": "风险严重度均值",
    "agent_semantic_risk_margin_cost_pressure_count": "毛利与成本压力风险",
    "agent_semantic_has_inventory_obsolescence": "存货减值风险",
    "agent_semantic_family_legal_compliance_max_severity": "法律合规最高严重度",
    "agent_interaction_supported_high_severity_share": "有证据支持的高严重风险占比",
    "market_advance_ratio_20d": "上市前市场上涨家数占比",
    "market_volatility_20d_pct": "上市前市场波动率",
}


class RuntimePredictionUnavailable(RuntimeError):
    """运行时模型或特征资源不可用。"""


def _resolve_resource(root: Path, environment_name: str, default: Path) -> Path:
    configured = os.getenv(environment_name, "").strip()
    path = Path(configured) if configured else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@lru_cache(maxsize=4)
def _load_bundle(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return load_model_bundle(Path(path_text))


def enrich_horizon_prediction_status(
    *,
    project_root: Path,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """为新旧预测记录补齐四个时间窗口的阈值和预警结论。"""

    root = project_root.resolve()
    payload = dict(prediction)
    raw_values = payload.get("horizon_predictions")
    values = dict(raw_values) if isinstance(raw_values, Mapping) else {}
    if values.get("h5") is None and payload.get("probability") is not None:
        values["h5"] = payload.get("probability")

    raw_thresholds = payload.get("horizon_thresholds")
    expose_thresholds = isinstance(raw_thresholds, Mapping) or payload.get("threshold") is not None
    thresholds = dict(raw_thresholds) if isinstance(raw_thresholds, Mapping) else {}
    if thresholds.get("h5") is None and payload.get("threshold") is not None:
        thresholds["h5"] = payload.get("threshold")

    model_paths = {
        "h1": root / "artifacts" / "models" / "unified_h1" / "model.joblib",
        "h5": _resolve_resource(root, "IPO_RISK_PREDICTION_MODEL", DEFAULT_MODEL_PATH),
        "h20": root / "artifacts" / "models" / "unified_h20" / "model.joblib",
        "h60": root / "artifacts" / "models" / "unified_h60" / "model.joblib",
    }
    for key, model_path in model_paths.items():
        if thresholds.get(key) is not None or not model_path.is_file():
            continue
        try:
            bundle = _load_bundle(str(model_path), model_path.stat().st_mtime_ns)
            thresholds[key] = float(bundle["threshold"])
        except Exception:  # 单个模型包不可读时，不阻断历史记录页面。
            thresholds[key] = None

    raw_alerts = payload.get("horizon_alerts")
    alerts = dict(raw_alerts) if isinstance(raw_alerts, Mapping) else {}
    for key in ("h1", "h5", "h20", "h60"):
        value = values.get(key)
        threshold = thresholds.get(key)
        try:
            alerts[key] = (
                float(value) >= float(threshold)
                if value is not None and threshold is not None
                else None
            )
        except (TypeError, ValueError):
            alerts[key] = None
    if alerts.get("h5") is None and isinstance(
        payload.get("predicted_significant_drop_5d"), bool
    ):
        alerts["h5"] = payload["predicted_significant_drop_5d"]

    if any(isinstance(alerts.get(key), bool) for key in ("h1", "h5", "h20", "h60")):
        payload["horizon_predictions"] = {
            key: values.get(key) for key in ("h1", "h5", "h20", "h60")
        }
        if expose_thresholds:
            payload["horizon_thresholds"] = {
                key: thresholds.get(key) for key in ("h1", "h5", "h20", "h60")
            }
        payload["horizon_alerts"] = {
            key: alerts.get(key) for key in ("h1", "h5", "h20", "h60")
        }
    return payload


@lru_cache(maxsize=4)
def _load_recall_guard_policy(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return load_high_risk_guard_policy(Path(path_text))


def _feature_values(features: Mapping[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, raw in (features or {}).items():
        if name.startswith("target_"):
            continue
        if isinstance(raw, Mapping) and "value" in raw:
            values[name] = raw.get("value")
        else:
            values[name] = raw
    return values


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if np.isscalar(value) and pd.isna(value):
        return None
    return value


def _display_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _risk_level(decision: str, probability: float) -> str:
    if decision == "high_risk":
        return "high"
    if decision == "low_risk":
        return "low"
    return "high" if probability >= 0.65 else "medium"

def _feature_scalar(raw: Any) -> int | float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw
    if isinstance(raw, dict):
        v = raw.get("value")
        return v if isinstance(v, (int, float)) else None
    return None


def _convert_native_to_hkd_mn(features: dict[str, Any]) -> None:
    """将 RAG/结构化产出的 _native 财务数值按 fx_to_hkd 换算为 _hkd_mn 契约名。

    不覆盖已有的 _hkd_mn 值。
    """
    native_fields = (
        "fundamental_revenue_latest_native",
        "fundamental_operating_cash_flow_latest_native",
        "fundamental_ending_cash_latest_native",
        "fundamental_net_profit_latest_native",
        "fundamental_research_development_latest_native",
    )
    has_native_value = any(_feature_scalar(features.get(name)) is not None for name in native_fields)
    fx_value = _feature_scalar(features.get("fundamental_fx_to_hkd"))
    if has_native_value and (fx_value is None or fx_value <= 0):
        # 推导汇率：从报账货币推断
        currency_val = _feature_scalar(features.get("fundamental_reporting_currency"))
        currency = str(currency_val).upper() if currency_val is not None else ""
        if currency == "HKD":
            fx_value = 1.0
        elif currency == "RMB" or currency == "CNY":
            fx_value = 1.1  # 近似港币/人民币
        else:
            fx_value = 1.0  # 默认港币
        features["fundamental_fx_to_hkd"] = fx_value
    for native_name, hkd_name in (
        ("fundamental_revenue_latest_native", "fundamental_revenue_latest_hkd_mn"),
        ("fundamental_operating_cash_flow_latest_native", "fundamental_operating_cash_flow_latest_hkd_mn"),
        ("fundamental_ending_cash_latest_native", "fundamental_ending_cash_latest_hkd_mn"),
        ("fundamental_net_profit_latest_native", "fundamental_net_profit_latest_hkd_mn"),
        ("fundamental_research_development_latest_native", "fundamental_research_development_latest_hkd_mn"),
    ):
        if hkd_name in features and _feature_scalar(features[hkd_name]) is not None:
            continue
        native_val = _feature_scalar(features.get(native_name))
        if native_val is not None and fx_value is not None:
            features[hkd_name] = round(float(native_val) * float(fx_value), 4)
    # Derive engineered ratios from now-available hkd values
    revenue = _feature_scalar(features.get("fundamental_revenue_latest_hkd_mn"))
    market_cap = _feature_scalar(features.get("offering_market_cap_mid_hkd"))
    net_proceeds = _feature_scalar(features.get("offering_net_proceeds_hkd"))
    ocf = _feature_scalar(features.get("fundamental_operating_cash_flow_latest_hkd_mn"))
    if revenue and revenue != 0:
        if market_cap is not None and _feature_scalar(features.get("engineered_price_to_sales")) is None:
            features["engineered_price_to_sales"] = round(market_cap / revenue, 4)
        if net_proceeds is not None and _feature_scalar(features.get("engineered_fundraising_to_revenue")) is None:
            features["engineered_fundraising_to_revenue"] = round(net_proceeds / revenue, 4)
        if ocf is not None and _feature_scalar(features.get("engineered_operating_cash_flow_margin")) is None:
            features["engineered_operating_cash_flow_margin"] = round(ocf / revenue, 4)



def generate_runtime_prediction(
    *,
    project_root: Path,
    company_id: str,
    company_name: str,
    run_id: str,
    features: Mapping[str, Any] | None,
    report_path: Path,
    prediction_as_of: date | None = None,
    feature_units: Mapping[str, str] | None = None,
    feature_formula_versions: Mapping[str, str] | None = None,
    model_path: Path | None = None,
    destination_path: Path | None = None,
    prediction_variant: str = "agent_enhanced",
) -> tuple[dict[str, Any], Path]:
    """只使用本次候选快照，按冻结模型契约运行五日风险预测。"""
    root = project_root.resolve()
    normalized_id = company_id.strip().zfill(5)
    model_path = model_path or _resolve_resource(
        root,
        "IPO_RISK_PREDICTION_MODEL",
        DEFAULT_MODEL_PATH,
    )
    model_path = model_path.resolve() if model_path.is_absolute() else (root / model_path).resolve()
    if not model_path.is_file():
        raise RuntimePredictionUnavailable(f"预测模型不存在：{model_path}")

    bundle = _load_bundle(str(model_path), model_path.stat().st_mtime_ns)
    contract = load_model_feature_contract(
        bundle,
        sidecar_path=model_path.with_name("feature_contract.json"),
    )
    working_features = dict(features or {})
    _convert_native_to_hkd_mn(working_features)
    projection = project_model_inputs(
        working_features,
        contract,
        prediction_as_of=prediction_as_of,
        feature_units=feature_units,
        feature_formula_versions=feature_formula_versions,
    )
    if projection.status == "blocked":
        detail = "；".join(projection.issues) or "模型输入不满足特征契约。"
        raise RuntimePredictionUnavailable(detail)

    candidate_values = _feature_values(working_features)
    candidate_feature_count = len(candidate_values)
    available_candidate_feature_count = sum(
        value is not None for value in candidate_values.values()
    )
    values = dict(candidate_values)
    values.update(projection.values)
    values.update({"company_id": normalized_id, "company_name": company_name})

    feature_names = list(contract.feature_names)

    prediction_frame = predict_feature_rows(bundle, pd.DataFrame([values]))
    prediction_row = prediction_frame.iloc[0]
    raw_probability = float(prediction_row["predicted_probability"])

    # 市场环境调节
    listing_date_raw = (
        prediction_as_of
        or (values.get("listing_date") or prediction_row.get("listing_date"))
    )
    daily_market_path = (
        root / "data" / "processed" / "modeling" / "daily_market_sentiment.csv"
    )
    if (
        isinstance(listing_date_raw, date)
        and daily_market_path.is_file()
    ):
        market_regime = detect_market_regime(
            str(daily_market_path), listing_date_raw
        )
        probability = adjust_probability(raw_probability, market_regime)
    else:
        market_regime = MarketRegimeResult(
            regime="neutral",
            label="市场数据不可用",
            adjustment_factor=1.0,
            market_return_60d_pct=None,
            market_volatility_60d_pct=None,
            market_return_20d_pct=None,
            description="未提供listing_date或缺少daily_market_sentiment.csv",
        )
        probability = raw_probability

    predicted = bool(int(prediction_row["predicted_significant_drop_5d"]))
    raw_decision = str(
        prediction_row.get(
            "predicted_decision",
            "high_risk" if predicted else "low_risk",
        )
    )
    decision = raw_decision
    requires_review = bool(prediction_row.get("prediction_requires_review", False)) or (
        projection.status == "review_required"
    )
    feature_snapshot = {name: _json_value(prediction_row.get(name)) for name in feature_names}
    available = sum(value is not None for value in feature_snapshot.values())
    ranking = [
        name
        for name in bundle.get("feature_ranking", feature_names)
        if name in feature_snapshot and feature_snapshot[name] is not None
    ]
    key_inputs = [
        {
            "feature": name,
            "label": _FEATURE_LABELS.get(name, name),
            "value": feature_snapshot[name],
        }
        for name in ranking[:5]
    ]
    warnings = list(projection.issues)
    completeness = available / len(feature_names)
    if requires_review:
        warnings.append("当前结果需要人工复核，系统不会将其表述为高置信度方向。")
    raw_confidence = max(probability, 1.0 - probability)
    displayed_confidence = min(raw_confidence, 0.70) if requires_review else raw_confidence

    guard_path = _resolve_resource(
        root,
        "IPO_RISK_RECALL_GUARD_POLICY",
        DEFAULT_RECALL_GUARD_POLICY_PATH,
    )
    try:
        if not guard_path.is_file():
            raise FileNotFoundError(guard_path)
        guard_policy = _load_recall_guard_policy(
            str(guard_path), guard_path.stat().st_mtime_ns
        )
        recall_guard = evaluate_high_risk_guard(values, guard_policy)
        recall_guard["enabled"] = True
        recall_guard["policy_path"] = _display_path(guard_path, project_root=root)
        recall_guard["caught_beyond_primary_model"] = bool(
            recall_guard["watchlist"] and decision != "high_risk"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        recall_guard = {
            "schema_version": "1.0",
            "enabled": False,
            "watchlist": False,
            "caught_beyond_primary_model": False,
            "reasons": [],
            "note": f"高召回保护层暂不可用：{type(exc).__name__}: {exc}",
        }
        warnings.append(recall_guard["note"])

    # 四窗口预测 (h1/h20/h60) — 复用同一特征行，但用各窗口专用模型
    horizon_predictions: dict[str, float] = {}
    horizon_thresholds: dict[str, float] = {}
    horizon_bundles: dict[str, dict[str, Any]] = {}
    for horizon_name, horizon_key in [
        ("h1", "unified_h1"),
        ("h20", "unified_h20"),
        ("h60", "unified_h60"),
    ]:
        horizon_model_path = root / "artifacts" / "models" / horizon_key / "model.joblib"
        if horizon_model_path.is_file():
            try:
                hbundle = _load_bundle(str(horizon_model_path), horizon_model_path.stat().st_mtime_ns)
                horizon_bundles[horizon_name] = hbundle
                hframe = predict_feature_rows(hbundle, pd.DataFrame([values]))
                hrow = hframe.iloc[0]
                horizon_predictions[horizon_name] = float(hrow["predicted_probability"])
                horizon_thresholds[horizon_name] = float(hbundle["threshold"])
            except Exception:
                horizon_predictions[horizon_name] = None
                horizon_thresholds[horizon_name] = None
        else:
            horizon_predictions[horizon_name] = None
            horizon_thresholds[horizon_name] = None

    payload = {
        "schema_version": "1.0",
        "prediction_type": "significant_drop_5d",
        "prediction_variant": prediction_variant,
        "run_id": run_id,
        "company_id": normalized_id,
        "company_name": company_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": str(bundle.get("model_name") or model_path.parent.name),
        "model_path": _display_path(model_path, project_root=root),
        "threshold": float(bundle["threshold"]),
        "raw_probability": raw_probability,
        "probability": probability,
        "probability_adjustment": market_regime.adjustment_factor,
        "market_regime": {
            "regime": market_regime.regime,
            "label": market_regime.label,
            "adjustment_factor": market_regime.adjustment_factor,
            "market_return_60d_pct": market_regime.market_return_60d_pct,
            "market_volatility_60d_pct": market_regime.market_volatility_60d_pct,
            "market_return_20d_pct": market_regime.market_return_20d_pct,
            "description": market_regime.description,
        },
        "horizon_predictions": {
            "h1": horizon_predictions.get("h1"),
            "h5": probability,
            "h20": horizon_predictions.get("h20"),
            "h60": horizon_predictions.get("h60"),
        },
        "horizon_thresholds": {
            "h1": horizon_thresholds.get("h1"),
            "h5": float(bundle["threshold"]),
            "h20": horizon_thresholds.get("h20"),
            "h60": horizon_thresholds.get("h60"),
        },
        "predicted_significant_drop_5d": predicted,
        "decision": decision,
        "requires_review": requires_review,
        "risk_level": _risk_level(decision, probability),
        "confidence": displayed_confidence,
        "feature_count": len(feature_names),
        "available_feature_count": available,
        "feature_completeness": completeness,
        "candidate_feature_count": candidate_feature_count,
        "available_candidate_feature_count": available_candidate_feature_count,
        "used_historical_feature_backfill": False,
        "matched_reference_features": False,
        "feature_contract": contract.to_payload(),
        "feature_readiness": projection.status,
        "feature_group_completeness": projection.group_completeness,
        "missing_model_features": list(projection.missing_features),
        "key_inputs": key_inputs,
        "feature_snapshot": feature_snapshot,
        "recall_guard": recall_guard,
        "warnings": warnings,
        "disclaimer": "方向性风险预警，不构成投资建议。",
    }
    payload = enrich_horizon_prediction_status(
        project_root=root,
        prediction=payload,
    )
    destination = destination_path or report_path.with_name("prediction_5d.json")
    write_json(payload, destination)
    return payload, destination


def generate_runtime_prediction_from_feature_file(
    *,
    project_root: Path,
    company_id: str,
    company_name: str,
    feature_path: Path,
    report_path: Path,
    prediction_variant: str,
    prediction_as_of_fallback: date | None = None,
    model_path: Path | None = None,
    destination_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """复用已保存的上市前特征快照生成预测，不触发PDF或Agent流程。"""

    payload = read_json(feature_path)
    raw_features = payload.get("features")
    if not isinstance(raw_features, dict):
        raise RuntimePredictionUnavailable("预测特征文件格式无效。")
    prediction_as_of_text = str(payload.get("prediction_as_of") or "").strip()
    try:
        prediction_as_of = (
            date.fromisoformat(prediction_as_of_text)
            if prediction_as_of_text
            else prediction_as_of_fallback
        )
    except ValueError as exc:
        raise RuntimePredictionUnavailable("预测特征中的时点格式无效。") from exc
    # 早期历史快照没有保存上市日期，因而也没有按上市日前窗口采集市场特征。
    # 仅在启用日期回退时，从赛事加工数据即时补齐这些客观特征；不重跑PDF或Agent，
    # 也不改写原始历史快照。
    if not prediction_as_of_text and prediction_as_of is not None:
        modeling_dir = project_root / "data" / "processed" / "modeling"
        market_features = build_listing_pre_market_feature_snapshot(
            modeling_dir / "daily_market_sentiment.csv",
            listing_date=prediction_as_of,
            windows=DEFAULT_MARKET_WINDOWS,
        )
        recent_ipo_features = build_recent_ipo_feature_snapshot(
            project_root
            / "data"
            / "processed"
            / "competition"
            / "market_labels.csv",
            listing_date=prediction_as_of,
        )
        raw_features = dict(raw_features)
        raw_features.update(
            {
                name: feature.model_dump(mode="json")
                for name, feature in {**market_features, **recent_ipo_features}.items()
            }
        )

    units = {
        name: str(value.get("unit"))
        for name, value in raw_features.items()
        if isinstance(value, Mapping) and value.get("unit") is not None
    }
    formulas = {
        name: str(value.get("formula_version"))
        for name, value in raw_features.items()
        if isinstance(value, Mapping) and value.get("formula_version") is not None
    }
    return generate_runtime_prediction(
        project_root=project_root,
        company_id=company_id,
        company_name=company_name,
        run_id=str(payload.get("run_id") or f"BASE-{company_id}"),
        features=raw_features,
        report_path=report_path,
        prediction_as_of=prediction_as_of,
        feature_units=units,
        feature_formula_versions=formulas,
        model_path=model_path,
        destination_path=destination_path,
        prediction_variant=prediction_variant,
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_RECALL_GUARD_POLICY_PATH",
    "RuntimePredictionUnavailable",
    "enrich_horizon_prediction_status",
    "generate_runtime_prediction",
    "generate_runtime_prediction_from_feature_file",
]
