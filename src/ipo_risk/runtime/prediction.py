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
from ipo_risk.modeling.prelisting import (
    load_model_feature_contract,
    project_model_inputs,
)
from ipo_risk.modeling.predict import load_model_bundle, predict_feature_rows
from ipo_risk.storage.files import read_json, write_json


DEFAULT_MODEL_PATH = Path(
    "artifacts/models/enhanced_blend4_rf_v4_listing/model.joblib"
)
DEFAULT_BASE_MODEL_PATH = Path(
    "artifacts/models/day5_business_base_v1/business_base_ensemble.joblib"
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
    projection = project_model_inputs(
        features or {},
        contract,
        prediction_as_of=prediction_as_of,
        feature_units=feature_units,
        feature_formula_versions=feature_formula_versions,
    )
    if projection.status == "blocked":
        detail = "；".join(projection.issues) or "模型输入不满足特征契约。"
        raise RuntimePredictionUnavailable(detail)

    candidate_values = _feature_values(features)
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
    probability = float(prediction_row["predicted_probability"])
    predicted = bool(int(prediction_row["predicted_significant_drop_5d"]))
    decision = str(
        prediction_row.get(
            "predicted_decision",
            "high_risk" if predicted else "low_risk",
        )
    )
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
        "probability": probability,
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
            else None
        )
    except ValueError as exc:
        raise RuntimePredictionUnavailable("预测特征中的时点格式无效。") from exc
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
    "DEFAULT_BASE_MODEL_PATH",
    "DEFAULT_RECALL_GUARD_POLICY_PATH",
    "RuntimePredictionUnavailable",
    "generate_runtime_prediction",
    "generate_runtime_prediction_from_feature_file",
]
