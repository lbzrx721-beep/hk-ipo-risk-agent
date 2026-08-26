"""上市后五日显著下跌的高召回保护层。

保护层不覆盖主模型概率和结论，只把具有多项历史高风险形态、但未被
主模型预警的公司加入观察名单，交给现有 Agent 证据链进一步核验。
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any


_FEATURE_LABELS = {
    "offering_offer_price_mid_hkd": "发行价处于历史低位",
    "offering_market_cap_mid_hkd": "预计市值处于历史低位",
    "fundamental_revenue_latest_native": "最近一期收入处于历史低位",
    "fundamental_ending_cash_latest_native": "期末现金处于历史低位",
    "customer_top5_ratio_pct": "前五大客户集中度较高",
    "offering_share_ratio_pct": "新发行股份比例较高",
}


def load_high_risk_guard_policy(path: Path) -> dict[str, Any]:
    """读取并校验高召回保护层策略。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("high-risk guard policy does not contain features")
    threshold = payload.get("watchlist_threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("high-risk guard policy has an invalid threshold")
    return payload


def evaluate_high_risk_guard(
    values: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """按历史分位计算稳定、方向受约束的风险观察分。"""
    policies = policy.get("features")
    if not isinstance(policies, Mapping):
        raise ValueError("high-risk guard policy does not contain features")

    signals: list[dict[str, Any]] = []
    risk_scores: list[float] = []
    available = 0
    for feature, feature_policy in policies.items():
        if not isinstance(feature_policy, Mapping):
            continue
        numeric = _number(_unwrap(values.get(str(feature))))
        if numeric is None:
            risk = 0.5
            percentile = None
        else:
            available += 1
            quantiles = _valid_quantiles(feature_policy.get("quantiles"))
            percentile = _approximate_percentile(numeric, quantiles)
            direction = str(feature_policy.get("direction", "down"))
            risk = percentile if direction == "up" else 1.0 - percentile
        risk_scores.append(risk)
        signals.append(
            {
                "feature": str(feature),
                "label": _FEATURE_LABELS.get(str(feature), str(feature)),
                "value": numeric,
                "historical_percentile": percentile,
                "risk_contribution": risk,
                "available": numeric is not None,
            }
        )

    score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
    minimum_available = int(policy.get("minimum_available_features", 4))
    threshold = float(policy["watchlist_threshold"])
    watchlist = available >= minimum_available and score >= threshold
    reasons = [
        signal["label"]
        for signal in sorted(
            signals,
            key=lambda item: float(item["risk_contribution"]),
            reverse=True,
        )
        if signal["available"] and float(signal["risk_contribution"]) >= 0.65
    ][:3]
    if watchlist and not reasons:
        reasons = ["多项上市前稳定因子组合后处于历史高风险区间"]

    return {
        "schema_version": "1.0",
        "policy_version": str(policy.get("policy_version", "unknown")),
        "watchlist": watchlist,
        "score": score,
        "threshold": threshold,
        "available_feature_count": available,
        "feature_count": len(risk_scores),
        "minimum_available_features": minimum_available,
        "reasons": reasons,
        "feature_signals": signals,
        "note": (
            "该结果是少漏报保护层，不改变主模型概率和方向；"
            "命中后应结合 Agent 原文证据进一步核验。"
        ),
    }


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _valid_quantiles(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("high-risk guard feature does not contain quantiles")
    result = tuple(numeric for item in value if (numeric := _number(item)) is not None)
    if len(result) < 2 or any(right < left for left, right in zip(result, result[1:])):
        raise ValueError("high-risk guard feature has invalid quantiles")
    return result


def _approximate_percentile(value: float, quantiles: tuple[float, ...]) -> float:
    """使用等距分位点近似经验分位，兼容重复分位值。"""
    if value <= quantiles[0]:
        return 0.0
    if value >= quantiles[-1]:
        return 1.0
    right = bisect_right(quantiles, value)
    left = max(0, right - 1)
    upper = min(right, len(quantiles) - 1)
    lower_value = quantiles[left]
    upper_value = quantiles[upper]
    lower_percentile = left / (len(quantiles) - 1)
    upper_percentile = upper / (len(quantiles) - 1)
    if upper_value <= lower_value:
        return upper_percentile
    fraction = (value - lower_value) / (upper_value - lower_value)
    return lower_percentile + fraction * (upper_percentile - lower_percentile)


__all__ = ["evaluate_high_risk_guard", "load_high_risk_guard_policy"]
