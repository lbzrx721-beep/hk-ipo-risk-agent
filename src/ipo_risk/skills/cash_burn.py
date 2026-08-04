"""现金消耗率和现金覆盖月数计算。

从 Agent 分析结果（RiskFinding）中提取 cash_burn / persistent_loss 风险发现，
结合结构化公司信息（融资净额等），输出供建模层消费的确定性特征。

所有函数均为纯函数，不调用 LLM，可单元测试。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ipo_risk.domain import FeatureValue, RiskFinding, RiskSeverity

# 严重度 → 数值分（供模型消费）
_SEVERITY_SCORE: dict[str, int] = {
    RiskSeverity.LOW: 1,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.HIGH: 3,
    RiskSeverity.CRITICAL: 4,
}

# 本模块关注的风险类型
_CASH_BURN_TYPES = frozenset({"cash_burn", "persistent_loss"})


def _severity_score(severity: RiskSeverity) -> int:
    return _SEVERITY_SCORE.get(severity, 0)


def extract_cash_burn_features(
    findings: list[RiskFinding],
    *,
    source_available_at: date | None = None,
    company_info: dict[str, Any] | None = None,
) -> dict[str, FeatureValue]:
    """从风险发现和公司信息中提取现金消耗相关特征。

    Parameters
    ----------
    findings:
        Agent 产出的全部 RiskFinding 列表（可包含多种 risk_type）。
    company_info:
        可选的结构化公司信息 JSON（data/raw/company_info/…/公司信息.json）。

    Returns
    -------
    dict[str, FeatureValue]
        特征名 → FeatureValue。缺失字段优雅跳过，不抛异常。
    """
    features: dict[str, FeatureValue] = {}

    # ---- 从 findings 中筛选现金消耗相关 ----
    cash_burn_findings = [
        f for f in findings if f.risk_type == "cash_burn"
    ]
    loss_findings = [
        f for f in findings if f.risk_type == "persistent_loss"
    ]
    if (cash_burn_findings or loss_findings) and source_available_at is None:
        raise ValueError("source_available_at is required for finding-derived features")

    # 现金消耗发现
    if cash_burn_findings:
        best = max(cash_burn_findings, key=lambda f: f.confidence)
        ref_date = source_available_at
        assert ref_date is not None
        features["has_cash_burn_finding"] = FeatureValue(
            value=True,
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}",
        )
        features["cash_burn_severity_score"] = FeatureValue(
            value=_severity_score(best.severity),
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}:severity",
        )
        features["cash_burn_confidence"] = FeatureValue(
            value=best.confidence,
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}:confidence",
        )
        features["cash_burn_finding_count"] = FeatureValue(
            value=len(cash_burn_findings),
            available_at=ref_date,
            source_ref="findings:cash_burn:count",
        )
        # 是否附带数值计算（如现金覆盖月数）
        has_calc = any(f.calculation is not None for f in cash_burn_findings)
        features["cash_burn_has_calculation"] = FeatureValue(
            value=has_calc,
            available_at=ref_date,
            source_ref="findings:cash_burn:has_calculation",
        )
        # 如果有 calculation，提取结果值
        for f in cash_burn_findings:
            if f.calculation is not None:
                result_val = _safe_float(f.calculation.result)
                if result_val is not None:
                    features["cash_burn_calculation_result"] = FeatureValue(
                        value=result_val,
                        available_at=ref_date,
                        source_ref=f"finding:{f.finding_id}:calculation",
                    )
                    break

    # 持续亏损发现
    if loss_findings:
        best = max(loss_findings, key=lambda f: f.confidence)
        ref_date = source_available_at
        assert ref_date is not None
        features["has_persistent_loss_finding"] = FeatureValue(
            value=True,
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}",
        )
        features["persistent_loss_severity_score"] = FeatureValue(
            value=_severity_score(best.severity),
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}:severity",
        )
        features["persistent_loss_confidence"] = FeatureValue(
            value=best.confidence,
            available_at=ref_date,
            source_ref=f"finding:{best.finding_id}:confidence",
        )

    # ---- 从 company_info 提取融资数据（现金跑道代理变量） ----
    if company_info:
        as_of = _parse_date(company_info.get("as_of_date"))
        offering = company_info.get("offering", {})
        if not isinstance(offering, dict):
            offering = {}

        net_proceeds = _safe_float(offering.get("net_proceeds_hkd"))
        if net_proceeds is not None and as_of is not None:
            features["net_proceeds_hkd"] = FeatureValue(
                value=net_proceeds,
                available_at=as_of,
                source_ref="company_info:offering.net_proceeds_hkd",
            )

        gross_proceeds = _safe_float(offering.get("gross_proceeds_hkd"))
        if gross_proceeds is not None and as_of is not None:
            features["gross_proceeds_hkd"] = FeatureValue(
                value=gross_proceeds,
                available_at=as_of,
                source_ref="company_info:offering.gross_proceeds_hkd",
            )

    return features


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> float | None:
    """安全转换为 float，失败返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if result == result and result not in (float("inf"), float("-inf")):
            return result
    return None


def _parse_date(value: object) -> date | None:
    """将 'YYYY-MM-DD' 字符串解析为 date，失败返回 None。"""
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    if isinstance(value, date):
        return value
    return None


__all__ = ["extract_cash_burn_features"]
