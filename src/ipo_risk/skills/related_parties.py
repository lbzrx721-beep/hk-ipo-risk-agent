"""关联方和关联交易结构化处理。

从 Agent 分析结果中提取 related_party_transaction 风险发现，
输出关联交易风险评分和结构化特征。

所有函数均为纯函数，不调用 LLM，可单元测试。
"""

from __future__ import annotations

from datetime import date

from ipo_risk.domain import FeatureValue, RiskFinding, RiskSeverity

_SEVERITY_SCORE: dict[str, int] = {
    RiskSeverity.LOW: 1,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.HIGH: 3,
    RiskSeverity.CRITICAL: 4,
}

_RPT_TYPE = "related_party_transaction"


def _severity_score(severity: RiskSeverity) -> int:
    return _SEVERITY_SCORE.get(severity, 0)


def extract_related_party_features(
    findings: list[RiskFinding],
    *,
    source_available_at: date | None = None,
) -> dict[str, FeatureValue]:
    """从风险发现中提取关联交易相关特征。

    Parameters
    ----------
    findings:
        Agent 产出的全部 RiskFinding 列表。

    Returns
    -------
    dict[str, FeatureValue]
        特征名 → FeatureValue。无相关发现时返回空字典。
    """
    features: dict[str, FeatureValue] = {}

    rpt_findings = [f for f in findings if f.risk_type == _RPT_TYPE]
    if not rpt_findings:
        return features
    if source_available_at is None:
        raise ValueError("source_available_at is required for finding-derived features")

    best = max(rpt_findings, key=lambda f: f.confidence)
    ref_date = source_available_at

    # 是否存在关联交易发现
    features["has_related_party_finding"] = FeatureValue(
        value=True,
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}",
    )

    # 最高置信度发现的严重度
    features["related_party_severity_score"] = FeatureValue(
        value=_severity_score(best.severity),
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}:severity",
    )

    # 最高置信度
    features["related_party_confidence"] = FeatureValue(
        value=best.confidence,
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}:confidence",
    )

    # 关联交易发现数量
    features["related_party_finding_count"] = FeatureValue(
        value=len(rpt_findings),
        available_at=ref_date,
        source_ref="findings:related_party:count",
    )

    # 最高严重度（跨所有关联交易发现）
    max_sev = max(_severity_score(f.severity) for f in rpt_findings)
    features["related_party_max_severity_score"] = FeatureValue(
        value=max_sev,
        available_at=ref_date,
        source_ref="findings:related_party:max_severity",
    )

    # 是否附带数值计算
    has_calc = any(f.calculation is not None for f in rpt_findings)
    features["related_party_has_calculation"] = FeatureValue(
        value=has_calc,
        available_at=ref_date,
        source_ref="findings:related_party:has_calculation",
    )

    # 关联交易风险综合评分 = 严重度 × 置信度（0-4）
    risk_score = _severity_score(best.severity) * best.confidence
    features["related_party_risk_score"] = FeatureValue(
        value=round(risk_score, 4),
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}:risk_score",
    )

    # 是否有 critical 级别的关联交易
    has_critical = any(
        f.severity == RiskSeverity.CRITICAL for f in rpt_findings
    )
    features["related_party_has_critical"] = FeatureValue(
        value=has_critical,
        available_at=ref_date,
        source_ref="findings:related_party:has_critical",
    )

    return features


__all__ = ["extract_related_party_features"]
