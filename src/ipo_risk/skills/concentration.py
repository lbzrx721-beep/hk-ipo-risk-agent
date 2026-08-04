"""客户、供应商和产品集中度计算。

从 Agent 分析结果中提取 customer_concentration / supplier_concentration /
single_product_dependency 风险发现，输出集中度相关特征。

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

# 集中度相关的风险类型
_CONCENTRATION_TYPES = frozenset({
    "customer_concentration",
    "supplier_concentration",
    "single_product_dependency",
})


def _severity_score(severity: RiskSeverity) -> int:
    return _SEVERITY_SCORE.get(severity, 0)


def extract_concentration_features(
    findings: list[RiskFinding],
    *,
    source_available_at: date | None = None,
) -> dict[str, FeatureValue]:
    """从风险发现中提取客户/供应商/产品集中度特征。

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

    # 按风险类型分组
    by_type: dict[str, list[RiskFinding]] = {}
    for f in findings:
        if f.risk_type in _CONCENTRATION_TYPES:
            by_type.setdefault(f.risk_type, []).append(f)
    if by_type and source_available_at is None:
        raise ValueError("source_available_at is required for finding-derived features")

    # 客户集中度
    _extract_single_type(
        features, by_type.get("customer_concentration", []),
        prefix="customer_concentration",
        source_available_at=source_available_at,
    )

    # 供应商集中度
    _extract_single_type(
        features, by_type.get("supplier_concentration", []),
        prefix="supplier_concentration",
        source_available_at=source_available_at,
    )

    # 单一产品依赖
    _extract_single_type(
        features, by_type.get("single_product_dependency", []),
        prefix="single_product_dependency",
        source_available_at=source_available_at,
    )

    # 汇总：集中度风险总数与最高严重度
    all_conc = [
        f for f in findings if f.risk_type in _CONCENTRATION_TYPES
    ]
    if all_conc:
        assert source_available_at is not None
        ref_date = source_available_at
        features["concentration_risk_count"] = FeatureValue(
            value=len(all_conc),
            available_at=ref_date,
            source_ref="findings:concentration:count",
        )
        max_sev = max(_severity_score(f.severity) for f in all_conc)
        features["concentration_max_severity_score"] = FeatureValue(
            value=max_sev,
            available_at=ref_date,
            source_ref="findings:concentration:max_severity",
        )
        # 综合集中度风险评分 = 各类型严重度之和（0-12）
        total_score = sum(
            max(_severity_score(f.severity) for f in type_findings)
            for type_findings in by_type.values()
        )
        features["concentration_total_severity_score"] = FeatureValue(
            value=total_score,
            available_at=ref_date,
            source_ref="findings:concentration:total_severity",
        )

    return features


def _extract_single_type(
    features: dict[str, FeatureValue],
    type_findings: list[RiskFinding],
    *,
    prefix: str,
    source_available_at: date | None,
) -> None:
    """为单一集中度风险类型提取特征并写入 features 字典。"""
    if not type_findings:
        return

    best = max(type_findings, key=lambda f: f.confidence)
    assert source_available_at is not None
    ref_date = source_available_at

    features[f"has_{prefix}"] = FeatureValue(
        value=True,
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}",
    )
    features[f"{prefix}_severity_score"] = FeatureValue(
        value=_severity_score(best.severity),
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}:severity",
    )
    features[f"{prefix}_confidence"] = FeatureValue(
        value=best.confidence,
        available_at=ref_date,
        source_ref=f"finding:{best.finding_id}:confidence",
    )
    features[f"{prefix}_finding_count"] = FeatureValue(
        value=len(type_findings),
        available_at=ref_date,
        source_ref=f"findings:{prefix}:count",
    )


__all__ = ["extract_concentration_features"]
