"""严格区分上市前预测特征与上市后标签/回测指标。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date
from statistics import mean
from typing import Any

from ipo_risk.domain import (
    AgentRole,
    EvidenceType,
    FeatureValue,
    RiskSeverity,
    ToolCallStatus,
    TraceStatus,
)
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.skills import (
    extract_cash_burn_features,
    extract_concentration_features,
    extract_market_features,
    extract_related_party_features,
    extract_valuation_features,
)


def build_prelisting_feature_snapshot(
    report: RiskAnalysisReport,
    *,
    company_info: dict[str, object],
    as_of_date: date,
) -> dict[str, FeatureValue]:
    """组装预测日可见的信息，绝不读取上市后行情。"""
    findings = list(report.risk_findings)
    source_available_at = report.source_document.available_at
    features: dict[str, FeatureValue] = {}
    for extracted in (
        build_agent_feature_snapshot(report, as_of_date=as_of_date),
        extract_cash_burn_features(
            findings,
            source_available_at=source_available_at,
            company_info=company_info,
        ),
        extract_concentration_features(
            findings,
            source_available_at=source_available_at,
        ),
        extract_related_party_features(
            findings,
            source_available_at=source_available_at,
        ),
        extract_valuation_features(company_info),
    ):
        _merge_features(features, extracted)

    future = sorted(
        name
        for name, feature in features.items()
        if feature.available_at > as_of_date
    )
    if future:
        raise ValueError(f"features available after as_of_date: {', '.join(future)}")
    return features


def build_agent_feature_snapshot(
    report: RiskAnalysisReport,
    *,
    as_of_date: date,
) -> dict[str, FeatureValue]:
    """把任意动态风险报告转换为稳定、可解释的预测特征。"""
    available_at = report.source_document.available_at
    if available_at > as_of_date:
        raise ValueError("risk report is available after as_of_date")

    findings = list(report.risk_findings)
    evidences = list(getattr(report, "evidences", ()))
    traces = list(getattr(report, "agent_traces", ()))
    severity_score = {
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }
    severity_counts = Counter(finding.severity for finding in findings)
    role_counts = Counter(finding.created_by for finding in findings)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    finding_confidences = [finding.confidence for finding in findings]
    evidence_confidences = [evidence.confidence for evidence in evidences]
    tool_calls = [call for trace in traces for call in trace.tool_calls]
    source_ref = f"analysis:{getattr(report, 'run_id', 'unknown')}"

    values: dict[str, Any] = {
        "agent_finding_count": len(findings),
        "agent_evidence_count": len(evidences),
        "agent_unique_risk_type_count": len({finding.risk_type for finding in findings}),
        "agent_dynamic_risk_finding_count": sum(
            finding.risk_type.startswith("open_") for finding in findings
        ),
        "agent_canonical_risk_finding_count": sum(
            not finding.risk_type.startswith("open_") for finding in findings
        ),
        "agent_unique_evidence_page_count": len({evidence.page for evidence in evidences}),
        "agent_finding_confidence_mean": (
            round(mean(finding_confidences), 8) if finding_confidences else 0.0
        ),
        "agent_finding_confidence_max": (
            max(finding_confidences) if finding_confidences else 0.0
        ),
        "agent_evidence_confidence_mean": (
            round(mean(evidence_confidences), 8) if evidence_confidences else 0.0
        ),
        "agent_evidence_per_finding": (
            round(len(evidences) / len(findings), 8) if findings else 0.0
        ),
        "agent_ocr_evidence_ratio": (
            round(
                sum(
                    evidence.evidence_type is EvidenceType.OCR_TEXT
                    for evidence in evidences
                )
                / len(evidences),
                8,
            )
            if evidences
            else 0.0
        ),
        "agent_calculation_count": sum(
            finding.calculation is not None for finding in findings
        ),
        "agent_uncertainty_count": sum(
            finding.uncertainty is not None for finding in findings
        ),
        "agent_max_severity_score": max(
            (severity_score[finding.severity] for finding in findings),
            default=0,
        ),
        "agent_trace_count": len(traces),
        "agent_failed_trace_count": sum(
            trace.status is TraceStatus.FAILED for trace in traces
        ),
        "agent_tool_call_count": len(tool_calls),
        "agent_failed_tool_call_count": sum(
            call.status is ToolCallStatus.FAILED for call in tool_calls
        ),
    }
    for severity in RiskSeverity:
        values[f"agent_severity_{severity.value}_count"] = severity_counts[severity]
    for role in (
        AgentRole.FINANCE,
        AgentRole.LEGAL,
        AgentRole.BUSINESS,
        AgentRole.MARKET,
    ):
        values[f"agent_role_{role.value}_finding_count"] = role_counts[role]

    for risk_type in sorted({finding.risk_type for finding in findings}):
        if risk_type.startswith("open_"):
            continue
        type_findings = [
            finding
            for finding in findings
            if finding.risk_type == risk_type
        ]
        referenced_evidence = {
            evidence_id
            for finding in type_findings
            for evidence_id in finding.evidence_ids
            if evidence_id in evidence_by_id
        }
        values[f"agent_has_risk_{risk_type}"] = 1
        values[f"agent_risk_{risk_type}_count"] = len(type_findings)
        values[f"agent_risk_{risk_type}_max_severity"] = max(
            severity_score[finding.severity] for finding in type_findings
        )
        values[f"agent_risk_{risk_type}_max_confidence"] = max(
            finding.confidence for finding in type_findings
        )
        values[f"agent_risk_{risk_type}_evidence_count"] = len(referenced_evidence)

    return {
        name: FeatureValue(
            value=value,
            available_at=available_at,
            source_ref=source_ref,
        )
        for name, value in values.items()
    }


def build_post_listing_metrics(
    market_data: dict[str, object],
    *,
    company_info: dict[str, object],
    horizons: tuple[int, ...] = (1, 5, 20, 60),
) -> dict[str, FeatureValue]:
    """单独构造上市后标签和回测指标，不可传入预测器。"""
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive trading-day windows")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")

    metrics: dict[str, FeatureValue] = {}
    for horizon in horizons:
        market = extract_market_features(
            market_data,
            company_info=company_info,
            window_days=horizon,
        )
        valuation = extract_valuation_features(
            company_info,
            market_data=market_data,
            window_days=horizon,
        )
        post_prefixes = ("first_day_", f"first_{horizon}d_")
        post_market = {
            name: value
            for name, value in market.items()
            if name.startswith(post_prefixes)
        }
        post_valuation = {
            name: value
            for name, value in valuation.items()
            if name.startswith(post_prefixes)
        }
        _merge_features(metrics, post_market)
        _merge_features(metrics, post_valuation)
    return metrics


def _merge_features(
    destination: dict[str, FeatureValue],
    incoming: Mapping[str, FeatureValue],
) -> None:
    for name, feature in incoming.items():
        existing = destination.get(name)
        if existing is None:
            destination[name] = feature
            continue
        if existing.value != feature.value or existing.available_at != feature.available_at:
            raise ValueError(f"conflicting feature values for {name}")


__all__ = [
    "build_agent_feature_snapshot",
    "build_post_listing_metrics",
    "build_prelisting_feature_snapshot",
]
