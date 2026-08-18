from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from ipo_risk.domain import AgentRole, Calculation, Evidence, EvidenceType
from ipo_risk.modeling.agent_precision import (
    build_precision_agent_feature_snapshot,
    reconcile_precision_agent_features,
)
from tests.unit.skills.conftest import make_finding


def _evidence(quote: str) -> Evidence:
    return Evidence(
        evidence_id="E-001",
        company_id="03696",
        document_id="prospectus",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=88,
        bbox=(10.0, 20.0, 300.0, 100.0),
        quote=quote,
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.LEGAL,
        confidence=0.95,
        available_at=date(2024, 1, 1),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _calculation(
    *,
    result: float,
    unit: str,
    formula: str,
    inputs: dict[str, float] | None = None,
    period: str = "2023",
) -> Calculation:
    values = inputs or {"a": 10.0, "b": 100.0}
    return Calculation(
        tool="deterministic_calculator",
        formula=formula,
        inputs=values,
        result=result,
        unit=unit,
        period=period,
        source_refs={name: "E-001" for name in values},
    )


def test_precision_snapshot_scores_strict_redemption_terms() -> None:
    finding = make_finding(
        risk_type="redemption_or_vam",
        claim="若公司未能于2025年12月31日前上市，投资者有权要求公司按8%年单利回购。",
        created_by=AgentRole.LEGAL,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("公司必须按投资成本加每年8%单利赎回股份。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-1",
    )

    assert features["agent_redemption_strictness_score"].value == 3
    assert features["agent_redemption_strictness_score"].source_ref.startswith("finding:F-001")


def test_precision_snapshot_keeps_absence_distinct_from_missing_ratios() -> None:
    features = build_precision_agent_feature_snapshot(
        [],
        [],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-2",
        redemption_assessed=True,
    )

    assert features["agent_redemption_strictness_score"].value == 0
    assert "agent_related_party_ratio_pct" not in features
    assert "agent_cash_runway_months" not in features


def test_precision_snapshot_extracts_reproducible_calculations_and_pipeline() -> None:
    runway = make_finding(
        finding_id="F-RUNWAY",
        risk_type="cash_burn",
        claim="期末现金按当前消耗速度可支持18个月。",
        calculation=_calculation(
            result=18.0,
            unit="months",
            formula="cash_balance / monthly_cash_burn",
            inputs={"cash_balance": 180.0, "monthly_cash_burn": 10.0},
        ),
    )
    related = make_finding(
        finding_id="F-RELATED",
        risk_type="related_party_transaction",
        claim="关联销售占收入12.5%。",
        calculation=_calculation(
            result=12.5,
            unit="percent",
            formula="related_sales / revenue * 100",
            inputs={"related_sales": 12.5, "revenue": 100.0},
        ),
        created_by=AgentRole.LEGAL,
    )
    pipeline = make_finding(
        finding_id="F-PIPELINE",
        risk_type="pipeline_uncertainty",
        claim="核心候选药物已进入临床III期。",
        created_by=AgentRole.BUSINESS,
    )

    features = build_precision_agent_feature_snapshot(
        [runway, related, pipeline],
        [_evidence("核心候选药物已进入临床III期。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-3",
    )

    assert features["agent_cash_runway_months"].value == 18.0
    assert features["agent_related_party_ratio_pct"].value == 12.5
    assert features["agent_pipeline_stage_score"].value == 4.0


def test_reconcile_does_not_relabel_rag_fields_as_agent_features() -> None:
    frame = pd.DataFrame(
        {
            "cash_runway_months": [15.0, None],
            "pipeline_stage_score": [4.0, 2.0],
            "agent_redemption_strictness_score": [None, 2.0],
            "fundamental_ending_cash_latest_native": [100.0, None],
            "fundamental_operating_cash_flow_latest_hkd_mn": [-20.0, None],
            "fundamental_fx_to_hkd": [1.0, None],
        }
    )

    result = reconcile_precision_agent_features(frame)

    assert result["agent_cash_runway_months"].isna().all()
    assert result["agent_pipeline_stage_score"].isna().all()
    assert pd.isna(result.loc[0, "agent_redemption_strictness_score"])
    assert result.loc[1, "agent_redemption_strictness_score"] == 2.0
    assert result["agent_related_party_ratio_pct"].isna().all()


def test_precision_snapshot_accepts_one_explicit_related_party_percentage() -> None:
    finding = make_finding(
        risk_type="related_party_transaction",
        claim="最近年度关联销售占总收入18.6%，定价公允性仍需关注。",
        created_by=AgentRole.LEGAL,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("最近年度关联销售占总收入18.6%。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-4",
    )

    assert features["agent_related_party_ratio_pct"].value == 18.6


def test_precision_snapshot_rejects_related_purchase_ratio_with_different_denominator() -> None:
    finding = make_finding(
        risk_type="related_party_transaction",
        claim="最近年度关联采购占总采购额18.6%。",
        created_by=AgentRole.LEGAL,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("最近年度关联采购占总采购额18.6%。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-4B",
    )

    assert "agent_related_party_ratio_pct" not in features


def test_precision_snapshot_rejects_unrelated_percentage_in_related_party_text() -> None:
    finding = make_finding(
        risk_type="related_party_transaction",
        claim="公司存在关联方贷款，贷款年利率为1%，需关注持续依赖。",
        created_by=AgentRole.LEGAL,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("该关联方贷款按年利率1%计息。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-5",
    )

    assert "agent_related_party_ratio_pct" not in features


def test_pipeline_uncertainty_does_not_mean_already_commercialized() -> None:
    finding = make_finding(
        risk_type="pipeline_uncertainty",
        claim="商业化仍存在重大不确定性，核心候选药物目前处于临床II期。",
        created_by=AgentRole.BUSINESS,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("核心候选药物目前处于临床II期。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-6",
    )

    assert features["agent_pipeline_stage_score"].value == 3.0


def test_terminated_redemption_right_is_not_scored_as_strict() -> None:
    finding = make_finding(
        risk_type="redemption_or_vam",
        claim="历史赎回权已终止。投资者于2023年5月12日持有公司8%股份。",
        created_by=AgentRole.LEGAL,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("相关特别权利已于上市前自动终止。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-7",
        redemption_assessed=True,
    )

    assert features["agent_redemption_strictness_score"].value == 0


def test_pipeline_future_commercialization_target_is_not_stage_six() -> None:
    finding = make_finding(
        risk_type="pipeline_uncertainty",
        claim=("公司预计获得已商业化公司资格，但核心产品尚未实现商业化，目前处于临床II期。"),
        created_by=AgentRole.BUSINESS,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("核心产品尚未实现商业化，目前处于临床II期。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-8",
    )

    assert features["agent_pipeline_stage_score"].value == 3.0


def test_pipeline_failed_phase_three_keeps_current_core_phase_two() -> None:
    failed = make_finding(
        finding_id="F-FAILED",
        risk_type="pipeline_uncertainty",
        claim="核心候选药物A的临床III期试验失败并已终止。",
        created_by=AgentRole.BUSINESS,
    )
    active = make_finding(
        finding_id="F-ACTIVE",
        risk_type="pipeline_uncertainty",
        claim="核心候选药物B目前处于临床II期。",
        created_by=AgentRole.BUSINESS,
    )

    features = build_precision_agent_feature_snapshot(
        [failed, active],
        [],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-8B",
    )

    assert features["agent_pipeline_stage_score"].value == 3.0
    assert features["agent_pipeline_stage_score"].source_ref.startswith("finding:F-ACTIVE")


def test_cash_runway_requires_cash_divided_by_burn_and_prefers_latest_period() -> None:
    old = make_finding(
        finding_id="F-OLD",
        risk_type="cash_burn",
        claim="旧报告期现金续航为24个月。",
        calculation=_calculation(
            result=24.0,
            unit="months",
            formula="cash_balance / monthly_cash_burn",
            inputs={"cash_balance": 240.0, "monthly_cash_burn": 10.0},
            period="2022-12-31",
        ),
    ).model_copy(update={"confidence": 0.99})
    latest = make_finding(
        finding_id="F-LATEST",
        risk_type="cash_burn",
        claim="最新报告期现金续航为12个月。",
        calculation=_calculation(
            result=12.0,
            unit="months",
            formula="cash_balance / monthly_cash_burn",
            inputs={"cash_balance": 120.0, "monthly_cash_burn": 10.0},
            period="2023-12-31",
        ),
    ).model_copy(update={"confidence": 0.80})
    unrelated = make_finding(
        finding_id="F-DEBT",
        risk_type="cash_burn",
        claim="债务将在18个月后到期。",
        calculation=_calculation(
            result=18.0,
            unit="months",
            formula="debt_maturity_months",
            inputs={"debt_maturity_months": 18.0},
            period="2024-01-01",
        ),
    )

    features = build_precision_agent_feature_snapshot(
        [old, latest, unrelated],
        [],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-8C",
    )

    assert features["agent_cash_runway_months"].value == 12.0
    assert features["agent_cash_runway_months"].source_ref.startswith("finding:F-LATEST")


def test_pipeline_unreached_phase_three_keeps_achieved_phase_two() -> None:
    finding = make_finding(
        risk_type="pipeline_uncertainty",
        claim="核心候选药物尚未进入临床III期，目前处于临床II期。",
        created_by=AgentRole.BUSINESS,
    )

    features = build_precision_agent_feature_snapshot(
        [finding],
        [_evidence("尚未进入临床III期，目前处于临床II期。")],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-9",
    )

    assert features["agent_pipeline_stage_score"].value == 3.0


def test_failed_specialist_finding_does_not_create_precision_feature() -> None:
    finding = make_finding(
        risk_type="pipeline_uncertainty",
        claim="核心候选药物处于临床III期。",
        created_by=AgentRole.BUSINESS,
    ).model_copy(update={"specialist_status": "technical_failure"})

    features = build_precision_agent_feature_snapshot(
        [finding],
        [],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-10",
    )

    assert "agent_pipeline_stage_score" not in features


def test_unassessed_redemption_absence_remains_missing() -> None:
    features = build_precision_agent_feature_snapshot(
        [],
        [],
        available_at=date(2024, 1, 1),
        source_ref="analysis:RUN-11",
        redemption_assessed=False,
    )

    assert "agent_redemption_strictness_score" not in features
