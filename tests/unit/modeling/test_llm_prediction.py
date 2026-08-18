from __future__ import annotations

import json
from datetime import date
from threading import Lock
from typing import Any

import pandas as pd
import pytest

from ipo_risk.modeling.llm_prediction import (
    HistoricalIPOCase,
    LLMPredictionCommittee,
    LLMPredictionSingleJudge,
    build_llm_prediction_payload,
    select_historical_cases,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "company_id": "00300",
        "company_name": "测试公司",
        "industry_primary": "工业",
        "prospectus_date": "2023-06-01",
        "target_observation_date_5d": "2023-07-01",
        "offering_offer_price_low_hkd": 8.0,
        "offering_offer_price_high_hkd": 10.0,
        "offering_gross_proceeds_mid_hkd": 900_000_000.0,
        "offering_net_proceeds_hkd": 820_000_000.0,
        "offering_market_cap_mid_hkd": 5_000_000_000.0,
        "offering_share_ratio_pct": 18.0,
        "fundamental_reporting_currency": "RMB",
        "fundamental_revenue_latest_hkd_mn": 1000.0,
        "fundamental_revenue_cagr_3y_pct": 20.0,
        "fundamental_gross_margin_latest_pct": 30.0,
        "fundamental_operating_cash_flow_latest_hkd_mn": -100.0,
        "market_return_20d_pct": -2.0,
        "market_volatility_20d_pct": 1.5,
        "recent_ipo_5d_drop_rate_180d": 0.4,
        "recent_ipo_5d_mean_return_180d_pct": -3.0,
        "recent_ipo_5d_observed_count_180d": 30,
        "recent_ipo_count_30d": 4,
        "pipeline_stage_score": None,
        "target_significant_drop_5d": 1,
        "target_return_vs_offer_5d_pct": -18.0,
    }
    row.update(overrides)
    return row


def test_payload_uses_readable_base_features_without_current_target() -> None:
    case = HistoricalIPOCase(
        company_id="00100",
        company_name="历史公司",
        prospectus_date=date(2022, 1, 1),
        outcome_available_at=date(2022, 2, 1),
        industry="工业",
        similarity=0.8,
        significant_drop_5d=True,
        return_vs_offer_5d_pct=-15.0,
        comparable_facts={"offer_price_mid_hkd": 7.0},
    )

    payload = build_llm_prediction_payload(_row(), report=None, historical_cases=(case,))

    assert payload["offering"]["market_cap_mid_hkd_mn"] == 5000.0
    assert payload["derived_ratios"]["price_to_sales"] == 5.0
    assert payload["agent_risks"] == []
    assert payload["calibration_context"]["market_base_rate_pct"] == 40.0
    assert payload["calibration_context"]["similar_case_event_rate_pct"] == 100.0
    assert 40.0 < payload["calibration_context"]["anchor_probability_pct"] < 50.0
    current_company = {key: value for key, value in payload.items() if key != "historical_cases"}
    serialized = json.dumps(current_company, ensure_ascii=False)
    assert "target_" not in serialized
    assert "-18" not in serialized


def test_historical_cases_are_strictly_prior_and_target_free_for_similarity() -> None:
    frame = pd.DataFrame(
        [
            _row(
                company_id="00100",
                company_name="过去同行",
                prospectus_date="2022-01-01",
                target_observation_date_5d="2022-02-01",
            ),
            _row(
                company_id="00101",
                company_name="过去异业",
                industry_primary="消费",
                prospectus_date="2022-06-01",
                target_observation_date_5d="2022-07-01",
                target_significant_drop_5d=0,
                target_return_vs_offer_5d_pct=5.0,
            ),
            _row(),
            _row(
                company_id="00200",
                company_name="结果尚未知",
                prospectus_date="2023-01-01",
                target_observation_date_5d="2023-07-01",
            ),
            _row(company_id="00400", company_name="未来公司", prospectus_date="2024-01-01"),
        ]
    )

    cases = select_historical_cases(frame, "00300", top_k=5)

    assert {case.company_id for case in cases} == {"00100", "00101"}
    assert all(case.prospectus_date < date(2023, 6, 1) for case in cases)
    assert cases[0].company_id == "00100"


class _FakeProvider:
    model_name = "fake-committee"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._lock = Lock()

    def complete_json(self, **request: Any) -> dict[str, Any]:
        system = request["system_prompt"]
        with self._lock:
            self.prompts.append(request["user_prompt"])
        if system == "bull":
            return {
                "stance": "bull",
                "risk_score": 35,
                "confidence": 0.7,
                "arguments": ["估值尚可"],
                "evidence_ids": [],
                "historical_case_ids": ["00100"],
                "uncertainties": [],
            }
        if system == "bear":
            return {
                "stance": "bear",
                "risk_score": 75,
                "confidence": 0.8,
                "arguments": ["现金流为负"],
                "evidence_ids": [],
                "historical_case_ids": ["00100"],
                "uncertainties": ["缺少Agent报告"],
            }
        return {
            "prospectus_risk_severity": 82,
            "risk_score": 68,
            "confidence": 0.76,
            "short_horizon_trigger_strength": "moderate",
            "short_horizon_triggers": ["上市窗口偏弱且相似案例下跌"],
            "long_horizon_risks": ["经营现金流持续承压"],
            "key_risk_factors": ["经营现金流为负且历史环境偏弱"],
            "positive_factors": ["营收仍增长"],
            "dissenting_opinions": ["多头认为估值尚可"],
            "evidence_ids": [],
            "supporting_case_ids": ["00100"],
            "reasoning_summary": "发行前现金流承压，历史相似案例出现显著下跌，多头意见不足以抵消风险。",
        }


def test_committee_runs_parallel_views_and_returns_fixed_contract() -> None:
    provider = _FakeProvider()
    case = HistoricalIPOCase(
        company_id="00100",
        company_name="历史公司",
        prospectus_date=date(2022, 1, 1),
        outcome_available_at=date(2022, 2, 1),
        industry="工业",
        similarity=0.8,
        significant_drop_5d=True,
        return_vs_offer_5d_pct=-15.0,
        comparable_facts={"market_cap_mid_hkd": 4500.0},
    )
    payload = build_llm_prediction_payload(_row(), report=None, historical_cases=(case,))
    committee = LLMPredictionCommittee(
        provider,
        bull_prompt="bull",
        bear_prompt="bear",
        decision_prompt="decision",
    )

    result = committee.predict(payload)

    assert result.risk_score == 68
    assert result.alert_threshold == 25
    assert result.prospectus_risk_severity == 82
    assert result.calibration_anchor_pct > 40.0
    assert result.risk_level == "high"
    assert result.predicted_significant_drop_5d is True
    assert result.historical_case_drop_rate == 1.0
    assert result.supporting_case_ids == ("00100",)
    assert result.model_name == "fake-committee"
    assert len(provider.prompts) == 3


def test_single_judge_uses_same_contract_with_one_model_call() -> None:
    provider = _FakeProvider()
    case = HistoricalIPOCase(
        company_id="00100",
        company_name="历史公司",
        prospectus_date=date(2022, 1, 1),
        outcome_available_at=date(2022, 2, 1),
        industry="工业",
        similarity=0.8,
        significant_drop_5d=True,
        return_vs_offer_5d_pct=-15.0,
        comparable_facts={"market_cap_mid_hkd": 4500.0},
    )
    payload = build_llm_prediction_payload(_row(), report=None, historical_cases=(case,))
    predictor = LLMPredictionSingleJudge(provider, prompt="single")

    result = predictor.predict(payload)

    assert result.method == "llm_single_judge"
    assert result.prompt_version == "llm-ipo-single-judge-v1"
    assert result.risk_score == 68
    assert result.predicted_significant_drop_5d is True
    assert result.bull_score is None
    assert result.bear_score is None
    assert len(provider.prompts) == 1


def test_decision_cannot_alert_without_short_horizon_channel() -> None:
    provider = _FakeProvider()
    case = HistoricalIPOCase(
        company_id="00100",
        company_name="历史公司",
        prospectus_date=date(2022, 1, 1),
        outcome_available_at=date(2022, 2, 1),
        industry="工业",
        similarity=0.8,
        significant_drop_5d=True,
        return_vs_offer_5d_pct=-15.0,
        comparable_facts={},
    )
    payload = build_llm_prediction_payload(_row(), report=None, historical_cases=(case,))
    committee = LLMPredictionCommittee(
        provider,
        bull_prompt="bull",
        bear_prompt="bear",
        decision_prompt="decision",
    )

    original = provider.complete_json

    def invalid_decision(**request: Any) -> dict[str, Any]:
        result = original(**request)
        if request["system_prompt"] == "decision":
            result["short_horizon_trigger_strength"] = "none"
            result["short_horizon_triggers"] = []
        return result

    provider.complete_json = invalid_decision  # type: ignore[method-assign]
    with pytest.raises(Exception, match="short-horizon channel"):
        committee.predict(payload)


def test_committee_rejects_hallucinated_case_id() -> None:
    provider = _FakeProvider()
    payload = build_llm_prediction_payload(_row(), report=None, historical_cases=())
    committee = LLMPredictionCommittee(
        provider,
        bull_prompt="bull",
        bear_prompt="bear",
        decision_prompt="decision",
    )

    with pytest.raises(Exception, match="unknown cases"):
        committee.predict(payload)
