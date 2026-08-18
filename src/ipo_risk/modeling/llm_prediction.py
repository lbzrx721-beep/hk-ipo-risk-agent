"""基于基础事实、Agent证据和历史先例的LLM直接预警。

该链路与LightGBM并列存在。它不读取待预测公司的上市后标签，只允许引用
招股书日前已经上市并完成五日观察的历史案例。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.agents.analysis import JsonChatProvider
from ipo_risk.modeling.historical_cases import (
    PgVectorHistoricalCaseStore,
    build_case_summary,
    case_feature_payload,
    industry_similarity,
    robust_feature_scales,
    robust_numeric_similarity,
)
from ipo_risk.modeling.industry import normalize_industry
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.storage.files import read_json


PROMPT_VERSION: Literal["llm-ipo-committee-v2.1"] = "llm-ipo-committee-v2.1"
SINGLE_JUDGE_PROMPT_VERSION: Literal["llm-ipo-single-judge-v1"] = (
    "llm-ipo-single-judge-v1"
)
BUSINESS_ALERT_THRESHOLD = 25

ShortHorizonTriggerStrength = Literal["none", "weak", "moderate", "strong"]
ProgressCallback = Callable[[str, str], None]

# 仅用上市前可获得的字段做相似度。禁止加入target_*及公司身份字段。
PEER_NUMERIC_FEATURES = (
    "offering_offer_price_mid_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "fundamental_revenue_latest_hkd_mn",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_gross_margin_latest_pct",
    "fundamental_operating_cash_flow_latest_hkd_mn",
    "market_return_20d_pct",
    "market_volatility_20d_pct",
    "recent_ipo_5d_drop_rate_180d",
    "recent_ipo_5d_mean_return_180d_pct",
    "recent_ipo_count_30d",
    "pipeline_stage_score",
)


class CommitteeMemo(BaseModel):
    """多头或空头Agent的结构化意见。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    stance: Literal["bull", "bear"]
    risk_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0)
    arguments: tuple[str, ...] = Field(min_length=1, max_length=6)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=12)
    historical_case_ids: tuple[str, ...] = Field(default=(), max_length=8)
    uncertainties: tuple[str, ...] = Field(default=(), max_length=5)


class CommitteeDecisionDraft(BaseModel):
    """决策Agent返回的最小草稿；确定性字段由代码补齐。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    prospectus_risk_severity: int = Field(
        ge=0,
        le=100,
        description="招股书中长期经营与合规风险的严重度，不等同于五日下跌概率。",
    )
    risk_score: int = Field(
        ge=0,
        le=100,
        description="上市后第5个交易日相对发行价收益率<=-10%的事件概率百分数。",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    short_horizon_trigger_strength: ShortHorizonTriggerStrength
    short_horizon_triggers: tuple[str, ...] = Field(default=(), max_length=5)
    long_horizon_risks: tuple[str, ...] = Field(default=(), max_length=6)
    key_risk_factors: tuple[str, ...] = Field(min_length=1, max_length=6)
    positive_factors: tuple[str, ...] = Field(default=(), max_length=5)
    dissenting_opinions: tuple[str, ...] = Field(default=(), max_length=4)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=15)
    supporting_case_ids: tuple[str, ...] = Field(default=(), max_length=8)
    reasoning_summary: str = Field(min_length=20, max_length=2_000)

    @model_validator(mode="after")
    def score_requires_short_horizon_channel(self) -> CommitteeDecisionDraft:
        if self.short_horizon_trigger_strength == "none" and self.risk_score >= 50:
            raise ValueError("risk_score cannot trigger an alert without a short-horizon channel")
        if self.short_horizon_trigger_strength == "none" and self.short_horizon_triggers:
            raise ValueError("short_horizon_triggers must be empty when strength is none")
        if self.short_horizon_trigger_strength != "none" and not self.short_horizon_triggers:
            raise ValueError("a non-empty short-horizon trigger is required")
        return self


class HistoricalIPOCase(BaseModel):
    """预测日以前已经可知结果的历史IPO案例。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    company_id: str
    company_name: str
    prospectus_date: date
    outcome_available_at: date | None = None
    industry: str | None = None
    industry_level_1: str | None = None
    industry_level_2: str | None = None
    similarity: float = Field(ge=0.0, le=1.0)
    significant_drop_5d: bool
    return_vs_offer_5d_pct: float | None = None
    comparable_facts: dict[str, Any]


class LLMPredictionResult(BaseModel):
    """可直接落盘、回测和交给报告Agent的固定预测契约。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["1.1"] = "1.1"
    method: Literal["llm_historical_committee", "llm_single_judge"] = (
        "llm_historical_committee"
    )
    prompt_version: Literal[
        "llm-ipo-committee-v2.1", "llm-ipo-single-judge-v1"
    ] = PROMPT_VERSION
    company_id: str
    company_name: str
    as_of_date: date
    risk_score: int = Field(ge=0, le=100)
    prospectus_risk_severity: int = Field(ge=0, le=100)
    calibration_anchor_pct: float = Field(ge=0.0, le=100.0)
    risk_level: Literal["low", "medium", "high"]
    predicted_significant_drop_5d: bool
    alert_threshold: int = Field(ge=1, le=99)
    confidence: float = Field(ge=0.0, le=1.0)
    short_horizon_trigger_strength: ShortHorizonTriggerStrength
    short_horizon_triggers: tuple[str, ...] = ()
    long_horizon_risks: tuple[str, ...] = ()
    historical_case_drop_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_case_ids: tuple[str, ...]
    key_risk_factors: tuple[str, ...] = Field(min_length=1)
    positive_factors: tuple[str, ...] = ()
    dissenting_opinions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reasoning_summary: str = Field(min_length=20)
    bull_score: int | None = Field(default=None, ge=0, le=100)
    bear_score: int | None = Field(default=None, ge=0, le=100)
    model_name: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime

    @model_validator(mode="after")
    def score_matches_business_decision(self) -> LLMPredictionResult:
        expected_level = _risk_level(self.risk_score)
        if self.risk_level != expected_level:
            raise ValueError("risk_level is inconsistent with risk_score")
        if self.predicted_significant_drop_5d != (self.risk_score >= self.alert_threshold):
            raise ValueError("prediction flag is inconsistent with risk_score")
        return self


class _CommitteeState(TypedDict):
    payload: dict[str, Any]
    bull: NotRequired[CommitteeMemo]
    bear: NotRequired[CommitteeMemo]
    decision: NotRequired[CommitteeDecisionDraft]


class LLMPredictionSingleJudge:
    """只调用一次模型完成五日风险判断，保留与委员会一致的输入和输出契约。"""

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4_096,
        retry_policy: RetryPolicy | None = None,
        progress_callback: ProgressCallback | None = None,
        alert_threshold: int = BUSINESS_ALERT_THRESHOLD,
    ) -> None:
        if not 1 <= alert_threshold <= 99:
            raise ValueError("alert_threshold must be between 1 and 99")
        self.provider = provider
        self.model_name = provider.model_name
        self._prompt = prompt
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._retry = retry_policy or RetryPolicy(
            max_retries=2,
            initial_seconds=1.0,
            backoff_factor=2.0,
            max_seconds=8.0,
            retry_if=retry_model_node_error,
        )
        self._progress = progress_callback
        self._alert_threshold = alert_threshold

    def predict(self, payload: Mapping[str, Any]) -> LLMPredictionResult:
        clean_payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        _assert_no_current_target(clean_payload)
        self._emit("llm_prediction:single:started", "单一决策Agent正在判断五日下跌风险。")
        request = {
            "constraints": {
                "target": "上市后第5个交易日相对发行价收益率不高于-10%",
                "probability_majority_threshold": 50,
                "business_alert_threshold": self._alert_threshold,
                "risk_score_semantics": "事件概率百分数，不是招股书风险严重度",
                "calibration_anchor": clean_payload["calibration_context"],
                "no_new_facts": True,
                "allowed_evidence_ids": _evidence_ids(clean_payload),
                "allowed_historical_case_ids": [
                    item["company_id"]
                    for item in clean_payload.get("historical_cases", [])
                ],
            },
            "company_context": clean_payload,
            "required_output": CommitteeDecisionDraft.model_json_schema(),
        }
        allowed_evidence = set(_evidence_ids(clean_payload))
        allowed_cases = {
            item["company_id"] for item in clean_payload.get("historical_cases", [])
        }

        def run(feedback: str | None) -> CommitteeDecisionDraft:
            request_payload = dict(request)
            if feedback:
                request_payload["correction_feedback"] = feedback
            raw = self.provider.complete_json(
                system_prompt=self._prompt,
                user_prompt=json.dumps(request_payload, ensure_ascii=False),
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
            draft = CommitteeDecisionDraft.model_validate(raw)
            unknown_evidence = set(draft.evidence_ids) - allowed_evidence
            unknown_cases = set(draft.supporting_case_ids) - allowed_cases
            if unknown_evidence:
                raise ValueError(
                    f"decision cited unknown evidence: {sorted(unknown_evidence)}"
                )
            if unknown_cases:
                raise ValueError(
                    f"decision cited unknown historical cases: {sorted(unknown_cases)}"
                )
            return draft

        draft = self._retry.run_with_feedback(run).value
        self._emit("llm_prediction:single:completed", "单一决策Agent预测已形成。")
        return _build_prediction_result(
            clean_payload,
            draft,
            model_name=self.model_name,
            alert_threshold=self._alert_threshold,
            method="llm_single_judge",
            prompt_version=SINGLE_JUDGE_PROMPT_VERSION,
        )

    def _emit(self, event: str, detail: str) -> None:
        if self._progress is not None:
            self._progress(event, detail)


class LLMPredictionCommittee:
    """并行运行多头/空头Agent，再由决策Agent裁决。"""

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        bull_prompt: str,
        bear_prompt: str,
        decision_prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4_096,
        retry_policy: RetryPolicy | None = None,
        progress_callback: ProgressCallback | None = None,
        alert_threshold: int = BUSINESS_ALERT_THRESHOLD,
    ) -> None:
        if not 1 <= alert_threshold <= 99:
            raise ValueError("alert_threshold must be between 1 and 99")
        self.provider = provider
        self.model_name = provider.model_name
        self._bull_prompt = bull_prompt
        self._bear_prompt = bear_prompt
        self._decision_prompt = decision_prompt
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._retry = retry_policy or RetryPolicy(
            max_retries=2,
            initial_seconds=1.0,
            backoff_factor=2.0,
            max_seconds=8.0,
            retry_if=retry_model_node_error,
        )
        self._progress = progress_callback
        self._alert_threshold = alert_threshold
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(_CommitteeState)
        builder.add_node("bull_agent", self._bull_node)
        builder.add_node("bear_agent", self._bear_node)
        builder.add_node("decision_agent", self._decision_node)
        builder.add_edge(START, "bull_agent")
        builder.add_edge(START, "bear_agent")
        builder.add_edge(["bull_agent", "bear_agent"], "decision_agent")
        builder.add_edge("decision_agent", END)
        return builder.compile()

    def predict(self, payload: Mapping[str, Any]) -> LLMPredictionResult:
        clean_payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        _assert_no_current_target(clean_payload)
        state = self._graph.invoke({"payload": clean_payload})
        bull = state["bull"]
        bear = state["bear"]
        draft = state["decision"]
        known_evidence = {
            item["evidence_id"]
            for finding in clean_payload.get("agent_risks", [])
            for item in finding.get("evidence", [])
        }
        known_cases = {item["company_id"] for item in clean_payload.get("historical_cases", [])}
        unknown_evidence = set(draft.evidence_ids) - known_evidence
        unknown_cases = set(draft.supporting_case_ids) - known_cases
        if unknown_evidence:
            raise ValueError(f"decision cited unknown evidence: {sorted(unknown_evidence)}")
        if unknown_cases:
            raise ValueError(f"decision cited unknown historical cases: {sorted(unknown_cases)}")
        return _build_prediction_result(
            clean_payload,
            draft,
            model_name=self.model_name,
            alert_threshold=self._alert_threshold,
            method="llm_historical_committee",
            prompt_version=PROMPT_VERSION,
            bull_score=bull.risk_score,
            bear_score=bear.risk_score,
        )

    def _bull_node(self, state: _CommitteeState) -> dict[str, CommitteeMemo]:
        self._emit("llm_prediction:bull:started", "多头Agent正在寻找风险缓释因素。")
        memo = self._memo("bull", self._bull_prompt, state["payload"])
        self._emit("llm_prediction:bull:completed", "多头Agent意见已形成。")
        return {"bull": memo}

    def _bear_node(self, state: _CommitteeState) -> dict[str, CommitteeMemo]:
        self._emit("llm_prediction:bear:started", "空头Agent正在识别五日下跌触发因素。")
        memo = self._memo("bear", self._bear_prompt, state["payload"])
        self._emit("llm_prediction:bear:completed", "空头Agent意见已形成。")
        return {"bear": memo}

    def _decision_node(self, state: _CommitteeState) -> dict[str, CommitteeDecisionDraft]:
        self._emit("llm_prediction:decision:started", "决策Agent正在结合历史案例裁决。")
        payload = {
            "constraints": {
                "target": "上市后第5个交易日相对发行价收益率不高于-10%",
                "probability_majority_threshold": 50,
                "business_alert_threshold": self._alert_threshold,
                "risk_score_semantics": "事件概率百分数，不是招股书风险严重度",
                "calibration_anchor": state["payload"]["calibration_context"],
                "no_new_facts": True,
                "allowed_evidence_ids": _evidence_ids(state["payload"]),
                "allowed_historical_case_ids": [
                    item["company_id"] for item in state["payload"].get("historical_cases", [])
                ],
            },
            "company_context": state["payload"],
            "bull_opinion": state["bull"].model_dump(mode="json"),
            "bear_opinion": state["bear"].model_dump(mode="json"),
            "required_output": CommitteeDecisionDraft.model_json_schema(),
        }
        allowed_evidence = set(_evidence_ids(state["payload"]))
        allowed_cases = {
            item["company_id"] for item in state["payload"].get("historical_cases", [])
        }

        def validate_references(draft: CommitteeDecisionDraft) -> None:
            unknown_evidence = set(draft.evidence_ids) - allowed_evidence
            unknown_cases = set(draft.supporting_case_ids) - allowed_cases
            if unknown_evidence:
                raise ValueError(f"decision cited unknown evidence: {sorted(unknown_evidence)}")
            if unknown_cases:
                raise ValueError(f"decision cited unknown historical cases: {sorted(unknown_cases)}")

        draft = self._call_model(
            self._decision_prompt,
            payload,
            CommitteeDecisionDraft,
            validator=validate_references,
        )
        self._emit("llm_prediction:decision:completed", "决策Agent预测已形成。")
        return {"decision": draft}

    def _memo(
        self,
        stance: Literal["bull", "bear"],
        system_prompt: str,
        payload: dict[str, Any],
    ) -> CommitteeMemo:
        request = {
            "stance": stance,
            "target": "上市后第5个交易日相对发行价收益率不高于-10%",
            "company_context": payload,
            "required_output": CommitteeMemo.model_json_schema(),
        }
        allowed_evidence = set(_evidence_ids(payload))
        allowed_cases = {item["company_id"] for item in payload.get("historical_cases", [])}

        def validate_references(memo: CommitteeMemo) -> None:
            if memo.stance != stance:
                raise ValueError(f"{stance} agent returned another stance")
            unknown_evidence = set(memo.evidence_ids) - allowed_evidence
            unknown_cases = set(memo.historical_case_ids) - allowed_cases
            if unknown_evidence:
                raise ValueError(f"{stance} agent cited unknown evidence: {sorted(unknown_evidence)}")
            if unknown_cases:
                raise ValueError(f"{stance} agent cited unknown cases: {sorted(unknown_cases)}")

        return self._call_model(
            system_prompt,
            request,
            CommitteeMemo,
            validator=validate_references,
        )

    def _call_model(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        *,
        validator: Callable[[Any], None] | None = None,
    ) -> Any:
        def run(feedback: str | None) -> Any:
            request_payload = dict(payload)
            if feedback:
                request_payload["correction_feedback"] = feedback
            raw = self.provider.complete_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(request_payload, ensure_ascii=False),
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
            value = schema.model_validate(raw)
            if validator is not None:
                validator(value)
            return value

        return self._retry.run_with_feedback(run).value

    def _emit(self, event: str, detail: str) -> None:
        if self._progress is not None:
            self._progress(event, detail)


def _build_prediction_result(
    payload: Mapping[str, Any],
    draft: CommitteeDecisionDraft,
    *,
    model_name: str,
    alert_threshold: int,
    method: Literal["llm_historical_committee", "llm_single_judge"],
    prompt_version: Literal["llm-ipo-committee-v2.1", "llm-ipo-single-judge-v1"],
    bull_score: int | None = None,
    bear_score: int | None = None,
) -> LLMPredictionResult:
    cases = payload.get("historical_cases", [])
    drop_rate = (
        sum(bool(item["significant_drop_5d"]) for item in cases) / len(cases)
        if cases
        else None
    )
    identity = payload["company"]
    calibration = payload["calibration_context"]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return LLMPredictionResult(
        method=method,
        prompt_version=prompt_version,
        company_id=identity["company_id"],
        company_name=identity["company_name"],
        as_of_date=identity["prospectus_date"],
        risk_score=draft.risk_score,
        prospectus_risk_severity=draft.prospectus_risk_severity,
        calibration_anchor_pct=calibration["anchor_probability_pct"],
        risk_level=_risk_level(draft.risk_score),
        predicted_significant_drop_5d=draft.risk_score >= alert_threshold,
        alert_threshold=alert_threshold,
        confidence=draft.confidence,
        short_horizon_trigger_strength=draft.short_horizon_trigger_strength,
        short_horizon_triggers=draft.short_horizon_triggers,
        long_horizon_risks=draft.long_horizon_risks,
        historical_case_drop_rate=(round(drop_rate, 6) if drop_rate is not None else None),
        supporting_case_ids=draft.supporting_case_ids,
        key_risk_factors=draft.key_risk_factors,
        positive_factors=draft.positive_factors,
        dissenting_opinions=draft.dissenting_opinions,
        evidence_ids=draft.evidence_ids,
        reasoning_summary=draft.reasoning_summary,
        bull_score=bull_score,
        bear_score=bear_score,
        model_name=model_name,
        input_sha256=hashlib.sha256(encoded).hexdigest(),
        generated_at=datetime.now(timezone.utc),
    )


def build_llm_prediction_payload(
    row: Mapping[str, Any],
    *,
    report: RiskAnalysisReport | None,
    historical_cases: Sequence[HistoricalIPOCase],
    max_risks: int = 24,
    max_evidence_per_risk: int = 2,
) -> dict[str, Any]:
    """把训练宽表转换为LLM可读单位，不暴露待预测公司的标签。"""
    company_id = str(row.get("company_id") or "").zfill(5)
    normalized_industry = normalize_industry(row.get("industry_primary"))
    company = {
        "company_id": company_id,
        "company_name": _text(row.get("company_name")) or company_id,
        "industry": _text(row.get("industry_primary")),
        "industry_level_1": normalized_industry.level_1,
        "industry_level_2": normalized_industry.level_2,
        "prospectus_date": _iso_date(row.get("prospectus_date")).isoformat(),
    }
    offering = {
        "offer_price_low_hkd_per_share": _number(row.get("offering_offer_price_low_hkd")),
        "offer_price_high_hkd_per_share": _number(row.get("offering_offer_price_high_hkd")),
        "offering_share_count": _number(row.get("offering_share_count")),
        "gross_proceeds_mid_hkd_mn": _hkd_millions(row.get("offering_gross_proceeds_mid_hkd")),
        "net_proceeds_hkd_mn": _hkd_millions(row.get("offering_net_proceeds_hkd")),
        "market_cap_mid_hkd_mn": _hkd_millions(row.get("offering_market_cap_mid_hkd")),
        "offering_share_ratio_pct": _number(row.get("offering_share_ratio_pct")),
        "cornerstone_ratio_pct": _number(row.get("cornerstone_ratio_pct")),
        "cornerstone_lockup_months": _number(row.get("cornerstone_lockup_months")),
    }
    fundamentals = {
        "reporting_currency": _text(row.get("fundamental_reporting_currency")),
        "latest_revenue_hkd_mn": _number(row.get("fundamental_revenue_latest_hkd_mn")),
        "revenue_cagr_3y_pct": _number(row.get("fundamental_revenue_cagr_3y_pct")),
        "gross_margin_latest_pct": _number(row.get("fundamental_gross_margin_latest_pct")),
        "operating_cash_flow_latest_hkd_mn": _number(
            row.get("fundamental_operating_cash_flow_latest_hkd_mn")
        ),
        "operating_cash_flow_to_revenue_pct": _number(
            row.get("fundamental_operating_cash_flow_to_revenue_latest_pct")
        ),
        "latest_net_profit_native_mn": _number(row.get("fundamental_net_profit_latest_native")),
        "ending_cash_native_mn": _number(row.get("fundamental_ending_cash_latest_native")),
        "research_development_native_mn": _number(
            row.get("fundamental_research_development_latest_native")
        ),
        "customer_top5_ratio_pct": _number(row.get("customer_top5_ratio_pct")),
        "supplier_top5_ratio_pct": _number(row.get("supplier_top5_ratio_pct")),
        "pipeline_stage": _text(row.get("pipeline_stage_name")),
    }
    market = {
        "market_return_5d_pct": _number(row.get("market_return_5d_pct")),
        "market_return_20d_pct": _number(row.get("market_return_20d_pct")),
        "market_return_60d_pct": _number(row.get("market_return_60d_pct")),
        "market_volatility_20d_pct": _number(row.get("market_volatility_20d_pct")),
        "market_advance_ratio_20d": _number(row.get("market_advance_ratio_20d")),
        "recent_ipo_5d_drop_rate_180d": _number(row.get("recent_ipo_5d_drop_rate_180d")),
        "recent_ipo_5d_mean_return_180d_pct": _number(
            row.get("recent_ipo_5d_mean_return_180d_pct")
        ),
        "recent_ipo_observed_count_180d": _number(row.get("recent_ipo_5d_observed_count_180d")),
        "recent_ipo_count_30d": _number(row.get("recent_ipo_count_30d")),
    }
    derived = _derived_readable_features(offering, fundamentals)
    serialized_cases = [case.model_dump(mode="json") for case in historical_cases]
    payload = {
        "prediction_contract": {
            "as_of": "招股书发布日期",
            "target": "上市后第5个交易日相对发行价收益率<=-10%",
            "amount_unit": "million HKD unless a field explicitly says native_mn",
            "percentage_unit": "percentage points, e.g. 12.5 means 12.5%",
            "missing_value": None,
        },
        "company": company,
        "offering": offering,
        "fundamentals": fundamentals,
        "derived_ratios": derived,
        "prelisting_market": market,
        "calibration_context": _build_calibration_context(market, serialized_cases),
        "agent_risks": _risk_payload(
            report,
            max_risks=max_risks,
            max_evidence_per_risk=max_evidence_per_risk,
        ),
        "historical_cases": serialized_cases,
    }
    _assert_no_current_target(payload)
    return payload


def _build_calibration_context(
    market: Mapping[str, Any],
    historical_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造只使用上市前可知信息的概率锚点，约束LLM不要把风险严重度当概率。"""
    market_rate = _probability(market.get("recent_ipo_5d_drop_rate_180d"))
    market_count = _non_negative_int(market.get("recent_ipo_observed_count_180d"))

    usable_cases = [
        item
        for item in historical_cases
        if item.get("significant_drop_5d") is not None
        and _number(item.get("similarity")) is not None
    ]
    case_count = len(usable_cases)
    case_event_count = sum(bool(item["significant_drop_5d"]) for item in usable_cases)
    case_rate = case_event_count / case_count if case_count else None
    similarity_weight = sum(float(item["similarity"]) for item in usable_cases)
    weighted_case_rate = (
        sum(
            float(item["similarity"]) * float(bool(item["significant_drop_5d"]))
            for item in usable_cases
        )
        / similarity_weight
        if similarity_weight > 0
        else None
    )

    sources: list[tuple[float, float]] = []
    if market_rate is not None:
        # 近期IPO覆盖约30家公司，代表当前发行窗口；权重应明显高于仅8个近邻的结果。
        sources.append((market_rate, 4.0 * min(max(market_count, 1) / 30.0, 1.0)))
    if weighted_case_rate is not None:
        sources.append((weighted_case_rate, min(similarity_weight / 5.0, 1.0)))
    if sources:
        total_weight = sum(weight for _, weight in sources)
        anchor = sum(value * weight for value, weight in sources) / total_weight
    else:
        # 训练期历史基准约为四分之一；只有两类动态锚点都缺失时才使用保守降级值。
        anchor = 0.25

    return {
        "event_definition": "上市后第5个交易日相对发行价收益率<=-10%",
        "anchor_probability_pct": round(anchor * 100.0, 2),
        "market_base_rate_pct": (
            round(market_rate * 100.0, 2) if market_rate is not None else None
        ),
        "market_observation_count": market_count,
        "similar_case_event_rate_pct": round(case_rate * 100.0, 2) if case_rate is not None else None,
        "similar_case_weighted_event_rate_pct": (
            round(weighted_case_rate * 100.0, 2)
            if weighted_case_rate is not None
            else None
        ),
        "similar_case_count": case_count,
        "similar_case_event_count": case_event_count,
        "similar_case_non_event_count": case_count - case_event_count,
        "interpretation": (
            "先从anchor_probability_pct出发，再依据具有明确五日价格传导渠道的因素调整；"
            "一般经营或合规风险只影响prospectus_risk_severity。"
        ),
    }


def select_historical_cases(
    frame: pd.DataFrame,
    company_id: str,
    *,
    top_k: int = 8,
    case_store: PgVectorHistoricalCaseStore | None = None,
) -> tuple[HistoricalIPOCase, ...]:
    """混合选择历史案例；无案例库时保留确定性的数值降级路径。"""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    normalized_id = str(company_id).zfill(5)
    ids = frame["company_id"].astype(str).str.zfill(5)
    matches = frame.loc[ids == normalized_id]
    if matches.empty:
        raise ValueError(f"company does not exist in feature matrix: {normalized_id}")
    current = matches.sort_values("prospectus_date").iloc[-1]
    current_date = pd.to_datetime(current["prospectus_date"], errors="raise")
    if case_store is not None:
        hybrid = _select_hybrid_historical_cases(
            frame,
            current,
            normalized_id,
            current_date=current_date.date(),
            top_k=top_k,
            case_store=case_store,
        )
        if hybrid:
            return hybrid
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    target = pd.to_numeric(frame.get("target_significant_drop_5d"), errors="coerce")
    observation_column = next(
        (
            name
            for name in ("target_observation_date_5d", "target_observation_date")
            if name in frame.columns
        ),
        None,
    )
    if observation_column is not None:
        observation_dates = pd.to_datetime(frame[observation_column], errors="coerce")
        outcome_is_known = observation_dates < current_date
    else:
        observation_dates = pd.Series(pd.NaT, index=frame.index)
        outcome_is_known = dates < current_date
    candidates = frame.loc[
        (dates < current_date) & outcome_is_known & target.notna() & (ids != normalized_id)
    ].copy()
    if candidates.empty:
        return ()

    usable = [name for name in PEER_NUMERIC_FEATURES if name in frame.columns]
    if not usable:
        raise ValueError("feature matrix has no supported peer features")
    historical = frame.loc[dates < current_date, usable].apply(pd.to_numeric, errors="coerce")
    medians = historical.median(axis=0).fillna(0.0)
    scales = historical.quantile(0.75) - historical.quantile(0.25)
    scales = scales.where(scales.abs() > 1e-9, historical.std(axis=0))
    scales = scales.where(scales.abs() > 1e-9, 1.0).fillna(1.0)
    current_values = pd.to_numeric(current[usable], errors="coerce")
    candidate_values = candidates[usable].apply(pd.to_numeric, errors="coerce")
    current_missing = current_values.isna()
    current_filled = current_values.fillna(medians)
    candidate_missing = candidate_values.isna()
    candidate_filled = candidate_values.fillna(medians)
    distance = ((candidate_filled - current_filled) / scales).clip(-8, 8).pow(2).mean(axis=1)
    missing_disagreement = candidate_missing.ne(current_missing, axis=1).mean(axis=1)
    distance = distance + 0.35 * missing_disagreement
    industry = normalize_industry(current.get("industry_primary"))
    if industry and "industry_primary" in candidates:
        candidate_industries = candidates["industry_primary"].map(normalize_industry)
        industry_bonus = candidate_industries.map(lambda item: industry_similarity(industry, item))
        distance = distance * (1.0 - 0.28 * industry_bonus)
    candidates["_distance"] = distance
    candidates = candidates.sort_values(["_distance", "prospectus_date"]).head(top_k)

    cases: list[HistoricalIPOCase] = []
    for _, item in candidates.iterrows():
        comparable = {
            _display_name(name): _number(item.get(name))
            for name in usable
            if _number(item.get(name)) is not None
        }
        cases.append(
            HistoricalIPOCase(
                company_id=str(item["company_id"]).zfill(5),
                company_name=_text(item.get("company_name")) or str(item["company_id"]),
                prospectus_date=_iso_date(item["prospectus_date"]),
                outcome_available_at=(
                    _iso_date(item[observation_column]) if observation_column is not None else None
                ),
                industry=_text(item.get("industry_primary")),
                industry_level_1=normalize_industry(item.get("industry_primary")).level_1,
                industry_level_2=normalize_industry(item.get("industry_primary")).level_2,
                similarity=round(1.0 / (1.0 + math.sqrt(max(float(item["_distance"]), 0))), 6),
                significant_drop_5d=bool(int(float(item["target_significant_drop_5d"]))),
                return_vs_offer_5d_pct=_number(
                    item.get(
                        _first_existing_name(
                            item.index,
                            (
                                "target_return_vs_offer_pct_5d",
                                "target_return_vs_offer_5d_pct",
                            ),
                        )
                    )
                ),
                comparable_facts=comparable,
            )
        )
    return tuple(cases)


def _select_hybrid_historical_cases(
    frame: pd.DataFrame,
    current: pd.Series,
    normalized_id: str,
    *,
    current_date: date,
    top_k: int,
    case_store: PgVectorHistoricalCaseStore,
) -> tuple[HistoricalIPOCase, ...]:
    """语义、数值与行业三路融合；历史结果不参与任何一路相似度。"""
    current_industry = normalize_industry(current.get("industry_primary"))
    current_features = case_feature_payload(current)
    summary = build_case_summary(
        current,
        industry=current_industry,
        feature_payload=current_features,
    )
    semantic_matches = case_store.search_semantic(
        case_summary_text=summary,
        current_company_id=normalized_id,
        current_prospectus_date=current_date,
        pool_size=max(200, top_k * 30),
    )
    if not semantic_matches:
        return ()
    scales = robust_feature_scales(frame, before=current_date)
    ranked: list[tuple[float, HistoricalIPOCase]] = []
    for match in semantic_matches:
        record = match.record
        numeric_score = robust_numeric_similarity(
            current_features,
            record.feature_payload,
            scales=scales,
        )
        industry_score = industry_similarity(current_industry, record.industry)
        hybrid_score = (
            0.50 * numeric_score + 0.35 * match.semantic_similarity + 0.15 * industry_score
        )
        comparable = {
            _display_name(name): round(value, 8) for name, value in record.feature_payload.items()
        }
        case = HistoricalIPOCase(
            company_id=record.company_id,
            company_name=record.company_name,
            prospectus_date=record.prospectus_date,
            outcome_available_at=record.outcome_available_at,
            industry=record.industry.raw,
            industry_level_1=record.industry.level_1,
            industry_level_2=record.industry.level_2,
            similarity=round(float(np.clip(hybrid_score, 0, 1)), 6),
            significant_drop_5d=record.significant_drop_5d,
            return_vs_offer_5d_pct=record.return_vs_offer_5d_pct,
            comparable_facts=comparable,
        )
        ranked.append((hybrid_score, case))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1].prospectus_date.toordinal(),
            item[1].company_id,
        )
    )
    return tuple(case for _, case in ranked[:top_k])


def load_latest_risk_report(project_root: Path, company_id: str) -> RiskAnalysisReport | None:
    """读取该公司最新成功落盘的风险报告。"""
    normalized = str(company_id).zfill(5)
    candidates = list(
        (project_root / "artifacts" / "analysis").glob(f"{normalized}_*/**/risk_analysis.json")
    )
    if not candidates:
        return None
    reports: list[tuple[datetime, RiskAnalysisReport]] = []
    for path in candidates:
        try:
            report = RiskAnalysisReport.model_validate(read_json(path))
        except (OSError, ValueError):
            continue
        reports.append((report.generated_at, report))
    return max(reports, key=lambda item: item[0])[1] if reports else None


def _risk_payload(
    report: RiskAnalysisReport | None,
    *,
    max_risks: int,
    max_evidence_per_risk: int,
) -> list[dict[str, Any]]:
    if report is None:
        return []
    evidence_by_id = {item.evidence_id: item for item in report.evidences}
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    findings = sorted(
        report.risk_findings,
        key=lambda item: (severity_rank[item.severity.value], item.confidence),
        reverse=True,
    )[:max_risks]
    result: list[dict[str, Any]] = []
    for finding in findings:
        evidence = []
        for evidence_id in finding.evidence_ids[:max_evidence_per_risk]:
            source = evidence_by_id.get(evidence_id)
            if source is None:
                continue
            evidence.append(
                {
                    "evidence_id": source.evidence_id,
                    "page": source.page,
                    "printed_page_label": source.printed_page_label,
                    "evidence_type": source.evidence_type.value,
                    "quote": source.quote[:800],
                    "table_id": source.table_id,
                    "table_cells": dict(zip(source.table_cell_refs, source.table_cell_values)),
                }
            )
        result.append(
            {
                "finding_id": finding.finding_id,
                "risk_type": finding.risk_type,
                "title": finding.title,
                "owner": finding.created_by.value,
                "severity": finding.severity.value,
                "confidence": finding.confidence,
                "claim": finding.claim,
                "uncertainty": finding.uncertainty,
                "specialist_status": finding.specialist_status,
                "evidence": evidence,
            }
        )
    return result


def _derived_readable_features(
    offering: Mapping[str, Any], fundamentals: Mapping[str, Any]
) -> dict[str, float | None]:
    revenue = _number(fundamentals.get("latest_revenue_hkd_mn"))
    market_cap = _number(offering.get("market_cap_mid_hkd_mn"))
    gross = _number(offering.get("gross_proceeds_mid_hkd_mn"))
    net = _number(offering.get("net_proceeds_hkd_mn"))
    ocf = _number(fundamentals.get("operating_cash_flow_latest_hkd_mn"))
    return {
        "price_to_sales": _safe_ratio(market_cap, revenue),
        "gross_fundraising_to_revenue": _safe_ratio(gross, revenue),
        "net_proceeds_to_market_cap": _safe_ratio(net, market_cap),
        "operating_cash_flow_margin": _safe_ratio(ocf, revenue),
        "cash_replenishment_ratio": _safe_ratio(net, abs(ocf) if ocf is not None else None),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return round(numerator / denominator, 6)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 8) if math.isfinite(number) else None


def _probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    return None


def _non_negative_int(value: Any) -> int:
    number = _number(value)
    return max(int(number), 0) if number is not None else 0


def _hkd_millions(value: Any) -> float | None:
    number = _number(value)
    return round(number / 1_000_000.0, 8) if number is not None else None


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _first_existing_name(columns: Sequence[str], names: Sequence[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def _iso_date(value: Any) -> date:
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.date()


def _risk_level(score: int) -> Literal["low", "medium", "high"]:
    if score < BUSINESS_ALERT_THRESHOLD:
        return "low"
    if score < 50:
        return "medium"
    return "high"


def _evidence_ids(payload: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            item["evidence_id"]
            for finding in payload.get("agent_risks", [])
            for item in finding.get("evidence", [])
        )
    )


def _display_name(name: str) -> str:
    return name.removeprefix("fundamental_").removeprefix("offering_")


def _assert_no_current_target(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, default=str).lower()
    forbidden = (
        "target_significant_drop",
        "target_return_vs_offer",
        "target_max_drawdown",
        "target_broke_issue_price",
        "target_return_vs_first_close",
    )
    # historical_cases内允许出现定义明确的历史结果，其他区域一律禁止。
    without_history = dict(payload)
    without_history.pop("historical_cases", None)
    current_text = json.dumps(without_history, ensure_ascii=False, default=str).lower()
    if any(name in current_text for name in forbidden):
        raise ValueError("current company post-listing target leaked into LLM payload")
    if "target_" in serialized and "historical_cases" not in payload:
        raise ValueError("target field leaked into LLM payload")


__all__ = [
    "CommitteeMemo",
    "BUSINESS_ALERT_THRESHOLD",
    "HistoricalIPOCase",
    "LLMPredictionCommittee",
    "LLMPredictionResult",
    "LLMPredictionSingleJudge",
    "PEER_NUMERIC_FEATURES",
    "PROMPT_VERSION",
    "SINGLE_JUDGE_PROMPT_VERSION",
    "build_llm_prediction_payload",
    "load_latest_risk_report",
    "select_historical_cases",
]
