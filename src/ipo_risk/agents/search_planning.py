"""专业Agent主动规划招股书语义检索问题的契约。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.agents.analysis import JsonChatProvider, TextDeltaCallback
from ipo_risk.domain import AgentRole
from ipo_risk.retrieval.search import RiskRule


class PlannedRiskSearch(BaseModel):
    """专业Agent为一类风险生成的一组语义搜索问题。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    queries: tuple[str, ...] = Field(min_length=1, max_length=8)
    coverage_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def queries_must_be_unique(self) -> PlannedRiskSearch:
        if len(self.queries) != len(set(self.queries)):
            raise ValueError("planned search queries must be unique")
        return self


class RoleSearchPlan(BaseModel):
    """一个专业Agent对其全部风险职责生成的搜索计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: AgentRole
    plans: tuple[PlannedRiskSearch, ...]

    @model_validator(mode="after")
    def risk_types_must_be_unique(self) -> RoleSearchPlan:
        risk_types = [plan.risk_type for plan in self.plans]
        if len(risk_types) != len(set(risk_types)):
            raise ValueError("planned risk types must be unique")
        return self

    def queries_by_risk(self) -> dict[str, tuple[str, ...]]:
        return {plan.risk_type: plan.queries for plan in self.plans}


class RoleSearchPlanner(Protocol):
    """角色Agent搜索规划器的最小可替换接口。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def plan(
        self,
        rules: Sequence[RiskRule],
        *,
        feedback: Mapping[str, Any] | None = None,
    ) -> RoleSearchPlan:
        """为同一角色拥有的风险生成语义检索问题。"""
        ...


class DeepSeekRoleSearchPlanner:
    """让专业Agent根据职责自行生成语义检索问题。"""

    provider_name = "deepseek"
    prompt_version = "role-search-planning-v1"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str | None = None,
        prompt_version: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = provider.model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._progress_callback = progress_callback
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._system_prompt = system_prompt or (
            "你是港股IPO招股书专业搜索Agent。请根据角色职责和风险定义，自主生成"
            "能够从招股书语义向量索引中发现证据的中文搜索问题。不要依赖固定关键词。"
            "每类风险生成3至6个互补问题，覆盖直接披露、同义表达、表格说明和交叉引用。"
            "只输出JSON对象，字段为role和plans；plans中每项包含risk_type、queries和"
            "coverage_note。coverage_note只描述覆盖范围，不输出思维链。"
        )

    def plan(
        self,
        rules: Sequence[RiskRule],
        *,
        feedback: Mapping[str, Any] | None = None,
    ) -> RoleSearchPlan:
        selected_rules = tuple(rules)
        if not selected_rules:
            raise ValueError("search planning requires at least one risk rule")
        role = selected_rules[0].owner
        if any(rule.owner is not role for rule in selected_rules):
            raise ValueError("one search plan can only contain one Agent role")
        payload: dict[str, Any] = {
            "role": role.value,
            "responsibilities": [
                {
                    "risk_type": rule.id,
                    "risk_name": rule.name,
                    "default_severity": rule.severity.value,
                    "risk_definition": rule.claim,
                    "evidence_fields": list(rule.evidence_fields),
                }
                for rule in selected_rules
            ],
            "requirements": {
                "queries_per_risk": "3-6",
                "semantic_questions_only": True,
                "cover_every_risk_type": True,
                "output_json": True,
            },
        }
        if feedback:
            payload["previous_search_feedback"] = dict(feedback)

        on_text_delta: TextDeltaCallback | None = None
        progress_callback = self._progress_callback
        if progress_callback is not None:

            def report_delta(delta: str) -> None:
                progress_callback(f"agent:{role.value}:delta", delta)

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": self._system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        plan = RoleSearchPlan.model_validate(self.provider.complete_json(**request))
        if plan.role is not role:
            raise ValueError("search planner returned another Agent role")
        expected = {rule.id for rule in selected_rules}
        returned = set(plan.queries_by_risk())
        if returned != expected:
            raise ValueError(
                "search planner must cover every assigned risk type: "
                f"expected={sorted(expected)}, returned={sorted(returned)}"
            )
        return plan


__all__ = [
    "PlannedRiskSearch",
    "DeepSeekRoleSearchPlanner",
    "RoleSearchPlan",
    "RoleSearchPlanner",
]
