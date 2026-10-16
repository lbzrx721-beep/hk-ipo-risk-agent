"""基于专业 Agent 提案形成最终、可追踪的风险裁决建议。"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ipo_risk.agents.analysis import JsonChatProvider, TextDeltaCallback
from ipo_risk.agents.prompting import prompt_audit, with_correction
from ipo_risk.domain import AgentRole, RiskSeverity


class DecisionVerdict(BaseModel):
    """决策模型对同一风险的一组角色提案给出的整合建议。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    accepted: bool
    claim: str = ""
    severity: RiskSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: tuple[str, ...] = ()
    uncertainty: str = ""
    rationale: str = Field(min_length=1)
    source_proposal_ids: tuple[str, ...] = ()
    needs_debate: bool = False
    challenge: str = ""
    challenge_role: AgentRole | None = None
    challenge_queries: tuple[str, ...] = Field(default=(), max_length=4)

    @field_validator("challenge_queries")
    @classmethod
    def challenge_queries_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("decision challenge_queries must be unique")
        return value

    @model_validator(mode="after")
    def validate_grounded_verdict(self) -> DecisionVerdict:
        if len(set(self.evidence_chunk_ids)) != len(self.evidence_chunk_ids):
            raise ValueError("decision evidence_chunk_ids must be unique")
        if len(set(self.source_proposal_ids)) != len(self.source_proposal_ids):
            raise ValueError("decision source_proposal_ids must be unique")
        if not self.source_proposal_ids:
            raise ValueError("every decision must cite at least one source proposal")
        if self.accepted and (
            not self.claim or not self.evidence_chunk_ids or not self.uncertainty
        ):
            raise ValueError(
                "an accepted decision requires claim, evidence, uncertainty and sources"
            )
        if not self.accepted and self.evidence_chunk_ids:
            raise ValueError("a rejected decision cannot cite accepted evidence")
        if self.needs_debate and (
            not self.challenge or self.challenge_role is None or not self.challenge_queries
        ):
            raise ValueError(
                "a debate request requires challenge, challenge_role and challenge_queries"
            )
        if not self.needs_debate and (
            self.challenge or self.challenge_role is not None or self.challenge_queries
        ):
            raise ValueError("debate fields are only allowed when needs_debate is true")
        return self


class DecisionBatch(BaseModel):
    """一次裁决调用返回的所有风险判断。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[DecisionVerdict, ...]

    @model_validator(mode="after")
    def risk_types_must_be_unique(self) -> DecisionBatch:
        risk_types = [decision.risk_type for decision in self.decisions]
        if len(risk_types) != len(set(risk_types)):
            raise ValueError("decision risk_types must be unique")
        return self


class DecisionAnalyzer(Protocol):
    """外部裁决模型需要实现的最小接口。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def adjudicate(self, proposals: Sequence[Any]) -> DecisionBatch:
        """对角色提案作出最终判断。"""
        ...


class DeepSeekDecisionAnalyzer:
    """要求模型仅基于专业提案及其已选原文给出最终整合建议。"""

    provider_name = "deepseek"
    prompt_version = "decision-adjudication-v4"

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
        self.provider_name = str(getattr(provider, "provider_name", "deepseek"))
        self.model_name = provider.model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._progress_callback = progress_callback
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._system_prompt = system_prompt or (
            "你是港股IPO风险决策Agent，逐项复核专业Agent提案并形成最终机器建议。"
            "只能使用专业Agent已选中的招股书原文，不得引入外部事实或新的证据ID。"
            "accepted=true表示纳入正式风险，accepted=false表示机器最终不纳入；"
            "原始提案仍保留在审计记录中。第一轮遇到实质冲突时用needs_debate、"
            "challenge、challenge_role和challenge_queries发起定向补证；补证后的最终轮"
            "不得再次发起辩论。uncertainty只描述证据边界，不得把任务交给人工。"
        )

    def prompt_audit(self) -> dict[str, str]:
        """返回决策节点实际使用的提示词审计信息。"""
        return prompt_audit(
            prompt=self._system_prompt,
            version=self.prompt_version,
        )

    def adjudicate(self, proposals: Sequence[Any]) -> DecisionBatch:
        return self.adjudicate_with_feedback(
            proposals,
            correction_feedback=None,
        )

    def adjudicate_with_feedback(
        self,
        proposals: Sequence[Any],
        *,
        correction_feedback: str | None,
        stream_marker: str = "",
    ) -> DecisionBatch:
        """整合当前决策批次，并支持把上一轮校验错误反馈给模型。"""
        if not proposals:
            return DecisionBatch(decisions=())
        proposal_payloads: list[dict[str, Any]] = []
        proposal_by_id: dict[str, Any] = {}
        expected_risk_types: set[str] = set()
        for proposal in proposals:
            proposal_by_id[proposal.proposal_id] = proposal
            analyzed = proposal.analyzed_risk
            expected_risk_types.add(analyzed.rule.id)
            evidence = [
                {
                    "chunk_id": chunk.chunk_id,
                    "physical_page": chunk.page,
                    "printed_page_label": chunk.printed_page_label,
                    "source_kind": chunk.chunk_kind,
                    "table_id": chunk.table_id,
                    "table_cell_refs": list(chunk.table_cell_refs),
                    "table_cells": dict(
                        zip(
                            chunk.table_cell_refs,
                            chunk.table_cell_values,
                            strict=True,
                        )
                    ),
                    "quote": chunk.text,
                }
                for chunk in analyzed.selected_chunks
            ]
            proposal_payloads.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "agent": proposal.agent.value,
                    "risk_type": analyzed.rule.id,
                    "risk_name": analyzed.rule.name,
                    "supported": analyzed.analysis.supported,
                    "claim": analyzed.analysis.claim,
                    "severity": analyzed.analysis.severity.value,
                    "confidence": analyzed.analysis.confidence,
                    "uncertainty": analyzed.analysis.uncertainty,
                    "evidence": evidence,
                }
            )
        allowed_evidence = list(
            dict.fromkeys(
                chunk.chunk_id
                for proposal in proposals
                for chunk in proposal.analyzed_risk.selected_chunks
            )
        )
        payload = with_correction(
            {
                "constraints": {
                    "required_risk_types": sorted(expected_risk_types),
                    "allowed_proposal_ids": list(proposal_by_id),
                    "allowed_evidence_chunk_ids": allowed_evidence,
                    "no_external_evidence": True,
                },
                "proposals": proposal_payloads,
            },
            correction_feedback,
        )
        on_text_delta: TextDeltaCallback | None = None
        progress_callback = self._progress_callback
        if progress_callback is not None:

            def report_delta(delta: str) -> None:
                progress_callback("agent:decision:delta", f"{stream_marker}{delta}")

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": self._system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        raw = self.provider.complete_json(**request)
        batch = DecisionBatch.model_validate(raw)
        returned_risk_types = {decision.risk_type for decision in batch.decisions}
        if returned_risk_types != expected_risk_types:
            raise ValueError(
                "decision must cover every submitted risk_type: "
                f"expected={sorted(expected_risk_types)}, "
                f"returned={sorted(returned_risk_types)}"
            )
        for decision in batch.decisions:
            unknown_proposals = set(decision.source_proposal_ids) - set(proposal_by_id)
            if unknown_proposals:
                raise ValueError(f"decision cited unknown proposals: {sorted(unknown_proposals)}")
            cited = [proposal_by_id[item] for item in decision.source_proposal_ids]
            if any(proposal.analyzed_risk.rule.id != decision.risk_type for proposal in cited):
                raise ValueError("decision cited a proposal for another risk_type")
            allowed_for_decision = {
                chunk.chunk_id
                for proposal in cited
                for chunk in proposal.analyzed_risk.selected_chunks
            }
            unknown_evidence = set(decision.evidence_chunk_ids) - allowed_for_decision
            if unknown_evidence:
                raise ValueError(
                    f"decision cited unknown evidence chunks: {sorted(unknown_evidence)}"
                )
        return batch


__all__ = [
    "DecisionAnalyzer",
    "DecisionBatch",
    "DecisionVerdict",
    "DeepSeekDecisionAnalyzer",
]
