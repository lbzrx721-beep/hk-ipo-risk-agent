"""根据公司业务画像判断通用风险规则是否适用。"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.agents.analysis import JsonChatProvider
from ipo_risk.agents.prompting import prompt_audit, with_correction
from ipo_risk.retrieval import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


class RiskApplicability(BaseModel):
    """一条风险规则针对当前公司的适用性判断。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    applicable: bool
    rationale: str = Field(min_length=2)


class CompanyRiskScope(BaseModel):
    """公司业务画像及全部通用风险的适用范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    industry: str = Field(min_length=2)
    business_model: str = Field(min_length=2)
    revenue_model: str = Field(min_length=2)
    risks: tuple[RiskApplicability, ...]

    @model_validator(mode="after")
    def risk_types_must_be_unique(self) -> CompanyRiskScope:
        risk_types = [risk.risk_type for risk in self.risks]
        if len(risk_types) != len(set(risk_types)):
            raise ValueError("risk scope contains duplicate risk types")
        return self

    @property
    def applicable_risk_types(self) -> tuple[str, ...]:
        return tuple(risk.risk_type for risk in self.risks if risk.applicable)

    @property
    def excluded_risk_types(self) -> tuple[str, ...]:
        return tuple(risk.risk_type for risk in self.risks if not risk.applicable)


class RiskScopePlanner(Protocol):
    """外部模型驱动的公司风险适用范围规划器。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def plan(
        self,
        rules: Sequence[RiskRule],
        chunks: Sequence[DocumentChunk],
    ) -> CompanyRiskScope: ...


def _build_document_context(
    chunks: Sequence[DocumentChunk],
    *,
    max_chars: int,
) -> str:
    """优先使用业务画像片段，构造有明确长度上限的公司画像上下文。"""
    profile_terms = (
        "我们是",
        "我們是",
        "公司是",
        "公司為",
        "主要业务",
        "主要業務",
        "业务模式",
        "業務模式",
        "产品",
        "產品",
        "服务",
        "服務",
        "解决方案",
        "解決方案",
        "收入",
        "收益",
        "客户",
        "客戶",
        "行业",
        "行業",
    )

    def context_priority(chunk: DocumentChunk) -> tuple[int, int, str]:
        normalized = "".join(chunk.text.split())
        profile_score = sum(term in normalized for term in profile_terms)
        return (-profile_score, chunk.page, chunk.chunk_id)

    parts: list[str] = []
    used = 0
    for chunk in sorted(chunks, key=context_priority):
        text = " ".join(chunk.text.split())
        if not text:
            continue
        part = (
            f"[{chunk.chunk_id} | PDF第{chunk.page}页"
            f" | 印刷页{chunk.printed_page_label or '-'}] {text}\n"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(part) > remaining:
            parts.append(part[:remaining])
            break
        parts.append(part)
        used += len(part)
    return "".join(parts)


class DeepSeekCompanyRiskScopePlanner:
    """识别公司画像，并排除动态候选中明显跨行业或主体错配的风险。"""

    provider_name = "deepseek"
    prompt_version = "company-risk-scope-v4"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        max_context_chars: int = 16_000,
        system_prompt: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        if max_context_chars < 200:
            raise ValueError("risk scoping context must contain at least 200 chars")
        self.provider = provider
        self.model_name = provider.model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_context_chars = max_context_chars
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._system_prompt = system_prompt or (
            "你是港股IPO公司画像与风险适用性核验Agent。风险候选已经由全文动态发现，"
            "你需要根据当前招股书识别行业、业务和收入模式，并排除明显属于其他公司、"
            "其他行业或错误主体的候选。此处不重新发明风险主题，也不判断证据强弱。必须"
            "覆盖输入的每个risk_type，只输出JSON对象。"
        )

    def prompt_audit(self) -> dict[str, str]:
        """返回公司画像与范围建议提示词的审计信息。"""
        return prompt_audit(
            prompt=self._system_prompt,
            version=self.prompt_version,
        )

    def plan(
        self,
        rules: Sequence[RiskRule],
        chunks: Sequence[DocumentChunk],
    ) -> CompanyRiskScope:
        return self.plan_with_feedback(
            rules,
            chunks,
            correction_feedback=None,
        )

    def plan_with_feedback(
        self,
        rules: Sequence[RiskRule],
        chunks: Sequence[DocumentChunk],
        *,
        correction_feedback: str | None,
    ) -> CompanyRiskScope:
        """生成公司画像和纳入建议，并支持校验错误反馈纠正。"""
        selected_rules = tuple(rules)
        if not selected_rules:
            raise ValueError("risk scoping requires at least one risk rule")
        if not chunks:
            raise ValueError("risk scoping requires prospectus chunks")
        first_chunk = chunks[0]
        payload: dict[str, Any] = with_correction(
            {
                "document_identity": {
                    "company_id": first_chunk.company_id,
                    "document_id": first_chunk.document_id,
                    "source_file": first_chunk.source_file,
                    "available_at": first_chunk.available_at.isoformat(),
                },
                "document_context": _build_document_context(
                    chunks,
                    max_chars=self._max_context_chars,
                ),
                "candidate_risks": [
                    {
                        "risk_type": rule.id,
                        "risk_name": rule.name,
                        "owner": rule.owner.value,
                        "default_severity": rule.severity.value,
                        "review_question": rule.claim,
                        "evidence_fields": list(rule.evidence_fields),
                    }
                    for rule in selected_rules
                ],
                "requirements": {
                    "cover_every_risk_type": True,
                    "topics_already_derived_from_document": True,
                    "exclude_only_clear_company_or_industry_mismatch": True,
                    "output_fields": [
                        "industry",
                        "business_model",
                        "revenue_model",
                        "risks",
                    ],
                },
            },
            correction_feedback,
        )
        scope = CompanyRiskScope.model_validate(
            self.provider.complete_json(
                system_prompt=self._system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
        )
        expected = {rule.id for rule in selected_rules}
        returned = {risk.risk_type for risk in scope.risks}
        if returned != expected:
            raise ValueError(
                "risk scope planner must cover every candidate risk type: "
                f"expected={sorted(expected)}, returned={sorted(returned)}"
            )
        return scope


__all__ = [
    "CompanyRiskScope",
    "DeepSeekCompanyRiskScopePlanner",
    "RiskApplicability",
    "RiskScopePlanner",
]
