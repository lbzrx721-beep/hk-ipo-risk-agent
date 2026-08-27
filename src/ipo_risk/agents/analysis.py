"""基于候选原文的结构化风险分析及DeepSeek适配层。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.agents.prompting import compose_prompt, prompt_audit, with_correction
from ipo_risk.agents.usage import ModelPricing, ModelUsageLedger
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


TextDeltaCallback = Callable[[str], None]
ProgressCallback = Callable[[str, str], None]


class JsonResponseDecodeError(ValueError):
    """保留模型原始文本，供特定业务节点恢复已完整生成的结构化对象。"""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


def _parse_json_object(content: str) -> dict[str, Any]:
    """兼容未启用 response_format 时常见的Markdown代码围栏。"""
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise JsonResponseDecodeError(
            f"chat API returned invalid JSON: {exc}",
            raw_content=content,
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("chat API JSON content must be an object")
    return parsed


class JsonChatProvider(Protocol):
    """结构化大模型调用需要实现的最小接口。"""

    model_name: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> dict[str, Any]:
        """返回已经解析的JSON对象。"""
        ...


class OpenAICompatibleJsonChatProvider:
    """通过OpenAI兼容ChatCompletions接口请求严格JSON。"""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        provider_name: str = "openai_compatible",
        request_extra_body: Mapping[str, Any] | None = None,
        supports_json_object: bool = True,
        reasoning_effort: str | None = None,
        reasoning_progress_callback: Callable[[], None] | None = None,
        usage_ledger: ModelUsageLedger | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        if client is None:
            if not api_key or not base_url:
                raise ValueError("api_key and base_url are required without a client")
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model_name = model_name
        self.provider_name = provider_name
        self._client = client
        self._request_extra_body = (
            dict(request_extra_body)
            if request_extra_body is not None
            else {"thinking": {"type": "disabled"}}
        )
        self._supports_json_object = supports_json_object
        self._reasoning_effort = reasoning_effort
        self._reasoning_progress_callback = reasoning_progress_callback
        self._usage_ledger = usage_ledger
        self._pricing = pricing

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if self._supports_json_object:
            request["response_format"] = {"type": "json_object"}
        if self._request_extra_body:
            request["extra_body"] = self._request_extra_body
        thinking = self._request_extra_body.get("thinking")
        thinking_disabled = (
            isinstance(thinking, Mapping)
            and thinking.get("type") == "disabled"
        )
        if self._reasoning_effort is not None and not thinking_disabled:
            request["reasoning_effort"] = self._reasoning_effort
        completions: Any = self._client.chat.completions
        ledger = self._usage_ledger
        if ledger is not None:
            ledger.begin_call(model_name=self.model_name, request_tag="json_chat")
        usage_source: Any | None = None
        try:
            if on_text_delta is not None:
                stream = completions.create(
                    **request,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                fragments: list[str] = []
                pending_fragments: list[str] = []
                pending_length = 0
                reasoning_reported = False
                for event in stream:
                    if getattr(event, "usage", None) is not None:
                        usage_source = event
                    choices = getattr(event, "choices", ())
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    reasoning_content = getattr(
                        delta,
                        "reasoning_content",
                        None,
                    )
                    if (
                        not reasoning_reported
                        and isinstance(reasoning_content, str)
                        and reasoning_content
                        and self._reasoning_progress_callback is not None
                    ):
                        self._reasoning_progress_callback()
                        reasoning_reported = True
                    content = getattr(delta, "content", None)
                    if not isinstance(content, str) or not content:
                        continue
                    fragments.append(content)
                    pending_fragments.append(content)
                    pending_length += len(content)
                    if pending_length >= 12 or "\n" in content:
                        on_text_delta("".join(pending_fragments))
                        pending_fragments.clear()
                        pending_length = 0
                if pending_fragments:
                    on_text_delta("".join(pending_fragments))
                content = "".join(fragments)
            else:
                response = completions.create(**request)
                usage_source = response
                content = response.choices[0].message.content
        except Exception:
            if ledger is not None:
                ledger.fail_call()
            raise
        if ledger is not None:
            input_tokens, output_tokens, cached_tokens = _extract_usage(usage_source)
            ledger.complete_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                pricing=self._pricing,
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat API returned empty JSON content")
        return _parse_json_object(content)


def _extract_usage(source: Any | None) -> tuple[int, int, int]:
    """兼容 OpenAI、DeepSeek 和其他 OpenAI 兼容响应中的 usage 字段。"""
    usage = getattr(source, "usage", None)
    if usage is None and isinstance(source, Mapping):
        usage = source.get("usage")
    if usage is None:
        return (0, 0, 0)

    def value(name: str) -> int:
        raw = usage.get(name, 0) if isinstance(usage, Mapping) else getattr(usage, name, 0)
        return int(raw or 0)

    details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage, Mapping)
        else getattr(usage, "prompt_tokens_details", None)
    )
    if isinstance(details, Mapping):
        cached = int(details.get("cached_tokens", 0) or 0)
    else:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return (value("prompt_tokens"), value("completion_tokens"), cached)


class EvidenceAnalysis(BaseModel):
    """模型对一条风险规则和候选证据作出的受约束判断。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    supported: bool
    claim: str = ""
    severity: RiskSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: tuple[str, ...] = ()
    uncertainty: str = ""

    @model_validator(mode="after")
    def validate_grounding(self) -> EvidenceAnalysis:
        if len(set(self.evidence_chunk_ids)) != len(self.evidence_chunk_ids):
            raise ValueError("evidence_chunk_ids must not contain duplicates")
        if self.supported and (
            not self.claim or not self.evidence_chunk_ids or not self.uncertainty
        ):
            raise ValueError("a supported analysis requires a claim, evidence and uncertainty")
        if not self.supported and (self.claim or self.evidence_chunk_ids):
            raise ValueError("an unsupported analysis cannot include a claim or accepted evidence")
        return self


class DeepSeekEvidenceAnalyzer:
    """要求DeepSeek只根据候选原文返回可验证的风险判断。"""

    provider_name = "deepseek"
    prompt_version = "evidence-analysis-v4"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str | None = None,
        role_prompts: Mapping[AgentRole, str] | None = None,
        prompt_version: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = provider.model_name
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._progress_callback = progress_callback
        self._role_prompts = dict(role_prompts or {})
        self._system_prompt = system_prompt or (
            "你是港股IPO招股书风险分析员。只能使用用户提供的候选原文，不得使用外部知识，"
            "不得编造数字、页码或证据。请输出JSON对象，字段严格为supported、claim、"
            "severity、confidence、evidence_chunk_ids、uncertainty。severity只能是low、"
            "medium、high、critical；confidence范围为0到1。证据不足时supported必须为false，"
            "claim和evidence_chunk_ids必须为空。输入中的review_question只是待核验问题，"
            "不得照抄为结论；必须同时检查候选原文中的反证、终止条款、改善趋势和限定条件。"
        )

    def prompt_audit(self, role: AgentRole) -> dict[str, str]:
        """返回该专业角色实际收到的提示词版本、指纹和快照。"""
        prompt = compose_prompt(self._system_prompt, self._role_prompts.get(role))
        return prompt_audit(prompt=prompt, version=self.prompt_version)

    def analyze(
        self,
        rule: RiskRule,
        candidates: Sequence[DocumentChunk],
    ) -> EvidenceAnalysis:
        return self.analyze_with_feedback(
            rule,
            candidates,
            correction_feedback=None,
        )

    def analyze_with_feedback(
        self,
        rule: RiskRule,
        candidates: Sequence[DocumentChunk],
        *,
        correction_feedback: str | None,
    ) -> EvidenceAnalysis:
        """根据候选原文分析，并在重试时接收上一轮结构校验反馈。"""
        if not candidates:
            raise ValueError("DeepSeek analysis requires at least one candidate")
        first_chunk = candidates[0]
        payload = with_correction(
            {
                "document_identity": {
                    "company_id": first_chunk.company_id,
                    "document_id": first_chunk.document_id,
                    "source_file": first_chunk.source_file,
                    "available_at": first_chunk.available_at.isoformat(),
                },
                "risk": {
                    "id": rule.id,
                    "name": rule.name,
                    "owner": rule.owner.value,
                    "default_severity": rule.severity.value,
                    "review_question": rule.claim,
                    "evidence_fields": list(rule.evidence_fields),
                    "search_queries": list(rule.search_queries),
                },
                "constraints": {
                    "allowed_evidence_chunk_ids": [chunk.chunk_id for chunk in candidates],
                    "must_consider_counterevidence": True,
                    "claim_must_be_derived_from_evidence": True,
                    "review_question_is_not_a_company_fact": True,
                },
                "candidates": [
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
                    for chunk in candidates
                ],
            },
            correction_feedback,
        )
        on_text_delta: TextDeltaCallback | None = None
        progress_callback = self._progress_callback
        if progress_callback is not None:

            def report_delta(delta: str) -> None:
                progress_callback(f"agent:{rule.owner.value}:delta", delta)

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": compose_prompt(
                self._system_prompt,
                self._role_prompts.get(rule.owner),
            ),
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        raw = self.provider.complete_json(
            **request,
        )
        result = EvidenceAnalysis.model_validate(raw)
        allowed = {chunk.chunk_id for chunk in candidates}
        unknown = set(result.evidence_chunk_ids) - allowed
        if unknown:
            raise ValueError(f"analysis cited unknown evidence chunks: {sorted(unknown)}")
        return result


__all__ = [
    "DeepSeekEvidenceAnalyzer",
    "EvidenceAnalysis",
    "JsonResponseDecodeError",
    "JsonChatProvider",
    "OpenAICompatibleJsonChatProvider",
    "ProgressCallback",
    "TextDeltaCallback",
]
