"""基于候选原文的结构化风险分析及DeepSeek适配层。"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.domain import RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


TextDeltaCallback = Callable[[str], None]
ProgressCallback = Callable[[str, str], None]


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
    ) -> None:
        if client is None:
            if not api_key or not base_url:
                raise ValueError("api_key and base_url are required without a client")
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model_name = model_name
        self._client = client

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
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        completions: Any = self._client.chat.completions
        if on_text_delta is not None:
            stream = completions.create(
                **request,
                stream=True,
            )
            fragments: list[str] = []
            pending_fragments: list[str] = []
            pending_length = 0
            for event in stream:
                choices = getattr(event, "choices", ())
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
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
            content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat API returned empty JSON content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("chat API JSON content must be an object")
        return parsed


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
            raise ValueError(
                "a supported analysis requires a claim, evidence and uncertainty"
            )
        if not self.supported and self.evidence_chunk_ids:
            raise ValueError("an unsupported analysis cannot cite evidence")
        return self


class DeepSeekEvidenceAnalyzer:
    """要求DeepSeek只根据候选原文返回可验证的风险判断。"""

    provider_name = "deepseek"
    prompt_version = "evidence-analysis-v1"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str | None = None,
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
        self._system_prompt = system_prompt or (
            "你是港股IPO招股书风险分析员。只能使用用户提供的候选原文，不得使用外部知识，"
            "不得编造数字、页码或证据。请输出JSON对象，字段严格为supported、claim、"
            "severity、confidence、evidence_chunk_ids、uncertainty。severity只能是low、"
            "medium、high、critical；confidence范围为0到1。证据不足时supported必须为false，"
            "claim和evidence_chunk_ids必须为空。"
        )

    def analyze(
        self,
        rule: RiskRule,
        candidates: Sequence[DocumentChunk],
    ) -> EvidenceAnalysis:
        if not candidates:
            raise ValueError("DeepSeek analysis requires at least one candidate")
        payload = {
            "risk": {
                "id": rule.id,
                "name": rule.name,
                "owner": rule.owner.value,
                "default_severity": rule.severity.value,
                "candidate_claim": rule.claim,
                "evidence_fields": list(rule.evidence_fields),
                "search_queries": list(rule.search_queries),
            },
            "constraints": {
                "allowed_evidence_chunk_ids": [chunk.chunk_id for chunk in candidates],
                "max_evidence": rule.max_evidence,
            },
            "candidates": [
                {
                    "chunk_id": chunk.chunk_id,
                    "physical_page": chunk.page,
                    "printed_page_label": chunk.printed_page_label,
                    "quote": chunk.text,
                }
                for chunk in candidates
            ],
        }
        on_text_delta: TextDeltaCallback | None = None
        progress_callback = self._progress_callback
        if progress_callback is not None:

            def report_delta(delta: str) -> None:
                progress_callback(f"agent:{rule.owner.value}:delta", delta)

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": self._system_prompt,
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
        if len(result.evidence_chunk_ids) > rule.max_evidence:
            raise ValueError("analysis cited more evidence than the risk rule allows")
        return result


__all__ = [
    "DeepSeekEvidenceAnalyzer",
    "EvidenceAnalysis",
    "JsonChatProvider",
    "OpenAICompatibleJsonChatProvider",
    "ProgressCallback",
    "TextDeltaCallback",
]
