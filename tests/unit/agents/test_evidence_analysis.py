from __future__ import annotations

from datetime import date
import json
from types import SimpleNamespace

import pytest

from ipo_risk.agents.analysis import (
    DeepSeekEvidenceAnalyzer,
    OpenAICompatibleJsonChatProvider,
)
from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryPolicy, retry_model_node_error


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        source_hash="a" * 64,
        page=30,
        printed_page_label="21",
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


def _rule() -> RiskRule:
    return RiskRule(
        id="cash_burn",
        name="现金消耗与融资依赖",
        owner=AgentRole.FINANCE,
        severity=RiskSeverity.HIGH,
        query="经营活动所用现金净额 负现金流量 研发活动",
        required_keywords=("負現金流量", "研發活動"),
        claim="公司在业绩记录期持续产生经营现金净流出。",
    )


class StubJsonChatProvider:
    model_name = "deepseek-v4-flash"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, object]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return {
            "supported": True,
            "claim": "公司持续产生经营现金净流出，且主要用于研发活动。",
            "severity": "high",
            "confidence": 0.91,
            "evidence_chunk_ids": ["C1"],
            "uncertainty": "未来融资规模仍取决于研发进展。",
        }


def test_deepseek_analyzer_returns_evidence_grounded_structured_result() -> None:
    provider = StubJsonChatProvider()
    analyzer = DeepSeekEvidenceAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
    )
    candidate = _chunk("C1", "我們的經營產生負現金流量，現金流出主要來自研發活動。")

    result = analyzer.analyze(_rule(), [candidate])

    assert result.supported
    assert result.evidence_chunk_ids == ("C1",)
    assert result.severity is RiskSeverity.HIGH
    assert result.confidence == 0.91
    assert "C1" in provider.user_prompt
    assert candidate.text in provider.user_prompt
    assert "JSON" in provider.system_prompt
    payload = json.loads(provider.user_prompt)
    assert payload["risk"]["review_question"] == _rule().claim
    assert "candidate_claim" not in payload["risk"]
    assert payload["constraints"]["must_consider_counterevidence"] is True
    assert payload["constraints"]["claim_must_be_derived_from_evidence"] is True


def test_role_charter_is_composed_and_fingerprinted_per_specialist() -> None:
    provider = StubJsonChatProvider()
    analyzer = DeepSeekEvidenceAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        role_prompts={AgentRole.FINANCE: "FINANCE_ROLE_UNIQUE_CHARTER"},
        prompt_version="evidence-analysis-v3",
    )

    analyzer.analyze(
        _rule(),
        [_chunk("C1", "我們的經營產生負現金流量。")],
    )

    assert "FINANCE_ROLE_UNIQUE_CHARTER" in provider.system_prompt
    audit = analyzer.prompt_audit(AgentRole.FINANCE)
    assert audit["prompt_version"] == "evidence-analysis-v3"
    assert len(audit["prompt_sha256"]) == 64
    assert audit["prompt_snapshot"] == provider.system_prompt


class CorrectingEvidenceProvider(StubJsonChatProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: object | None = None,
    ) -> dict[str, object]:
        self.calls += 1
        payload = json.loads(user_prompt)
        self.prompts.append(payload)
        allowed = payload["constraints"]["allowed_evidence_chunk_ids"]
        return {
            "supported": True,
            "claim": "公司持续产生经营现金净流出。",
            "severity": "high",
            "confidence": 0.9,
            "evidence_chunk_ids": allowed,
            "uncertainty": "未来现金需求仍需核验。",
        }


def test_evidence_analysis_preserves_all_legal_evidence_without_retry() -> None:
    provider = CorrectingEvidenceProvider()
    analyzer = DeepSeekEvidenceAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
    )
    rule = _rule().model_copy(update={"max_evidence": 1})
    chunks = [
        _chunk("C1", "经营活动所用现金净额主要来自研发活动现金流出。"),
        _chunk("C2", "经营活动所用现金净额持续来自研究及研发活动现金流出。"),
    ]
    engine = GroundedAnalysisEngine(
        chunks,
        [rule],
        evidence_analyzer=analyzer,
        lexical_top_k=5,
        rerank_top_k=2,
        analysis_retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    )

    analyzed = engine.analyze()[0]

    assert provider.calls == 1
    assert set(analyzed.analysis.evidence_chunk_ids) == {"C1", "C2"}
    assert analyzed.analysis_attempts == 1
    assert not analyzed.analysis_retry_errors
    assert "max_evidence" not in provider.prompts[0]["constraints"]


class UnknownEvidenceProvider(StubJsonChatProvider):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, object]:
        return {
            "supported": True,
            "claim": "这条结论引用了不存在的证据。",
            "severity": "high",
            "confidence": 0.99,
            "evidence_chunk_ids": ["C-NOT-PROVIDED"],
            "uncertainty": "无",
        }


def test_deepseek_analyzer_rejects_evidence_not_present_in_candidates() -> None:
    analyzer = DeepSeekEvidenceAnalyzer(
        UnknownEvidenceProvider(),
        temperature=0.1,
        max_output_tokens=1024,
    )

    with pytest.raises(ValueError, match="unknown evidence chunks"):
        analyzer.analyze(
            _rule(),
            [_chunk("C1", "我們的經營產生負現金流量。")],
        )


class FakeChatCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        message = SimpleNamespace(content='{"supported": false, "reason": "insufficient"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_compatible_chat_provider_enables_deepseek_json_output() -> None:
    completions = FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="deepseek-v4-flash",
    )

    result = provider.complete_json(
        system_prompt="输出JSON",
        user_prompt="分析候选证据",
        temperature=0.1,
        max_output_tokens=1024,
    )

    assert result == {"supported": False, "reason": "insufficient"}
    assert completions.request["response_format"] == {"type": "json_object"}
    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_compatible_chat_provider_omits_reasoning_effort_when_thinking_disabled() -> None:
    completions = FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="kimi-k3",
        reasoning_effort="low",
    )

    provider.complete_json(
        system_prompt="输出JSON",
        user_prompt="批量核验候选风险",
        temperature=0.1,
        max_output_tokens=1024,
    )

    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in completions.request


def test_openai_compatible_chat_provider_keeps_reasoning_effort_when_thinking_enabled() -> None:
    completions = FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="kimi-k3",
        request_extra_body={},
        reasoning_effort="low",
    )

    provider.complete_json(
        system_prompt="输出JSON",
        user_prompt="批量核验候选风险",
        temperature=0.1,
        max_output_tokens=1024,
    )

    assert "extra_body" not in completions.request
    assert completions.request["reasoning_effort"] == "low"


def test_openai_compatible_chat_provider_can_omit_unsupported_json_mode() -> None:
    completions = FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="deepseek-v4-flash",
        supports_json_object=False,
    )

    result = provider.complete_json(
        system_prompt="只输出JSON对象",
        user_prompt="分析候选证据",
        temperature=0.1,
        max_output_tokens=1024,
    )

    assert result == {"supported": False, "reason": "insufficient"}
    assert "response_format" not in completions.request


class FencedJsonChatCompletions(FakeChatCompletions):
    def create(self, **kwargs: object) -> SimpleNamespace:
        response = super().create(**kwargs)
        response.choices[
            0
        ].message.content = '```json\n{"supported": false, "reason": "insufficient"}\n```'
        return response


def test_openai_compatible_chat_provider_accepts_fenced_json_without_json_mode() -> None:
    completions = FencedJsonChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="deepseek-v4-flash",
        supports_json_object=False,
    )

    result = provider.complete_json(
        system_prompt="只输出JSON对象",
        user_prompt="分析候选证据",
        temperature=0.1,
        max_output_tokens=1024,
    )

    assert result == {"supported": False, "reason": "insufficient"}


class FakeStreamingChatCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.request = kwargs
        fragments = (
            '{"supported": false,',
            ' "reason": "insufficient"}',
        )
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=fragment))])
            for fragment in fragments
        )


class ReasoningThenContentChatCompletions(FakeStreamingChatCompletions):
    def create(self, **kwargs: object) -> object:
        self.request = kwargs
        deltas = (
            SimpleNamespace(content=None, reasoning_content="正在核验主体和期间"),
            SimpleNamespace(content='{"supported": false,', reasoning_content=None),
            SimpleNamespace(content=' "reason": "insufficient"}', reasoning_content=None),
        )
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
            for delta in deltas
        )


def test_openai_compatible_chat_provider_streams_and_reassembles_json() -> None:
    completions = FakeStreamingChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="deepseek-v4-flash",
    )
    fragments: list[str] = []

    result = provider.complete_json(
        system_prompt="输出JSON",
        user_prompt="分析候选证据",
        temperature=0.1,
        max_output_tokens=1024,
        on_text_delta=fragments.append,
    )

    assert result == {"supported": False, "reason": "insufficient"}
    assert fragments == [
        '{"supported": false,',
        ' "reason": "insufficient"}',
    ]
    assert completions.request["stream"] is True


def test_openai_compatible_provider_reports_reasoning_activity_without_leaking_it() -> None:
    completions = ReasoningThenContentChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    reasoning_events: list[str] = []
    provider = OpenAICompatibleJsonChatProvider(
        client=client,
        model_name="kimi-k3",
        reasoning_progress_callback=lambda: reasoning_events.append("reasoning"),
    )
    fragments: list[str] = []

    result = provider.complete_json(
        system_prompt="输出JSON",
        user_prompt="分析候选证据",
        temperature=0.1,
        max_output_tokens=1024,
        on_text_delta=fragments.append,
    )

    assert result == {"supported": False, "reason": "insufficient"}
    assert reasoning_events == ["reasoning"]
    assert "正在核验主体和期间" not in "".join(fragments)


class DeltaReportingProvider(StubJsonChatProvider):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: object | None = None,
    ) -> dict[str, object]:
        if callable(on_text_delta):
            on_text_delta('{"claim":"现金消耗风险"}')
        return super().complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def test_deepseek_analyzer_routes_model_deltas_to_owning_agent() -> None:
    progress_events: list[tuple[str, str]] = []
    analyzer = DeepSeekEvidenceAnalyzer(
        DeltaReportingProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        progress_callback=lambda stage, detail: progress_events.append((stage, detail)),
    )

    analyzer.analyze(
        _rule(),
        [_chunk("C1", "我們的經營產生負現金流量，現金流出主要來自研發活動。")],
    )

    assert progress_events == [("agent:finance:delta", '{"claim":"现金消耗风险"}')]
