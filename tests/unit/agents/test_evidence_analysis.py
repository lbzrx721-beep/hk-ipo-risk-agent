from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from ipo_risk.agents.analysis import (
    DeepSeekEvidenceAnalyzer,
    OpenAICompatibleJsonChatProvider,
)
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


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
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=fragment))
                ]
            )
            for fragment in fragments
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
        progress_callback=lambda stage, detail: progress_events.append(
            (stage, detail)
        ),
    )

    analyzer.analyze(
        _rule(),
        [_chunk("C1", "我們的經營產生負現金流量，現金流出主要來自研發活動。")],
    )

    assert progress_events == [
        ("agent:finance:delta", '{"claim":"现金消耗风险"}')
    ]
