from __future__ import annotations

import json
from datetime import date
from threading import Barrier, Lock

from ipo_risk.agents.analysis import JsonResponseDecodeError
from ipo_risk.agents.discovery import DeepSeekOpenRiskDiscoverer
from ipo_risk.domain import AgentRole
from ipo_risk.retrieval import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryPolicy, retry_model_node_error


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=8,
        printed_page_label="3",
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


def _known_rule() -> RiskRule:
    return RiskRule(
        id="cash_burn",
        name="现金消耗",
        owner=AgentRole.FINANCE,
        query="现金消耗",
        required_keywords=("现金",),
        claim="公司存在现金消耗风险。",
    )


class _DiscoveryProvider:
    model_name = "deepseek-v4-flash"

    def __init__(self) -> None:
        self.payload: dict[str, object] = {}

    def complete_json(self, **request: object) -> dict[str, object]:
        self.payload = json.loads(str(request["user_prompt"]))
        records = self.payload.get("candidate_records")
        if isinstance(records, list):
            groups: dict[tuple[str, str], list[dict[str, object]]] = {}
            for record in records:
                assert isinstance(record, dict)
                key = (str(record["owner"]), str(record["title"]))
                groups.setdefault(key, []).append(record)
            return {
                "consolidated_risks": [
                    {
                        "source_candidate_ids": [
                            str(record["candidate_id"])
                            for record in sorted(
                                group,
                                key=lambda item: -float(item["confidence"]),
                            )
                        ],
                        "title": group[0]["title"],
                        "owner": group[0]["owner"],
                        "severity": group[0]["severity"],
                        "claim": group[0]["claim"],
                        "confidence": max(float(record["confidence"]) for record in group),
                        "evidence_chunk_ids": list(
                            dict.fromkeys(
                                str(chunk_id)
                                for record in sorted(
                                    group,
                                    key=lambda item: -float(item["confidence"]),
                                )
                                for chunk_id in record["evidence_chunk_ids"]
                            )
                        )[:3],
                        "search_queries": list(
                            dict.fromkeys(
                                str(query) for record in group for query in record["search_queries"]
                            )
                        )[:6],
                        "anchor_terms": list(
                            dict.fromkeys(
                                str(term) for record in group for term in record["anchor_terms"]
                            )
                        )[:6],
                        "uncertainty": group[0]["uncertainty"],
                        "related_catalog_risk_type": None,
                    }
                    for group in groups.values()
                ]
            }
        allowed = self.payload["constraints"]["allowed_evidence_chunk_ids"]
        return {
            "candidates": [
                {
                    "title": "独家数据授权终止风险",
                    "owner": "legal",
                    "severity": "high",
                    "claim": "公司的核心训练数据依赖可被提前终止的独家授权。",
                    "confidence": 0.86,
                    "evidence_chunk_ids": [allowed[0]],
                    "search_queries": ["核心训练数据的独家授权是否可以提前终止"],
                    "anchor_terms": ["独家授权", "提前终止"],
                    "uncertainty": "尚需核对其他授权协议是否能够替代。",
                    "related_known_risk_type": None,
                }
            ]
        }


def test_discovery_returns_a_grounded_dynamic_rule_for_the_responsible_agent() -> None:
    provider = _DiscoveryProvider()
    source = _chunk("C-OPEN-1", "核心训练数据来自独家授权，许可方可以提前终止协议。")
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
    )

    result = discoverer.discover([source], [_known_rule()])

    assert len(result.risks) == 1
    risk = result.risks[0]
    assert risk.risk_type.startswith("open_")
    assert risk.owner is AgentRole.LEGAL
    assert risk.evidence_chunk_ids == ("C-OPEN-1",)
    assert risk.to_rule().owner is AgentRole.LEGAL
    assert provider.payload["catalog_risk_definitions"] == [
        {
            "id": "cash_burn",
            "name": "现金消耗",
            "owner": "finance",
            "claim": "公司存在现金消耗风险。",
        }
    ]
    assert result.raw_candidate_count == 1
    assert result.consolidated_candidate_count == 1
    assert result.consolidation_mode == "semantic_model"
    assert result.consolidation_mappings[0].source_candidate_ids == ("R-0001-01",)


class _VectorGuidedProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if "document_outline_and_samples" in payload:
            return {
                "queries": [
                    {
                        "query": "核心训练数据授权是否可以提前终止",
                        "reason": "目录样本披露了独家数据授权安排。",
                    },
                    {
                        "query": "授权终止后是否存在替代数据安排",
                        "reason": "需要同时定位缓释措施和反证。",
                    },
                ]
            }
        return super().complete_json(**request)


def test_vector_guided_discovery_reads_retrieved_original_chunks_and_keeps_schema() -> None:
    provider = _VectorGuidedProvider()
    chunks = [
        _chunk("C-OPEN-1", "公司业务概览。"),
        _chunk("C-OPEN-2", "核心训练数据来自独家授权，许可方可以提前终止协议。"),
        _chunk("C-OPEN-3", "公司正在评估替代数据来源。"),
    ]
    selector_calls: list[tuple[tuple[str, ...], dict[str, int]]] = []

    def select_corpus(
        queries: tuple[str, ...],
        **settings: int,
    ) -> tuple[DocumentChunk, ...]:
        selector_calls.append((queries, settings))
        return (chunks[1], chunks[2])

    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
        retrieval_query_count=4,
    )

    result = discoverer.discover_with_retrieval(
        chunks,
        [_known_rule()],
        corpus_selector=select_corpus,
    )

    assert selector_calls[0][0][:2] == (
        "核心训练数据授权是否可以提前终止",
        "授权终止后是否存在替代数据安排",
    )
    assert len(selector_calls[0][0]) == 4
    assert result.discovery_mode == "vector_guided_original_text"
    assert result.source_chunk_count == 3
    assert result.selected_chunk_count == 2
    assert result.retrieval_query_count == 4
    assert result.risks[0].evidence_chunk_ids == ("C-OPEN-2",)


class _HallucinatingProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = super().complete_json(**request)
        payload["candidates"][0]["evidence_chunk_ids"] = ["C-NOT-IN-PDF"]
        return payload


def test_discovery_drops_a_batch_that_cites_unknown_evidence() -> None:
    discoverer = DeepSeekOpenRiskDiscoverer(
        _HallucinatingProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
    )

    result = discoverer.discover(
        [_chunk("C-OPEN-1", "核心训练数据来自独家授权。")],
        [_known_rule()],
    )

    assert result.risks == ()
    assert "unknown evidence chunks" in result.batch_errors[0]


class _PartiallyInvalidDiscoveryProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        result = super().complete_json(**request)
        invalid = dict(result["candidates"][0])
        invalid["title"] = "引用不存在原文的候选"
        invalid["evidence_chunk_ids"] = ["C-NOT-IN-PDF"]
        result["candidates"].append(invalid)
        return result


def test_discovery_preserves_valid_candidates_when_one_item_is_invalid() -> None:
    result = DeepSeekOpenRiskDiscoverer(
        _PartiallyInvalidDiscoveryProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
        max_batch_passes=1,
    ).discover(
        [_chunk("C-OPEN-1", "核心训练数据来自独家授权。")],
        [_known_rule()],
    )

    assert result.batch_errors == ()
    assert result.raw_candidate_count == 1
    assert result.raw_candidates[0].title == "独家数据授权终止风险"
    assert any("引用不存在原文的候选" in warning for warning in result.batch_warnings)


class _TruncatedDiscoveryProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        allowed = payload["constraints"]["allowed_evidence_chunk_ids"]
        complete_candidate = json.dumps(
            {
                "title": "独家数据授权终止风险",
                "owner": "legal",
                "severity": "high",
                "claim": "公司的核心训练数据依赖可被提前终止的独家授权。",
                "confidence": 0.86,
                "evidence_chunk_ids": [allowed[0]],
                "search_queries": ["核心训练数据的独家授权是否可以提前终止"],
                "anchor_terms": ["独家授权", "提前终止"],
                "uncertainty": "尚需核对其他授权协议是否能够替代。",
                "related_known_risk_type": None,
            },
            ensure_ascii=False,
        )
        raise JsonResponseDecodeError(
            "response was truncated",
            raw_content=(f'{{"candidates":[{complete_candidate},{{"title":"下一条尚未生成完成"'),
        )


def test_discovery_salvages_complete_candidates_from_a_truncated_response() -> None:
    result = DeepSeekOpenRiskDiscoverer(
        _TruncatedDiscoveryProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
        max_batch_passes=1,
    ).discover(
        [_chunk("C-OPEN-1", "核心训练数据来自独家授权。")],
        [_known_rule()],
    )

    assert result.batch_errors == ()
    assert result.raw_candidate_count == 1
    assert result.raw_candidates[0].title == "独家数据授权终止风险"
    assert any("响应尾部未闭合" in warning for warning in result.batch_warnings)


class _FourEvidenceProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        result = super().complete_json(**request)
        allowed = payload["constraints"]["allowed_evidence_chunk_ids"]
        result["candidates"][0]["evidence_chunk_ids"] = allowed[:4]
        return result


def test_discovery_accepts_four_grounded_evidence_ids_without_dropping_batch() -> None:
    chunks = [
        _chunk(f"C-OPEN-{index}", f"独家授权条款原文{index}，包含终止与替代安排。")
        for index in range(1, 5)
    ]
    result = DeepSeekOpenRiskDiscoverer(
        _FourEvidenceProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=2_000,
    ).discover(chunks, [_known_rule()])

    assert result.batch_errors == ()
    assert result.raw_candidates[0].evidence_chunk_ids == tuple(chunk.chunk_id for chunk in chunks)


class _RepeatedRiskProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete_json(self, **request: object) -> dict[str, object]:
        request_payload = json.loads(str(request["user_prompt"]))
        if isinstance(request_payload.get("candidate_records"), list):
            return super().complete_json(**request)
        self.calls += 1
        payload = super().complete_json(**request)
        payload["candidates"][0]["search_queries"] = [f"第{self.calls}批独家授权终止风险"]
        return payload


class _RememberingRiskProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_payloads: list[dict[str, object]] = []

    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        self.discovery_payloads.append(payload)
        result = super().complete_json(**request)
        call_number = len(self.discovery_payloads)
        result["candidates"][0]["title"] = f"风险主题{call_number}"
        result["candidates"][0]["claim"] = f"当前批次披露风险主题{call_number}。"
        return result


def test_later_discovery_batches_receive_prior_document_risk_memory() -> None:
    provider = _RememberingRiskProvider()
    events: list[tuple[str, str]] = []
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=500,
        wave_size=1,
        progress_callback=lambda stage, detail: events.append((stage, detail)),
    )
    chunks = [
        _chunk("C-OPEN-1", "第一章节披露风险。" * 40),
        _chunk("C-OPEN-2", "第二章节披露风险。" * 40),
    ]

    result = discoverer.discover(chunks, [_known_rule()])

    assert result.batch_count == 2
    assert provider.discovery_payloads[0]["prior_document_candidates"] == []
    prior = provider.discovery_payloads[1]["prior_document_candidates"]
    assert isinstance(prior, list)
    assert prior[0]["candidate_id"] == "R-0001-01"
    assert prior[0]["title"] == "风险主题1"
    assert [stage for stage, _ in events].count("agent:discovery:batch_completed") == 2
    assert any("[1/2]" in detail and "累计1条" in detail for _, detail in events)


class _WaveProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.barrier = Barrier(2)
        self.lock = Lock()
        self.active = 0
        self.max_active = 0
        self.discovery_payloads: dict[int, dict[str, object]] = {}

    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        scan_state = payload["document_scan_state"]
        batch_index = int(scan_state["batch_index"])
        self.discovery_payloads[batch_index] = payload
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if batch_index <= 2:
            self.barrier.wait(timeout=3)
        try:
            result = super().complete_json(**request)
            result["candidates"][0]["title"] = f"波次风险{batch_index}"
            result["candidates"][0]["claim"] = f"第{batch_index}批披露独立风险。"
            return result
        finally:
            with self.lock:
                self.active -= 1


def test_discovery_scans_two_batches_in_parallel_then_shares_wave_memory() -> None:
    provider = _WaveProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=500,
        wave_size=2,
    )
    chunks = [
        _chunk(f"C-OPEN-{index}", f"第{index}章节披露独立风险。" * 40) for index in range(1, 4)
    ]

    result = discoverer.discover(chunks, [_known_rule()])

    assert result.batch_count == 3
    assert provider.max_active == 2
    assert provider.discovery_payloads[1]["prior_document_candidates"] == []
    assert provider.discovery_payloads[2]["prior_document_candidates"] == []
    third_prior = provider.discovery_payloads[3]["prior_document_candidates"]
    assert isinstance(third_prior, list)
    assert {item["candidate_id"] for item in third_prior} == {
        "R-0001-01",
        "R-0002-01",
    }


def test_discovery_scans_every_batch_and_merges_duplicate_evidence() -> None:
    provider = _RepeatedRiskProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=500,
    )
    chunks = [
        _chunk("C-OPEN-1", "独家授权可以提前终止。" * 25),
        _chunk("C-OPEN-2", "核心数据没有替代来源。" * 25),
    ]

    result = discoverer.discover(chunks, [_known_rule()])

    assert provider.calls == 2
    assert result.batch_count == 2
    assert len(result.risks) == 1
    assert result.risks[0].evidence_chunk_ids == ("C-OPEN-1", "C-OPEN-2")
    assert result.risks[0].search_queries == (
        "第1批独家授权终止风险",
        "第2批独家授权终止风险",
    )


class _ContinuationProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_calls = 0
        self.second_payload: dict[str, object] = {}

    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        self.discovery_calls += 1
        result = super().complete_json(**request)
        candidate = result["candidates"][0]
        candidate["title"] = f"独立重大风险{self.discovery_calls}"
        candidate["claim"] = f"原文直接支持独立重大风险{self.discovery_calls}。"
        candidate["search_queries"] = [f"核验独立重大风险{self.discovery_calls}"]
        candidate["anchor_terms"] = [f"风险{self.discovery_calls}"]
        result["has_more"] = self.discovery_calls == 1
        result["coverage_note"] = "仍有候选" if result["has_more"] else "已覆盖"
        if self.discovery_calls == 2:
            self.second_payload = payload
        return result


def test_discovery_continues_when_model_reports_more_material_risks() -> None:
    provider = _ContinuationProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=2_048,
        batch_char_limit=2_000,
        max_batch_passes=3,
    )

    result = discoverer.discover(
        [_chunk("C-OPEN-1", "同一批次披露两项相互独立的重大风险。")],
        [_known_rule()],
    )

    assert provider.discovery_calls == 2
    assert len(result.raw_candidates) == 2
    assert provider.second_payload["scan_pass"] == 2
    assert len(provider.second_payload["already_returned_candidates"]) == 1
    assert {candidate.candidate_id for candidate in result.raw_candidates} == {
        "R-0001-01",
        "R-0001-02",
    }


class _CorrectingDiscoveryProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_calls = 0
        self.correction: object = None

    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return super().complete_json(**request)
        self.discovery_calls += 1
        result = super().complete_json(**request)
        if self.discovery_calls == 1:
            result["candidates"][0]["evidence_chunk_ids"] = [
                f"C-OPEN-{index}" for index in range(1, 10)
            ]
        else:
            self.correction = payload.get("correction")
        return result


def test_discovery_retry_receives_validation_feedback_and_corrects_output() -> None:
    provider = _CorrectingDiscoveryProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=2_048,
        batch_char_limit=2_000,
        retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    )
    chunks = [_chunk(f"C-OPEN-{index}", f"风险事实{index}。") for index in range(1, 10)]

    result = discoverer.discover(chunks, [_known_rule()])

    assert provider.discovery_calls == 2
    assert result.raw_candidate_count == 1
    assert isinstance(provider.correction, dict)
    assert "at most 8" in json.dumps(provider.correction, ensure_ascii=False)


class _LaterHighConfidenceProvider(_RepeatedRiskProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        request_payload = json.loads(str(request["user_prompt"]))
        payload = super().complete_json(**request)
        if not isinstance(request_payload.get("candidate_records"), list):
            payload["candidates"][0]["confidence"] = 0.99 if self.calls == 4 else 0.7
        return payload


def test_duplicate_merge_keeps_primary_candidate_evidence_first() -> None:
    provider = _LaterHighConfidenceProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
        batch_char_limit=500,
    )
    chunks = [_chunk(f"C-OPEN-{index}", "独家授权可以提前终止。" * 25) for index in range(1, 5)]

    result = discoverer.discover(chunks, [_known_rule()])

    assert result.risks[0].confidence == 0.99
    assert result.risks[0].evidence_chunk_ids == (
        "C-OPEN-4",
        "C-OPEN-1",
        "C-OPEN-2",
    )


class _SemanticDuplicateProvider(_DiscoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        records = payload.get("candidate_records")
        if isinstance(records, list):
            return {
                "consolidated_risks": [
                    {
                        "source_candidate_ids": [str(record["candidate_id"]) for record in records],
                        "title": "持续亏损及经营现金净流出风险",
                        "owner": "finance",
                        "severity": "high",
                        "claim": "公司持续亏损且经营活动现金流为负。",
                        "confidence": 0.91,
                        "evidence_chunk_ids": [
                            str(records[0]["evidence_chunk_ids"][0]),
                            str(records[1]["evidence_chunk_ids"][0]),
                        ],
                        "search_queries": ["公司持续亏损及现金净流出的变化趋势"],
                        "anchor_terms": ["持续亏损", "经营活动现金流量净额"],
                        "uncertainty": "仍需核验上市所得款项的缓释程度。",
                        "related_catalog_risk_type": "cash_burn",
                    }
                ]
            }
        self.batch_calls += 1
        allowed = payload["constraints"]["allowed_evidence_chunk_ids"]
        titles = (
            ("连续净亏损风险", "公司连续报告净亏损。", "持续亏损"),
            (
                "经营活动现金流长期为负",
                "经营活动现金流量净额持续为负。",
                "经营活动现金流量净额",
            ),
        )
        title, claim, anchor = titles[self.batch_calls - 1]
        return {
            "candidates": [
                {
                    "title": title,
                    "owner": "finance",
                    "severity": "high",
                    "claim": claim,
                    "confidence": 0.91,
                    "evidence_chunk_ids": [allowed[0]],
                    "search_queries": [title],
                    "anchor_terms": [anchor],
                    "uncertainty": "仍需核验后续变化。",
                    "related_known_risk_type": None,
                }
            ]
        }


def test_global_model_consolidates_semantic_duplicates_and_preserves_mapping() -> None:
    provider = _SemanticDuplicateProvider()
    discoverer = DeepSeekOpenRiskDiscoverer(
        provider,
        temperature=0.1,
        max_output_tokens=2_048,
        batch_char_limit=500,
    )
    chunks = [
        _chunk("C-LOSS", "公司连续报告净亏损。" * 35),
        _chunk("C-CASH", "经营活动现金流量净额持续为负。" * 30),
    ]

    result = discoverer.discover(chunks, [_known_rule()])

    assert result.raw_candidate_count == 2
    assert result.consolidated_candidate_count == 1
    assert result.risks[0].title == "持续亏损及经营现金净流出风险"
    assert result.risks[0].related_catalog_risk_type == "cash_burn"
    assert result.consolidation_mappings[0].source_candidate_ids == (
        "R-0001-01",
        "R-0002-01",
    )


class _DistinctCandidateProvider(_DiscoveryProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        records = payload.get("candidate_records")
        if isinstance(records, list):
            return {
                "consolidated_risks": [
                    {
                        "source_candidate_ids": [record["candidate_id"]],
                        "title": record["title"],
                        "owner": record["owner"],
                        "severity": record["severity"],
                        "claim": record["claim"],
                        "confidence": record["confidence"],
                        "evidence_chunk_ids": record["evidence_chunk_ids"],
                        "search_queries": record["search_queries"],
                        "anchor_terms": record["anchor_terms"],
                        "uncertainty": record["uncertainty"],
                        "related_catalog_risk_type": None,
                    }
                    for record in records
                ]
            }
        allowed = payload["constraints"]["allowed_evidence_chunk_ids"]
        chunk_number = int(str(allowed[0]).rsplit("-", 1)[-1])
        return {
            "candidates": [
                {
                    "title": f"公司特异风险{chunk_number}",
                    "owner": "business",
                    "severity": "medium",
                    "claim": f"原文直接披露公司特异风险事项{chunk_number}。",
                    "confidence": 0.8,
                    "evidence_chunk_ids": [allowed[0]],
                    "search_queries": [f"核验公司特异风险事项{chunk_number}"],
                    "anchor_terms": [f"事项{chunk_number}"],
                    "uncertainty": "仍需核验缓释安排。",
                    "related_known_risk_type": None,
                }
            ]
        }


def test_distinct_candidates_are_not_silently_truncated_before_human_review() -> None:
    discoverer = DeepSeekOpenRiskDiscoverer(
        _DistinctCandidateProvider(),
        temperature=0.1,
        max_output_tokens=8_192,
        batch_char_limit=500,
    )
    chunks = [
        _chunk(f"C-RISK-{index}", f"事项{index}存在独立风险。" * 40) for index in range(1, 13)
    ]

    result = discoverer.discover(chunks, [_known_rule()])

    assert result.raw_candidate_count == 12
    assert result.consolidated_candidate_count == 12
    assert len(result.risks) == 12
    assert len(result.consolidation_mappings) == 12
    assert {
        candidate_id
        for mapping in result.consolidation_mappings
        for candidate_id in mapping.source_candidate_ids
    } == {f"R-{index:04d}-01" for index in range(1, 13)}


class _InvalidConsolidationProvider(_DistinctCandidateProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        payload = json.loads(str(request["user_prompt"]))
        if isinstance(payload.get("candidate_records"), list):
            return {
                "consolidated_risks": [
                    {
                        "source_candidate_ids": ["R-NOT-REAL"],
                        "title": "无效合并",
                        "owner": "business",
                        "severity": "medium",
                        "claim": "该输出引用了不存在的原始候选。",
                        "confidence": 0.8,
                        "evidence_chunk_ids": ["C-NOT-REAL"],
                        "search_queries": ["无效查询"],
                        "anchor_terms": ["无效锚点"],
                        "uncertainty": "输出无效。",
                        "related_catalog_risk_type": None,
                    }
                ]
            }
        return super().complete_json(**request)


def test_failed_global_consolidation_keeps_all_candidates_with_audited_fallback() -> None:
    discoverer = DeepSeekOpenRiskDiscoverer(
        _InvalidConsolidationProvider(),
        temperature=0.1,
        max_output_tokens=2_048,
        batch_char_limit=500,
    )
    chunks = [_chunk(f"C-RISK-{index}", f"事项{index}存在独立风险。" * 40) for index in range(1, 4)]

    result = discoverer.discover(chunks, [_known_rule()])

    assert len(result.risks) == 3
    assert result.consolidation_mode == "exact_title_fallback"
    assert result.consolidation_error
    assert len(result.consolidation_mappings) == 3
