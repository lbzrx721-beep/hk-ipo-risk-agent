from __future__ import annotations

from datetime import date

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import SearchHit
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryPolicy


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
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
        query="现金消耗 研发投入 融资需求",
        required_keywords=("負現金流量", "研發活動"),
        claim="公司存在资金消耗和后续融资需求。",
    )


class SemanticMeaningProvider:
    model_name = "test-semantic"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "研发" in text or "現金" in text else [0.0, 1.0])
        return vectors


class FixedSemanticSearcher:
    def __init__(self, hit: DocumentChunk) -> None:
        self.hit = hit
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        self.queries.append(query)
        return [SearchHit(chunk=self.hit, score=0.95, rank=1)]


def test_discovery_corpus_keeps_vector_hit_neighbors_and_section_coverage() -> None:
    chunks = [
        _chunk(f"C{index}", f"原文段落{index}，包含当前章节的公司披露信息。")
        for index in range(1, 9)
    ]
    chunks[0] = chunks[0].model_copy(update={"is_heading": True, "text": "业务概览"})
    chunks[5] = chunks[5].model_copy(update={"is_heading": True, "text": "重大合同"})
    semantic = FixedSemanticSearcher(chunks[3])
    engine = GroundedAnalysisEngine(
        chunks,
        [_rule()],
        semantic_retriever=semantic,
    )

    selected = engine.select_discovery_corpus(
        ["重大合同中的终止条件"],
        semantic_top_k_per_query=3,
        lexical_top_k_per_query=1,
        neighbor_radius=1,
        coverage_section_count=2,
        max_chunks=7,
    )
    selected_ids = {chunk.chunk_id for chunk in selected}

    assert semantic.queries == ["重大合同中的终止条件"]
    assert {"C3", "C4", "C5"}.issubset(selected_ids)
    assert "C1" in selected_ids
    assert len(selected) <= 7


class SelectingAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        selected = next(chunk for chunk in candidates if "研发" in chunk.text)
        return EvidenceAnalysis(
            supported=True,
            claim="公司研发开支巨大，存在持续资金消耗和融资依赖。",
            severity=RiskSeverity.HIGH,
            confidence=0.89,
            evidence_chunk_ids=(selected.chunk_id,),
            uncertainty="尚需结合现金余额判断覆盖期。",
        )


def test_engine_allows_deepseek_to_select_semantic_evidence_without_exact_keywords() -> None:
    chunks = [
        _chunk("C1", "公司研发开支巨大，可能需要继续筹集资金。"),
        _chunk("C2", "董事会负责公司治理。"),
    ]
    engine = GroundedAnalysisEngine(
        chunks,
        [_rule()],
        embedding_provider=SemanticMeaningProvider(),
        evidence_analyzer=SelectingAnalyzer(),
        lexical_top_k=1,
        semantic_top_k=2,
        rerank_top_k=2,
    )

    results = engine.analyze()

    assert len(results) == 1
    assert results[0].selected_chunks[0].chunk_id == "C1"
    assert results[0].retrieval_mode == "hybrid"
    assert results[0].analysis_mode == "deepseek"


class FailingAnalyzer(SelectingAnalyzer):
    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        raise RuntimeError("temporary DeepSeek failure")


def test_engine_falls_back_to_grounded_rule_when_deepseek_fails() -> None:
    exact_evidence = _chunk(
        "C1",
        "公司的經營產生負現金流量，而且現金流出主要來自研發活動。",
    )
    engine = GroundedAnalysisEngine(
        [exact_evidence],
        [_rule()],
        evidence_analyzer=FailingAnalyzer(),
        lexical_top_k=5,
    )

    result = engine.analyze()[0]

    assert result.analysis.supported
    assert result.selected_chunks == (exact_evidence,)
    assert result.analysis_mode == "rule_fallback"
    assert result.analysis_error == "temporary DeepSeek failure"


class FlakyValidationAnalyzer(SelectingAnalyzer):
    def __init__(self) -> None:
        self.calls = 0

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("model returned invalid evidence JSON")
        return super().analyze(rule, candidates)


def test_engine_retries_entire_model_node_after_validation_failure() -> None:
    analyzer = FlakyValidationAnalyzer()
    delays: list[float] = []
    engine = GroundedAnalysisEngine(
        [_chunk("C1", "公司研发开支巨大，可能需要继续筹集资金。")],
        [_rule()],
        embedding_provider=SemanticMeaningProvider(),
        evidence_analyzer=analyzer,
        analysis_retry_policy=RetryPolicy(
            max_retries=2,
            initial_seconds=0.5,
            retry_if=lambda error: True,
            sleep=delays.append,
        ),
    )

    result = engine.analyze()[0]

    assert result.analysis_mode == "deepseek"
    assert result.analysis_attempts == 2
    assert result.analysis_error is None
    assert analyzer.calls == 2
    assert delays == [0.5]


class CapturingAnalyzer(SelectingAnalyzer):
    def __init__(self) -> None:
        self.candidate_ids: tuple[str, ...] = ()

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        self.candidate_ids = tuple(chunk.chunk_id for chunk in candidates)
        selected = candidates[0]
        return EvidenceAnalysis(
            supported=True,
            claim="Cash runway disclosure indicates continued financing dependence.",
            severity=RiskSeverity.HIGH,
            confidence=0.84,
            evidence_chunk_ids=(selected.chunk_id,),
            uncertainty="The exact runway requires calculation.",
        )


def test_bm25_multi_query_still_feeds_deepseek_without_exact_fallback_keywords() -> None:
    analyzer = CapturingAnalyzer()
    rule = _rule().model_copy(
        update={
            "query_variants": (
                "现金跑道 未来融资",
                "流动性 资本需求",
            )
        }
    )
    relevant = _chunk(
        "C-RUNWAY",
        "公司的现金跑道可能不足，预计需要未来融资以支持持续研发。",
    )
    engine = GroundedAnalysisEngine(
        [relevant, _chunk("C-GOV", "The board oversees corporate governance.")],
        [rule],
        evidence_analyzer=analyzer,
        lexical_top_k=5,
        rerank_top_k=3,
    )

    result = engine.analyze()[0]

    assert result.analysis_mode == "deepseek"
    assert result.retrieval_mode == "bm25_multi_query"
    assert analyzer.candidate_ids[0] == "C-RUNWAY"
    assert result.search_queries == (
        "现金消耗 研发投入 融资需求",
        "现金跑道 未来融资",
        "流动性 资本需求",
    )


def test_agent_queries_fall_back_to_fixed_bm25_when_vector_search_is_unavailable() -> None:
    relevant = _chunk("C-FIXED", "公司现金消耗较快，研发投入带来后续融资需求。")
    rule = _rule().model_copy(update={"required_keywords": ("不存在的硬锚点",)})
    engine = GroundedAnalysisEngine(
        [relevant],
        [rule],
        evidence_analyzer=SelectingAnalyzer(),
    )

    result = engine.analyze_rules(
        (rule,),
        search_queries_by_risk={
            "cash_burn": ("Agent自主生成但当前没有向量服务的问题",)
        },
    )[0]

    assert result.retrieval_mode == "fixed_query_fallback"
    assert result.candidates[0].chunk == relevant
    assert result.search_queries == ("Agent自主生成但当前没有向量服务的问题",)
