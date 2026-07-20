"""将混合检索与证据约束分析组合成可降级的单规则引擎。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import BM25Retriever, SearchHit, tokenize_text
from ipo_risk.retrieval.search import RiskRule, select_rule_candidates
from ipo_risk.retrieval.semantic import (
    EmbeddingProvider,
    HybridRetriever,
    HybridSearchHit,
    SemanticRetriever,
)
from ipo_risk.resilience import (
    RetryExhaustedError,
    RetryPolicy,
    retry_model_node_error,
)


class EvidenceAnalyzer(Protocol):
    """外部证据分析器的最小接口。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def analyze(
        self,
        rule: RiskRule,
        candidates: Sequence[DocumentChunk],
    ) -> EvidenceAnalysis:
        """根据候选证据判断风险是否成立。"""
        ...


@dataclass(frozen=True)
class RetrievalCandidate:
    """供审计记录使用的检索候选及两路分数。"""

    chunk: DocumentChunk
    score: float
    rank: int
    lexical_score: float | None = None
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None


@dataclass(frozen=True)
class AnalyzedRisk:
    """一条规则从候选召回到证据选择的完整结果。"""

    rule: RiskRule
    analysis: EvidenceAnalysis
    candidates: tuple[RetrievalCandidate, ...]
    selected_chunks: tuple[DocumentChunk, ...]
    retrieval_mode: str
    analysis_mode: str
    analysis_attempts: int
    search_queries: tuple[str, ...]
    retrieval_error: str | None = None
    analysis_error: str | None = None


def _candidate_from_hit(hit: SearchHit | HybridSearchHit) -> RetrievalCandidate:
    if isinstance(hit, HybridSearchHit):
        return RetrievalCandidate(
            chunk=hit.chunk,
            score=hit.score,
            rank=hit.rank,
            lexical_score=hit.lexical_score,
            lexical_rank=hit.lexical_rank,
            semantic_score=hit.semantic_score,
            semantic_rank=hit.semantic_rank,
        )
    return RetrievalCandidate(
        chunk=hit.chunk,
        score=hit.score,
        rank=hit.rank,
        lexical_score=hit.score,
        lexical_rank=hit.rank,
    )


def _deterministic_analysis(
    rule: RiskRule,
    hits: Sequence[SearchHit],
) -> EvidenceAnalysis:
    if not hits:
        return EvidenceAnalysis(
            supported=False,
            claim="",
            severity=rule.severity,
            confidence=0.0,
            evidence_chunk_ids=(),
            uncertainty="未找到同时满足全部锚定关键词的原文证据。",
        )
    ranks = [hit.rank for hit in hits]
    confidence = max(0.72, 0.92 - 0.01 * (max(ranks) - 1))
    return EvidenceAnalysis(
        supported=True,
        claim=rule.claim,
        severity=rule.severity,
        confidence=round(confidence, 2),
        evidence_chunk_ids=tuple(hit.chunk.chunk_id for hit in hits),
        uncertainty="规则与BM25生成的候选结论，需由投研人员复核确认。",
    )


class GroundedAnalysisEngine:
    """优先使用混合检索和LLM，失败时回退到原有确定性规则。"""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        rules: list[RiskRule],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        evidence_analyzer: EvidenceAnalyzer | None = None,
        embedding_cache_dir: Path | None = None,
        lexical_top_k: int = 20,
        semantic_top_k: int = 20,
        rerank_top_k: int = 8,
        analysis_retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("GroundedAnalysisEngine requires chunks")
        if lexical_top_k < 1 or semantic_top_k < 1 or rerank_top_k < 1:
            raise ValueError("retrieval limits must be positive")
        self._rules = list(rules)
        self._lexical = BM25Retriever(chunks)
        self._semantic: SemanticRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        self._retrieval_error: str | None = None
        self._evidence_analyzer = evidence_analyzer
        self._lexical_top_k = lexical_top_k
        self._rerank_top_k = rerank_top_k
        self._analysis_retry_policy = analysis_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )

        if embedding_provider is not None:
            try:
                semantic = SemanticRetriever(
                    chunks,
                    embedding_provider,
                    cache_dir=embedding_cache_dir,
                )
                self._semantic = semantic
                self._hybrid = HybridRetriever(
                    self._lexical,
                    semantic,
                    lexical_top_k=lexical_top_k,
                    semantic_top_k=semantic_top_k,
                )
            except Exception as exc:
                self._retrieval_error = str(exc)

    def _hybrid_candidates(self, rule: RiskRule) -> list[RetrievalCandidate]:
        if self._hybrid is None:
            return []
        by_chunk: dict[str, RetrievalCandidate] = {}
        for query in rule.search_queries:
            for hit in self._hybrid.search(query, top_k=self._rerank_top_k):
                if hit.chunk.is_heading or len("".join(hit.chunk.text.split())) < 20:
                    continue
                candidate = _candidate_from_hit(hit)
                current = by_chunk.get(hit.chunk.chunk_id)
                if current is None or candidate.score > current.score:
                    by_chunk[hit.chunk.chunk_id] = candidate
        return sorted(
            by_chunk.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.rank,
                candidate.chunk.chunk_id,
            ),
        )[: self._rerank_top_k]

    def _semantic_candidates(
        self,
        queries: Sequence[str],
    ) -> list[RetrievalCandidate]:
        if self._semantic is None:
            return []
        by_chunk: dict[str, RetrievalCandidate] = {}
        for query in queries:
            for hit in self._semantic.search(query, top_k=self._rerank_top_k):
                if hit.chunk.is_heading or len("".join(hit.chunk.text.split())) < 20:
                    continue
                candidate = RetrievalCandidate(
                    chunk=hit.chunk,
                    score=hit.score,
                    rank=hit.rank,
                    semantic_score=hit.score,
                    semantic_rank=hit.rank,
                )
                current = by_chunk.get(hit.chunk.chunk_id)
                if current is None or candidate.score > current.score:
                    by_chunk[hit.chunk.chunk_id] = candidate
        return sorted(
            by_chunk.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.rank,
                candidate.chunk.chunk_id,
            ),
        )[: self._rerank_top_k]

    def _lexical_candidates(self, rule: RiskRule) -> list[RetrievalCandidate]:
        by_chunk: dict[str, RetrievalCandidate] = {}
        for query in rule.search_queries:
            query_tokens = set(tokenize_text(query))
            for hit in self._lexical.search(query, top_k=self._lexical_top_k):
                if hit.chunk.is_heading or len("".join(hit.chunk.text.split())) < 20:
                    continue
                overlap = len(query_tokens & set(tokenize_text(hit.chunk.text)))
                if overlap == 0:
                    continue
                candidate = replace(
                    _candidate_from_hit(hit),
                    score=hit.score + overlap / max(1, len(query_tokens)),
                )
                current = by_chunk.get(hit.chunk.chunk_id)
                if current is None or candidate.score > current.score:
                    by_chunk[hit.chunk.chunk_id] = candidate
        return sorted(
            by_chunk.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.rank,
                candidate.chunk.chunk_id,
            ),
        )[: self._rerank_top_k]

    def analyze_rules(
        self,
        rules: Sequence[RiskRule] | None = None,
        *,
        search_queries_by_risk: Mapping[str, Sequence[str]] | None = None,
    ) -> list[AnalyzedRisk]:
        """分析指定规则；未指定时分析初始化阶段加载的全部规则。"""
        results: list[AnalyzedRisk] = []
        selected_rules = self._rules if rules is None else list(rules)
        for rule in selected_rules:
            retrieval_error = self._retrieval_error
            planned_queries = tuple(
                dict.fromkeys(
                    query.strip()
                    for query in (search_queries_by_risk or {}).get(rule.id, ())
                    if query.strip()
                )
            )
            search_queries = planned_queries or rule.search_queries
            exact_hits = select_rule_candidates(
                rule,
                self._lexical,
                top_k=self._lexical_top_k,
            )
            if planned_queries:
                candidates: list[RetrievalCandidate] = []
                if self._semantic is not None:
                    try:
                        candidates = self._semantic_candidates(planned_queries)
                    except Exception as exc:
                        retrieval_error = str(exc)
                if candidates:
                    retrieval_mode = "agentic_semantic"
                else:
                    candidates = self._lexical_candidates(rule)
                    retrieval_mode = "fixed_query_fallback"
            elif self._hybrid is not None:
                retrieval_mode = (
                    "hybrid_multi_query"
                    if len(rule.search_queries) > 1
                    else "hybrid"
                )
                candidates = self._hybrid_candidates(rule)
            else:
                retrieval_mode = (
                    "bm25_multi_query"
                    if len(rule.search_queries) > 1
                    else "bm25"
                )
                candidates = self._lexical_candidates(rule)
            if not candidates and not planned_queries:
                candidates = self._lexical_candidates(rule)

            analysis_error: str | None = None
            analysis_attempts = 0
            analyzer = self._evidence_analyzer
            if analyzer is not None and candidates:
                try:
                    outcome = self._analysis_retry_policy.run(
                        lambda: analyzer.analyze(
                            rule,
                            [candidate.chunk for candidate in candidates],
                        )
                    )
                    analysis = outcome.value
                    analysis_attempts = outcome.attempts
                    analysis_mode = analyzer.provider_name
                except RetryExhaustedError as error:
                    analysis_attempts = error.attempts
                    analysis_error = str(error.last_error)
                    analysis = _deterministic_analysis(rule, exact_hits)
                    analysis_mode = "rule_fallback"
            else:
                analysis = _deterministic_analysis(rule, exact_hits)
                analysis_mode = "rule"
                analysis_attempts = 1

            chunk_by_id = {
                candidate.chunk.chunk_id: candidate.chunk for candidate in candidates
            }
            if analysis_mode.startswith("rule"):
                chunk_by_id.update({hit.chunk.chunk_id: hit.chunk for hit in exact_hits})
            unknown = set(analysis.evidence_chunk_ids) - set(chunk_by_id)
            if unknown:
                raise ValueError(
                    f"analysis selected chunks outside retrieval candidates: {sorted(unknown)}"
                )
            selected_chunks = tuple(
                chunk_by_id[chunk_id] for chunk_id in analysis.evidence_chunk_ids
            )
            results.append(
                AnalyzedRisk(
                    rule=rule,
                    analysis=analysis,
                    candidates=tuple(candidates),
                    selected_chunks=selected_chunks,
                    retrieval_mode=retrieval_mode,
                    analysis_mode=analysis_mode,
                    analysis_attempts=analysis_attempts,
                    search_queries=search_queries,
                    retrieval_error=retrieval_error,
                    analysis_error=analysis_error,
                )
            )
        return results

    def analyze(self) -> list[AnalyzedRisk]:
        """兼容原有调用方式，分析初始化阶段加载的全部规则。"""
        return self.analyze_rules()


__all__ = [
    "AnalyzedRisk",
    "EvidenceAnalyzer",
    "GroundedAnalysisEngine",
    "RetrievalCandidate",
]
