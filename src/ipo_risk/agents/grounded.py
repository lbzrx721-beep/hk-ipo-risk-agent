"""将混合检索与证据约束分析组合成可降级的单规则引擎。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.specialist import (
    SpecialistAssessment,
    SpecialistBatchAnalyzer,
    SpecialistStatus,
    SpecialistTask,
)
from ipo_risk.agents.usage import ModelBudgetExceededError
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


class SemanticSearcher(Protocol):
    """语义检索器的最小接口，NumPy与pgvector检索器均满足。"""

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """返回按余弦相似度降序排列的语义候选段落。"""
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
    prompt_version: str | None = None
    prompt_sha256: str | None = None
    prompt_snapshot: str | None = None
    analysis_retry_errors: tuple[str, ...] = ()
    specialist_status: SpecialistStatus = SpecialistStatus.EVIDENCE_INSUFFICIENT
    followup_queries: tuple[str, ...] = ()


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


def _open_risk_fail_closed(
    rule: RiskRule,
    reason: str,
) -> EvidenceAnalysis:
    return EvidenceAnalysis(
        supported=False,
        claim="",
        severity=rule.severity,
        confidence=0.0,
        evidence_chunk_ids=(),
        uncertainty=(f"开放风险必须由专业Agent完成语义证据核验；本次未生成结论：{reason}。"),
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
        specialist_analyzer: SpecialistBatchAnalyzer | None = None,
        specialist_followup_rounds: int = 1,
        embedding_cache_dir: Path | None = None,
        semantic_retriever: SemanticSearcher | None = None,
        lexical_top_k: int = 20,
        semantic_top_k: int = 20,
        rerank_top_k: int = 8,
        analysis_retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("GroundedAnalysisEngine requires chunks")
        if lexical_top_k < 1 or semantic_top_k < 1 or rerank_top_k < 1:
            raise ValueError("retrieval limits must be positive")
        if evidence_analyzer is not None and specialist_analyzer is not None:
            raise ValueError(
                "configure either single-risk evidence_analyzer or specialist_analyzer"
            )
        if specialist_followup_rounds < 0:
            raise ValueError("specialist_followup_rounds must be non-negative")
        self._rules = list(rules)
        self._chunks = tuple(chunks)
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_index = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        self._lexical = BM25Retriever(chunks)
        self._semantic: SemanticSearcher | None = None
        self._hybrid: HybridRetriever | None = None
        self._retrieval_error: str | None = None
        self._evidence_analyzer = evidence_analyzer
        self._specialist_analyzer = specialist_analyzer
        self._specialist_followup_rounds = specialist_followup_rounds
        self._lexical_top_k = lexical_top_k
        self._rerank_top_k = rerank_top_k
        self._analysis_retry_policy = analysis_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )

        if semantic_retriever is not None:
            # 外部传入的语义检索器（如 PgVectorRetriever）优先使用
            self._semantic = semantic_retriever
            self._hybrid = HybridRetriever(
                self._lexical,
                semantic_retriever,
                lexical_top_k=lexical_top_k,
                semantic_top_k=semantic_top_k,
            )
        elif embedding_provider is not None:
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

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """返回本次分析的不可变全文切片，供开放式发现 Agent 扫描。"""
        return self._chunks

    def enable_bm25_fallback(self, reason: str) -> None:
        """按使用者选择关闭本公司的语义通道，后续证据召回只走BM25。"""
        self._semantic = None
        self._hybrid = None
        self._retrieval_error = reason

    def select_discovery_corpus(
        self,
        queries: Sequence[str],
        *,
        semantic_top_k_per_query: int = 12,
        lexical_top_k_per_query: int = 6,
        neighbor_radius: int = 1,
        coverage_section_count: int = 24,
        max_chunks: int = 360,
    ) -> tuple[DocumentChunk, ...]:
        """为风险发现构造向量优先、保留原文上下文的高召回语料包。

        向量命中只负责定位，交给模型阅读的仍是 ``DocumentChunk.text`` 原文。
        同时加入相邻段落与跨章节抽样，避免单纯 Top-K 检索遗漏分散披露。
        """
        if semantic_top_k_per_query < 1 or lexical_top_k_per_query < 1:
            raise ValueError("discovery retrieval limits must be positive")
        if neighbor_radius < 0:
            raise ValueError("neighbor_radius must be non-negative")
        if coverage_section_count < 1 or max_chunks < 1:
            raise ValueError("discovery coverage limits must be positive")

        normalized_queries = tuple(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        )
        seed_ids: list[str] = []
        seen_seed_ids: set[str] = set()

        def add_seed(chunk_id: str) -> None:
            if chunk_id in self._chunk_by_id and chunk_id not in seen_seed_ids:
                seen_seed_ids.add(chunk_id)
                seed_ids.append(chunk_id)

        if self._semantic is not None:
            try:
                for query in normalized_queries:
                    for hit in self._semantic.search(
                        query,
                        top_k=semantic_top_k_per_query,
                    ):
                        add_seed(hit.chunk.chunk_id)
            except Exception as exc:  # noqa: BLE001
                self._retrieval_error = str(exc)

        # 精确词和数字容易被向量相似度稀释，BM25只作为召回补充工具。
        for query in normalized_queries:
            for hit in self._lexical.search(query, top_k=lexical_top_k_per_query):
                if hit.score <= 0:
                    continue
                add_seed(hit.chunk.chunk_id)

        primary_ids: list[str] = []
        seen_primary_ids: set[str] = set()
        for chunk_id in seed_ids:
            index = self._chunk_index[chunk_id]
            for neighbor_index in range(
                max(0, index - neighbor_radius),
                min(len(self._chunks), index + neighbor_radius + 1),
            ):
                neighbor_id = self._chunks[neighbor_index].chunk_id
                if neighbor_id not in seen_primary_ids:
                    seen_primary_ids.add(neighbor_id)
                    primary_ids.append(neighbor_id)

        headings = [chunk for chunk in self._chunks if chunk.is_heading]
        coverage_anchors = headings if headings else list(self._chunks)
        if len(coverage_anchors) > coverage_section_count:
            denominator = max(1, coverage_section_count - 1)
            indexes = {
                round(index * (len(coverage_anchors) - 1) / denominator)
                for index in range(coverage_section_count)
            }
            coverage_anchors = [coverage_anchors[index] for index in sorted(indexes)]

        coverage_ids: list[str] = []
        seen_coverage_ids: set[str] = set()
        for anchor in coverage_anchors:
            anchor_index = self._chunk_index[anchor.chunk_id]
            for index in range(
                anchor_index,
                min(len(self._chunks), anchor_index + 3),
            ):
                chunk_id = self._chunks[index].chunk_id
                if chunk_id not in seen_coverage_ids:
                    seen_coverage_ids.add(chunk_id)
                    coverage_ids.append(chunk_id)

        # 给章节覆盖预留容量，不能让高相似度命中挤掉整份文档的结构性兜底。
        coverage_limit = min(len(coverage_ids), max(1, max_chunks // 3))
        reserved_coverage = coverage_ids[:coverage_limit]
        selected_ids = primary_ids[: max(0, max_chunks - coverage_limit)]
        selected_seen = set(selected_ids)
        for chunk_id in reserved_coverage:
            if chunk_id not in selected_seen and len(selected_ids) < max_chunks:
                selected_seen.add(chunk_id)
                selected_ids.append(chunk_id)

        # 检索结果过窄时，按全文位置均匀补足少量原文，不重新扫描整份文档。
        minimum_corpus_size = min(
            max_chunks,
            max(80, coverage_section_count * 3),
            len(self._chunks),
        )
        if len(selected_ids) < minimum_corpus_size:
            denominator = max(1, minimum_corpus_size - 1)
            for sample_index in range(minimum_corpus_size):
                chunk_index = round(sample_index * (len(self._chunks) - 1) / denominator)
                chunk_id = self._chunks[chunk_index].chunk_id
                if chunk_id not in selected_seen:
                    selected_seen.add(chunk_id)
                    selected_ids.append(chunk_id)
                if len(selected_ids) >= minimum_corpus_size:
                    break

        selected_ids = selected_ids[:max_chunks]
        return tuple(
            sorted(
                (self._chunk_by_id[chunk_id] for chunk_id in selected_ids),
                key=lambda chunk: self._chunk_index[chunk.chunk_id],
            )
        )

    @property
    def profile_retrieval_mode(self) -> str:
        return "hybrid" if self._semantic is not None else "bm25"

    def retrieve_profile_context(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 4,
        max_chars: int = 24000,
        neighbor_radius: int = 1,
    ) -> tuple[DocumentChunk, ...]:
        """按主题均衡取回原文；标题仅用于定位，不随机抽样、不截断证据。"""
        buckets: list[list[DocumentChunk]] = []
        for query in dict.fromkeys(queries):
            query_tokens = set(tokenize_text(query))
            lexical = [hit for hit in self._lexical.search(query, top_k=top_k)
                       if query_tokens & set(tokenize_text(hit.chunk.text))]
            semantic: list[SearchHit] = []
            if self._semantic is not None:
                try:
                    semantic = self._semantic.search(query, top_k=top_k)
                except Exception as exc:  # noqa: BLE001 - preserve lexical retrieval
                    self.enable_bm25_fallback(str(exc))
            # Both channels contribute without comparing incompatible score scales.
            seeds: list[DocumentChunk] = []
            for rank in range(max(len(lexical), len(semantic))):
                for hits in (lexical, semantic):
                    if rank < len(hits) and hits[rank].chunk.chunk_id in self._chunk_by_id:
                        seeds.append(self._chunk_by_id[hits[rank].chunk.chunk_id])
            neighbors: list[DocumentChunk] = []
            for seed in seeds:
                index = self._chunk_index[seed.chunk_id]
                for offset in range(-neighbor_radius, neighbor_radius + 1):
                    position = index + offset
                    if 0 <= position < len(self._chunks):
                        neighbor = self._chunks[position]
                        if (neighbor.document_id == seed.document_id
                                and abs(neighbor.page - seed.page) <= 1):
                            neighbors.append(neighbor)
            buckets.append([*seeds, *neighbors])
        selected: dict[str, DocumentChunk] = {}
        used = 0
        for rank in range(max((len(bucket) for bucket in buckets), default=0)):
            for bucket in buckets:
                if rank >= len(bucket):
                    continue
                candidate = bucket[rank]
                if candidate.is_heading or candidate.chunk_id in selected:
                    continue
                if used + len(candidate.text) > max_chars:
                    continue
                selected[candidate.chunk_id] = candidate
                used += len(candidate.text)
        return tuple(selected.values())

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
        seed_chunk_ids_by_risk: Mapping[str, Sequence[str]] | None = None,
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
            seed_chunk_ids = tuple(
                dict.fromkeys(
                    chunk_id.strip()
                    for chunk_id in (seed_chunk_ids_by_risk or {}).get(rule.id, ())
                    if chunk_id.strip()
                )
            )
            unknown_seeds = set(seed_chunk_ids) - set(self._chunk_by_id)
            if unknown_seeds:
                raise ValueError(f"discovery seeded unknown chunks: {sorted(unknown_seeds)}")
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
                        self.enable_bm25_fallback(retrieval_error)
                if candidates:
                    retrieval_mode = "agentic_semantic"
                else:
                    candidates = self._lexical_candidates(rule.model_copy(update={
                        "query": planned_queries[0], "query_variants": planned_queries[1:]
                    }))
                    retrieval_mode = "agentic_bm25"
                    if not candidates:
                        candidates = self._lexical_candidates(rule)
                        retrieval_mode = "fixed_query_fallback"
            elif self._hybrid is not None:
                retrieval_mode = "hybrid_multi_query" if len(rule.search_queries) > 1 else "hybrid"
                try:
                    candidates = self._hybrid_candidates(rule)
                except Exception as exc:  # noqa: BLE001
                    retrieval_error = str(exc)
                    self.enable_bm25_fallback(retrieval_error)
                    candidates = self._lexical_candidates(rule)
                    retrieval_mode = "bm25_fallback"
            else:
                retrieval_mode = "bm25_multi_query" if len(rule.search_queries) > 1 else "bm25"
                candidates = self._lexical_candidates(rule)
            if not candidates and not planned_queries:
                candidates = self._lexical_candidates(rule)
            if seed_chunk_ids:
                seeded = [
                    RetrievalCandidate(
                        chunk=self._chunk_by_id[chunk_id],
                        score=1.0,
                        rank=0,
                    )
                    for chunk_id in seed_chunk_ids
                ]
                seeded_ids = set(seed_chunk_ids)
                candidates = (
                    seeded
                    + [
                        candidate
                        for candidate in candidates
                        if candidate.chunk.chunk_id not in seeded_ids
                    ]
                )[: self._rerank_top_k]
                retrieval_mode = f"{retrieval_mode}_with_discovery_seed"

            analysis_error: str | None = None
            analysis_attempts = 0
            analysis_retry_errors: tuple[str, ...] = ()
            analyzer = self._evidence_analyzer
            if analyzer is not None and candidates:
                try:
                    candidate_chunks = [candidate.chunk for candidate in candidates]
                    analyze_with_feedback = getattr(
                        analyzer,
                        "analyze_with_feedback",
                        None,
                    )
                    if callable(analyze_with_feedback):
                        outcome = self._analysis_retry_policy.run_with_feedback(
                            lambda correction_feedback: analyze_with_feedback(
                                rule,
                                candidate_chunks,
                                correction_feedback=correction_feedback,
                            )
                        )
                    else:
                        outcome = self._analysis_retry_policy.run(
                            lambda: analyzer.analyze(
                                rule,
                                candidate_chunks,
                            )
                        )
                    analysis = outcome.value
                    analysis_attempts = outcome.attempts
                    analysis_retry_errors = outcome.errors
                    analysis_mode = analyzer.provider_name
                except RetryExhaustedError as error:
                    if isinstance(error.last_error, ModelBudgetExceededError):
                        raise error.last_error
                    analysis_attempts = error.attempts
                    analysis_error = str(error.last_error)
                    analysis_retry_errors = (analysis_error,)
                    if rule.requires_model_verification or rule.id.startswith(("open_", "check_")):
                        analysis = _open_risk_fail_closed(rule, analysis_error)
                        analysis_mode = "open_fail_closed"
                    else:
                        analysis = _deterministic_analysis(rule, exact_hits)
                        analysis_mode = "rule_fallback"
            else:
                if rule.requires_model_verification or rule.id.startswith(("open_", "check_")):
                    analysis_error = (
                        "专业证据模型不可用"
                        if analyzer is None
                        else "没有可供专业证据模型核验的候选原文"
                    )
                    analysis = _open_risk_fail_closed(rule, analysis_error)
                    analysis_mode = "open_fail_closed"
                else:
                    analysis = _deterministic_analysis(rule, exact_hits)
                    analysis_mode = "rule"
                    analysis_attempts = 1

            chunk_by_id = {candidate.chunk.chunk_id: candidate.chunk for candidate in candidates}
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
            audit: dict[str, str] = {}
            prompt_audit_getter = getattr(analyzer, "prompt_audit", None)
            if (
                analyzer is not None
                and analysis_mode == analyzer.provider_name
                and callable(prompt_audit_getter)
            ):
                audit = prompt_audit_getter(rule.owner)
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
                    prompt_version=audit.get("prompt_version"),
                    prompt_sha256=audit.get("prompt_sha256"),
                    prompt_snapshot=audit.get("prompt_snapshot"),
                    analysis_retry_errors=analysis_retry_errors,
                    specialist_status=(
                        SpecialistStatus.SUPPORTED
                        if analysis.supported
                        else SpecialistStatus.EVIDENCE_INSUFFICIENT
                    ),
                )
            )
        return results

    def analyze_rules_batched(
        self,
        rules: Sequence[RiskRule],
        *,
        search_queries_by_risk: Mapping[str, Sequence[str]] | None = None,
        seed_chunk_ids_by_risk: Mapping[str, Sequence[str]] | None = None,
    ) -> list[AnalyzedRisk]:
        """先检索全部风险，再由一个专业角色按批次核验。

        只有 ``needs_more_evidence`` 会触发至多一轮额外语义检索和第二个批次；模型失败不会
        回退为算法自动支持，候选风险以 ``evidence_insufficient`` 留在审计账本。
        """
        analyzer = self._specialist_analyzer
        if analyzer is None:
            return self.analyze_rules(
                rules,
                search_queries_by_risk=search_queries_by_risk,
                seed_chunk_ids_by_risk=seed_chunk_ids_by_risk,
            )
        if not rules:
            return []

        # specialist_analyzer 与 single-risk analyzer 互斥，因此这里复用现有检索实现时不会
        # 产生任何付费证据分析调用。
        retrieved = self.analyze_rules(
            rules,
            search_queries_by_risk=search_queries_by_risk,
            seed_chunk_ids_by_risk=seed_chunk_ids_by_risk,
        )
        retrieved_by_risk = {item.rule.id: item for item in retrieved}
        tasks = [
            SpecialistTask(
                rule=item.rule,
                candidates=tuple(candidate.chunk for candidate in item.candidates),
            )
            for item in retrieved
            if item.candidates
        ]
        result = analyzer.analyze_batch(tasks)
        assessments = result.by_risk()
        attempts_by_risk = dict(result.attempts_by_risk)
        errors_by_risk = dict(result.errors_by_risk)

        for item in retrieved:
            if item.rule.id not in assessments:
                assessments[item.rule.id] = SpecialistAssessment(
                    risk_type=item.rule.id,
                    status=SpecialistStatus.EVIDENCE_INSUFFICIENT,
                    severity=item.rule.severity,
                    confidence=0.0,
                    uncertainty="未召回可供专业Agent核验的候选原文，风险仍保留在审计账本。",
                )
                attempts_by_risk[item.rule.id] = 0
                errors_by_risk[item.rule.id] = ()

        for _ in range(self._specialist_followup_rounds):
            followup = [
                assessment
                for assessment in assessments.values()
                if assessment.status is SpecialistStatus.NEEDS_MORE_EVIDENCE
                and assessment.followup_queries
            ]
            if not followup:
                break
            followup_tasks: list[SpecialistTask] = []
            for assessment in followup:
                original = retrieved_by_risk[assessment.risk_type]
                extra = self.analyze_rules(
                    (original.rule,),
                    search_queries_by_risk={
                        original.rule.id: assessment.followup_queries,
                    },
                    seed_chunk_ids_by_risk=seed_chunk_ids_by_risk,
                )[0]
                merged_candidates = _merge_retrieval_candidates(
                    original.candidates,
                    extra.candidates,
                )
                retrieved_by_risk[assessment.risk_type] = replace(
                    original,
                    candidates=merged_candidates,
                    retrieval_mode=f"{original.retrieval_mode}+agent_followup",
                    search_queries=tuple(
                        dict.fromkeys((*original.search_queries, *assessment.followup_queries))
                    ),
                )
                followup_tasks.append(
                    SpecialistTask(
                        rule=original.rule,
                        candidates=tuple(candidate.chunk for candidate in merged_candidates),
                    )
                )
            followup_result = analyzer.analyze_batch(followup_tasks)
            for assessment in followup_result.assessments:
                assessments[assessment.risk_type] = assessment
            for risk_type, attempts in followup_result.attempts_by_risk.items():
                attempts_by_risk[risk_type] = attempts_by_risk.get(risk_type, 0) + attempts
            for risk_type, errors in followup_result.errors_by_risk.items():
                errors_by_risk[risk_type] = (
                    *errors_by_risk.get(risk_type, ()),
                    *errors,
                )

        analyzed: list[AnalyzedRisk] = []
        for rule in rules:
            base = retrieved_by_risk[rule.id]
            assessment = assessments[rule.id]
            supported = assessment.status is SpecialistStatus.SUPPORTED
            analysis = EvidenceAnalysis(
                supported=supported,
                claim=assessment.claim if supported else "",
                severity=assessment.severity,
                confidence=assessment.confidence,
                evidence_chunk_ids=(assessment.evidence_chunk_ids if supported else ()),
                uncertainty=assessment.uncertainty,
            )
            chunk_by_id = {
                candidate.chunk.chunk_id: candidate.chunk for candidate in base.candidates
            }
            selected = tuple(chunk_by_id[chunk_id] for chunk_id in analysis.evidence_chunk_ids)
            audit = analyzer.prompt_audit(rule.owner)
            retry_errors = errors_by_risk.get(rule.id, ())
            analyzed.append(
                replace(
                    base,
                    analysis=analysis,
                    selected_chunks=selected,
                    analysis_mode=(
                        f"{analyzer.provider_name}_batch" if base.candidates else "no_candidate"
                    ),
                    analysis_attempts=attempts_by_risk.get(rule.id, 0),
                    analysis_error=(
                        retry_errors[-1]
                        if retry_errors
                        and assessment.status is SpecialistStatus.EVIDENCE_INSUFFICIENT
                        else None
                    ),
                    prompt_version=audit.get("prompt_version"),
                    prompt_sha256=audit.get("prompt_sha256"),
                    prompt_snapshot=audit.get("prompt_snapshot"),
                    analysis_retry_errors=retry_errors,
                    specialist_status=assessment.status,
                    followup_queries=assessment.followup_queries,
                )
            )
        return analyzed

    def analyze(self) -> list[AnalyzedRisk]:
        """兼容原有调用方式，分析初始化阶段加载的全部规则。"""
        return self.analyze_rules()


def _merge_retrieval_candidates(
    primary: Sequence[RetrievalCandidate],
    extra: Sequence[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    """按已有顺序合并两轮召回，保留每个 chunk 的最佳分数。"""
    by_id: dict[str, RetrievalCandidate] = {}
    order: list[str] = []
    for candidate in (*primary, *extra):
        chunk_id = candidate.chunk.chunk_id
        current = by_id.get(chunk_id)
        if current is None:
            order.append(chunk_id)
            by_id[chunk_id] = candidate
        elif candidate.score > current.score:
            by_id[chunk_id] = candidate
    return tuple(by_id[chunk_id] for chunk_id in order)


__all__ = [
    "AnalyzedRisk",
    "EvidenceAnalyzer",
    "GroundedAnalysisEngine",
    "RetrievalCandidate",
]
