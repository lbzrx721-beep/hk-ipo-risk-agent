"""单份招股书从 PDF 到可核验风险 JSON 的第一阶段流水线。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import uuid4

import fitz
from opencc import OpenCC
from pydantic import JsonValue

from ipo_risk.agents.analysis import (
    OpenAICompatibleJsonChatProvider,
)
from ipo_risk.agents.decision import DecisionAnalyzer, DeepSeekDecisionAnalyzer
from ipo_risk.agents.discovery import DeepSeekOpenRiskDiscoverer, RiskDiscoverer
from ipo_risk.agents.grounded import EvidenceAnalyzer, GroundedAnalysisEngine
from ipo_risk.agents.intervention import (
    LlmFailureRecoveryController,
    RiskScopeReviewController,
)
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator
from ipo_risk.agents.specialist import (
    DeepSeekSpecialistBatchAnalyzer,
    SpecialistBatchAnalyzer,
)
from ipo_risk.agents.usage import (
    ModelBudget,
    ModelPricing,
    ModelUsageLedger,
    ModelUsageSnapshot,
)
from ipo_risk.agents.risk_scoping import (
    DeepSeekCompanyRiskScopePlanner,
    RiskScopePlanner,
)
from ipo_risk.agents.search_planning import (
    DeepSeekRoleSearchPlanner,
    RoleSearchPlanner,
)
from ipo_risk.calculations import build_calculation
from ipo_risk.config import load_config
from ipo_risk.domain import (
    AgentRole,
    Evidence,
    EvidenceType,
    FeatureValue,
    RiskFinding,
    ToolCall,
    ToolCallStatus,
    TraceStatus,
    VectorIndexRecord,
    VerificationStatus,
)
from ipo_risk.parser.document_parser import PageResult, parse_document
from ipo_risk.parser.evidence_renderer import render_evidence_screenshot
from ipo_risk.parser.layout import extract_printed_page_label
from ipo_risk.modeling.prelisting import (
    collect_prelisting_candidates,
)
from ipo_risk.modeling.rag_fundamentals import RagFundamentalExtractor
from ipo_risk.modeling.structured_fundamentals import (
    extract_structured_fundamental_snapshot,
)
from ipo_risk.reporting import (
    AnalysisSummary,
    RiskAnalysisReport,
    SourceDocument,
    write_risk_report,
)
from ipo_risk.retrieval import (
    DocumentChunk,
    build_page_chunks,
    load_risk_rules,
)
from ipo_risk.retrieval.semantic import (
    EmbeddingProgressCallback,
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SemanticRetriever,
)
from ipo_risk.retrieval.pgvector import PgVectorRetriever
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.runtime.prediction import (
    generate_runtime_prediction,
)
from ipo_risk.storage.files import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


from ipo_risk.storage.ingestion import register_company_sources
from ipo_risk.storage.paths import DocumentPaths
from ipo_risk.storage.repositories import CatalogRepository


class _UnavailableRiskDiscoverer:
    """把缺少模型配置也转换为可由页面处理的模型故障。"""

    prompt_version = "document-first-risk-discovery-v6"

    def __init__(self, *, provider_name: str, model_name: str, reason: str) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._reason = reason

    def discover(self, chunks: Any, known_rules: Any) -> Any:  # noqa: ARG002
        raise RuntimeError(self._reason)


_TO_SIMPLIFIED = OpenCC("t2s")
ProgressCallback = Callable[[str, str], None]
PreparationArtifactCallback = Callable[[Mapping[str, Any]], None]


class _DeferredSemanticRetriever:
    """后台建库，风险发现首次检索原文时再等待索引就绪。"""

    def __init__(
        self,
        future: Future[SemanticRetriever | PgVectorRetriever | None],
        progress_callback: ProgressCallback | None,
    ) -> None:
        self._future = future
        self._progress_callback = progress_callback
        self._join_lock = Lock()
        self._join_reported = False

    def search(self, query: str, top_k: int = 20) -> list[Any]:
        if not self._future.done():
            with self._join_lock:
                if not self._join_reported:
                    self._join_reported = True
                    _emit_progress(
                        self._progress_callback,
                        "embedding_join_wait",
                        "风险发现正在等待内存语义索引，以检索风险原文与相邻上下文。",
                    )
        retriever = self._future.result()
        if self._join_reported:
            with self._join_lock:
                if self._join_reported:
                    self._join_reported = False
                    _emit_progress(
                        self._progress_callback,
                        "embedding_join_completed",
                        "内存语义索引已接入，风险发现开始构建高召回原文包。",
                    )
        return [] if retriever is None else retriever.search(query, top_k=top_k)

    def close(self) -> None:
        """分析结束后释放pgvector连接。"""
        retriever = self._future.result()
        if isinstance(retriever, PgVectorRetriever):
            retriever.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, detail)


def _portable_path(path: Path, project_root: Path) -> str:
    """项目内路径保存为 POSIX 相对路径，项目外路径保存为绝对路径。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _page_record(
    page: PageResult,
    *,
    company_id: str,
    document_id: str,
    source_hash: str,
) -> dict[str, Any]:
    record = asdict(page)
    record["company_id"] = company_id
    record["document_id"] = document_id
    record["source_hash"] = source_hash
    record["printed_page_label"] = extract_printed_page_label(
        page.text_blocks,
        page_height=page.height,
    )
    return record


@dataclass(frozen=True)
class AnalysisArtifacts:
    """流水线结果以及可直接打开的产物路径。"""

    report: RiskAnalysisReport
    report_path: Path
    pages_path: Path
    chunks_path: Path
    tables_path: Path | None = None
    embedding_dir: Path | None = None
    prelisting_features_path: Path | None = None
    prediction_path: Path | None = None
    model_usage: ModelUsageSnapshot | None = None


@dataclass(frozen=True)
class PreparedProspectus:
    """可在批量预处理与正式Agent分析之间复用的文档级产物。"""

    pages: tuple[dict[str, Any], ...]
    tables: tuple[dict[str, Any], ...]
    chunks: tuple[DocumentChunk, ...]
    paths: DocumentPaths
    cache_hit: bool
    started_at: datetime
    ended_at: datetime


_PREPARATION_SCHEMA_VERSION = 1


def _prepare_prospectus_source(
    *,
    pdf_path: Path,
    company_id: str,
    company_folder: str,
    document_id: str,
    source_file: str,
    source_hash: str,
    available_at: date,
    interim_root: Path,
    artifacts_root: Path,
    min_text_chars_before_ocr: int,
    progress_callback: ProgressCallback | None,
) -> PreparedProspectus:
    """解析PDF并落盘；相同文档与解析参数再次运行时直接复用。"""
    started_at = _now()
    paths = DocumentPaths.build(
        interim_root=interim_root,
        artifacts_root=artifacts_root,
        company_folder=company_folder,
        document_id=document_id,
    )
    manifest_path = paths.chunks_path.with_name("preparation.json")
    expected_manifest = {
        "schema_version": _PREPARATION_SCHEMA_VERSION,
        "source_hash": source_hash,
        "min_text_chars_before_ocr": min_text_chars_before_ocr,
    }
    if (
        manifest_path.is_file()
        and paths.pages_path.is_file()
        and paths.tables_path.is_file()
        and paths.chunks_path.is_file()
    ):
        try:
            manifest = read_json(manifest_path)
            if all(manifest.get(key) == value for key, value in expected_manifest.items()):
                pages = tuple(read_jsonl(paths.pages_path))
                tables = tuple(read_jsonl(paths.tables_path))
                chunks = tuple(
                    DocumentChunk.model_validate(item) for item in read_jsonl(paths.chunks_path)
                )
                if pages and chunks:
                    _emit_progress(
                        progress_callback,
                        "parsing_cache_hit",
                        (f"已复用解析缓存：{len(pages)}页，{len(chunks)}个文本切片。"),
                    )
                    return PreparedProspectus(
                        pages=pages,
                        tables=tables,
                        chunks=chunks,
                        paths=paths,
                        cache_hit=True,
                        started_at=started_at,
                        ended_at=_now(),
                    )
        except (OSError, TypeError, ValueError):
            # 缓存不完整或版本不兼容时安全重建，不污染正式分析。
            pass

    pages_buffer: list[dict[str, Any]] = []
    tables_buffer: list[dict[str, Any]] = []
    chunks_buffer: list[DocumentChunk] = []
    with fitz.open(pdf_path) as source_document:
        total_pages = source_document.page_count
    _emit_progress(progress_callback, "parsing_started", "开始解析PDF页面与坐标。")
    parsed_pages = parse_document(
        pdf_path,
        min_chars=min_text_chars_before_ocr,
    )
    for page_number, page_result in enumerate(parsed_pages, start=1):
        pages_buffer.append(
            _page_record(
                page_result,
                company_id=company_id,
                document_id=document_id,
                source_hash=source_hash,
            )
        )
        tables_buffer.extend(
            {
                **asdict(table),
                "company_id": company_id,
                "document_id": document_id,
                "source_hash": source_hash,
            }
            for table in page_result.tables
        )
        chunks_buffer.extend(
            build_page_chunks(
                page_result,
                company_id=company_id,
                document_id=document_id,
                source_file=source_file,
                source_hash=source_hash,
                available_at=available_at,
            )
        )
        _emit_progress(
            progress_callback,
            "parsing_progress",
            (
                f"[{page_number}/{total_pages}] 已解析PDF第{page_number}页，"
                "正在构建文本、坐标与视觉区域索引。"
            ),
        )
    if not chunks_buffer:
        raise RuntimeError("prospectus parsing produced no searchable chunks")
    write_jsonl(pages_buffer, paths.pages_path)
    write_jsonl(tables_buffer, paths.tables_path)
    write_jsonl(
        [chunk.model_dump(mode="json") for chunk in chunks_buffer],
        paths.chunks_path,
    )
    write_json(
        {
            **expected_manifest,
            "page_count": len(pages_buffer),
            "table_count": len(tables_buffer),
            "chunk_count": len(chunks_buffer),
        },
        manifest_path,
    )
    return PreparedProspectus(
        pages=tuple(pages_buffer),
        tables=tuple(tables_buffer),
        chunks=tuple(chunks_buffer),
        paths=paths,
        cache_hit=False,
        started_at=started_at,
        ended_at=_now(),
    )


def analyze_prospectus(
    *,
    pdf_path: Path,
    company_id: str,
    company_name: str,
    company_folder: str,
    document_id: str,
    available_at: date,
    project_root: Path,
    interim_root: Path,
    artifacts_root: Path,
    risk_rules_path: Path,
    min_findings: int = 1,
    min_text_chars_before_ocr: int = 80,
    render_dpi: int = 150,
    expected_source_hash: str | None = None,
    company_info: Mapping[str, object] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    rag_fundamental_extractor: RagFundamentalExtractor | None = None,
    evidence_analyzer: EvidenceAnalyzer | None = None,
    specialist_analyzer: SpecialistBatchAnalyzer | None = None,
    specialist_followup_rounds: int = 1,
    decision_analyzer: DecisionAnalyzer | None = None,
    search_planner: RoleSearchPlanner | None = None,
    risk_discoverer: RiskDiscoverer | None = None,
    risk_scope_planner: RiskScopePlanner | None = None,
    risk_scope_review_controller: RiskScopeReviewController | None = None,
    llm_failure_recovery_controller: LlmFailureRecoveryController | None = None,
    decision_conflict_batch_size: int = 12,
    vector_store_backend: str = "numpy",
    postgres_dsn: str | None = None,
    postgres_schema: str = "ipo_risk",
    postgres_connect_timeout_seconds: float = 10.0,
    lexical_top_k: int = 80,
    semantic_top_k: int = 20,
    rerank_top_k: int = 8,
    max_model_retries: int = 0,
    retry_initial_seconds: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_max_seconds: float = 8.0,
    progress_callback: ProgressCallback | None = None,
    model_usage_ledger: ModelUsageLedger | None = None,
) -> AnalysisArtifacts:
    """解析一份 PDF，召回证据并输出带截图的结构化风险报告。"""
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if min_findings < 1:
        raise ValueError("min_findings must be positive")

    run_started = _now()
    run_id = f"RUN-{company_id}-{run_started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    source_hash = sha256_file(pdf_path)
    if expected_source_hash and source_hash != expected_source_hash.lower():
        raise ValueError("prospectus SHA-256 does not match company metadata")
    source_file = _portable_path(pdf_path, project_root)

    prepared = _prepare_prospectus_source(
        pdf_path=pdf_path,
        company_id=company_id,
        company_folder=company_folder,
        document_id=document_id,
        source_file=source_file,
        source_hash=source_hash,
        available_at=available_at,
        interim_root=interim_root,
        artifacts_root=artifacts_root,
        min_text_chars_before_ocr=min_text_chars_before_ocr,
        progress_callback=progress_callback,
    )
    pages = list(prepared.pages)
    tables = list(prepared.tables)
    chunks = list(prepared.chunks)
    parse_started = prepared.started_at
    parse_ended = prepared.ended_at
    visual_region_count = sum(len(page.get("visual_regions", [])) for page in pages)

    document_paths = prepared.paths
    _emit_progress(
        progress_callback,
        "parsing_completed",
        (
            f"{'解析缓存复用' if prepared.cache_hit else 'PDF解析完成'}："
            f"{len(pages)}页，{len(chunks)}个文本切片，"
            f"结构化{len(tables)}张表格，并识别{visual_region_count}个"
            "表格/图片/流程图候选。"
        ),
    )

    tool_calls: list[ToolCall] = [
        ToolCall(
            tool_call_id=f"TC-{company_id}-parse",
            tool_name="parse_document",
            started_at=parse_started,
            ended_at=parse_ended,
            inputs={
                "source_file": source_file,
                "min_text_chars_before_ocr": min_text_chars_before_ocr,
            },
            output={
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "structured_table_count": len(tables),
                "visual_region_count": visual_region_count,
                "visual_detection_mode": "local_geometry_first",
                "cache_hit": prepared.cache_hit,
            },
            status=ToolCallStatus.SUCCEEDED,
        )
    ]

    rules = load_risk_rules(risk_rules_path)
    analysis_started = _now()
    agent_start_detail = (
        "开始动态向量检索、原文风险发现、专业Agent核验与决策裁决。"
        if risk_discoverer is not None
        else f"动态发现未启用，使用{len(rules)}类目录风险启动兜底分析。"
    )
    _emit_progress(progress_callback, "agents_started", agent_start_detail)
    if risk_discoverer is None:
        _emit_progress(
            progress_callback,
            "agent:discovery:skipped",
            "全文动态风险发现未启用；固定目录仅作为兜底继续运行。",
        )

    # ------------------------------------------------------------------
    # 语义索引只在这里构建一次。失败后直接降级 BM25，避免整篇文档重复调用。
    # ------------------------------------------------------------------
    import logging as _logging

    _logger = _logging.getLogger(__name__)
    embedding_cache_dir = (
        document_paths.embedding_dir(embedding_provider.model_name)
        if embedding_provider is not None
        else None
    )
    semantic_retriever: (
        SemanticRetriever | PgVectorRetriever | _DeferredSemanticRetriever | None
    ) = None
    vector_executor: ThreadPoolExecutor | None = None
    vector_future: Future[SemanticRetriever | PgVectorRetriever | None] | None = None
    if embedding_provider is not None:

        def build_semantic_index() -> SemanticRetriever | PgVectorRetriever | None:
            _emit_progress(
                progress_callback,
                "embedding_started",
                f"后台正在建立语义索引：{len(chunks)}个招股书切片。",
            )
            try:
                if vector_store_backend == "pgvector":
                    if not postgres_dsn:
                        raise ValueError("postgres_dsn is required for pgvector")
                    pgvector = PgVectorRetriever(
                        embedding_provider,
                        dsn=postgres_dsn,
                        schema=postgres_schema,
                        connect_timeout_seconds=postgres_connect_timeout_seconds,
                    )
                    if pgvector.activate_document(
                        document_id,
                        source_hash=source_hash,
                        expected_chunk_count=len(chunks),
                    ):
                        added = 0
                    else:
                        memory_retriever = SemanticRetriever(
                            chunks,
                            embedding_provider,
                            cache_dir=embedding_cache_dir,
                        )
                        added = pgvector.add_chunks(
                            chunks,
                            skip_existing_document=False,
                            precomputed_embeddings=memory_retriever.document_vectors,
                            company_name=company_name,
                        )
                    retriever = pgvector
                    action = "复用完整文档" if added == 0 else f"新增{added}条向量"
                    _emit_progress(
                        progress_callback,
                        "vector_store_indexed",
                        f"pgvector索引就绪：{action}。",
                    )
                elif vector_store_backend == "numpy":
                    memory_retriever = SemanticRetriever(
                        chunks,
                        embedding_provider,
                        cache_dir=embedding_cache_dir,
                    )
                    retriever = memory_retriever
                else:
                    raise ValueError("vector_store_backend must be numpy or pgvector")
                _emit_progress(
                    progress_callback,
                    "embedding_completed",
                    (f"后台语义索引已就绪：{len(chunks)}条，{embedding_provider.dimensions}维。"),
                )
                return retriever
            except Exception as exc:  # noqa: BLE001
                _logger.warning("语义索引不可用，降级到BM25: %s", exc, exc_info=True)
                _emit_progress(
                    progress_callback,
                    "embedding_fallback",
                    f"语义索引失败，已降级到关键词检索：{exc}",
                )
                return None

        vector_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="semantic-index",
        )
        vector_future = vector_executor.submit(build_semantic_index)
        semantic_retriever = _DeferredSemanticRetriever(
            vector_future,
            progress_callback,
        )

    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        evidence_analyzer=evidence_analyzer,
        specialist_analyzer=specialist_analyzer,
        specialist_followup_rounds=specialist_followup_rounds,
        embedding_cache_dir=embedding_cache_dir,
        semantic_retriever=semantic_retriever,
        lexical_top_k=lexical_top_k,
        semantic_top_k=semantic_top_k,
        rerank_top_k=rerank_top_k,
        analysis_retry_policy=RetryPolicy(
            max_retries=max_model_retries,
            initial_seconds=retry_initial_seconds,
            backoff_factor=retry_backoff_factor,
            max_seconds=retry_max_seconds,
            retry_if=retry_model_node_error,
        ),
    )
    try:
        multi_agent_result = MultiAgentOrchestrator(
            engine,
            rules,
            decision_analyzer=decision_analyzer,
            decision_retry_policy=RetryPolicy(
                max_retries=max_model_retries,
                initial_seconds=retry_initial_seconds,
                backoff_factor=retry_backoff_factor,
                max_seconds=retry_max_seconds,
                retry_if=retry_model_node_error,
            ),
            search_planner=search_planner,
            risk_discoverer=risk_discoverer,
            risk_scope_planner=risk_scope_planner,
            risk_scope_review_controller=risk_scope_review_controller,
            llm_failure_recovery_controller=llm_failure_recovery_controller,
            decision_conflict_batch_size=decision_conflict_batch_size,
            risk_scope_retry_policy=RetryPolicy(
                max_retries=max_model_retries,
                initial_seconds=retry_initial_seconds,
                backoff_factor=retry_backoff_factor,
                max_seconds=retry_max_seconds,
                retry_if=retry_model_node_error,
            ),
            search_retry_policy=RetryPolicy(
                max_retries=max_model_retries,
                initial_seconds=retry_initial_seconds,
                backoff_factor=retry_backoff_factor,
                max_seconds=retry_max_seconds,
                retry_if=retry_model_node_error,
            ),
        ).run(
            run_id=run_id,
            source_ref=source_file,
            progress_callback=progress_callback,
        )
        if vector_future is not None:
            vector_future.result()
    finally:
        if vector_executor is not None:
            vector_executor.shutdown(wait=True)
    analyzed_risks = [proposal.analyzed_risk for proposal in multi_agent_result.proposals]
    decision_by_risk = {decision.risk_type: decision for decision in multi_agent_result.decisions}
    supported_decision_count = sum(
        decision.analysis.supported for decision in multi_agent_result.decisions
    )
    unresolved_decision_count = len(multi_agent_result.decisions) - supported_decision_count
    _emit_progress(
        progress_callback,
        "agents_completed",
        (
            f"风险整合完成：机器采纳{supported_decision_count}条正式风险，"
            f"{unresolved_decision_count}条未纳入；原始提案完整保留审计。"
        ),
    )
    analysis_ended = _now()
    if model_usage_ledger is not None:
        usage_snapshot = model_usage_ledger.snapshot()
        tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{company_id}-model-usage",
                tool_name="model_usage_budget",
                started_at=analysis_started,
                ended_at=analysis_ended,
                inputs={"scope": "single_company_analysis"},
                output=cast(JsonValue, usage_snapshot.as_dict()),
                status=ToolCallStatus.SUCCEEDED,
            )
        )
    evidences: list[Evidence] = []
    findings: list[RiskFinding] = []
    evidence_display: dict[str, str] = {}
    evidence_by_chunk: dict[str, Evidence] = {}
    screenshot_dir = document_paths.evidence_dir
    _emit_progress(
        progress_callback,
        "evidence_started",
        "开始生成证据对象、高亮截图与候选报告。",
    )

    for analyzed in analyzed_risks:
        rule = analyzed.rule
        tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{company_id}-retrieval-{rule.id}",
                tool_name="evidence_retrieval",
                started_at=analysis_started,
                ended_at=analysis_ended,
                inputs={
                    "risk_type": rule.id,
                    "query": rule.query,
                    "search_queries": list(analyzed.search_queries),
                    "required_keywords": list(rule.required_keywords),
                    "lexical_top_k": lexical_top_k,
                    "semantic_top_k": semantic_top_k,
                    "rerank_top_k": rerank_top_k,
                    "max_retries": (
                        getattr(embedding_provider, "max_retries", 0)
                        if embedding_provider is not None
                        else 0
                    ),
                },
                output={
                    "mode": analyzed.retrieval_mode,
                    "embedding_model": (
                        embedding_provider.model_name if embedding_provider is not None else None
                    ),
                    "fallback_error": analyzed.retrieval_error,
                    "embedding_retries": (
                        int(getattr(embedding_provider, "retry_count", 0))
                        if embedding_provider is not None
                        else 0
                    ),
                    "embedding_requests": (
                        int(getattr(embedding_provider, "request_count", 0))
                        if embedding_provider is not None
                        else 0
                    ),
                    "candidates": [
                        {
                            "chunk_id": candidate.chunk.chunk_id,
                            "page": candidate.chunk.page,
                            "score": round(candidate.score, 6),
                            "rank": candidate.rank,
                            "lexical_score": (
                                round(candidate.lexical_score, 6)
                                if candidate.lexical_score is not None
                                else None
                            ),
                            "lexical_rank": candidate.lexical_rank,
                            "semantic_score": (
                                round(candidate.semantic_score, 6)
                                if candidate.semantic_score is not None
                                else None
                            ),
                            "semantic_rank": candidate.semantic_rank,
                        }
                        for candidate in analyzed.candidates
                    ],
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )
        tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{company_id}-analysis-{rule.id}",
                tool_name="evidence_analysis",
                started_at=analysis_started,
                ended_at=analysis_ended,
                inputs={
                    "risk_type": rule.id,
                    "candidate_chunk_ids": [
                        candidate.chunk.chunk_id for candidate in analyzed.candidates
                    ],
                },
                output={
                    "mode": analyzed.analysis_mode,
                    "model": (
                        evidence_analyzer.model_name
                        if evidence_analyzer is not None
                        and analyzed.analysis_mode == evidence_analyzer.provider_name
                        else None
                    ),
                    "prompt_version": (
                        analyzed.prompt_version
                        or (
                            evidence_analyzer.prompt_version
                            if evidence_analyzer is not None
                            and analyzed.analysis_mode == evidence_analyzer.provider_name
                            else None
                        )
                    ),
                    "prompt_sha256": analyzed.prompt_sha256,
                    "prompt_snapshot": analyzed.prompt_snapshot,
                    "supported": analyzed.analysis.supported,
                    "selected_chunk_ids": list(analyzed.analysis.evidence_chunk_ids),
                    "severity": analyzed.analysis.severity.value,
                    "confidence": analyzed.analysis.confidence,
                    "uncertainty": analyzed.analysis.uncertainty,
                    "fallback_error": analyzed.analysis_error,
                    "attempts": analyzed.analysis_attempts,
                    "retries": max(0, analyzed.analysis_attempts - 1),
                    "repair_errors": list(analyzed.analysis_retry_errors),
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )
        decision = decision_by_risk.get(rule.id)
        if decision is None:
            continue

        final_analysis = decision.analysis
        if not final_analysis.supported:
            _emit_progress(
                progress_callback,
                "decision_excluded",
                f"风险{rule.id}未通过机器裁决，不纳入正式风险卡；原提案保留审计。",
            )
            continue
        selected_chunk_by_id = {
            chunk.chunk_id: chunk for chunk in decision.selected.selected_chunks
        }
        selected_chunks = tuple(
            selected_chunk_by_id[chunk_id] for chunk_id in final_analysis.evidence_chunk_ids
        )
        finding_claim = final_analysis.claim
        finding_uncertainty = final_analysis.uncertainty

        finding_evidence_ids: list[str] = []
        for chunk in selected_chunks:
            evidence = evidence_by_chunk.get(chunk.chunk_id)
            if evidence is None:
                evidence_id = f"E-{company_id}-{len(evidences) + 1:04d}"
                screenshot_path = screenshot_dir / f"{evidence_id}.png"
                screenshot_started = _now()
                rendered = render_evidence_screenshot(
                    pdf_path=pdf_path,
                    page=chunk.page,
                    bbox=chunk.bbox,
                    output_path=screenshot_path,
                    dpi=render_dpi,
                )
                screenshot_ended = _now()
                if rendered is None:
                    raise RuntimeError(f"failed to render evidence screenshot for {chunk.chunk_id}")
                screenshot_ref = _portable_path(rendered, project_root)
                tool_calls.append(
                    ToolCall(
                        tool_call_id=f"TC-{company_id}-render-{evidence_id}",
                        tool_name="render_evidence_screenshot",
                        started_at=screenshot_started,
                        ended_at=screenshot_ended,
                        inputs={
                            "page": chunk.page,
                            "bbox": list(chunk.bbox),
                            "chunk_id": chunk.chunk_id,
                        },
                        output={"screenshot_path": screenshot_ref},
                        status=ToolCallStatus.SUCCEEDED,
                    )
                )
                evidence = Evidence(
                    evidence_id=evidence_id,
                    company_id=company_id,
                    document_id=document_id,
                    source_file=source_file,
                    source_hash=source_hash,
                    page=chunk.page,
                    printed_page_label=chunk.printed_page_label,
                    bbox=chunk.bbox,
                    quote=chunk.text,
                    evidence_type=(
                        EvidenceType.TABLE
                        if chunk.chunk_kind == "table"
                        else (EvidenceType.OCR_TEXT if chunk.is_ocr else EvidenceType.TEXT)
                    ),
                    extractor=AgentRole.DOCUMENT_PARSER,
                    confidence=0.82 if chunk.is_ocr else 0.97,
                    available_at=available_at,
                    screenshot_path=screenshot_ref,
                    table_id=chunk.table_id,
                    table_cell_refs=chunk.table_cell_refs,
                    table_cell_values=chunk.table_cell_values,
                )
                evidence_by_chunk[chunk.chunk_id] = evidence
                evidences.append(evidence)
                evidence_display[evidence.evidence_id] = _TO_SIMPLIFIED.convert(evidence.quote)
            finding_evidence_ids.append(evidence.evidence_id)

        calculation = None
        if rule.calculation is not None:
            if company_info is None:
                # 缺少结构化资料时跳过数值复算，保留风险结论，不中断整体分析
                _emit_progress(
                    progress_callback,
                    "calculation_skipped",
                    f"规则{rule.id}需要结构化资料做数值复算，但未提供company_info，已跳过。",
                )
            else:
                try:
                    calculation = build_calculation(rule.calculation, company_info)
                except ValueError as exc:
                    # 新上传公司可能缺少该计算所需的结构化字段（如基石配售数据）。
                    # 数值复算只是可选增强，不应中断整体分析；跳过复算并保留风险结论。
                    _emit_progress(
                        progress_callback,
                        "calculation_skipped",
                        f"规则{rule.id}的数值复算输入缺失，已跳过：{exc}",
                    )
        findings.append(
            RiskFinding(
                finding_id=f"F-{company_id}-{len(findings) + 1:04d}",
                company_id=company_id,
                risk_type=rule.id,
                title=rule.name,
                severity=final_analysis.severity,
                claim=finding_claim,
                evidence_ids=tuple(finding_evidence_ids),
                status=VerificationStatus.PENDING_VERIFICATION,
                confidence=round(final_analysis.confidence, 2),
                created_by=rule.owner,
                calculation=calculation,
                uncertainty=finding_uncertainty,
                specialist_status=decision.specialist_status.value,
            )
        )

    if supported_decision_count < min_findings:
        coverage_warning = (
            f"本次仅形成{supported_decision_count}条有充分证据风险，"
            f"低于期望值{min_findings}；另有{unresolved_decision_count}条"
            "证据不足候选未纳入正式风险，原始提案保留审计。"
        )
        _emit_progress(
            progress_callback,
            "coverage_warning",
            coverage_warning,
        )
        tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{company_id}-coverage-warning",
                tool_name="risk_coverage_gate",
                started_at=analysis_started,
                ended_at=_now(),
                inputs={"minimum_expected_findings": min_findings},
                output={
                    "actual_grounded_findings": supported_decision_count,
                    "threshold_met": False,
                    "action": "preserve_report_and_candidate_ledger",
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )

    run_ended = _now()
    trace = multi_agent_result.root_trace.model_copy(
        update={
            "started_at": run_started,
            "ended_at": run_ended,
            "input_refs": (source_file,),
            "tool_calls": (
                *tuple(tool_calls),
                *multi_agent_result.root_trace.tool_calls,
            ),
            "output_refs": tuple(finding.finding_id for finding in findings),
            "status": TraceStatus.SUCCEEDED,
            "decision": (
                f"多角色Agent与决策Agent机器采纳{len(findings)}条有原文、页码、"
                "坐标和截图的正式候选风险；人工复核仅做质量确认。"
            ),
        }
    )
    child_traces = tuple(
        item
        for item in multi_agent_result.traces
        if item.trace_id != multi_agent_result.root_trace.trace_id
    )
    report = RiskAnalysisReport(
        run_id=run_id,
        company_id=company_id,
        company_name=company_name,
        generated_at=run_ended,
        source_document=SourceDocument(
            document_id=document_id,
            source_file=source_file,
            source_hash=source_hash,
            available_at=available_at,
            page_count=len(pages),
            chunk_count=len(chunks),
        ),
        summary=AnalysisSummary(
            evidence_count=len(evidences),
            finding_count=len(findings),
            screenshot_count=sum(evidence.screenshot_path is not None for evidence in evidences),
        ),
        evidences=tuple(evidences),
        risk_findings=tuple(findings),
        agent_traces=(trace, *child_traces),
        evidence_display_zh_hans=evidence_display,
    )
    prelisting_features_path: Path | None = None
    prelisting_feature_snapshot: Mapping[str, object] = {}
    prelisting_feature_units: Mapping[str, str] = {}
    prelisting_feature_formula_versions: Mapping[str, str] = {}
    prediction_as_of: date | None = None
    if company_info is not None:
        try:
            raw_as_of = company_info.get("as_of_date")
            source_as_of: date | None = None
            if isinstance(raw_as_of, datetime):
                source_as_of = raw_as_of.date()
            elif isinstance(raw_as_of, date):
                source_as_of = raw_as_of
            elif isinstance(raw_as_of, str):
                try:
                    source_as_of = date.fromisoformat(raw_as_of.strip())
                except ValueError:
                    source_as_of = None
            raw_listing_date = company_info.get("listing_date")
            listing_date: date | None = None
            if isinstance(raw_listing_date, datetime):
                listing_date = raw_listing_date.date()
            elif isinstance(raw_listing_date, date):
                listing_date = raw_listing_date
            elif isinstance(raw_listing_date, str):
                try:
                    listing_date = date.fromisoformat(raw_listing_date.strip())
                except ValueError:
                    listing_date = None
            candidate_as_of = listing_date or source_as_of
            prediction_as_of = listing_date
            if candidate_as_of is not None:
                supplemental_features: dict[str, FeatureValue] = {}
                if rag_fundamental_extractor is not None:
                    rag_sidecar_path = document_paths.prelisting_features_path.with_name(
                        "rag_fundamental_features.json"
                    )
                    _emit_progress(
                        progress_callback,
                        "rag_features_started",
                        "正在从本公司pgvector招股书索引检索预测所需财务、发行与业务事实。",
                    )
                    try:
                        existing_operating_cash_flow: float | None = None
                        structured_for_rag = extract_structured_fundamental_snapshot(
                            project_root,
                            available_at=report.source_document.available_at,
                            table_path=document_paths.tables_path,
                            chunk_path=document_paths.chunks_path,
                        )
                        operating_cash_flow = structured_for_rag.features.get(
                            "fundamental_operating_cash_flow_latest_native"
                        )
                        if operating_cash_flow is not None and isinstance(
                            operating_cash_flow.value, (int, float)
                        ):
                            existing_operating_cash_flow = float(operating_cash_flow.value)
                        rag_record = rag_fundamental_extractor.extract(
                            manifest_row={
                                "company_id": company_id,
                                "company_name": company_name,
                                "prospectus_date": report.source_document.available_at.isoformat(),
                                "source_hash": report.source_document.source_hash,
                            },
                            document_id=document_id,
                            chunks=chunks,
                            existing_operating_cash_flow=existing_operating_cash_flow,
                            progress=lambda detail: _emit_progress(
                                progress_callback,
                                "rag_features_progress",
                                detail,
                            ),
                        )
                        write_json(rag_record, rag_sidecar_path, pretty=True)
                        supplemental_features = {
                            name: FeatureValue.model_validate(payload)
                            for name, payload in dict(rag_record.get("features") or {}).items()
                        }
                        available_rag_count = sum(
                            feature.value is not None
                            for feature in supplemental_features.values()
                        )
                        _emit_progress(
                            progress_callback,
                            "rag_features_completed",
                            f"本公司RAG事实提取完成：取得{available_rag_count}项可用字段，审计记录已保存。",
                        )
                    except Exception as exc:  # noqa: BLE001 - RAG补充失败不丢风险报告
                        _logger.warning("当前公司RAG预测特征提取失败: %s", exc)
                        _emit_progress(
                            progress_callback,
                            "rag_features_failed",
                            f"本公司RAG事实提取失败，继续使用表格和公式结果：{exc}",
                        )
                _emit_progress(
                    progress_callback,
                    "features_started",
                    (
                        "正在自动采集发行、财务、文档、上市前市场、近期IPO、"
                        "Agent风险与数据质量候选特征。"
                    ),
                )
                candidate_snapshot = collect_prelisting_candidates(
                    report,
                    project_root=project_root,
                    company_info=dict(company_info),
                    prediction_as_of=candidate_as_of,
                    tables_path=document_paths.tables_path,
                    chunks_path=document_paths.chunks_path,
                    listing_date=listing_date,
                    supplemental_features=supplemental_features,
                    progress_callback=lambda current, total, detail: _emit_progress(
                        progress_callback,
                        "features_progress",
                        f"[{current}/{total}] {detail}",
                    ),
                )
                write_json(
                    candidate_snapshot.to_payload(
                        run_id=run_id,
                        company_id=company_id,
                    ),
                    document_paths.prelisting_features_path,
                    pretty=True,
                )
                prelisting_features_path = document_paths.prelisting_features_path
                prelisting_feature_snapshot = candidate_snapshot.features
                prelisting_feature_units = candidate_snapshot.feature_units
                prelisting_feature_formula_versions = candidate_snapshot.feature_formula_versions
                _emit_progress(
                    progress_callback,
                    "features_completed",
                    (
                        "上市前宽候选特征快照已生成："
                        f"{len(candidate_snapshot.features)}项；"
                        "预测时将按模型契约固定取值。"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("上市前特征快照生成失败，不影响风险报告: %s", exc)
            _emit_progress(
                progress_callback,
                "features_failed",
                f"风险报告已完成，但上市前特征快照生成失败：{exc}",
            )
    write_risk_report(report, document_paths.report_path)
    prediction_path: Path | None = None
    try:
        _emit_progress(
            progress_callback,
            "prediction_started",
            "正在合并上市前基础特征与最新Agent风险特征。",
        )
        prediction, prediction_path = generate_runtime_prediction(
            project_root=project_root,
            company_id=company_id,
            company_name=company_name,
            run_id=run_id,
            features=prelisting_feature_snapshot,
            report_path=document_paths.report_path,
            prediction_as_of=prediction_as_of,
            feature_units=prelisting_feature_units,
            feature_formula_versions=prelisting_feature_formula_versions,
        )
        _emit_progress(
            progress_callback,
            "prediction_completed",
            (
                "五日显著下跌预测完成："
                f"概率{prediction['probability']:.1%}。"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - 风险报告不能因预测资源失败而丢失
        _logger.warning("五日显著下跌预测失败，不影响风险报告: %s", exc)
        _emit_progress(
            progress_callback,
            "prediction_failed",
            f"风险与证据已保存，但五日显著下跌预测暂不可用：{exc}",
        )
    _emit_progress(
        progress_callback,
        "completed",
        f"报告生成完成：{len(findings)}条风险，{len(evidences)}条原文证据。",
    )
    embedding_dir = None
    if embedding_provider is not None:
        candidate_embedding_dir = document_paths.embedding_dir(embedding_provider.model_name)
        required_index_files = (
            candidate_embedding_dir / "vectors.npy",
            candidate_embedding_dir / "metadata.json",
            candidate_embedding_dir / "manifest.json",
        )
        if all(path.is_file() for path in required_index_files):
            embedding_dir = candidate_embedding_dir
    if isinstance(semantic_retriever, _DeferredSemanticRetriever):
        semantic_retriever.close()
    if rag_fundamental_extractor is not None:
        rag_fundamental_extractor.retriever.close()
    return AnalysisArtifacts(
        report=report,
        report_path=document_paths.report_path,
        pages_path=document_paths.pages_path,
        chunks_path=document_paths.chunks_path,
        tables_path=document_paths.tables_path,
        embedding_dir=embedding_dir,
        prelisting_features_path=prelisting_features_path,
        prediction_path=prediction_path,
        model_usage=(model_usage_ledger.snapshot() if model_usage_ledger is not None else None),
    )


def analyze_company(
    company_folder: str,
    *,
    project_root: Path | None = None,
    document_id: str | None = None,
    min_findings: int = 1,
    risk_scope_review_controller: RiskScopeReviewController | None = None,
    llm_failure_recovery_controller: LlmFailureRecoveryController | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisArtifacts:
    """登记公司资料，并分析指定或最新的正式招股书版本。"""
    config = load_config()
    root = (project_root or config.project_root).resolve()
    repository = CatalogRepository(root / config.paths.database)
    ingestion = register_company_sources(
        company_folder,
        project_root=root,
        repository=repository,
    )
    prospectuses = [
        item
        for item in ingestion.documents
        if item.document_type == "prospectus" and item.media_type == "application/pdf"
    ]
    if document_id is not None:
        prospectuses = [item for item in prospectuses if item.document_id == document_id]
        if not prospectuses:
            raise ValueError(f"prospectus document_id is not registered: {document_id}")
    if not prospectuses:
        raise RuntimeError(f"no registered prospectus PDF for {company_folder}")
    document = max(
        prospectuses,
        key=lambda item: (item.available_at, item.document_date, item.document_id),
    )
    pdf_path = root / document.local_file
    company_info = read_json(root / ingestion.company.metadata_file)

    embedding_provider: EmbeddingProvider | None = None
    if config.embedding.is_configured:
        # 首次向量化可能耗时较长，逐批上报百分比避免界面看似卡死。
        # detail 中嵌入 [completed/total]，供进度视图解析并映射到总进度条。
        embedding_progress: EmbeddingProgressCallback | None = None
        if progress_callback is not None:

            def embedding_progress(completed: int, total: int) -> None:
                percent = round(completed / total * 100) if total else 100
                progress_callback(
                    "embedding_progress",
                    f"首次向量化 [{completed}/{total}] · {percent}%",
                )

        embedding_provider = OpenAICompatibleEmbeddingProvider(
            api_key=config.embedding.require_api_key(),
            base_url=config.embedding.base_url,
            model_name=config.embedding.model_name,
            api_model_name=config.embedding.request_model_name,
            dimensions=config.embedding.dimensions,
            batch_size=config.embedding.batch_size,
            request_concurrency=config.embedding.request_concurrency,
            timeout_seconds=config.embedding.timeout_seconds,
            max_retries=config.embedding.max_retries,
            retry_initial_seconds=config.embedding.retry_initial_seconds,
            retry_backoff_factor=config.embedding.retry_backoff_factor,
            retry_max_seconds=config.embedding.retry_max_seconds,
            progress_callback=embedding_progress,
        )

    evidence_analyzer: EvidenceAnalyzer | None = None
    specialist_analyzer: SpecialistBatchAnalyzer | None = None
    decision_analyzer: DecisionAnalyzer | None = None
    search_planner: RoleSearchPlanner | None = None
    risk_discoverer: RiskDiscoverer | None = None
    risk_scope_planner: RiskScopePlanner | None = None
    rag_fundamental_extractor: RagFundamentalExtractor | None = None
    model_usage_ledger: ModelUsageLedger | None = None
    if (
        config.agents.enable_open_discovery
        and llm_failure_recovery_controller is not None
        and not config.llm.is_configured
    ):
        risk_discoverer = _UnavailableRiskDiscoverer(
            provider_name=config.llm.provider,
            model_name=config.llm.model_name,
            reason=(
                f"{config.llm.provider} API Key 未配置，风险发现LLM无法启动；"
                "请配置密钥并重启服务后重新分析，或选择BM25降级继续。"
            ),
        )
    if config.llm.is_configured:
        budget_config = config.agents.model_budget
        model_usage_ledger = ModelUsageLedger(
            budget=ModelBudget(
                max_calls=budget_config.max_calls,
                max_input_tokens=budget_config.max_input_tokens,
                max_output_tokens=budget_config.max_output_tokens,
                max_cost_rmb=budget_config.max_cost_rmb,
            )
        )
        role_prompts = {
            role: (root / "prompts" / "roles" / f"{role.value}_v2.md").read_text(encoding="utf-8")
            for role in (
                AgentRole.FINANCE,
                AgentRole.LEGAL,
                AgentRole.BUSINESS,
                AgentRole.MARKET,
            )
        }
        chat_provider = OpenAICompatibleJsonChatProvider(
            api_key=config.llm.require_api_key(),
            base_url=config.llm.base_url,
            model_name=config.llm.model_name,
            provider_name=config.llm.provider,
            request_extra_body=(None if config.llm.disable_thinking else {}),
            supports_json_object=config.llm.supports_json_object,
            reasoning_effort=config.llm.reasoning_effort,
            usage_ledger=model_usage_ledger,
            pricing=ModelPricing(
                input_per_million_rmb=config.llm.input_price_per_million_rmb,
                output_per_million_rmb=config.llm.output_price_per_million_rmb,
                cached_input_per_million_rmb=(config.llm.cached_input_price_per_million_rmb),
            ),
        )
        # 风险发现由主 DeepSeek 模型分批扫描，四个专业角色再各用一次
        # specialist_llm 长上下文模型核验。
        specialist_config = (
            config.specialist_llm
            if config.specialist_llm is not None and config.specialist_llm.is_configured
            else config.llm
        )

        def build_specialist_provider(
            role: AgentRole,
        ) -> OpenAICompatibleJsonChatProvider:
            def report_reasoning_started() -> None:
                if progress_callback is not None:
                    progress_callback(
                        f"agent:{role.value}:reasoning",
                        "模型正在推理，尚未输出结构化正文。",
                    )

            return OpenAICompatibleJsonChatProvider(
                api_key=specialist_config.require_api_key(),
                base_url=specialist_config.base_url,
                model_name=specialist_config.model_name,
                provider_name=specialist_config.provider,
                request_extra_body=(None if specialist_config.disable_thinking else {}),
                supports_json_object=specialist_config.supports_json_object,
                reasoning_effort=specialist_config.reasoning_effort,
                reasoning_progress_callback=(
                    report_reasoning_started if progress_callback is not None else None
                ),
                usage_ledger=model_usage_ledger,
                pricing=ModelPricing(
                    input_per_million_rmb=(specialist_config.input_price_per_million_rmb),
                    output_per_million_rmb=(specialist_config.output_price_per_million_rmb),
                    cached_input_per_million_rmb=(
                        specialist_config.cached_input_price_per_million_rmb
                    ),
                ),
            )

        specialist_providers = {
            role: build_specialist_provider(role)
            for role in (
                AgentRole.FINANCE,
                AgentRole.LEGAL,
                AgentRole.BUSINESS,
                AgentRole.MARKET,
            )
        }
        specialist_provider = specialist_providers[AgentRole.FINANCE]
        if embedding_provider is not None and config.vector_store.backend == "pgvector":
            rag_fundamental_extractor = RagFundamentalExtractor(
                retriever=PgVectorRetriever(
                    embedding_provider,
                    dsn=config.vector_store.require_postgres_dsn(),
                    schema=config.vector_store.postgres_schema,
                    connect_timeout_seconds=(
                        config.vector_store.postgres_connect_timeout_seconds
                    ),
                ),
                provider=specialist_provider,
                system_prompt=(
                    root / "prompts" / "extraction" / "rag_fundamentals_v1.md"
                ).read_text(encoding="utf-8"),
                max_output_tokens=min(specialist_config.max_output_tokens, 12_000),
                retry_policy=RetryPolicy(
                    max_retries=config.agents.max_retries,
                    initial_seconds=config.agents.retry_initial_seconds,
                    backoff_factor=config.agents.retry_backoff_factor,
                    max_seconds=config.agents.retry_max_seconds,
                    retry_if=retry_model_node_error,
                ),
            )
        specialist_analyzer = DeepSeekSpecialistBatchAnalyzer(
            specialist_provider,
            temperature=specialist_config.temperature,
            max_output_tokens=specialist_config.max_output_tokens,
            system_prompt=(root / "prompts" / "decision" / "evidence_analysis_v5.md").read_text(
                encoding="utf-8"
            ),
            role_prompts=role_prompts,
            providers_by_role=specialist_providers,
            prompt_version="specialist-evidence-batch-v3",
            progress_callback=progress_callback,
            batch_size=config.agents.specialist_batch_size,
            retry_policy=RetryPolicy(
                max_retries=config.agents.max_retries,
                initial_seconds=config.agents.retry_initial_seconds,
                backoff_factor=config.agents.retry_backoff_factor,
                max_seconds=config.agents.retry_max_seconds,
                retry_if=retry_model_node_error,
            ),
        )
        risk_scope_planner = DeepSeekCompanyRiskScopePlanner(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(root / "prompts" / "discovery" / "company_risk_scope_v4.md").read_text(
                encoding="utf-8"
            ),
            prompt_version="company-risk-scope-v4",
        )
        search_planner = DeepSeekRoleSearchPlanner(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(root / "prompts" / "search" / "role_search_planning_v4.md").read_text(
                encoding="utf-8"
            ),
            role_prompts=role_prompts,
            prompt_version="role-search-planning-v4",
            progress_callback=progress_callback,
        )
        decision_analyzer = DeepSeekDecisionAnalyzer(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(root / "prompts" / "decision" / "decision_adjudication_v4.md").read_text(
                encoding="utf-8"
            ),
            prompt_version="decision-adjudication-v4",
            progress_callback=progress_callback,
        )
        if config.agents.enable_open_discovery:
            risk_discoverer = DeepSeekOpenRiskDiscoverer(
                chat_provider,
                temperature=config.llm.temperature,
                max_output_tokens=config.llm.max_output_tokens,
                system_prompt=(
                    root / "prompts" / "discovery" / "document_risk_discovery_v6.md"
                ).read_text(encoding="utf-8"),
                consolidation_system_prompt=(
                    root / "prompts" / "discovery" / "risk_consolidation_v3.md"
                ).read_text(encoding="utf-8"),
                retrieval_planning_system_prompt=(
                    root / "prompts" / "discovery" / "vector_query_planning_v2.md"
                ).read_text(encoding="utf-8"),
                prompt_version="document-first-risk-discovery-v6",
                progress_callback=progress_callback,
                retry_policy=RetryPolicy(
                    max_retries=config.agents.max_retries,
                    initial_seconds=config.agents.retry_initial_seconds,
                    backoff_factor=config.agents.retry_backoff_factor,
                    max_seconds=config.agents.retry_max_seconds,
                    retry_if=retry_model_node_error,
                ),
                batch_char_limit=config.agents.discovery_batch_chars,
                wave_size=config.agents.discovery_wave_size,
                max_batch_passes=config.agents.discovery_max_batch_passes,
                consolidation_batch_size=(config.agents.consolidation_batch_size),
                retrieval_query_count=(config.agents.discovery_retrieval_query_count),
                retrieval_top_k=config.agents.discovery_retrieval_top_k,
                retrieval_lexical_top_k=(config.agents.discovery_retrieval_lexical_top_k),
                retrieval_neighbor_radius=(config.agents.discovery_retrieval_neighbor_radius),
                retrieval_coverage_sections=(config.agents.discovery_coverage_sections),
                retrieval_max_selected_chunks=(config.agents.discovery_max_selected_chunks),
            )

    artifacts = analyze_prospectus(
        pdf_path=pdf_path,
        company_id=ingestion.company.company_id,
        company_name=ingestion.company.company_name_zh,
        company_folder=company_folder,
        document_id=document.document_id,
        available_at=document.available_at,
        project_root=root,
        interim_root=root / config.paths.interim_dir,
        artifacts_root=root / config.paths.artifacts_dir,
        risk_rules_path=root / "configs" / "risks.yaml",
        min_findings=min_findings,
        min_text_chars_before_ocr=config.parser.min_text_chars_before_ocr,
        render_dpi=config.parser.render_dpi,
        expected_source_hash=document.source_hash,
        company_info=company_info,
        embedding_provider=embedding_provider,
        rag_fundamental_extractor=rag_fundamental_extractor,
        evidence_analyzer=evidence_analyzer,
        specialist_analyzer=specialist_analyzer,
        specialist_followup_rounds=config.agents.specialist_followup_rounds,
        decision_analyzer=decision_analyzer,
        search_planner=search_planner,
        risk_discoverer=risk_discoverer,
        risk_scope_planner=risk_scope_planner,
        risk_scope_review_controller=risk_scope_review_controller,
        llm_failure_recovery_controller=llm_failure_recovery_controller,
        decision_conflict_batch_size=config.agents.decision_conflict_batch_size,
        vector_store_backend=config.vector_store.backend,
        postgres_dsn=(
            config.vector_store.require_postgres_dsn()
            if config.vector_store.backend == "pgvector"
            else None
        ),
        postgres_schema=config.vector_store.postgres_schema,
        postgres_connect_timeout_seconds=(config.vector_store.postgres_connect_timeout_seconds),
        lexical_top_k=config.retrieval.lexical_top_k,
        semantic_top_k=config.retrieval.semantic_top_k,
        rerank_top_k=config.retrieval.rerank_top_k,
        max_model_retries=config.agents.max_retries,
        retry_initial_seconds=config.agents.retry_initial_seconds,
        retry_backoff_factor=config.agents.retry_backoff_factor,
        retry_max_seconds=config.agents.retry_max_seconds,
        progress_callback=progress_callback,
        model_usage_ledger=model_usage_ledger,
    )
    repository.save_analysis_report(
        artifacts.report,
        _portable_path(artifacts.report_path, root),
    )
    if embedding_provider is not None and artifacts.embedding_dir is not None:
        index_suffix = sha256(
            f"{embedding_provider.model_name}:{embedding_provider.dimensions}".encode()
        ).hexdigest()[:12]
        repository.upsert_vector_index(
            VectorIndexRecord(
                index_id=f"VI-{document.document_id}-{index_suffix}",
                document_id=document.document_id,
                source_hash=document.source_hash,
                embedding_provider=config.embedding.provider,
                embedding_model=embedding_provider.model_name,
                dimensions=embedding_provider.dimensions,
                chunk_count=artifacts.report.source_document.chunk_count,
                vectors_path=_portable_path(
                    artifacts.embedding_dir / "vectors.npy",
                    root,
                ),
                metadata_path=_portable_path(
                    artifacts.embedding_dir / "metadata.json",
                    root,
                ),
                manifest_path=_portable_path(
                    artifacts.embedding_dir / "manifest.json",
                    root,
                ),
                status="ready",
                created_at=artifacts.report.generated_at,
            )
        )
    return artifacts


@dataclass(frozen=True)
class CompanyPreparationResult:
    """批量队列在正式Agent分析前完成的可复用预处理摘要。"""

    company_folder: str
    document_id: str
    page_count: int
    chunk_count: int
    embedding_ready: bool
    embedding_cache_hit: bool
    parser_cache_hit: bool
    base_feature_path: str | None = None
    base_prediction: dict[str, Any] | None = None
    base_prediction_path: str | None = None
    base_prediction_status: str = "unavailable"
    base_prediction_error: str | None = None


def prepare_company_semantic_index(
    company_folder: str,
    *,
    project_root: Path | None = None,
    request_concurrency: int | None = None,
    embedding_provider: OpenAICompatibleEmbeddingProvider | None = None,
    progress_callback: ProgressCallback | None = None,
    artifact_callback: PreparationArtifactCallback | None = None,
) -> CompanyPreparationResult:
    """只解析并向量化一家公司，不调用风险或专业Agent。"""
    config = load_config()
    root = (project_root or config.project_root).resolve()
    repository = CatalogRepository(root / config.paths.database)
    ingestion = register_company_sources(
        company_folder,
        project_root=root,
        repository=repository,
    )
    prospectuses = [
        item
        for item in ingestion.documents
        if item.document_type == "prospectus" and item.media_type == "application/pdf"
    ]
    if not prospectuses:
        raise RuntimeError(f"no registered prospectus PDF for {company_folder}")
    document = max(
        prospectuses,
        key=lambda item: (item.available_at, item.document_date, item.document_id),
    )
    pdf_path = root / document.local_file
    source_hash = sha256_file(pdf_path)
    if source_hash != document.source_hash.lower():
        raise ValueError("prospectus SHA-256 does not match company metadata")
    prepared = _prepare_prospectus_source(
        pdf_path=pdf_path,
        company_id=ingestion.company.company_id,
        company_folder=company_folder,
        document_id=document.document_id,
        source_file=_portable_path(pdf_path, root),
        source_hash=source_hash,
        available_at=document.available_at,
        interim_root=root / config.paths.interim_dir,
        artifacts_root=root / config.paths.artifacts_dir,
        min_text_chars_before_ocr=config.parser.min_text_chars_before_ocr,
        progress_callback=progress_callback,
    )
    # 预处理阶段只负责PDF解析和向量索引。五日预测统一在Agent分析、
    # 当前公司RAG事实提取和决策裁决全部完成后执行。
    del artifact_callback
    embedding_ready = False
    embedding_cache_hit = False
    if config.embedding.is_configured:
        provider = (
            embedding_provider
            if embedding_provider is not None
            else OpenAICompatibleEmbeddingProvider(
                api_key=config.embedding.require_api_key(),
                base_url=config.embedding.base_url,
                model_name=config.embedding.model_name,
                api_model_name=config.embedding.request_model_name,
                dimensions=config.embedding.dimensions,
                batch_size=config.embedding.batch_size,
                request_concurrency=(
                    request_concurrency
                    if request_concurrency is not None
                    else config.embedding.request_concurrency
                ),
                timeout_seconds=config.embedding.timeout_seconds,
                max_retries=config.embedding.max_retries,
                retry_initial_seconds=config.embedding.retry_initial_seconds,
                retry_backoff_factor=config.embedding.retry_backoff_factor,
                retry_max_seconds=config.embedding.retry_max_seconds,
                progress_callback=(
                    (
                        lambda completed, total: _emit_progress(
                            progress_callback,
                            "embedding_progress",
                            (
                                f"预向量化 [{completed}/{total}] · "
                                f"{round(completed / total * 100) if total else 100}%"
                            ),
                        )
                    )
                    if progress_callback is not None
                    else None
                ),
            )
        )
        _emit_progress(
            progress_callback,
            "embedding_started",
            f"开始预向量化{len(prepared.chunks)}个招股书切片。",
        )
        SemanticRetriever(
            list(prepared.chunks),
            provider,
            cache_dir=prepared.paths.embedding_dir(provider.model_name),
        )
        embedding_cache_hit = provider.request_count == 0
        embedding_ready = True
        _emit_progress(
            progress_callback,
            "embedding_completed",
            f"预向量化完成：{len(prepared.chunks)}条。",
        )
    return CompanyPreparationResult(
        company_folder=company_folder,
        document_id=document.document_id,
        page_count=len(prepared.pages),
        chunk_count=len(prepared.chunks),
        embedding_ready=embedding_ready,
        embedding_cache_hit=embedding_cache_hit,
        parser_cache_hit=prepared.cache_hit,
    )


__all__ = [
    "AnalysisArtifacts",
    "CompanyPreparationResult",
    "PreparationArtifactCallback",
    "ProgressCallback",
    "analyze_company",
    "analyze_prospectus",
    "prepare_company_semantic_index",
]
