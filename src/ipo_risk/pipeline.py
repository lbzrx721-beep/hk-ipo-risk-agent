"""单份招股书从 PDF 到可核验风险 JSON 的第一阶段流水线。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from opencc import OpenCC

from ipo_risk.agents.analysis import (
    DeepSeekEvidenceAnalyzer,
    OpenAICompatibleJsonChatProvider,
)
from ipo_risk.agents.decision import DecisionAnalyzer, DeepSeekDecisionAnalyzer
from ipo_risk.agents.grounded import EvidenceAnalyzer, GroundedAnalysisEngine
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator
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
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.storage.files import read_json, sha256_file, write_jsonl
from ipo_risk.storage.ingestion import register_company_sources
from ipo_risk.storage.paths import DocumentPaths
from ipo_risk.storage.repositories import CatalogRepository


_TO_SIMPLIFIED = OpenCC("t2s")
ProgressCallback = Callable[[str, str], None]


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
    embedding_dir: Path | None = None


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
    min_findings: int = 5,
    min_text_chars_before_ocr: int = 80,
    render_dpi: int = 150,
    expected_source_hash: str | None = None,
    company_info: Mapping[str, object] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    evidence_analyzer: EvidenceAnalyzer | None = None,
    decision_analyzer: DecisionAnalyzer | None = None,
    search_planner: RoleSearchPlanner | None = None,
    lexical_top_k: int = 80,
    semantic_top_k: int = 20,
    rerank_top_k: int = 8,
    max_model_retries: int = 0,
    retry_initial_seconds: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_max_seconds: float = 8.0,
    progress_callback: ProgressCallback | None = None,
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

    pages: list[dict[str, Any]] = []
    chunks: list[DocumentChunk] = []
    parse_started = _now()
    _emit_progress(progress_callback, "parsing_started", "开始解析PDF页面与坐标。")
    for page_result in parse_document(
        pdf_path,
        min_chars=min_text_chars_before_ocr,
    ):
        pages.append(
            _page_record(
                page_result,
                company_id=company_id,
                document_id=document_id,
                source_hash=source_hash,
            )
        )
        chunks.extend(
            build_page_chunks(
                page_result,
                company_id=company_id,
                document_id=document_id,
                source_file=source_file,
                source_hash=source_hash,
                available_at=available_at,
            )
        )
    parse_ended = _now()
    if not chunks:
        raise RuntimeError("prospectus parsing produced no searchable chunks")

    document_paths = DocumentPaths.build(
        interim_root=interim_root,
        artifacts_root=artifacts_root,
        company_folder=company_folder,
        document_id=document_id,
    )
    write_jsonl(pages, document_paths.pages_path)
    write_jsonl(
        [chunk.model_dump(mode="json") for chunk in chunks],
        document_paths.chunks_path,
    )
    _emit_progress(
        progress_callback,
        "parsing_completed",
        f"PDF解析完成：{len(pages)}页，{len(chunks)}个文本切片。",
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
            output={"page_count": len(pages), "chunk_count": len(chunks)},
            status=ToolCallStatus.SUCCEEDED,
        )
    ]

    rules = load_risk_rules(risk_rules_path)
    analysis_started = _now()
    _emit_progress(
        progress_callback,
        "agents_started",
        f"开始运行{len(rules)}类风险的专业Agent与决策Agent。",
    )
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        embedding_provider=embedding_provider,
        evidence_analyzer=evidence_analyzer,
        embedding_cache_dir=(
            document_paths.embedding_dir(embedding_provider.model_name)
            if embedding_provider is not None
            else None
        ),
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
    analyzed_risks = [
        proposal.analyzed_risk for proposal in multi_agent_result.proposals
    ]
    decision_by_risk = {
        decision.risk_type: decision for decision in multi_agent_result.decisions
    }
    _emit_progress(
        progress_callback,
        "agents_completed",
        f"Agent裁决完成：接受{len(decision_by_risk)}条有证据风险。",
    )
    analysis_ended = _now()
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
                        embedding_provider.model_name
                        if embedding_provider is not None
                        else None
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
                        evidence_analyzer.prompt_version
                        if evidence_analyzer is not None
                        and analyzed.analysis_mode == evidence_analyzer.provider_name
                        else None
                    ),
                    "supported": analyzed.analysis.supported,
                    "selected_chunk_ids": list(
                        analyzed.analysis.evidence_chunk_ids
                    ),
                    "severity": analyzed.analysis.severity.value,
                    "confidence": analyzed.analysis.confidence,
                    "uncertainty": analyzed.analysis.uncertainty,
                    "fallback_error": analyzed.analysis_error,
                    "attempts": analyzed.analysis_attempts,
                    "retries": max(0, analyzed.analysis_attempts - 1),
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )
        decision = decision_by_risk.get(rule.id)
        if decision is None:
            continue

        final_analysis = decision.analysis
        selected_chunk_by_id = {
            chunk.chunk_id: chunk for chunk in analyzed.selected_chunks
        }
        selected_chunks = tuple(
            selected_chunk_by_id[chunk_id]
            for chunk_id in final_analysis.evidence_chunk_ids
        )

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
                    raise RuntimeError(
                        f"failed to render evidence screenshot for {chunk.chunk_id}"
                    )
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
                        EvidenceType.OCR_TEXT if chunk.is_ocr else EvidenceType.TEXT
                    ),
                    extractor=AgentRole.DOCUMENT_PARSER,
                    confidence=0.82 if chunk.is_ocr else 0.97,
                    available_at=available_at,
                    screenshot_path=screenshot_ref,
                )
                evidence_by_chunk[chunk.chunk_id] = evidence
                evidences.append(evidence)
                evidence_display[evidence.evidence_id] = _TO_SIMPLIFIED.convert(
                    evidence.quote
                )
            finding_evidence_ids.append(evidence.evidence_id)

        calculation = None
        if rule.calculation is not None:
            if company_info is None:
                raise ValueError(
                    f"company_info is required by calculation rule: {rule.id}"
                )
            calculation = build_calculation(rule.calculation, company_info)
        findings.append(
            RiskFinding(
                finding_id=f"F-{company_id}-{len(findings) + 1:04d}",
                company_id=company_id,
                risk_type=rule.id,
                title=rule.name,
                severity=final_analysis.severity,
                claim=final_analysis.claim,
                evidence_ids=tuple(finding_evidence_ids),
                status=VerificationStatus.PENDING_VERIFICATION,
                confidence=round(final_analysis.confidence, 2),
                created_by=rule.owner,
                calculation=calculation,
                uncertainty=final_analysis.uncertainty,
            )
        )

    if len(findings) < min_findings:
        raise RuntimeError(
            f"only {len(findings)} grounded findings were generated; "
            f"at least {min_findings} required"
        )

    run_ended = _now()
    trace = multi_agent_result.root_trace.model_copy(
        update={
            "started_at": run_started,
            "ended_at": run_ended,
            "input_refs": (source_file,),
            "tool_calls": tuple(tool_calls),
            "output_refs": tuple(finding.finding_id for finding in findings),
            "status": TraceStatus.SUCCEEDED,
            "decision": (
                f"多角色Agent与决策Agent生成{len(findings)}条有原文、页码、"
                "坐标和截图的候选风险，全部进入人工复核。"
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
            screenshot_count=sum(
                evidence.screenshot_path is not None for evidence in evidences
            ),
        ),
        evidences=tuple(evidences),
        risk_findings=tuple(findings),
        agent_traces=(trace, *child_traces),
        evidence_display_zh_hans=evidence_display,
    )
    write_risk_report(report, document_paths.report_path)
    _emit_progress(
        progress_callback,
        "completed",
        f"报告生成完成：{len(findings)}条风险，{len(evidences)}条原文证据。",
    )
    embedding_dir = None
    if embedding_provider is not None:
        candidate_embedding_dir = document_paths.embedding_dir(
            embedding_provider.model_name
        )
        required_index_files = (
            candidate_embedding_dir / "vectors.npy",
            candidate_embedding_dir / "metadata.json",
            candidate_embedding_dir / "manifest.json",
        )
        if all(path.is_file() for path in required_index_files):
            embedding_dir = candidate_embedding_dir
    return AnalysisArtifacts(
        report=report,
        report_path=document_paths.report_path,
        pages_path=document_paths.pages_path,
        chunks_path=document_paths.chunks_path,
        embedding_dir=embedding_dir,
    )


def analyze_company(
    company_folder: str,
    *,
    project_root: Path | None = None,
    document_id: str | None = None,
    min_findings: int = 5,
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
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            api_key=config.embedding.require_api_key(),
            base_url=config.embedding.base_url,
            model_name=config.embedding.model_name,
            dimensions=config.embedding.dimensions,
            batch_size=config.embedding.batch_size,
            max_retries=config.agents.max_retries,
            retry_initial_seconds=config.agents.retry_initial_seconds,
            retry_backoff_factor=config.agents.retry_backoff_factor,
            retry_max_seconds=config.agents.retry_max_seconds,
        )

    evidence_analyzer: EvidenceAnalyzer | None = None
    decision_analyzer: DecisionAnalyzer | None = None
    search_planner: RoleSearchPlanner | None = None
    if config.llm.is_configured:
        chat_provider = OpenAICompatibleJsonChatProvider(
            api_key=config.llm.require_api_key(),
            base_url=config.llm.base_url,
            model_name=config.llm.model_name,
        )
        evidence_analyzer = DeepSeekEvidenceAnalyzer(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(
                root / "prompts" / "decision" / "evidence_analysis_v1.md"
            ).read_text(encoding="utf-8"),
            prompt_version="evidence-analysis-v1",
            progress_callback=progress_callback,
        )
        search_planner = DeepSeekRoleSearchPlanner(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(
                root / "prompts" / "search" / "role_search_planning_v1.md"
            ).read_text(encoding="utf-8"),
            prompt_version="role-search-planning-v1",
            progress_callback=progress_callback,
        )
        decision_analyzer = DeepSeekDecisionAnalyzer(
            chat_provider,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(
                root / "prompts" / "decision" / "decision_adjudication_v1.md"
            ).read_text(encoding="utf-8"),
            prompt_version="decision-adjudication-v1",
            progress_callback=progress_callback,
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
        evidence_analyzer=evidence_analyzer,
        decision_analyzer=decision_analyzer,
        search_planner=search_planner,
        lexical_top_k=config.retrieval.lexical_top_k,
        semantic_top_k=config.retrieval.semantic_top_k,
        rerank_top_k=config.retrieval.rerank_top_k,
        max_model_retries=config.agents.max_retries,
        retry_initial_seconds=config.agents.retry_initial_seconds,
        retry_backoff_factor=config.agents.retry_backoff_factor,
        retry_max_seconds=config.agents.retry_max_seconds,
        progress_callback=progress_callback,
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


__all__ = [
    "AnalysisArtifacts",
    "ProgressCallback",
    "analyze_company",
    "analyze_prospectus",
]
