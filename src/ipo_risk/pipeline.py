"""单份招股书从 PDF 到可核验风险 JSON 的第一阶段流水线。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from opencc import OpenCC

from ipo_risk.calculations import build_calculation
from ipo_risk.config import load_config
from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    Evidence,
    EvidenceType,
    RiskFinding,
    ToolCall,
    ToolCallStatus,
    TraceStatus,
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
    BM25Retriever,
    DocumentChunk,
    build_page_chunks,
    load_risk_rules,
    select_rule_candidates,
)
from ipo_risk.storage.files import read_json, sha256_file, write_jsonl
from ipo_risk.storage.ingestion import register_company_sources
from ipo_risk.storage.paths import DocumentPaths
from ipo_risk.storage.repositories import CatalogRepository


_TO_SIMPLIFIED = OpenCC("t2s")


def _now() -> datetime:
    return datetime.now(timezone.utc)


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

    retriever = BM25Retriever(chunks)
    rules = load_risk_rules(risk_rules_path)
    evidences: list[Evidence] = []
    findings: list[RiskFinding] = []
    evidence_display: dict[str, str] = {}
    evidence_by_chunk: dict[str, Evidence] = {}
    screenshot_dir = document_paths.evidence_dir

    for rule in rules:
        retrieval_started = _now()
        hits = select_rule_candidates(rule, retriever, top_k=80)
        retrieval_ended = _now()
        tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{company_id}-bm25-{rule.id}",
                tool_name="bm25_search",
                started_at=retrieval_started,
                ended_at=retrieval_ended,
                inputs={
                    "risk_type": rule.id,
                    "query": rule.query,
                    "required_keywords": list(rule.required_keywords),
                    "top_k": 80,
                },
                output={
                    "selected": [
                        {
                            "chunk_id": hit.chunk.chunk_id,
                            "page": hit.chunk.page,
                            "score": round(hit.score, 6),
                            "rank": hit.rank,
                        }
                        for hit in hits
                    ]
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )
        if not hits:
            continue

        finding_evidence_ids: list[str] = []
        ranks: list[int] = []
        for hit in hits:
            chunk = hit.chunk
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
            ranks.append(hit.rank)

        finding_confidence = max(0.72, 0.92 - 0.01 * (max(ranks) - 1))
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
                severity=rule.severity,
                claim=rule.claim,
                evidence_ids=tuple(finding_evidence_ids),
                status=VerificationStatus.PENDING_VERIFICATION,
                confidence=round(finding_confidence, 2),
                created_by=rule.owner,
                calculation=calculation,
                uncertainty="规则与BM25生成的候选结论，需由投研人员复核确认。",
            )
        )

    if len(findings) < min_findings:
        raise RuntimeError(
            f"only {len(findings)} grounded findings were generated; "
            f"at least {min_findings} required"
        )

    run_ended = _now()
    trace = AgentTrace(
        trace_id=f"T-{company_id}-0001",
        run_id=run_id,
        agent=AgentRole.ORCHESTRATOR,
        started_at=run_started,
        ended_at=run_ended,
        input_refs=(source_file,),
        tool_calls=tuple(tool_calls),
        output_refs=tuple(finding.finding_id for finding in findings),
        status=TraceStatus.SUCCEEDED,
        decision=(
            f"生成{len(findings)}条有原文、页码、坐标和截图的候选风险，"
            "全部进入人工复核。"
        ),
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
        agent_traces=(trace,),
        evidence_display_zh_hans=evidence_display,
    )
    write_risk_report(report, document_paths.report_path)
    return AnalysisArtifacts(
        report=report,
        report_path=document_paths.report_path,
        pages_path=document_paths.pages_path,
        chunks_path=document_paths.chunks_path,
    )


def analyze_company(
    company_folder: str,
    *,
    project_root: Path | None = None,
    document_id: str | None = None,
    min_findings: int = 5,
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
    )
    repository.save_analysis_report(
        artifacts.report,
        _portable_path(artifacts.report_path, root),
    )
    return artifacts


__all__ = ["AnalysisArtifacts", "analyze_company", "analyze_prospectus"]
