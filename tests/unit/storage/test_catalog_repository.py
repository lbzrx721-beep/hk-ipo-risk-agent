"""SQLite资料目录与分析结果持久化测试。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    CompanyRecord,
    DatasetRecord,
    DocumentRecord,
    Evidence,
    EvidenceType,
    RiskFinding,
    RiskSeverity,
    TraceStatus,
    VectorIndexRecord,
    VerificationStatus,
)
from ipo_risk.reporting import AnalysisSummary, RiskAnalysisReport, SourceDocument
from ipo_risk.storage.repositories import CatalogRepository


NOW = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


def _company() -> CompanyRecord:
    return CompanyRecord(
        company_id="03696",
        company_folder="03696_英矽智能",
        ticker="03696.HK",
        exchange="HKEX",
        market="Main Board",
        company_name_zh="英矽智能",
        company_name_en="InSilico Medicine",
        listing_date=date(2025, 12, 30),
        metadata_file="data/raw/company_info/03696_英矽智能/公司信息.json",
        metadata_hash="a" * 64,
        created_at=NOW,
        updated_at=NOW,
    )


def _document(
    document_id: str = "doc_03696_prospectus_2025-12-18_bbbbbbbb",
    document_type: str = "prospectus",
    available_at: date = date(2025, 12, 18),
    local_file: str = "data/raw/prospectus/03696_英矽智能/招股书.pdf",
    source_hash: str = "b" * 64,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        company_id="03696",
        document_type=document_type,
        title="测试文档",
        local_file=local_file,
        source_hash=source_hash,
        document_date=available_at,
        available_at=available_at,
        language="zh-Hant",
        source_kind="official_hkex_public_filing",
        source_url="https://example.com/document.pdf",
        media_type="application/pdf",
        file_size_bytes=1024,
        page_count=10,
        text_layer_status="present",
        metadata={"source": "unit-test"},
        registered_at=NOW,
    )


def test_catalog_is_idempotent_and_point_in_time_selectable(tmp_path: Path) -> None:
    repository = CatalogRepository(tmp_path / "app.db")
    repository.initialize()
    repository.initialize()
    company = _company()
    prospectus = _document()
    allotment = _document(
        document_id="doc_03696_allotment_results_2025-12-29_cccccccc",
        document_type="allotment_results",
        available_at=date(2025, 12, 29),
        local_file="data/raw/company_info/03696_英矽智能/配发结果.pdf",
        source_hash="c" * 64,
    )
    dataset = DatasetRecord(
        dataset_id="ds_03696_market_daily_2026-07-17_dddddddd",
        company_id="03696",
        dataset_type="market_daily",
        title="日线行情",
        local_file="data/raw/market/03696_英矽智能/日线行情.json",
        source_hash="d" * 64,
        available_at=date(2026, 7, 17),
        file_format="json",
        period_start=date(2025, 12, 30),
        period_end=date(2026, 7, 16),
        row_count=132,
        usage_scope="post_listing_label_and_backtest_only",
        pre_listing_eligible=False,
        registered_at=NOW,
    )

    repository.upsert_company(company)
    repository.upsert_company(company)
    repository.upsert_document(prospectus)
    repository.upsert_document(prospectus)
    repository.upsert_document(allotment)
    repository.upsert_dataset(dataset)
    repository.upsert_dataset(dataset)

    before_allotment = repository.list_documents(
        "03696",
        available_on_or_before=date(2025, 12, 20),
    )
    before_listing_datasets = repository.list_datasets(
        "03696",
        available_on_or_before=date(2025, 12, 29),
        pre_listing_only=True,
    )

    assert [item.document_id for item in before_allotment] == [prospectus.document_id]
    assert before_listing_datasets == ()
    counts = repository.counts()
    assert counts.companies == 1
    assert counts.documents == 2
    assert counts.datasets == 1


def test_vector_index_and_report_are_registered(tmp_path: Path) -> None:
    repository = CatalogRepository(tmp_path / "app.db")
    repository.initialize()
    company = _company()
    document = _document()
    repository.upsert_company(company)
    repository.upsert_document(document)
    vector_index = VectorIndexRecord(
        index_id="vi_03696_bbbbbbbb_qwen37_1024",
        document_id=document.document_id,
        source_hash=document.source_hash,
        embedding_provider="dashscope",
        embedding_model="qwen3.7-text-embedding",
        dimensions=1024,
        chunk_count=6091,
        vectors_path="data/interim/embeddings/03696/doc/embeddings.npy",
        metadata_path="data/interim/embeddings/03696/doc/metadata.jsonl",
        manifest_path="data/interim/embeddings/03696/doc/manifest.json",
        created_at=NOW,
    )
    repository.upsert_vector_index(vector_index)

    evidence = Evidence(
        evidence_id="E-03696-0001",
        company_id="03696",
        document_id=document.document_id,
        source_file=document.local_file,
        source_hash=document.source_hash,
        page=30,
        bbox=(10.0, 20.0, 100.0, 80.0),
        quote="测试证据原文",
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.97,
        available_at=document.available_at,
        created_at=NOW,
    )
    finding = RiskFinding(
        finding_id="F-03696-0001",
        company_id="03696",
        risk_type="cash_burn",
        title="现金消耗风险",
        severity=RiskSeverity.HIGH,
        claim="现金消耗可能造成融资压力。",
        evidence_ids=(evidence.evidence_id,),
        status=VerificationStatus.PENDING_VERIFICATION,
        confidence=0.85,
        created_by=AgentRole.FINANCE,
        created_at=NOW,
    )
    trace = AgentTrace(
        trace_id="T-03696-0001",
        run_id="RUN-03696-TEST",
        agent=AgentRole.ORCHESTRATOR,
        started_at=NOW,
        ended_at=NOW,
        output_refs=(finding.finding_id,),
        status=TraceStatus.SUCCEEDED,
        decision="生成一条候选风险。",
    )
    report = RiskAnalysisReport(
        run_id="RUN-03696-TEST",
        company_id="03696",
        company_name="英矽智能",
        generated_at=NOW,
        source_document=SourceDocument(
            document_id=document.document_id,
            source_file=document.local_file,
            source_hash=document.source_hash,
            available_at=document.available_at,
            page_count=10,
            chunk_count=20,
        ),
        summary=AnalysisSummary(
            evidence_count=1,
            finding_count=1,
            screenshot_count=0,
        ),
        evidences=(evidence,),
        risk_findings=(finding,),
        agent_traces=(trace,),
        evidence_display_zh_hans={evidence.evidence_id: evidence.quote},
    )

    report_path = "artifacts/analysis/03696/test/risk_analysis.json"
    repository.save_analysis_report(report, report_path)
    repository.save_analysis_report(report, report_path)

    assert repository.get_vector_index(vector_index.index_id) == vector_index
    counts = repository.counts()
    assert counts.vector_indexes == 1
    assert counts.analysis_runs == 1
    assert counts.evidence == 1
    assert counts.risk_findings == 1
