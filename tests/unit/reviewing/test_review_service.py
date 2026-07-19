"""人工复核状态机与SQLite审计记录测试。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import fitz
import pytest
from typer.testing import CliRunner

from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    CompanyRecord,
    DocumentRecord,
    Evidence,
    EvidenceType,
    ReviewDecision,
    RiskFinding,
    RiskSeverity,
    TraceStatus,
    VerificationStatus,
)
from ipo_risk.reporting import (
    AnalysisSummary,
    RiskAnalysisReport,
    SourceDocument,
    write_risk_report,
)
from ipo_risk.reviewing import IncompleteReviewError, ReviewBlockedError, ReviewService
from ipo_risk.storage.files import sha256_file
from ipo_risk.storage.repositories import CatalogRepository
from scripts.review_findings import app as review_app


NOW = datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc)


def _prepared_run(tmp_path: Path) -> tuple[ReviewService, CatalogRepository, RiskFinding]:
    pdf_path = tmp_path / "source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_text((50, 60), "Customer concentration remains material")
    pdf.save(pdf_path)
    pdf.close()
    source_hash = sha256_file(pdf_path)

    repository = CatalogRepository(tmp_path / "data" / "app.db")
    repository.initialize()
    repository.upsert_company(
        CompanyRecord(
            company_id="TEST",
            company_folder="TEST_测试公司",
            company_name_zh="测试公司",
            metadata_file="company.json",
            metadata_hash="a" * 64,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    document = DocumentRecord(
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        company_id="TEST",
        document_type="prospectus",
        title="测试招股书",
        local_file=pdf_path.name,
        source_hash=source_hash,
        document_date=date(2025, 1, 1),
        available_at=date(2025, 1, 1),
        media_type="application/pdf",
        file_size_bytes=pdf_path.stat().st_size,
        page_count=1,
        metadata_status="declared",
        registered_at=NOW,
    )
    repository.upsert_document(document)
    evidence = Evidence(
        evidence_id="E-TEST-0001",
        company_id="TEST",
        document_id=document.document_id,
        source_file=pdf_path.name,
        source_hash=source_hash,
        page=1,
        bbox=(40.0, 40.0, 280.0, 80.0),
        quote="Customer concentration remains material",
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
        created_at=NOW,
    )
    finding = RiskFinding(
        finding_id="F-TEST-0001",
        company_id="TEST",
        risk_type="customer_concentration",
        title="客户集中度风险",
        severity=RiskSeverity.HIGH,
        claim="客户集中可能影响经营稳定性。",
        evidence_ids=(evidence.evidence_id,),
        status=VerificationStatus.PENDING_VERIFICATION,
        confidence=0.85,
        created_by=AgentRole.FINANCE,
        created_at=NOW,
    )
    trace = AgentTrace(
        trace_id="T-TEST-0001",
        run_id="RUN-TEST-001",
        agent=AgentRole.ORCHESTRATOR,
        started_at=NOW,
        ended_at=NOW,
        output_refs=(finding.finding_id,),
        status=TraceStatus.SUCCEEDED,
        decision="生成候选风险。",
    )
    report = RiskAnalysisReport(
        run_id=trace.run_id,
        company_id="TEST",
        company_name="测试公司",
        generated_at=NOW,
        source_document=SourceDocument(
            document_id=document.document_id,
            source_file=document.local_file,
            source_hash=source_hash,
            available_at=date(2025, 1, 1),
            page_count=1,
            chunk_count=1,
        ),
        summary=AnalysisSummary(evidence_count=1, finding_count=1, screenshot_count=0),
        evidences=(evidence,),
        risk_findings=(finding,),
        agent_traces=(trace,),
        evidence_display_zh_hans={evidence.evidence_id: evidence.quote},
    )
    report_path = tmp_path / "artifacts" / "risk_analysis.json"
    write_risk_report(report, report_path)
    repository.save_analysis_report(report, report_path.relative_to(tmp_path).as_posix())
    return ReviewService(repository=repository, project_root=tmp_path), repository, finding


def test_confirmed_finding_is_verified_and_review_is_retrievable(tmp_path: Path) -> None:
    service, repository, finding = _prepared_run(tmp_path)

    review = service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
        comment="原文、期间和风险表述一致。",
    )

    assert review.resulting_finding.status is VerificationStatus.VERIFIED
    assert repository.get_finding("RUN-TEST-001", finding.finding_id) == (
        review.resulting_finding
    )
    assert repository.get_review(review.review_id) == review


def test_rejected_finding_keeps_failed_verification_in_the_audit_log(
    tmp_path: Path,
) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(source_path.read_bytes() + b"tampered")

    review = service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.REJECTED,
        reviewer="analyst@example.com",
        comment="源文件哈希与分析时版本不一致，驳回。",
    )

    assert review.verification.passed is False
    assert review.resulting_finding.status is VerificationStatus.REJECTED
    assert repository.list_reviews("RUN-TEST-001", finding_id=finding.finding_id) == (
        review,
    )


def test_edited_finding_replaces_payload_and_preserves_identity(tmp_path: Path) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    edited = finding.model_copy(
        update={
            "claim": "经人工修订，客户集中度可能对经营稳定性造成重大影响。",
            "confidence": 0.93,
        }
    )

    review = service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.EDITED,
        reviewer="analyst@example.com",
        comment="根据原文调整风险措辞和置信度。",
        edited_finding=edited,
    )

    stored = repository.get_finding("RUN-TEST-001", finding.finding_id)
    assert stored == review.resulting_finding
    assert stored is not None
    assert stored.claim == edited.claim
    assert stored.confidence == 0.93
    assert stored.status is VerificationStatus.VERIFIED


def test_numeric_finding_cannot_be_confirmed_without_reproducible_calculation(
    tmp_path: Path,
) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    numeric = finding.model_copy(update={"claim": "客户集中度达到90%。"})
    repository.save_analysis_report(
        RiskAnalysisReport(
            run_id="RUN-TEST-001",
            company_id="TEST",
            company_name="测试公司",
            generated_at=NOW,
            source_document=SourceDocument(
                document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
                source_file="source.pdf",
                source_hash=sha256_file(tmp_path / "source.pdf"),
                available_at=date(2025, 1, 1),
                page_count=1,
                chunk_count=1,
            ),
            summary=AnalysisSummary(evidence_count=1, finding_count=1, screenshot_count=0),
            evidences=repository.list_evidence("RUN-TEST-001"),
            risk_findings=(numeric,),
            agent_traces=(),
            evidence_display_zh_hans={"E-TEST-0001": "Customer concentration remains material"},
        ),
        "artifacts/risk_analysis.json",
    )

    with pytest.raises(ReviewBlockedError) as exc_info:
        service.review_finding(
            run_id="RUN-TEST-001",
            finding_id=finding.finding_id,
            decision=ReviewDecision.CONFIRMED,
            reviewer="analyst@example.com",
        )

    assert any(
        check.name == "calculation" and not check.passed
        for check in exc_info.value.verification.checks
    )

    assert repository.list_reviews("RUN-TEST-001") == ()
    stored = repository.get_finding("RUN-TEST-001", finding.finding_id)
    assert stored is not None
    assert stored.status is VerificationStatus.PENDING_VERIFICATION


def test_verified_report_contains_only_terminal_verified_findings(tmp_path: Path) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
    )

    report_path = service.export_verified_report("RUN-TEST-001")
    report = RiskAnalysisReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    assert report_path.name == "verified_risk_analysis.json"
    assert report.summary.requires_human_review is False
    assert report.risk_findings[0].status is VerificationStatus.VERIFIED
    assert {evidence.evidence_id for evidence in report.evidences} == set(
        report.risk_findings[0].evidence_ids
    )
    assert repository.get_verified_report_path("RUN-TEST-001") == (
        "artifacts/verified_risk_analysis.json"
    )


def test_verified_report_is_blocked_while_any_finding_is_pending(tmp_path: Path) -> None:
    service, _, finding = _prepared_run(tmp_path)

    with pytest.raises(IncompleteReviewError) as exc_info:
        service.export_verified_report("RUN-TEST-001")

    assert exc_info.value.finding_ids == (finding.finding_id,)


def test_review_cli_lists_pending_findings(tmp_path: Path) -> None:
    _prepared_run(tmp_path)

    result = CliRunner().invoke(
        review_app,
        [
            "list",
            "--run-id",
            "RUN-TEST-001",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "F-TEST-0001" in result.stdout
    assert "pending_verification" in result.stdout


def test_saved_candidate_report_cannot_reset_a_completed_review(tmp_path: Path) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
    )
    candidate_path = tmp_path / "artifacts" / "risk_analysis.json"
    candidate = RiskAnalysisReport.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )

    repository.save_analysis_report(candidate, "artifacts/risk_analysis.json")

    stored = repository.get_finding("RUN-TEST-001", finding.finding_id)
    assert stored is not None
    assert stored.status is VerificationStatus.VERIFIED
    assert len(repository.list_reviews("RUN-TEST-001")) == 1
