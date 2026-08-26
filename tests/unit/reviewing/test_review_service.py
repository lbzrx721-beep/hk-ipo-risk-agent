"""人工复核状态机与SQLite审计记录测试。"""

from __future__ import annotations

import json
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
from ipo_risk.reviewing import IncompleteReviewError, ReviewService
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
    assert repository.get_finding("RUN-TEST-001", finding.finding_id) == (review.resulting_finding)
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
    assert repository.list_reviews("RUN-TEST-001", finding_id=finding.finding_id) == (review,)


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


def test_disclosed_numeric_finding_can_be_confirmed_without_derived_calculation(
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

    review = service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
    )

    assert review.verification.passed is True
    assert any(
        check.name == "calculation" and check.passed and check.detail.startswith("not applicable")
        for check in review.verification.checks
    )
    stored = repository.get_finding("RUN-TEST-001", finding.finding_id)
    assert stored is not None
    assert stored.status is VerificationStatus.VERIFIED


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
    assert report.agent_traces == ()
    assert repository.get_verified_report_path("RUN-TEST-001") == (
        "artifacts/verified_risk_analysis.json"
    )


def test_verified_export_bundle_separates_pdf_report_and_full_audit(
    tmp_path: Path,
) -> None:
    service, _, finding = _prepared_run(tmp_path)
    prediction_path = tmp_path / "artifacts" / "prediction_5d.json"
    prediction_path.write_text(
        json.dumps(
            {
                "probability": 0.72,
                "decision": "high_risk",
                "confidence": 0.72,
                "feature_completeness": 0.875,
                "disclaimer": "方向性风险预警，不构成投资建议。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
        comment="原文、期间和风险表述一致。",
    )

    bundle = service.export_verified_bundle("RUN-TEST-001")
    compact_report = RiskAnalysisReport.model_validate_json(
        bundle.json_path.read_text(encoding="utf-8")
    )
    audit = json.loads(bundle.audit_path.read_text(encoding="utf-8"))

    assert bundle.pdf_path.name == "verified_risk_report.pdf"
    assert bundle.pdf_path.read_bytes().startswith(b"%PDF")
    assert compact_report.agent_traces == ()
    assert audit["export_type"] == "analysis_audit_bundle"
    assert len(audit["candidate_report"]["agent_traces"]) == 1
    assert len(audit["reviews"]) == 1
    assert audit["prediction"]["probability"] == 0.72

    document = fitz.open(bundle.pdf_path)
    pdf_text = "".join(page.get_text() for page in document)
    document.close()
    assert "客户集中度风险" in pdf_text
    assert "上市后五日风险预警" in pdf_text
    assert finding.claim in pdf_text
    assert "Customer concentration remains material" in pdf_text


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
    candidate = RiskAnalysisReport.model_validate_json(candidate_path.read_text(encoding="utf-8"))

    repository.save_analysis_report(candidate, "artifacts/risk_analysis.json")

    stored = repository.get_finding("RUN-TEST-001", finding.finding_id)
    assert stored is not None
    assert stored.status is VerificationStatus.VERIFIED
    assert len(repository.list_reviews("RUN-TEST-001")) == 1


def test_review_workbench_lists_run_progress_and_finding_details(tmp_path: Path) -> None:
    service, _, finding = _prepared_run(tmp_path)
    (tmp_path / "artifacts" / "prediction_5d.json").write_text(
        json.dumps({"probability": 0.35}, ensure_ascii=False),
        encoding="utf-8",
    )

    runs = service.list_runs()
    detail = service.get_finding_detail("RUN-TEST-001", finding.finding_id)

    assert len(runs) == 1
    assert runs[0].run_id == "RUN-TEST-001"
    assert runs[0].company_name == "测试公司"
    assert runs[0].total_findings == 1
    assert runs[0].pending_findings == 1
    assert runs[0].reviewed_findings == 0
    assert runs[0].complete is False
    assert runs[0].prediction == {"probability": 0.35}
    assert detail.finding == finding
    assert detail.evidences[0].evidence_id == "E-TEST-0001"
    assert detail.verification.passed is True
    assert detail.reviews == ()
    assert detail.agent_traces[0].agent is AgentRole.ORCHESTRATOR


def test_historical_run_can_rerun_prediction_from_saved_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, repository, _ = _prepared_run(tmp_path)
    feature_path = tmp_path / "artifacts" / "prelisting_features.json"
    feature_path.write_text(
        json.dumps(
            {
                "run_id": "RUN-TEST-001",
                "company_id": "TEST",
                "prediction_as_of": "2025-01-01",
                "features": {"agent_score": {"value": 2.0, "unit": "score"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_prediction(**kwargs):
        destination = kwargs["destination_path"]
        payload = {
            "prediction_type": "significant_drop_5d",
            "prediction_variant": "agent_enhanced",
            "run_id": "RUN-TEST-001",
            "company_id": "TEST",
            "model_name": "test-model",
            "probability": 0.63,
        }
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return payload, destination

    monkeypatch.setattr(
        "ipo_risk.reviewing.generate_runtime_prediction_from_feature_file",
        fake_prediction,
    )

    prediction, summary = service.rerun_prediction(
        "RUN-TEST-001",
        variant="agent_enhanced",
    )

    assert prediction["probability"] == 0.63
    assert summary.agent_feature_available is True
    assert summary.prediction == prediction
    records = repository.list_predictions(run_id="RUN-TEST-001")
    assert len(records) == 1
    assert records[0].risk_probability == 0.63


def test_delete_analysis_run_removes_derived_state_but_preserves_source(
    tmp_path: Path,
) -> None:
    service, repository, finding = _prepared_run(tmp_path)
    service.review_finding(
        run_id="RUN-TEST-001",
        finding_id=finding.finding_id,
        decision=ReviewDecision.CONFIRMED,
        reviewer="analyst@example.com",
    )
    bundle = service.export_verified_bundle("RUN-TEST-001")
    source_path = tmp_path / "source.pdf"
    report_path = tmp_path / "artifacts" / "risk_analysis.json"

    result = service.delete_analysis_run("RUN-TEST-001")

    assert result.run_id == "RUN-TEST-001"
    assert result.recovery_path is not None
    assert result.recovery_path.is_dir()
    assert repository.get_analysis_run("RUN-TEST-001") is None
    assert repository.list_findings("RUN-TEST-001") == ()
    assert repository.list_reviews("RUN-TEST-001") == ()
    assert source_path.is_file()
    assert not report_path.exists()
    assert not bundle.pdf_path.exists()
    assert not bundle.json_path.exists()
    assert not bundle.audit_path.exists()
    recovered_names = {path.name for path in result.recovery_path.rglob("*")}
    assert {
        report_path.name,
        bundle.pdf_path.name,
        bundle.json_path.name,
        bundle.audit_path.name,
    } <= recovered_names


def test_delete_analysis_run_preserves_files_referenced_by_another_run(
    tmp_path: Path,
) -> None:
    service, repository, _ = _prepared_run(tmp_path)
    report_path = tmp_path / "artifacts" / "risk_analysis.json"
    first_report = RiskAnalysisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    second_trace = first_report.agent_traces[0].model_copy(
        update={"run_id": "RUN-TEST-002", "trace_id": "T-TEST-0002"}
    )
    second_report = first_report.model_copy(
        update={"run_id": "RUN-TEST-002", "agent_traces": (second_trace,)}
    )
    repository.save_analysis_report(
        second_report,
        report_path.relative_to(tmp_path).as_posix(),
    )

    result = service.delete_analysis_run("RUN-TEST-001")

    assert repository.get_analysis_run("RUN-TEST-001") is None
    assert repository.get_analysis_run("RUN-TEST-002") is not None
    assert report_path.is_file()
    assert report_path.resolve() in result.skipped_paths
