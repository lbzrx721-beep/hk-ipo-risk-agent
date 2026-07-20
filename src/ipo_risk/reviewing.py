"""Deterministic verification and immutable human-review state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator

from ipo_risk.agents.verifier import verify_finding as run_verification
from ipo_risk.domain import (
    AgentTrace,
    Evidence,
    FindingReview,
    FindingVerification,
    ReviewDecision,
    RiskFinding,
    VerificationStatus,
)
from ipo_risk.reporting import AnalysisSummary, RiskAnalysisReport, write_risk_report
from ipo_risk.storage.files import read_json, write_json
from ipo_risk.storage.paths import safe_path_segment
from ipo_risk.storage.repositories import CatalogRepository


class ReviewBlockedError(ValueError):
    """Raised when confirmation or editing fails deterministic verification."""

    def __init__(self, verification: FindingVerification) -> None:
        self.verification = verification
        failed = [check.name for check in verification.checks if not check.passed]
        failed.extend(
            f"evidence:{result.evidence_id}"
            for result in verification.evidence_verifications
            if not result.passed
        )
        super().__init__(f"finding verification failed: {', '.join(failed)}")


class IncompleteReviewError(ValueError):
    """Raised when a final report is requested before every finding is reviewed."""

    def __init__(self, finding_ids: tuple[str, ...]) -> None:
        self.finding_ids = finding_ids
        super().__init__(f"findings still require review: {', '.join(finding_ids)}")


@dataclass(frozen=True)
class ReviewRunSummary:
    """人工复核工作台中的一条运行及进度摘要。"""

    run_id: str
    company_id: str
    company_name: str
    as_of_date: str
    total_findings: int
    pending_findings: int
    verified_findings: int
    rejected_findings: int
    reviewed_findings: int
    complete: bool
    report_path: str | None
    verified_report_path: str | None


@dataclass(frozen=True)
class ReviewFindingDetail:
    """页面展示一条风险所需的结论、证据、核验和审计记录。"""

    finding: RiskFinding
    evidences: tuple[Evidence, ...]
    verification: FindingVerification
    reviews: tuple[FindingReview, ...]
    agent_traces: tuple[AgentTrace, ...]


class ReviewAnnotationBundle(BaseModel):
    """一次人工复核进度的不可变、可追溯标注快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    annotation_version: str
    exported_at: datetime
    run_id: str
    company_id: str
    company_name: str
    document_id: str
    source_hash: str
    complete: bool
    findings: tuple[RiskFinding, ...]
    evidences: tuple[Evidence, ...]
    reviews: tuple[FindingReview, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> ReviewAnnotationBundle:
        known_evidence = {evidence.evidence_id for evidence in self.evidences}
        referenced_evidence = {
            evidence_id
            for finding in self.findings
            for evidence_id in finding.evidence_ids
        }
        if referenced_evidence - known_evidence:
            raise ValueError("annotation findings reference missing evidence")
        if any(review.run_id != self.run_id for review in self.reviews):
            raise ValueError("annotation review belongs to another run")
        pending = any(
            finding.status
            in {
                VerificationStatus.PROPOSED,
                VerificationStatus.PENDING_VERIFICATION,
            }
            for finding in self.findings
        )
        if self.complete == pending:
            raise ValueError("complete must match current finding states")
        return self


class ReviewService:
    """Small public interface for checking and reviewing stored findings."""

    def __init__(
        self,
        *,
        repository: CatalogRepository,
        project_root: Path,
        annotations_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.project_root = project_root.resolve()
        self.annotations_root = (
            annotations_root or self.project_root / "data" / "annotations"
        ).resolve()

    def list_runs(self) -> tuple[ReviewRunSummary, ...]:
        summaries: list[ReviewRunSummary] = []
        for run in self.repository.list_analysis_runs(status="succeeded"):
            findings = self.repository.list_findings(run.run_id)
            if not findings:
                continue
            company = self.repository.get_company(run.company_id)
            pending = sum(
                finding.status
                in {
                    VerificationStatus.PROPOSED,
                    VerificationStatus.PENDING_VERIFICATION,
                }
                for finding in findings
            )
            verified = sum(
                finding.status is VerificationStatus.VERIFIED for finding in findings
            )
            rejected = sum(
                finding.status is VerificationStatus.REJECTED for finding in findings
            )
            summaries.append(
                ReviewRunSummary(
                    run_id=run.run_id,
                    company_id=run.company_id,
                    company_name=(company.company_name_zh if company else run.company_id),
                    as_of_date=run.as_of_date.isoformat(),
                    total_findings=len(findings),
                    pending_findings=pending,
                    verified_findings=verified,
                    rejected_findings=rejected,
                    reviewed_findings=verified + rejected,
                    complete=pending == 0,
                    report_path=run.report_path,
                    verified_report_path=run.verified_report_path,
                )
            )
        return tuple(summaries)

    def list_findings(self, run_id: str) -> tuple[RiskFinding, ...]:
        return self.repository.list_findings(run_id)

    def get_agent_traces(self, run_id: str) -> tuple[AgentTrace, ...]:
        """从候选报告读取不可变 Agent 轨迹，供复核和审计展示。"""
        report_reference = self.repository.get_analysis_report_path(run_id)
        if report_reference is None:
            return ()
        report_path = Path(report_reference)
        if not report_path.is_absolute():
            report_path = self.project_root / report_path
        if not report_path.is_file():
            return ()
        report = RiskAnalysisReport.model_validate(read_json(report_path))
        return report.agent_traces

    def get_finding_detail(
        self,
        run_id: str,
        finding_id: str,
    ) -> ReviewFindingDetail:
        finding = self.repository.get_finding(run_id, finding_id)
        if finding is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")
        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=finding.evidence_ids,
        )
        return ReviewFindingDetail(
            finding=finding,
            evidences=evidences,
            verification=run_verification(
                finding,
                evidences,
                project_root=self.project_root,
            ),
            reviews=self.repository.list_reviews(run_id, finding_id=finding_id),
            agent_traces=self.get_agent_traces(run_id),
        )

    def verify_finding(self, run_id: str, finding_id: str) -> FindingVerification:
        finding = self.repository.get_finding(run_id, finding_id)
        if finding is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")
        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=finding.evidence_ids,
        )
        return run_verification(finding, evidences, project_root=self.project_root)

    def review_finding(
        self,
        *,
        run_id: str,
        finding_id: str,
        decision: ReviewDecision,
        reviewer: str,
        comment: str | None = None,
        edited_finding: RiskFinding | None = None,
    ) -> FindingReview:
        original = self.repository.get_finding(run_id, finding_id)
        if original is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")

        if decision is ReviewDecision.EDITED:
            if edited_finding is None:
                raise ValueError("edited review requires edited_finding")
            if edited_finding.finding_id != original.finding_id:
                raise ValueError("edited finding must keep finding_id")
            if edited_finding.company_id != original.company_id:
                raise ValueError("edited finding must keep company_id")
            candidate = edited_finding.model_copy(
                update={"status": VerificationStatus.PENDING_VERIFICATION}
            )
        else:
            if edited_finding is not None:
                raise ValueError("edited_finding is only valid for an edited review")
            candidate = original.model_copy(
                update={"status": VerificationStatus.PENDING_VERIFICATION}
            )

        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=candidate.evidence_ids,
        )
        verification = run_verification(
            candidate,
            evidences,
            project_root=self.project_root,
        )
        if decision is not ReviewDecision.REJECTED and not verification.passed:
            raise ReviewBlockedError(verification)

        status = (
            VerificationStatus.REJECTED
            if decision is ReviewDecision.REJECTED
            else VerificationStatus.VERIFIED
        )
        resulting_finding = candidate.model_copy(update={"status": status})
        review = FindingReview(
            review_id=f"REV-{uuid4().hex}",
            run_id=run_id,
            finding_id=finding_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
            verification=verification,
            resulting_finding=resulting_finding,
        )
        self.repository.save_review(review)
        return review

    def export_verified_report(
        self,
        run_id: str,
        *,
        output_path: Path | None = None,
    ) -> Path:
        """Export a final report only after every finding reaches a terminal state."""
        candidate_reference = self.repository.get_analysis_report_path(run_id)
        if candidate_reference is None:
            raise KeyError(f"analysis run does not exist or has no report: {run_id}")
        candidate_path = Path(candidate_reference)
        if not candidate_path.is_absolute():
            candidate_path = self.project_root / candidate_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        candidate = RiskAnalysisReport.model_validate(read_json(candidate_path))

        stored_findings = self.repository.list_findings(run_id)
        incomplete = tuple(
            finding.finding_id
            for finding in stored_findings
            if finding.status
            in {
                VerificationStatus.PROPOSED,
                VerificationStatus.PENDING_VERIFICATION,
            }
        )
        if incomplete:
            raise IncompleteReviewError(incomplete)

        verified_findings = tuple(
            finding
            for finding in stored_findings
            if finding.status is VerificationStatus.VERIFIED
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for finding in verified_findings
                for evidence_id in finding.evidence_ids
            )
        )
        evidences = self.repository.list_evidence(run_id, evidence_ids=evidence_ids)
        evidence_display = {
            evidence.evidence_id: candidate.evidence_display_zh_hans.get(
                evidence.evidence_id,
                evidence.quote,
            )
            for evidence in evidences
        }
        final_report = RiskAnalysisReport(
            run_id=candidate.run_id,
            company_id=candidate.company_id,
            company_name=candidate.company_name,
            generated_at=datetime.now(timezone.utc),
            source_document=candidate.source_document,
            summary=AnalysisSummary(
                evidence_count=len(evidences),
                finding_count=len(verified_findings),
                screenshot_count=sum(
                    evidence.screenshot_path is not None for evidence in evidences
                ),
                requires_human_review=False,
            ),
            evidences=evidences,
            risk_findings=verified_findings,
            agent_traces=candidate.agent_traces,
            evidence_display_zh_hans=evidence_display,
        )
        destination = output_path or candidate_path.with_name("verified_risk_analysis.json")
        if not destination.is_absolute():
            destination = self.project_root / destination
        write_risk_report(final_report, destination)
        try:
            stored_path = destination.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            stored_path = str(destination.resolve())
        self.repository.set_verified_report_path(run_id, stored_path)
        return destination

    def export_review_annotations(
        self,
        run_id: str,
        *,
        output_path: Path | None = None,
    ) -> Path:
        """把当前风险、证据和审计记录导出为一个新的版本化标注快照。"""
        candidate_reference = self.repository.get_analysis_report_path(run_id)
        if candidate_reference is None:
            raise KeyError(f"analysis run does not exist or has no report: {run_id}")
        candidate_path = Path(candidate_reference)
        if not candidate_path.is_absolute():
            candidate_path = self.project_root / candidate_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        candidate = RiskAnalysisReport.model_validate(read_json(candidate_path))

        findings = self.repository.list_findings(run_id)
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for finding in findings
                for evidence_id in finding.evidence_ids
            )
        )
        evidences = self.repository.list_evidence(run_id, evidence_ids=evidence_ids)
        reviews = self.repository.list_reviews(run_id)
        complete = all(
            finding.status
            in {VerificationStatus.VERIFIED, VerificationStatus.REJECTED}
            for finding in findings
        )
        exported_at = datetime.now(timezone.utc)
        version = f"{exported_at:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
        bundle = ReviewAnnotationBundle(
            annotation_version=version,
            exported_at=exported_at,
            run_id=run_id,
            company_id=candidate.company_id,
            company_name=candidate.company_name,
            document_id=candidate.source_document.document_id,
            source_hash=candidate.source_document.source_hash,
            complete=complete,
            findings=findings,
            evidences=evidences,
            reviews=reviews,
        )

        if output_path is None:
            company_segment = safe_path_segment(candidate.company_id, "company_id")
            run_segment = safe_path_segment(run_id, "run_id")
            destination = (
                self.annotations_root
                / company_segment
                / run_segment
                / f"review_annotations_{version}.json"
            )
        else:
            destination = output_path
            if not destination.is_absolute():
                destination = self.project_root / destination
        return write_json(bundle.model_dump(mode="json"), destination, pretty=True)


__all__ = [
    "IncompleteReviewError",
    "ReviewBlockedError",
    "ReviewAnnotationBundle",
    "ReviewFindingDetail",
    "ReviewRunSummary",
    "ReviewService",
]
