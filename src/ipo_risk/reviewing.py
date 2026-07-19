"""Deterministic verification and immutable human-review state transitions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ipo_risk.agents.verifier import verify_finding as run_verification
from ipo_risk.domain import (
    FindingReview,
    FindingVerification,
    ReviewDecision,
    RiskFinding,
    VerificationStatus,
)
from ipo_risk.reporting import AnalysisSummary, RiskAnalysisReport, write_risk_report
from ipo_risk.storage.files import read_json
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


class ReviewService:
    """Small public interface for checking and reviewing stored findings."""

    def __init__(self, *, repository: CatalogRepository, project_root: Path) -> None:
        self.repository = repository
        self.project_root = project_root.resolve()

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


__all__ = ["IncompleteReviewError", "ReviewBlockedError", "ReviewService"]
