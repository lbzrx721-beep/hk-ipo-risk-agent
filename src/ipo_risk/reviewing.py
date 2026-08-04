"""Deterministic verification and immutable human-review state transitions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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
from ipo_risk.reporting import write_verified_pdf_report
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


@dataclass(frozen=True)
class AnalysisDeletionResult:
    """一次分析删除后的可恢复位置和路径处理结果。"""

    run_id: str
    recovery_path: Path | None
    moved_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerifiedExportBundle:
    """Human PDF, compact verified JSON and the separate complete audit."""

    pdf_path: Path
    json_path: Path
    audit_path: Path


class ReviewService:
    """Small public interface for checking and reviewing stored findings."""

    def __init__(
        self,
        *,
        repository: CatalogRepository,
        project_root: Path,
    ) -> None:
        self.repository = repository
        self.project_root = project_root.resolve()

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
            verified = sum(finding.status is VerificationStatus.VERIFIED for finding in findings)
            rejected = sum(finding.status is VerificationStatus.REJECTED for finding in findings)
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

    def delete_analysis_run(self, run_id: str) -> AnalysisDeletionResult:
        """删除指定运行，派生文件移入项目回收区并保留原始文档。"""
        run = self.repository.get_analysis_run(run_id)
        if run is None:
            raise KeyError(f"analysis run does not exist: {run_id}")

        run_segment = safe_path_segment(run_id, "run_id")
        candidates: list[tuple[Path, str, Path]] = []
        missing: list[Path] = []
        skipped: list[Path] = []

        def resolve_project_reference(reference: str) -> Path:
            path = Path(reference)
            if not path.is_absolute():
                path = self.project_root / path
            return path.resolve()

        def related_export_paths(reference: str) -> tuple[Path, ...]:
            primary = resolve_project_reference(reference)
            if primary.name != "verified_risk_analysis.json":
                return (primary,)
            return (
                primary,
                primary.with_name("verified_risk_report.pdf"),
                primary.with_name("analysis_audit_bundle.json"),
            )

        # A document or derived asset may be reused by multiple analysis runs.
        # Such files remain active; only the selected run's database links are removed.
        protected_paths = {
            resolve_project_reference(document.local_file)
            for document in self.repository.list_documents(run.company_id)
        }
        for other_run in self.repository.list_analysis_runs():
            if other_run.run_id == run_id:
                continue
            protected_paths.update(
                related_path
                for reference in (
                    other_run.report_path,
                    other_run.verified_report_path,
                )
                if reference
                for related_path in related_export_paths(reference)
            )
            protected_paths.update(
                resolve_project_reference(evidence.screenshot_path)
                for evidence in self.repository.list_evidence(other_run.run_id)
                if evidence.screenshot_path
            )

        def add_project_candidate(reference: str) -> None:
            source = resolve_project_reference(reference)
            if not source.is_relative_to(self.project_root) or source in protected_paths:
                skipped.append(source)
            elif source.exists():
                candidates.append((source, "project", source.relative_to(self.project_root)))
            else:
                missing.append(source)

        if run.report_path:
            add_project_candidate(run.report_path)
        if run.verified_report_path:
            for related_path in related_export_paths(run.verified_report_path):
                add_project_candidate(str(related_path))

        for evidence in self.repository.list_evidence(run_id):
            if evidence.screenshot_path:
                add_project_candidate(evidence.screenshot_path)

        unique_candidates: list[tuple[Path, str, Path]] = []
        seen_sources: set[Path] = set()
        for candidate in candidates:
            if candidate[0] not in seen_sources:
                unique_candidates.append(candidate)
                seen_sources.add(candidate[0])

        recovery_path: Path | None = None
        moved: list[tuple[Path, Path]] = []
        if unique_candidates:
            version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
            recovery_path = self.project_root / ".trash" / "analysis_runs" / run_segment / version
            try:
                for source, category, relative_path in unique_candidates:
                    destination = recovery_path / category / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    moved.append((source, destination))
                if not self.repository.delete_analysis_run(run_id):
                    raise KeyError(f"analysis run does not exist: {run_id}")
            except Exception:
                for source, destination in reversed(moved):
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        shutil.move(str(destination), str(source))
                if recovery_path.exists():
                    shutil.rmtree(recovery_path)
                raise
        elif not self.repository.delete_analysis_run(run_id):
            raise KeyError(f"analysis run does not exist: {run_id}")

        return AnalysisDeletionResult(
            run_id=run_id,
            recovery_path=recovery_path,
            moved_paths=tuple(source for source, _ in moved),
            missing_paths=tuple(dict.fromkeys(missing)),
            skipped_paths=tuple(dict.fromkeys(skipped)),
        )

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
            finding for finding in stored_findings if finding.status is VerificationStatus.VERIFIED
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id for finding in verified_findings for evidence_id in finding.evidence_ids
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
            agent_traces=(),
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

    def export_verified_bundle(
        self,
        run_id: str,
        *,
        output_directory: Path | None = None,
    ) -> VerifiedExportBundle:
        """Export a concise PDF/JSON plus a separately labelled full audit."""
        candidate_reference = self.repository.get_analysis_report_path(run_id)
        if candidate_reference is None:
            raise KeyError(f"analysis run does not exist or has no report: {run_id}")
        candidate_path = Path(candidate_reference)
        if not candidate_path.is_absolute():
            candidate_path = self.project_root / candidate_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)

        directory = output_directory or candidate_path.parent
        if not directory.is_absolute():
            directory = self.project_root / directory
        directory.mkdir(parents=True, exist_ok=True)
        compact_path = self.export_verified_report(
            run_id,
            output_path=directory / "verified_risk_analysis.json",
        )
        compact_report = RiskAnalysisReport.model_validate(read_json(compact_path))
        candidate_report = RiskAnalysisReport.model_validate(read_json(candidate_path))
        stored_findings = self.repository.list_findings(run_id)
        reviews = self.repository.list_reviews(run_id)

        audit_path = directory / "analysis_audit_bundle.json"
        write_json(
            {
                "schema_version": "1.0",
                "export_type": "analysis_audit_bundle",
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "candidate_report": candidate_report.model_dump(mode="json"),
                "final_findings": [finding.model_dump(mode="json") for finding in stored_findings],
                "reviews": [review.model_dump(mode="json") for review in reviews],
            },
            audit_path,
        )

        pdf_path = directory / "verified_risk_report.pdf"
        write_verified_pdf_report(
            compact_report,
            pdf_path,
            reviews=reviews,
            project_root=self.project_root,
        )
        return VerifiedExportBundle(
            pdf_path=pdf_path,
            json_path=compact_path,
            audit_path=audit_path,
        )


__all__ = [
    "AnalysisDeletionResult",
    "IncompleteReviewError",
    "ReviewBlockedError",
    "ReviewFindingDetail",
    "ReviewRunSummary",
    "ReviewService",
    "VerifiedExportBundle",
]
