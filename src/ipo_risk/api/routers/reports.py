"""查询、复核和导出风险报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ipo_risk.domain import AgentRole, ReviewDecision, RiskFinding
from ipo_risk.reviewing import (
    IncompleteReviewError,
    ReviewBlockedError,
    ReviewService,
)
from ipo_risk.retrying import FindingRetryService, finding_has_retryable_failure


class FindingReviewRequest(BaseModel):
    decision: ReviewDecision
    reviewer: str = Field(min_length=1)
    comment: str | None = None
    title: str | None = None
    severity: str | None = None
    claim: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class HistoricalPredictionRequest(BaseModel):
    variant: Literal["auto", "base", "agent_enhanced"] = "auto"


def _model_payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else value


def _resolve_project_file(project_root: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def create_reports_router(
    service: ReviewService,
    retry_service: FindingRetryService,
) -> APIRouter:
    router = APIRouter(prefix="/reports", tags=["reports"])

    @router.get("/runs")
    def list_runs() -> dict[str, object]:
        return {"runs": [summary.__dict__ for summary in service.list_runs()]}

    @router.post("/runs/{run_id}/prediction")
    def rerun_historical_prediction(
        run_id: str,
        request: HistoricalPredictionRequest,
    ) -> dict[str, object]:
        try:
            prediction, summary = service.rerun_prediction(
                run_id,
                variant=request.variant,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"prediction": prediction, "run": summary.__dict__}

    @router.get("/runs/{run_id}/findings")
    def list_findings(run_id: str) -> dict[str, object]:
        findings = service.list_findings(run_id)
        if not findings:
            raise HTTPException(status_code=404, detail="未找到该分析运行或风险结论。")
        return {"findings": [finding.model_dump(mode="json") for finding in findings]}

    @router.get("/runs/{run_id}/findings/{finding_id}")
    def finding_detail(run_id: str, finding_id: str) -> dict[str, object]:
        try:
            detail = service.get_finding_detail(run_id, finding_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "finding": detail.finding.model_dump(mode="json"),
            "retryable_failure": finding_has_retryable_failure(detail.finding),
            "evidences": [item.model_dump(mode="json") for item in detail.evidences],
            "verification": detail.verification.model_dump(mode="json"),
            "reviews": [item.model_dump(mode="json") for item in detail.reviews],
            "agent_traces": [item.model_dump(mode="json") for item in detail.agent_traces],
        }

    def start_retry(**kwargs: object) -> dict[str, object]:
        try:
            return {"job": retry_service.start(**kwargs)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/retry-jobs")
    def retry_jobs(run_id: str) -> dict[str, object]:
        return {"jobs": retry_service.list_jobs(run_id)}

    @router.post("/runs/{run_id}/findings/{finding_id}/retry")
    def retry_finding(run_id: str, finding_id: str) -> dict[str, object]:
        return start_retry(
            run_id=run_id,
            finding_ids=(finding_id,),
            agent=None,
            scope="single_finding",
        )

    @router.post("/runs/{run_id}/agents/{agent}/retry")
    def retry_agent(run_id: str, agent: str) -> dict[str, object]:
        try:
            role = AgentRole(agent)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="未知的专业Agent。") from exc
        if role not in {
            AgentRole.FINANCE,
            AgentRole.LEGAL,
            AgentRole.BUSINESS,
            AgentRole.MARKET,
        }:
            raise HTTPException(status_code=422, detail="该角色不是可重试的专业Agent。")
        return start_retry(
            run_id=run_id,
            finding_ids=None,
            agent=role,
            scope=f"agent:{role.value}",
        )

    @router.post("/runs/{run_id}/retry-technical-failures")
    def retry_all_technical_failures(run_id: str) -> dict[str, object]:
        return start_retry(
            run_id=run_id,
            finding_ids=None,
            agent=None,
            scope="all_technical_failures",
        )

    @router.post("/runs/{run_id}/findings/{finding_id}/review")
    def review_finding(
        run_id: str,
        finding_id: str,
        request: FindingReviewRequest,
    ) -> dict[str, object]:
        edited_finding: RiskFinding | None = None
        if request.decision is ReviewDecision.EDITED:
            original = service.repository.get_finding(run_id, finding_id)
            if original is None:
                raise HTTPException(status_code=404, detail="未找到需要编辑的风险结论。")
            payload = original.model_dump(mode="python")
            for key, value in {
                "title": request.title,
                "severity": request.severity,
                "claim": request.claim,
                "confidence": request.confidence,
            }.items():
                if value is not None:
                    payload[key] = value
            edited_finding = RiskFinding.model_validate(payload)
        try:
            review = service.review_finding(
                run_id=run_id,
                finding_id=finding_id,
                decision=request.decision,
                reviewer=request.reviewer.strip(),
                comment=request.comment.strip() if request.comment else None,
                edited_finding=edited_finding,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewBlockedError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "确定性核验未通过，可驳回或先补充证据。",
                    "verification": exc.verification.model_dump(mode="json"),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"review": _model_payload(review)}

    @router.post("/runs/{run_id}/export")
    def export_run(run_id: str) -> dict[str, object]:
        try:
            bundle = service.export_verified_bundle(run_id)
        except IncompleteReviewError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": "仍有风险未复核。", "finding_ids": exc.finding_ids},
            ) from exc
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "files": {
                "pdf": f"/api/reports/runs/{run_id}/downloads/pdf",
                "json": f"/api/reports/runs/{run_id}/downloads/json",
                "audit": f"/api/reports/runs/{run_id}/downloads/audit",
            },
            "paths": {
                "pdf": str(bundle.pdf_path),
                "json": str(bundle.json_path),
                "audit": str(bundle.audit_path),
            },
        }

    @router.get("/runs/{run_id}/downloads/{kind}")
    def download_run(
        run_id: str,
        kind: Literal["pdf", "json", "audit"],
    ) -> FileResponse:
        reference = service.repository.get_verified_report_path(run_id)
        if not reference:
            raise HTTPException(status_code=404, detail="该运行尚未导出正式报告。")
        verified_json = _resolve_project_file(service.project_root, reference)
        names = {
            "pdf": "verified_risk_report.pdf",
            "json": "verified_risk_analysis.json",
            "audit": "analysis_audit_bundle.json",
        }
        target = verified_json.with_name(names[kind])
        if not target.is_file():
            raise HTTPException(status_code=404, detail="导出文件不存在。")
        media_types = {
            "pdf": "application/pdf",
            "json": "application/json",
            "audit": "application/json",
        }
        return FileResponse(target, media_type=media_types[kind], filename=target.name)

    @router.get("/runs/{run_id}/evidence/{evidence_id}/image")
    def evidence_image(run_id: str, evidence_id: str) -> FileResponse:
        evidences = service.repository.list_evidence(run_id, evidence_ids=(evidence_id,))
        if not evidences or not evidences[0].screenshot_path:
            raise HTTPException(status_code=404, detail="该证据没有高亮截图。")
        target = _resolve_project_file(service.project_root, evidences[0].screenshot_path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="证据截图文件不存在。")
        return FileResponse(target)

    return router


__all__ = [
    "FindingReviewRequest",
    "HistoricalPredictionRequest",
    "create_reports_router",
]
