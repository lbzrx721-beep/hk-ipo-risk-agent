"""跨页面生命周期的批量分析工作区注册表。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException

from ipo_risk.agents.intervention import (
    LlmFailureRecoveryDecision,
    RiskScopeReviewDecision,
)
from ipo_risk.runtime.batch_analysis_job import (
    BatchAnalysisJob,
    BatchCompany,
    BatchUploadSpec,
    load_batch_companies,
)
from ipo_risk.runtime.progress import build_analysis_progress_snapshot


def _workspace_id() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    return f"web_batch_{timestamp}_{uuid4().hex[:6]}"


@dataclass(frozen=True)
class WorkspaceSegment:
    segment_id: str
    job: BatchAnalysisJob


class BatchWorkspace:
    """把后续追加的PDF统一呈现在一个前端队列中。"""

    def __init__(self, *, workspace_id: str, project_root: Path) -> None:
        self.workspace_id = workspace_id
        self.project_root = project_root.resolve()
        self.created_at = datetime.now().astimezone()
        self._segments: list[WorkspaceSegment] = []
        self._lock = RLock()

    def _next_segment_id(self) -> str:
        return f"s{len(self._segments) + 1:02d}"

    def add_job(self, job: BatchAnalysisJob) -> WorkspaceSegment:
        with self._lock:
            segment = WorkspaceSegment(self._next_segment_id(), job)
            self._segments.append(segment)
        job.start()
        return segment

    def append_uploads(
        self,
        uploads: tuple[BatchUploadSpec, ...],
        *,
        reasoning_model: Literal["deepseek", "deepseek_flash", "deepseek_pro", "kimi"] = "deepseek_flash",
        agent_models: Mapping[str, str] | None = None,
    ) -> None:
        if not uploads:
            raise ValueError("至少需要上传一份PDF。")
        with self._lock:
            sequence = len(self._segments) + 1
        experiment_name = f"{self.workspace_id}_s{sequence:02d}"
        self.add_job(
            BatchAnalysisJob.from_uploads(
                project_root=self.project_root,
                experiment_name=experiment_name,
                uploads=uploads,
                reasoning_model=reasoning_model,
                agent_models=agent_models,
            )
        )

    def pause_after_current(self) -> None:
        for segment in self._segment_snapshot():
            if segment.job.snapshot().active:
                segment.job.pause_after_current()

    def resume(self) -> None:
        resumed = False
        for segment in self._segment_snapshot():
            snapshot = segment.job.snapshot()
            if snapshot.active:
                continue
            if snapshot.state in {"paused", "failed"}:
                if snapshot.state == "failed":
                    segment.job.retry_failed()
                segment.job.resume()
                resumed = True
        if not resumed and self.state() == "completed":
            raise ValueError("该批次已经全部完成。")

    def state(self) -> str:
        snapshots = [segment.job.snapshot() for segment in self._segment_snapshot()]
        if not snapshots:
            return "pending"
        if any(item.state == "pausing" for item in snapshots):
            return "pausing"
        if any(item.active for item in snapshots):
            return "running"
        if any(item.state == "paused" for item in snapshots):
            return "paused"
        if any(item.state == "failed" for item in snapshots):
            return "failed"
        if all(item.state == "completed" for item in snapshots):
            return "completed"
        return "pending"

    def snapshot_payload(self) -> dict[str, Any]:
        companies: list[dict[str, Any]] = []
        phases: list[dict[str, Any]] = []
        for segment in self._segment_snapshot():
            snapshot = segment.job.snapshot()
            phases.append(
                {
                    "segment_id": segment.segment_id,
                    "state": snapshot.state,
                    "phase": snapshot.phase,
                    "phase_detail": snapshot.phase_detail,
                    "active": snapshot.active,
                    "error": snapshot.error,
                    "agent_models": dict(getattr(segment.job, "agent_models", {})),
                    "model_assignment_locked": True,
                }
            )
            for company in snapshot.companies:
                events = segment.job.events_for(company.company_folder)
                payload = asdict(company)
                payload["company_key"] = f"{segment.segment_id}:{company.company_id}"
                payload["segment_id"] = segment.segment_id
                review = company.risk_scope_review
                payload["risk_scope_review"] = review.model_dump(mode="json") if review else None
                recovery = company.llm_failure_recovery
                payload["llm_failure_recovery"] = (
                    recovery.model_dump(mode="json") if recovery else None
                )
                progress = build_analysis_progress_snapshot(
                    events,
                    status=_progress_status(company.state, bool(events)),
                    error=company.error,
                    include_outputs=False,
                )
                progress_elapsed = progress.get("elapsed_seconds")
                if events:
                    payload["duration_seconds"] = progress_elapsed
                payload["duration_updated_at"] = datetime.now().astimezone().isoformat()
                payload["review_wait_seconds"] = progress.get(
                    "review_wait_seconds",
                    0,
                )
                # 模型正文通过独立结构化事件按变化推送；公司快照只保留轻量状态，
                # 避免每秒重复传输不断增长的流式 JSON。
                for key in (
                    "events",
                    "outputs",
                    "elapsed_seconds",
                    "review_wait_seconds",
                ):
                    progress.pop(key, None)
                payload["progress"] = progress
                companies.append(payload)
        counts: dict[str, int] = {}
        for company_payload in companies:
            state = str(company_payload["state"])
            counts[state] = counts.get(state, 0) + 1
        return {
            "batch_id": self.workspace_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state(),
            "company_count": len(companies),
            "counts": counts,
            "companies": companies,
            "segments": phases,
        }

    def events_payload(self, company_key: str) -> list[dict[str, str]]:
        segment, company = self._resolve_company(company_key)
        return [
            {
                "stage": event.stage,
                "detail": event.detail,
                "occurred_at": event.occurred_at.isoformat(),
            }
            for event in segment.job.events_for(company.company_folder)
        ]

    def structured_events_payload(
        self,
        company_key: str,
    ) -> list[dict[str, str]]:
        """返回需要跨最近事件窗口长期保留的Agent结构化快照。"""
        segment, company = self._resolve_company(company_key)
        events = segment.job.events_for(company.company_folder)
        return [
            {
                "stage": event.stage,
                "detail": event.detail,
                "occurred_at": event.occurred_at.isoformat(),
            }
            for event in events
            if _is_durable_agent_event(event.stage)
        ]

    def submit_risk_scope_review(
        self,
        company_key: str,
        decision: RiskScopeReviewDecision,
    ) -> None:
        segment, company = self._resolve_company(company_key)
        segment.job.submit_risk_scope_review(company.company_folder, decision)

    def submit_llm_failure_recovery(
        self,
        company_key: str,
        decision: LlmFailureRecoveryDecision,
    ) -> None:
        segment, company = self._resolve_company(company_key)
        segment.job.submit_llm_failure_recovery(company.company_folder, decision)

    def rerun_prediction(
        self,
        company_key: str,
        *,
        variant: Literal["auto", "base", "agent_enhanced"] = "auto",
    ) -> dict[str, Any]:
        segment, company = self._resolve_company(company_key)
        return segment.job.rerun_prediction(
            company.company_folder,
            variant=variant,
        )

    def _resolve_company(
        self,
        company_key: str,
    ) -> tuple[WorkspaceSegment, BatchCompany]:
        try:
            segment_id, company_id = company_key.split(":", 1)
        except ValueError as exc:
            raise KeyError(f"无效公司标识：{company_key}") from exc
        for segment in self._segment_snapshot():
            if segment.segment_id != segment_id:
                continue
            for company in segment.job.companies:
                if company.company_id == company_id:
                    return segment, company
        raise KeyError(f"批次中不存在公司：{company_key}")

    def _segment_snapshot(self) -> tuple[WorkspaceSegment, ...]:
        with self._lock:
            return tuple(self._segments)


class BatchWorkspaceRegistry:
    """FastAPI进程内的任务注册表；任务本身仍持续写入磁盘。"""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._workspaces: dict[str, BatchWorkspace] = {}
        self._lock = RLock()

    def catalog(self) -> tuple[BatchCompany, ...]:
        return load_batch_companies(self.project_root)

    @staticmethod
    def company_to_dict(company: BatchCompany) -> dict[str, str]:
        return asdict(company)

    def create_from_uploads(
        self,
        uploads: tuple[BatchUploadSpec, ...],
        *,
        reasoning_model: Literal["deepseek", "deepseek_flash", "deepseek_pro", "kimi"] = "deepseek_flash",
        agent_models: Mapping[str, str] | None = None,
    ) -> BatchWorkspace:
        if not uploads:
            raise ValueError("至少需要上传一份PDF。")
        workspace = self._new_workspace()
        workspace.append_uploads(
            uploads,
            reasoning_model=reasoning_model,
            agent_models=agent_models,
        )
        return workspace

    def create_from_catalog(
        self,
        company_folders: tuple[str, ...],
    ) -> BatchWorkspace:
        catalog = {item.company_folder: item for item in self.catalog()}
        missing = [item for item in company_folders if item not in catalog]
        if missing:
            raise KeyError(f"公司不在赛事清单中：{', '.join(missing)}")
        companies = tuple(catalog[item] for item in company_folders)
        workspace = self._new_workspace()
        workspace.add_job(
            BatchAnalysisJob(
                project_root=self.project_root,
                experiment_name=f"{workspace.workspace_id}_s01",
                companies=companies,
            )
        )
        return workspace

    def require(self, batch_id: str) -> BatchWorkspace:
        with self._lock:
            workspace = self._workspaces.get(batch_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="分析批次不存在或服务已重启。")
        return workspace

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            workspaces = tuple(self._workspaces.values())
        return [workspace.snapshot_payload() for workspace in reversed(workspaces)]

    def _new_workspace(self) -> BatchWorkspace:
        workspace = BatchWorkspace(
            workspace_id=_workspace_id(),
            project_root=self.project_root,
        )
        with self._lock:
            self._workspaces[workspace.workspace_id] = workspace
        return workspace


def _progress_status(state: str, has_events: bool) -> str:
    if state == "completed":
        return "completed"
    if state == "failed":
        return "failed"
    if state in {"paused", "pausing"}:
        return "paused"
    if state == "awaiting_review":
        return "awaiting_review"
    if state == "awaiting_recovery":
        return "awaiting_recovery"
    if has_events or state in {"preparing", "ready", "running"}:
        return "running"
    return "pending"


def _is_durable_agent_event(stage: str) -> bool:
    if not stage.startswith("agent:"):
        return stage in {
            "risk_scoping_completed",
            "risk_scoping_fallback",
            "risk_scope_review_requested",
            "risk_scope_review_completed",
            "llm_recovery_requested",
            "llm_recovery_retry_selected",
            "llm_recovery_bm25_selected",
            "analysis_model_selected",
            "debate_record",
            "features_started",
            "features_completed",
            "features_failed",
            "prediction_started",
            "prediction_completed",
            "prediction_failed",
            "completed",
        }
    action = stage.rsplit(":", maxsplit=1)[-1]
    return action in {
        "structured",
        "risk_catalog",
        "raw_completed",
        "consolidated",
        "completed",
        "degraded",
        "failed",
        "skipped",
    }


__all__ = ["BatchWorkspace", "BatchWorkspaceRegistry", "WorkspaceSegment"]
