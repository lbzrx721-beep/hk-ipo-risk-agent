"""与Streamlit页面重跑解耦的单次招股书分析任务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING, Literal, Protocol

from ipo_risk.agents.intervention import (
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
)
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.retrieval.persistence import PersistenceSnapshot
from ipo_risk.uploads import UploadedProspectus

if TYPE_CHECKING:
    from ipo_risk.pipeline import AnalysisArtifacts


AnalysisJobStatus = Literal[
    "pending",
    "running",
    "awaiting_review",
    "completed",
    "paused",
    "failed",
]


class UploadService(Protocol):
    project_root: Path

    def register(
        self,
        *,
        file_name: str,
        content: bytes,
        company_id: str,
        company_name: str,
        document_date: date,
    ) -> UploadedProspectus: ...


@dataclass(frozen=True)
class AnalysisProgressEvent:
    stage: str
    detail: str
    occurred_at: datetime


@dataclass(frozen=True)
class AnalysisJobSnapshot:
    status: AnalysisJobStatus
    events: tuple[AnalysisProgressEvent, ...]
    artifacts: AnalysisArtifacts | None
    error: str | None
    vector_persistence: PersistenceSnapshot | None = None
    risk_scope_review: RiskScopeReviewRequest | None = None


class AnalysisJob:
    """在后台线程运行流水线，并提供线程安全的只读状态快照。"""

    def __init__(
        self,
        *,
        upload_service: UploadService,
        analyze: Callable[..., AnalysisArtifacts],
        file_name: str,
        content: bytes,
        company_id: str,
        company_name: str,
        document_date: date,
    ) -> None:
        self._upload_service = upload_service
        self._analyze = analyze
        self._file_name = file_name
        self._content = content
        self._company_id = company_id
        self._company_name = company_name
        self._document_date = document_date
        self._lock = Lock()
        self._status: AnalysisJobStatus = "pending"
        self._events: list[AnalysisProgressEvent] = []
        self._delta_event_indexes: dict[str, int] = {}
        self._artifacts: AnalysisArtifacts | None = None
        self._error: str | None = None
        self._thread: Thread | None = None
        self._risk_scope_review_controller = RiskScopeReviewController()

    def start(self) -> None:
        """启动且只启动一次后台分析。"""
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("analysis job has already been started")
            self._status = "running"
            self._thread = Thread(
                target=self._run,
                name=f"ipo-analysis-{self._company_id}",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def _emit(self, stage: str, detail: str) -> None:
        event = AnalysisProgressEvent(
            stage=stage,
            detail=detail,
            occurred_at=datetime.now().astimezone(),
        )
        with self._lock:
            if stage.endswith(":prompt"):
                role_prefix = stage.rsplit(":", maxsplit=1)[0] + ":delta|"
                for key in tuple(self._delta_event_indexes):
                    if key.startswith(role_prefix) and not key.startswith(
                        "agent:discovery:delta|"
                    ):
                        self._delta_event_indexes.pop(key, None)
            stream_key, delta = _delta_stream(stage, detail)
            if stream_key is not None:
                existing_index = self._delta_event_indexes.get(stream_key)
                if existing_index is not None:
                    previous = self._events[existing_index]
                    marker = detail[: len(detail) - len(delta)]
                    previous_delta = previous.detail
                    if marker and previous_delta.startswith(marker):
                        previous_delta = previous_delta[len(marker) :]
                    self._events[existing_index] = AnalysisProgressEvent(
                        stage=stage,
                        detail=f"{marker}{previous_delta}{delta}",
                        occurred_at=event.occurred_at,
                    )
                    return
                self._delta_event_indexes[stream_key] = len(self._events)
            self._events.append(event)

    def _run(self) -> None:
        try:
            self._emit("upload_started", "正在校验PDF并登记不可变文档版本。")
            registered = self._upload_service.register(
                file_name=self._file_name,
                content=self._content,
                company_id=self._company_id,
                company_name=self._company_name,
                document_date=self._document_date,
            )
            source_action = (
                "复用已登记的同一文件" if registered.reused else "保存新的原始文件"
            )
            self._emit(
                "upload_completed",
                f"已{source_action}：{registered.page_count}页，"
                f"文档ID {registered.document_id}。",
            )
            artifacts = self._analyze(
                registered.company_folder,
                project_root=self._upload_service.project_root,
                document_id=registered.document_id,
                progress_callback=self._emit,
                risk_scope_review_controller=self._risk_scope_review_controller,
            )
        except ModelBudgetExceededError as exc:
            self._emit("model_budget_paused", str(exc))
            with self._lock:
                self._status = "paused"
                self._error = str(exc)
            return
        except Exception as exc:
            with self._lock:
                self._status = "failed"
                self._error = str(exc)
            return
        with self._lock:
            self._artifacts = artifacts
            self._status = "completed"

    def snapshot(self) -> AnalysisJobSnapshot:
        scope_review = self._risk_scope_review_controller.pending_request()
        with self._lock:
            artifacts = self._artifacts
            persistence_task = (
                getattr(artifacts, "vector_persistence_task", None)
                if artifacts is not None
                else None
            )
            return AnalysisJobSnapshot(
                status=(
                    "awaiting_review"
                    if self._status == "running" and scope_review is not None
                    else self._status
                ),
                events=tuple(self._events),
                artifacts=artifacts,
                error=self._error,
                vector_persistence=(
                    persistence_task.snapshot()
                    if persistence_task is not None
                    else None
                ),
                risk_scope_review=scope_review,
            )

    def submit_risk_scope_review(
        self,
        decision: RiskScopeReviewDecision,
    ) -> None:
        """提交页面确认结果并恢复等待中的LangGraph节点。"""
        self._risk_scope_review_controller.submit(decision)

    def wait_for_risk_scope_review(self, timeout: float | None = None) -> bool:
        """仅供测试与受控协调等待人工确认节点出现。"""
        return self._risk_scope_review_controller.wait_for_request(timeout)

    def wait(self, timeout: float | None = None) -> bool:
        """等待测试或受控关闭；页面渲染不应调用此方法阻塞。"""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()


def _delta_stream(stage: str, detail: str) -> tuple[str | None, str]:
    """把高频Token事件归并为每个模型流一个快照，避免事件列表无限增长。"""
    if not stage.endswith(":delta"):
        return None, detail
    marker = ""
    delta = detail
    if detail.startswith("[["):
        marker_end = detail.find("]]")
        if marker_end >= 0:
            marker_end += 2
            marker = detail[:marker_end]
            delta = detail[marker_end:]
    return f"{stage}|{marker or 'default'}", delta


__all__ = [
    "AnalysisJob",
    "AnalysisJobSnapshot",
    "AnalysisJobStatus",
    "AnalysisProgressEvent",
]
