"""与Streamlit页面重跑解耦的历史招股书批量分析任务。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from threading import Event, Lock, Thread
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from ipo_risk.agents.intervention import (
    LlmFailureRecoveryController,
    LlmFailureRecoveryDecision,
    LlmFailureRecoveryRequest,
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
)
from ipo_risk.agents.model_routing import normalize_agent_models
from ipo_risk.storage.files import read_json, write_json
from ipo_risk.runtime.analysis_job import AnalysisProgressEvent
from ipo_risk.runtime.prediction import (
    generate_runtime_prediction,
)

if TYPE_CHECKING:
    from ipo_risk.experiments import AgentFeatureExperimentResult


BatchQueueState = Literal[
    "pending",
    "running",
    "pausing",
    "paused",
    "completed",
    "failed",
]
ReasoningModel = Literal["deepseek", "deepseek_flash", "deepseek_pro", "kimi"]


@dataclass(frozen=True)
class BatchCompany:
    company_id: str
    company_folder: str
    company_name: str
    prospectus_date: str
    split: str


@dataclass(frozen=True)
class BatchUploadSpec:
    """用户拖入批量队列的一份、已确认元数据的招股书。"""

    file_name: str
    content: bytes
    company_id: str
    company_name: str
    document_date: date
    source_hash: str
    page_count: int
    listing_date: date | None = None


@dataclass(frozen=True)
class BatchCompanySnapshot:
    company_id: str
    company_folder: str
    company_name: str
    prospectus_date: str
    split: str
    state: str
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    run_id: str | None = None
    finding_count: int | None = None
    evidence_count: int | None = None
    feature_path: str | None = None
    base_feature_path: str | None = None
    base_prediction: dict[str, Any] | None = None
    base_prediction_path: str | None = None
    base_prediction_status: str | None = None
    base_prediction_error: str | None = None
    prediction: dict[str, Any] | None = None
    prediction_path: str | None = None
    prediction_status: str | None = None
    prediction_error: str | None = None
    error: str | None = None
    preparation_state: str | None = None
    preparation_error: str | None = None
    risk_scope_review: RiskScopeReviewRequest | None = None
    llm_failure_recovery: LlmFailureRecoveryRequest | None = None
    agent_models: dict[str, str] | None = None
    model_assignment_locked: bool = True


@dataclass(frozen=True)
class BatchAnalysisJobSnapshot:
    experiment_name: str
    state: BatchQueueState
    companies: tuple[BatchCompanySnapshot, ...]
    active: bool
    phase: Literal["preparing", "analyzing", "finished"]
    phase_detail: str | None
    pause_reason: str | None
    error: str | None

    @property
    def current_company_folder(self) -> str | None:
        return next(
            (
                company.company_folder
                for company in self.companies
                if company.state
                in {
                    "awaiting_recovery",
                    "awaiting_review",
                    "running",
                    "preparing",
                }
            ),
            None,
        )


ExperimentRunner = Callable[..., "AgentFeatureExperimentResult"]


def _default_experiment_runner(
    *args: Any,
    **kwargs: Any,
) -> AgentFeatureExperimentResult:
    """真正启动批次时才加载完整分析流水线。"""
    from ipo_risk.experiments import run_agent_feature_experiment

    return run_agent_feature_experiment(*args, **kwargs)


class BatchAnalysisJob:
    """在后台受控并行运行公司队列，并在公司边界安全暂停。"""

    def __init__(
        self,
        *,
        project_root: Path,
        experiment_name: str,
        companies: tuple[BatchCompany, ...],
        uploads: tuple[BatchUploadSpec, ...] = (),
        runner: ExperimentRunner = _default_experiment_runner,
        reasoning_model: ReasoningModel = "deepseek_flash",
        agent_models: Mapping[str, str] | None = None,
    ) -> None:
        if not companies:
            raise ValueError("batch analysis requires at least one company")
        locked_agent_models = normalize_agent_models(
            agent_models,
            legacy_reasoning_model=reasoning_model,
        )
        self.project_root = project_root.resolve()
        self.experiment_name = experiment_name
        self.companies = companies
        self._uploads = uploads
        self._runner = runner
        self.reasoning_model = reasoning_model
        self.agent_models = MappingProxyType(locked_agent_models)
        self._lock = Lock()
        self._pause_requested = Event()
        self._thread: Thread | None = None
        self._error: str | None = None
        self._preparing = bool(uploads)
        self._prepared_uploads = not uploads
        self._phase_detail: str | None = "等待后台登记上传文件。" if uploads else None
        self._company_rows: tuple[dict[str, Any], ...] = ()
        self._event_offsets: dict[str, int] = {}
        self._event_cache: dict[str, list[AnalysisProgressEvent]] = {}
        self._delta_indexes: dict[str, dict[str, int]] = {}
        self._risk_scope_review_controllers = {
            company.company_folder: RiskScopeReviewController() for company in companies
        }
        self._llm_failure_recovery_controllers = {
            company.company_folder: LlmFailureRecoveryController() for company in companies
        }
        self._registration_states = (
            {company.company_folder: "pending" for company in companies} if uploads else {}
        )

    @classmethod
    def from_uploads(
        cls,
        *,
        project_root: Path,
        experiment_name: str,
        uploads: tuple[BatchUploadSpec, ...],
        runner: ExperimentRunner = _default_experiment_runner,
        reasoning_model: ReasoningModel = "deepseek_flash",
        agent_models: Mapping[str, str] | None = None,
    ) -> BatchAnalysisJob:
        """建立上传批次；文件登记与正式分析都在后台线程中执行。"""
        if not uploads:
            raise ValueError("batch uploads must not be empty")
        companies = tuple(
            BatchCompany(
                company_id=upload.company_id,
                company_folder=f"upload_{upload.source_hash[:16]}",
                company_name=upload.company_name,
                prospectus_date=upload.document_date.isoformat(),
                split="upload",
            )
            for upload in uploads
        )
        return cls(
            project_root=project_root,
            experiment_name=experiment_name,
            companies=companies,
            uploads=uploads,
            runner=runner,
            reasoning_model=reasoning_model,
            agent_models=agent_models,
        )

    @property
    def experiment_dir(self) -> Path:
        return self.project_root / "artifacts" / "experiments" / self.experiment_name

    @property
    def status_path(self) -> Path:
        return self.experiment_dir / "status.json"

    def start(self) -> None:
        """启动新队列；重复调用不会并行启动第二条线程。"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("batch analysis job is already running")
            self._pause_requested.clear()
            self._error = None
            self._thread = Thread(
                target=self._run,
                name=f"ipo-batch-{self.experiment_name}",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def pause_after_current(self) -> None:
        """请求在当前运行公司完整落盘后暂停，不强杀模型请求。"""
        self._pause_requested.set()

    def submit_risk_scope_review(
        self,
        company_folder: str,
        decision: RiskScopeReviewDecision,
    ) -> None:
        """提交一家公司的风险范围确认，不影响其他公司流水线。"""
        controller = self._risk_scope_review_controllers.get(company_folder)
        if controller is None:
            raise KeyError(f"unknown batch company: {company_folder}")
        controller.submit(decision)

    def submit_llm_failure_recovery(
        self,
        company_folder: str,
        decision: LlmFailureRecoveryDecision,
    ) -> None:
        """提交一家公司的模型故障恢复选择，不影响其他公司。"""
        controller = self._llm_failure_recovery_controllers.get(company_folder)
        if controller is None:
            raise KeyError(f"unknown batch company: {company_folder}")
        controller.submit(decision)

    def resume(self) -> None:
        """从已完成公司之后继续运行剩余队列。"""
        self.start()

    def retry_failed(self) -> int:
        """把失败或中断的公司恢复为待处理，返回重置数量。"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("cannot retry companies while batch is running")
        if not self.status_path.is_file():
            return 0
        status = read_json(self.status_path)
        reset_count = 0
        for item in status.get("companies", {}).values():
            if item.get("state") not in {"failed", "paused", "running"}:
                continue
            item["state"] = "pending"
            for key in (
                "started_at",
                "ended_at",
                "duration_seconds",
                "error",
                "traceback",
            ):
                item.pop(key, None)
            reset_count += 1
        if reset_count:
            status["queue_state"] = "paused"
            status["pause_reason"] = "失败项已恢复为待处理，请点击继续队列。"
            write_json(status, self.status_path)
        return reset_count

    def snapshot(self) -> BatchAnalysisJobSnapshot:
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            runtime_error = self._error
            preparing = self._preparing
            phase_detail = self._phase_detail
            companies = self.companies
            review_controllers = dict(self._risk_scope_review_controllers)
            recovery_controllers = dict(self._llm_failure_recovery_controllers)
            registration_states = dict(self._registration_states)
        raw_status = read_json(self.status_path) if self.status_path.is_file() else {}
        raw_companies = raw_status.get("companies", {})
        snapshots: list[BatchCompanySnapshot] = []
        for company in companies:
            raw = raw_companies.get(company.company_folder, {})
            raw_state = str(raw.get("state", "pending"))
            preparation_state = _optional_text(
                raw.get("preparation_state") or registration_states.get(company.company_folder)
            )
            scope_review = review_controllers.get(company.company_folder)
            pending_scope_review = (
                scope_review.pending_request() if scope_review is not None else None
            )
            recovery = recovery_controllers.get(company.company_folder)
            pending_recovery = recovery.pending_request() if recovery is not None else None
            if raw_state == "running" and pending_recovery is not None:
                company_state = "awaiting_recovery"
            elif raw_state == "running" and pending_scope_review is not None:
                company_state = "awaiting_review"
            elif raw_state == "pending" and preparation_state in {
                "registering",
                "registered",
                "running",
            }:
                company_state = "preparing"
            elif raw_state == "pending" and preparation_state == "completed":
                company_state = "ready"
            else:
                company_state = raw_state
            snapshots.append(
                BatchCompanySnapshot(
                    company_id=company.company_id,
                    company_folder=company.company_folder,
                    company_name=company.company_name,
                    prospectus_date=company.prospectus_date,
                    split=company.split,
                    state=company_state,
                    started_at=_optional_text(raw.get("started_at")),
                    ended_at=_optional_text(raw.get("ended_at")),
                    duration_seconds=_optional_float(raw.get("duration_seconds")),
                    run_id=_optional_text(raw.get("run_id")),
                    finding_count=_optional_int(raw.get("finding_count")),
                    evidence_count=_optional_int(raw.get("evidence_count")),
                    feature_path=_optional_text(raw.get("feature_path")),
                    base_feature_path=_optional_text(raw.get("base_feature_path")),
                    base_prediction=(
                        dict(raw["base_prediction"])
                        if isinstance(raw.get("base_prediction"), dict)
                        else None
                    ),
                    base_prediction_path=_optional_text(raw.get("base_prediction_path")),
                    base_prediction_status=_optional_text(raw.get("base_prediction_status")),
                    base_prediction_error=_optional_text(raw.get("base_prediction_error")),
                    prediction=(
                        dict(raw["prediction"]) if isinstance(raw.get("prediction"), dict) else None
                    ),
                    prediction_path=_optional_text(raw.get("prediction_path")),
                    prediction_status=_optional_text(raw.get("prediction_status")),
                    prediction_error=_optional_text(raw.get("prediction_error")),
                    error=_optional_text(raw.get("error")),
                    preparation_state=preparation_state,
                    preparation_error=_optional_text(raw.get("preparation_error")),
                    risk_scope_review=pending_scope_review,
                    llm_failure_recovery=pending_recovery,
                    agent_models=dict(self.agent_models),
                    model_assignment_locked=True,
                )
            )
        if active and self._pause_requested.is_set():
            state: BatchQueueState = "pausing"
        elif active:
            state = "running"
        elif runtime_error:
            state = "failed"
        else:
            raw_state = str(raw_status.get("queue_state", "pending"))
            state = (
                cast(BatchQueueState, raw_state)
                if raw_state in {"pending", "running", "paused", "completed", "failed"}
                else "pending"
            )
        return BatchAnalysisJobSnapshot(
            experiment_name=self.experiment_name,
            state=state,
            companies=tuple(snapshots),
            active=active,
            phase=("preparing" if preparing else ("analyzing" if active else "finished")),
            phase_detail=phase_detail,
            pause_reason=_optional_text(raw_status.get("pause_reason")),
            error=runtime_error,
        )

    def events_for(self, company_folder: str) -> tuple[AnalysisProgressEvent, ...]:
        company = next(
            (item for item in self.companies if item.company_folder == company_folder),
            None,
        )
        if company is None:
            return ()
        path = self.experiment_dir / "events" / f"{company.company_id}.jsonl"
        with self._lock:
            events = self._event_cache.setdefault(company.company_id, [])
            offset = self._event_offsets.get(company.company_id, 0)
            delta_indexes = self._delta_indexes.setdefault(company.company_id, {})
            if not path.is_file():
                return tuple(events)
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                for line in handle:
                    try:
                        payload = json.loads(line)
                        event = AnalysisProgressEvent(
                            stage=str(payload["stage"]),
                            detail=str(payload["detail"]),
                            occurred_at=datetime.fromisoformat(str(payload["timestamp"])),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    _append_coalesced_event(events, delta_indexes, event)
                self._event_offsets[company.company_id] = handle.tell()
            return tuple(events)

    def feature_payload(self, company_id: str) -> dict[str, Any] | None:
        path = self.experiment_dir / "companies" / f"{company_id}.json"
        return read_json(path) if path.is_file() else None

    def rerun_prediction(
        self,
        company_folder: str,
        *,
        variant: Literal["auto", "base", "agent_enhanced"] = "auto",
    ) -> dict[str, Any]:
        """复用已保存特征重算预测，不重新解析PDF，也不重新调用Agent。"""

        company = next(
            (item for item in self.companies if item.company_folder == company_folder),
            None,
        )
        if company is None:
            raise KeyError(f"unknown batch company: {company_folder}")
        if not self.status_path.is_file():
            raise RuntimeError("公司特征尚未就绪。")
        status = read_json(self.status_path)
        current = status.get("companies", {}).get(company_folder)
        if not isinstance(current, dict):
            raise RuntimeError("公司状态不存在。")
        effective_variant = variant
        if effective_variant == "auto":
            effective_variant = (
                "agent_enhanced"
                if current.get("prelisting_features_path")
                else "base"
            )
        path_key = (
            "prelisting_features_path"
            if effective_variant == "agent_enhanced"
            else "base_feature_path"
        )
        raw_feature_path = _optional_text(current.get(path_key))
        if raw_feature_path is None:
            raise RuntimeError(
                "Agent增强特征尚未生成。"
                if effective_variant == "agent_enhanced"
                else "基础特征尚未生成，请等待PDF解析完成。"
            )
        feature_path = Path(raw_feature_path)
        if not feature_path.is_absolute():
            feature_path = self.project_root / feature_path
        if not feature_path.is_file():
            raise RuntimeError(f"预测特征文件不存在：{feature_path}")
        feature_payload = read_json(feature_path)
        raw_features = feature_payload.get("features")
        if not isinstance(raw_features, dict):
            raise RuntimeError("预测特征文件格式无效。")
        prediction_as_of_text = _optional_text(feature_payload.get("prediction_as_of"))
        prediction_as_of = (
            date.fromisoformat(prediction_as_of_text)
            if prediction_as_of_text is not None
            else None
        )
        units = {
            name: str(value.get("unit"))
            for name, value in raw_features.items()
            if isinstance(value, dict) and value.get("unit") is not None
        }
        formulas = {
            name: str(value.get("formula_version"))
            for name, value in raw_features.items()
            if isinstance(value, dict) and value.get("formula_version") is not None
        }
        destination = feature_path.with_name(
            "prediction_5d.json"
            if effective_variant == "agent_enhanced"
            else "prediction_5d_base.json"
        )
        prediction, prediction_path = generate_runtime_prediction(
            project_root=self.project_root,
            company_id=company.company_id,
            company_name=company.company_name,
            run_id=str(feature_payload.get("run_id") or f"BASE-{company.company_id}"),
            features=raw_features,
            report_path=destination.with_name("risk_analysis.json"),
            prediction_as_of=prediction_as_of,
            feature_units=units,
            feature_formula_versions=formulas,
            model_path=None,
            destination_path=destination,
            prediction_variant=effective_variant,
        )
        prefix = ""
        current[f"{prefix}prediction"] = prediction
        current[f"{prefix}prediction_path"] = _portable_runtime_path(
            prediction_path,
            self.project_root,
        )
        current[f"{prefix}prediction_status"] = "completed"
        current.pop(f"{prefix}prediction_error", None)
        write_json(status, self.status_path)
        return prediction

    def _run(self) -> None:
        try:
            if not self._prepared_uploads:
                self._prepare_uploads()
            with self._lock:
                companies = self.companies
                company_rows = self._company_rows
                self._phase_detail = (
                    "正在并行解析并向量化待处理PDF；就绪后将受控并行运行公司Agent。"
                )
            self._runner(
                self.project_root,
                experiment_name=self.experiment_name,
                company_folders=tuple(company.company_folder for company in companies),
                company_rows=company_rows,
                continue_on_error=True,
                pause_requested=self._pause_requested.is_set,
                preparation_progress=self._on_preparation_progress,
                risk_scope_review_controllers=(self._risk_scope_review_controllers),
                llm_failure_recovery_controllers=(self._llm_failure_recovery_controllers),
                reasoning_model=self.reasoning_model,
                agent_models=dict(self.agent_models),
            )
            with self._lock:
                self._phase_detail = None
        except Exception as exc:  # noqa: BLE001 - 后台队列必须向页面暴露错误
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._preparing = False
                self._phase_detail = None

    def _on_preparation_progress(
        self,
        completed: int,
        total: int,
        company_folder: str,
        state: str,
    ) -> None:
        """把并行预处理进度转换为页面可读的一行状态。"""
        state_text = "已就绪" if state == "completed" else "将由正式分析重试"
        with self._lock:
            self._phase_detail = (
                f"文档预处理 {completed}/{total}：{company_folder} {state_text}；"
                "已就绪公司将按并发上限启动Agent。"
            )

    def _prepare_uploads(self) -> None:
        """在后台登记拖入的PDF，并转换为实验队列可消费的公司记录。"""
        from ipo_risk.uploads import ProspectusUploadService

        upload_service = ProspectusUploadService(project_root=self.project_root)
        prepared_companies: list[BatchCompany] = []
        company_rows: list[dict[str, Any]] = []
        total = len(self._uploads)
        placeholder_companies = self.companies
        for index, upload in enumerate(self._uploads, start=1):
            placeholder_folder = placeholder_companies[index - 1].company_folder
            with self._lock:
                self._registration_states[placeholder_folder] = "registering"
                self._phase_detail = f"正在登记第{index}/{total}份：{upload.file_name}"
            registered = upload_service.register(
                file_name=upload.file_name,
                content=upload.content,
                company_id=upload.company_id,
                company_name=upload.company_name,
                document_date=upload.document_date,
                listing_date=upload.listing_date,
            )
            company = BatchCompany(
                company_id=upload.company_id,
                company_folder=registered.company_folder,
                company_name=upload.company_name,
                prospectus_date=upload.document_date.isoformat(),
                split="upload",
            )
            prepared_companies.append(company)
            with self._lock:
                self._registration_states[placeholder_folder] = "registered"
            company_rows.append(
                {
                    "company_id": company.company_id,
                    "company_folder": company.company_folder,
                    "company_name": company.company_name,
                    "prospectus_date": company.prospectus_date,
                    "source_hash": registered.source_hash,
                    "split": company.split,
                    "label_status": "not_required",
                }
            )
        with self._lock:
            self.companies = tuple(prepared_companies)
            self._company_rows = tuple(company_rows)
            self._risk_scope_review_controllers = {
                company.company_folder: RiskScopeReviewController()
                for company in prepared_companies
            }
            self._llm_failure_recovery_controllers = {
                company.company_folder: LlmFailureRecoveryController()
                for company in prepared_companies
            }
            self._registration_states = {
                company.company_folder: "registered" for company in prepared_companies
            }
            self._prepared_uploads = True
            self._preparing = False
            self._phase_detail = "资料登记完成，正在并行准备公司分析。"


def load_batch_companies(project_root: Path) -> tuple[BatchCompany, ...]:
    """加载赛事清单中可分析的公司，不读取上市后标签值。"""
    manifest_path = project_root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    if not manifest_path.is_file():
        return ()
    companies: list[BatchCompany] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            company_folder = str(payload.get("company_folder") or "").strip()
            raw_company_id = str(payload.get("company_id") or "").strip()
            if not company_folder or not raw_company_id:
                continue
            company_id = raw_company_id.zfill(5)
            companies.append(
                BatchCompany(
                    company_id=company_id,
                    company_folder=company_folder,
                    company_name=str(payload.get("company_name") or company_folder),
                    prospectus_date=str(payload.get("prospectus_date") or ""),
                    split=str(payload.get("split") or "unassigned"),
                )
            )
    return tuple(
        sorted(
            companies,
            key=lambda item: (
                item.split,
                item.prospectus_date,
                item.company_id,
            ),
        )
    )


def _append_coalesced_event(
    events: list[AnalysisProgressEvent],
    delta_indexes: dict[str, int],
    event: AnalysisProgressEvent,
) -> None:
    if event.stage in {
        "parsing_progress",
        "embedding_progress",
        "rag_features_progress",
        "features_progress",
    }:
        stream_key = f"progress|{event.stage}"
        existing_index = delta_indexes.get(stream_key)
        if existing_index is None:
            delta_indexes[stream_key] = len(events)
            events.append(event)
        else:
            events[existing_index] = event
        return
    stream_key, marker, delta = _delta_stream(event.stage, event.detail)
    if stream_key is None:
        events.append(event)
        return
    existing_index = delta_indexes.get(stream_key)
    if existing_index is None:
        delta_indexes[stream_key] = len(events)
        events.append(event)
        return
    previous = events[existing_index]
    previous_delta = previous.detail
    if marker and previous_delta.startswith(marker):
        previous_delta = previous_delta[len(marker) :]
    events[existing_index] = AnalysisProgressEvent(
        stage=event.stage,
        detail=f"{marker}{previous_delta}{delta}",
        occurred_at=event.occurred_at,
    )


def _delta_stream(stage: str, detail: str) -> tuple[str | None, str, str]:
    if not stage.endswith(":delta"):
        return None, "", detail
    marker = ""
    delta = detail
    if detail.startswith("[["):
        marker_end = detail.find("]]")
        if marker_end >= 0:
            marker_end += 2
            marker = detail[:marker_end]
            delta = detail[marker_end:]
    return f"{stage}|{marker or 'default'}", marker, delta


def _optional_text(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _portable_runtime_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


__all__ = [
    "BatchAnalysisJob",
    "BatchAnalysisJobSnapshot",
    "BatchCompany",
    "BatchCompanySnapshot",
    "BatchQueueState",
    "BatchUploadSpec",
    "load_batch_companies",
]
