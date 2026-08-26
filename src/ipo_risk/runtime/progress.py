"""分析运行状态归约与前端快照构建。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import json
import re
from typing import Any, Literal, Protocol


_AGENT_SPECS = {
    "discovery": ("风险发现 Agent", ":material/travel_explore:"),
    "finance": ("财务 Agent", ":material/account_balance:"),
    "legal": ("法律 Agent", ":material/gavel:"),
    "business": ("业务 Agent", ":material/domain:"),
    "market": ("市场 Agent", ":material/candlestick_chart:"),
    "decision": ("决策 Agent", ":material/hub:"),
}
# 总进度条上只标注 Agent 当前所处的步骤，不展示模型生成的具体内容。
_AGENT_STEP_LABELS = {
    "started": "接收任务",
    "search_planned": "语义检索",
    "search_fallback": "检索兜底",
    "retrieval_planning": "动态检索规划",
    "retrieval_plan_fallback": "检索规划兜底",
    "retrieval_ready": "风险原文包就绪",
    "prompt": "请求模型",
    "delta": "流式生成",
    "structured": "风险卡片已更新",
    "progress": "检索分析证据",
    "batch_completed": "风险原文包已完成",
    "completed": "已完成",
    "failed": "失败后降级",
    "degraded": "部分批次降级",
    "raw_completed": "批次发现完成",
    "consolidated": "全局语义合并",
    "skipped": "未启用",
}
_STAGE_LABELS = {
    "upload_started": "资料登记",
    "upload_completed": "资料登记",
    "parsing_started": "PDF解析",
    "parsing_progress": "PDF解析",
    "parsing_completed": "PDF解析",
    "agents_started": "任务编排",
    "agents_parallel_started": "专业Agent并行调度",
    "embedding_started": "向量索引",
    "embedding_cache_hit": "向量索引",
    "embedding_progress": "向量索引",
    "embedding_completed": "向量索引",
    "embedding_fallback": "向量索引降级",
    "embedding_join_wait": "证据检索准备",
    "embedding_join_completed": "证据检索准备",
    "vector_store_indexed": "向量索引",
    "risk_scoping_started": "公司画像与风险范围",
    "risk_scoping_completed": "公司画像与风险范围",
    "risk_scoping_fallback": "风险范围识别降级",
    "risk_scope_review_requested": "人工确认风险范围",
    "risk_scope_review_completed": "人工确认风险范围",
    "llm_recovery_requested": "模型故障等待处理",
    "llm_recovery_retry_selected": "模型故障恢复",
    "llm_recovery_bm25_selected": "BM25降级确认",
    "agents_completed": "风险整合",
    "calculation_skipped": "数值复算",
    "evidence_started": "证据报告",
    "features_started": "上市前特征",
    "features_progress": "上市前特征",
    "features_completed": "上市前特征",
    "features_failed": "上市前特征",
    "prediction_started": "五日下跌预测",
    "prediction_completed": "五日下跌预测",
    "prediction_failed": "五日下跌预测",
    "model_budget_paused": "模型预算安全暂停",
    "coverage_warning": "风险覆盖提醒",
    "completed": "分析完成",
}
_PROGRESS_PATTERN = re.compile(r"\[(?P<current>\d+)/(?P<total>\d+)]")
_DISCOVERY_DELTA_PATTERN = re.compile(
    r"^\[\[(?P<kind>batch|consolidation):"
    r"(?P<current>\d+)/(?P<total>\d+)]](?P<delta>.*)$",
    re.DOTALL,
)
_OUTPUT_SEPARATOR = "\n\n--- 新的风险任务 ---\n"
_DISCOVERY_RAW_PATTERN = re.compile(
    r"(?:全文(?P<full_batches>\d+)个批次|"
    r"(?P<retrieved_batches>\d+)个风险原文包)共形成"
    r"(?P<raw>\d+)条原始候选"
)
_DISCOVERY_CONSOLIDATED_PATTERN = re.compile(
    r"(?P<raw>\d+)条原始候选.+形成(?P<consolidated>\d+)条待确认风险"
)
_DISCOVERY_BATCH_PATTERN = re.compile(
    r"\[(?P<current>\d+)/(?P<total>\d+)] "
    r"(?:本批|章节包\d+|风险原文包\d+)新增"
    r"(?P<new>\d+)条，累计(?P<raw>\d+)条风险候选"
)
_SCOPING_PATTERN = re.compile(r"保留(?P<included>\d+)条.+排除(?P<excluded>\d+)条")
_HUMAN_SCOPE_PATTERN = re.compile(r"保留及新增(?P<selected>\d+)条")
_ROLE_ASSIGNMENT_PATTERN = re.compile(r"开始分析(?P<count>\d+)类风险")
_FINAL_ACCEPTED_PATTERN = re.compile(r"(?:接受|形成|进入人工复核，?)(?P<count>\d+)条")
_AGENT_OVERALL_RANGES = {
    "discovery": (30, 53),
    "decision": (83, 88),
}
_PARALLEL_AGENT_ROLES = ("finance", "legal", "business", "market")
_PARALLEL_AGENT_OVERALL_RANGE = (58, 83)


def _parallel_agent_overall(progress_by_role: dict[str, int]) -> int:
    """按四条并行分支的平均完成度映射专业 Agent 总进度。"""
    start, end = _PARALLEL_AGENT_OVERALL_RANGE
    total = sum(max(0, min(progress_by_role.get(role, 0), 100)) for role in _PARALLEL_AGENT_ROLES)
    fraction = total / (len(_PARALLEL_AGENT_ROLES) * 100)
    return round(start + (end - start) * fraction)


class ProgressEvent(Protocol):
    """构建只读进度快照所需的最小事件接口。"""

    @property
    def stage(self) -> str: ...

    @property
    def detail(self) -> str: ...

    @property
    def occurred_at(self) -> datetime: ...


AnalysisStatus = Literal[
    "pending",
    "running",
    "awaiting_recovery",
    "awaiting_review",
    "completed",
    "paused",
    "failed",
]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class AnalysisProgressState:
    """纯内存进度归约器；计算状态时不向浏览器发送任何中间UI更新。"""

    def __init__(self) -> None:
        self._events: list[str] = []
        self._max_percent = 0
        self._overall_text = "等待开始"
        self._current_activity = "等待分析事件"
        self._started_at: datetime | None = None
        self._last_event_at: datetime | None = None
        self._review_pause_started_at: datetime | None = None
        self._review_wait_seconds = 0
        self._background_tasks: dict[str, dict[str, object]] = {
            "embedding": {
                "state": "pending",
                "progress": 0,
                "message": "等待PDF解析完成后启动。",
            },
        }
        self._agents: dict[str, dict[str, object]] = {
            role: {
                "state": "pending",
                "message": "等待编排器分发任务。",
                "progress": 0,
                "progress_text": "尚未开始",
                "sequence_current": 0,
            }
            for role in _AGENT_SPECS
        }
        self._outputs = {role: "" for role in _AGENT_SPECS}
        self._discovery_outputs: dict[str, str] = {}
        self._role_batch_outputs: dict[str, dict[str, str]] = {
            role: {} for role in _AGENT_SPECS if role != "discovery"
        }
        self._risk_titles: dict[str, dict[str, str]] = {role: {} for role in _AGENT_SPECS}
        self._discovery_failures: list[str] = []
        self._funnel: dict[str, object] = {
            "document_batches": 0,
            "raw_candidates": 0,
            "consolidated_candidates": 0,
            "scoped_in": 0,
            "scoped_out": 0,
            "human_selected": 0,
            "role_assignments": {role: 0 for role in _PARALLEL_AGENT_ROLES},
            "final_accepted": 0,
        }

    def _set_overall(self, percent: int, text: str) -> None:
        bounded = max(0, min(percent, 100))
        if bounded < self._max_percent:
            return
        self._max_percent = bounded
        self._overall_text = text

    def _set_agent_progress(self, role: str, percent: int, text: str) -> None:
        agent = self._agents[role]
        bounded = max(0, min(percent, 100))
        current = _as_int(agent["progress"])
        if bounded < current:
            return
        agent["progress"] = bounded
        agent["progress_text"] = text

    def _append_event(
        self,
        label: str,
        detail: str,
        occurred_at: datetime,
    ) -> None:
        timestamp = occurred_at.strftime("%H:%M:%S")
        self._events.append(f"[{timestamp}] {label:<8} | {detail}")

    def _update_agent(self, role: str, action: str, detail: str) -> None:
        if role not in self._agents:
            return
        agent = self._agents[role]
        if action == "started":
            agent["state"] = "running"
            self._set_agent_progress(role, 5, "正在接收任务")
        elif action == "search_planned":
            agent["state"] = "running"
            self._set_agent_progress(role, 25, "Agent 已生成语义搜索问题")
        elif action == "search_fallback":
            agent["state"] = "running"
            self._set_agent_progress(role, 25, "固定查询兜底运行中")
        elif action == "retrieval_planning":
            agent["state"] = "running"
            self._set_agent_progress(
                role,
                12,
                "正在生成公司特异的原文检索问题",
            )
        elif action == "retrieval_plan_fallback":
            agent["state"] = "running"
            self._set_agent_progress(
                role,
                15,
                "使用文档目录与样本构造检索问题",
            )
        elif action == "retrieval_ready":
            agent["state"] = "running"
            self._set_agent_progress(
                role,
                22,
                "向量命中原文与章节覆盖样本已就绪",
            )
        elif action == "prompt":
            agent["state"] = "running"
            match = _PROGRESS_PATTERN.search(detail)
            if match is not None and role == "discovery":
                current = int(match.group("current"))
                total = max(1, int(match.group("total")))
                percent = min(90, max(15, round(current / total * 90)))
                progress_text = f"风险原文包 {current}/{total}"
            else:
                percent = 30
                progress_text = "正在检索证据并请求模型"
            self._set_agent_progress(role, percent, progress_text)
            if self._outputs[role]:
                self._outputs[role] += "\n\n--- 新的风险任务 ---\n"
        elif action == "model_requested":
            agent["state"] = "running"
            self._set_agent_progress(role, 40, "模型请求已发送")
        elif action == "reasoning":
            agent["state"] = "running"
            self._set_agent_progress(role, 45, "模型正在推理")
        elif action == "delta":
            agent["state"] = "running"
            self._set_agent_progress(role, 50, "模型正在流式生成")
            delta_match = _DISCOVERY_DELTA_PATTERN.match(detail)
            if delta_match is None:
                self._outputs[role] += detail
            elif role == "discovery":
                stream_key = f"{delta_match.group('kind')}:{delta_match.group('current')}"
                self._discovery_outputs[stream_key] = self._discovery_outputs.get(
                    stream_key, ""
                ) + delta_match.group("delta")
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _stream_key, content in sorted(
                        self._discovery_outputs.items(),
                        key=lambda item: (
                            0 if item[0].startswith("batch:") else 1,
                            int(item[0].split(":", maxsplit=1)[1]),
                        ),
                    )
                )
            else:
                stream_key = f"{delta_match.group('kind')}:{delta_match.group('current')}"
                role_outputs = self._role_batch_outputs[role]
                role_outputs[stream_key] = role_outputs.get(stream_key, "") + delta_match.group(
                    "delta"
                )
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _key, content in sorted(
                        role_outputs.items(),
                        key=lambda item: int(item[0].split(":", maxsplit=1)[1]),
                    )
                )
        elif action == "structured":
            agent["state"] = "running"
            self._set_agent_progress(role, 55, "本批风险卡片已生成")
            structured_match = _DISCOVERY_DELTA_PATTERN.match(detail)
            if structured_match is None:
                self._outputs[role] = detail
            elif role == "discovery":
                stream_key = f"{structured_match.group('kind')}:{structured_match.group('current')}"
                self._discovery_outputs[stream_key] = structured_match.group("delta")
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _stream_key, content in sorted(
                        self._discovery_outputs.items(),
                        key=lambda item: (
                            0 if item[0].startswith("batch:") else 1,
                            int(item[0].split(":", maxsplit=1)[1]),
                        ),
                    )
                )
            else:
                stream_key = f"{structured_match.group('kind')}:{structured_match.group('current')}"
                role_outputs = self._role_batch_outputs[role]
                role_outputs[stream_key] = structured_match.group("delta")
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _key, content in sorted(
                        role_outputs.items(),
                        key=lambda item: int(item[0].split(":", maxsplit=1)[1]),
                    )
                )
        elif action == "progress":
            match = _PROGRESS_PATTERN.search(detail)
            percent = 50
            progress_text = "正在检索和分析证据"
            if match is not None:
                current = int(match.group("current"))
                total = max(1, int(match.group("total")))
                percent = min(95, max(10, round(current / total * 90)))
                progress_text = f"风险任务 {current}/{total}"
            self._set_agent_progress(role, percent, progress_text)
        elif action == "batch_completed":
            agent["state"] = "running"
            match = _PROGRESS_PATTERN.search(detail)
            percent = 50
            progress_text = "风险原文包分析完成"
            if match is not None:
                current = int(match.group("current"))
                total = max(1, int(match.group("total")))
                percent = min(92, max(15, round(current / total * 90)))
                progress_text = f"风险原文包 {current}/{total} · 已形成候选"
            self._set_agent_progress(role, percent, progress_text)
        elif action == "completed":
            agent["state"] = "complete"
            self._set_agent_progress(role, 100, "任务完成")
        elif action == "failed":
            agent["state"] = "failed"
            self._set_agent_progress(role, 100, "已降级继续分析")
        elif action == "degraded":
            agent["state"] = "degraded"
            self._set_agent_progress(role, 100, "扫描完成，部分批次失败")
        elif action == "raw_completed":
            agent["state"] = "running"
            self._set_agent_progress(role, 92, "批次扫描完成，正在全局合并")
        elif action == "consolidated":
            agent["state"] = "running"
            self._set_agent_progress(role, 98, "全局语义合并完成")
        elif action == "risk_catalog":
            try:
                values = json.loads(detail)
            except json.JSONDecodeError:
                values = {}
            if isinstance(values, dict):
                self._risk_titles[role].update(
                    {
                        str(risk_type): str(title)
                        for risk_type, title in values.items()
                        if str(risk_type).strip() and str(title).strip()
                    }
                )
        elif action == "batch_failed":
            agent["state"] = "running"
            self._discovery_failures.append(detail)
        elif action == "skipped":
            agent["state"] = "skipped"
            self._set_agent_progress(
                role,
                100,
                ("跳过全文动态发现" if role == "discovery" else "本次未分配风险"),
            )
        if action not in {"delta", "structured"}:
            agent["message"] = detail
        elif action == "structured":
            agent["message"] = "本批结构化风险卡片已更新。"

    def _update_funnel(self, stage: str, detail: str) -> None:
        raw_match = _DISCOVERY_RAW_PATTERN.search(detail)
        if raw_match is not None:
            batch_count = (
                raw_match.group("full_batches") or raw_match.group("retrieved_batches") or "0"
            )
            self._funnel["document_batches"] = int(batch_count)
            self._funnel["raw_candidates"] = int(raw_match.group("raw"))

        batch_match = _DISCOVERY_BATCH_PATTERN.search(detail)
        if batch_match is not None:
            self._funnel["document_batches"] = int(batch_match.group("total"))
            self._funnel["raw_candidates"] = int(batch_match.group("raw"))

        consolidated_match = _DISCOVERY_CONSOLIDATED_PATTERN.search(detail)
        if consolidated_match is not None:
            self._funnel["raw_candidates"] = int(consolidated_match.group("raw"))
            self._funnel["consolidated_candidates"] = int(consolidated_match.group("consolidated"))

        if stage == "risk_scoping_completed":
            scope_match = _SCOPING_PATTERN.search(detail)
            if scope_match is not None:
                self._funnel["scoped_in"] = int(scope_match.group("included"))
                self._funnel["scoped_out"] = int(scope_match.group("excluded"))

        if stage == "risk_scope_review_completed":
            human_match = _HUMAN_SCOPE_PATTERN.search(detail)
            if human_match is not None:
                self._funnel["human_selected"] = int(human_match.group("selected"))

        parts = stage.split(":")
        if (
            len(parts) == 3
            and parts[0] == "agent"
            and parts[1] in _PARALLEL_AGENT_ROLES
            and parts[2] == "started"
        ):
            role_match = _ROLE_ASSIGNMENT_PATTERN.search(detail)
            if role_match is not None:
                assignments = self._funnel["role_assignments"]
                if isinstance(assignments, dict):
                    assignments[parts[1]] = int(role_match.group("count"))

        if stage == "agent:decision:completed":
            decision_match = _FINAL_ACCEPTED_PATTERN.search(detail)
            if decision_match is not None:
                self._funnel["final_accepted"] = int(decision_match.group("count"))

    def emit(self, stage: str, detail: str, *, occurred_at: datetime) -> None:
        """把单个事件折叠进当前状态，不产生页面增量指令。"""
        if self._started_at is None:
            self._started_at = occurred_at
        self._last_event_at = occurred_at
        if stage in {"risk_scope_review_requested", "llm_recovery_requested"}:
            self._review_pause_started_at = occurred_at
        elif (
            stage
            in {
                "risk_scope_review_completed",
                "llm_recovery_retry_selected",
                "llm_recovery_bm25_selected",
            }
            and self._review_pause_started_at is not None
        ):
            self._review_wait_seconds += max(
                0,
                round((occurred_at - self._review_pause_started_at).total_seconds()),
            )
            self._review_pause_started_at = None
        parts = stage.split(":")
        if len(parts) == 3 and parts[0] == "agent":
            role, action = parts[1], parts[2]
            progress_match = _PROGRESS_PATTERN.search(detail)
            agent = self._agents.get(role)
            if (
                progress_match is not None
                and agent is not None
                and action in {"prompt", "progress", "batch_completed"}
            ):
                current = int(progress_match.group("current"))
                prior = _as_int(agent.get("sequence_current", 0))
                if current < prior:
                    return
                agent["sequence_current"] = current
        self._update_funnel(stage, detail)
        if stage == "embedding_join_wait":
            self._current_activity = f"证据检索准备 · {detail}"
            self._append_event("证据检索准备", detail, occurred_at)
            return
        if stage == "embedding_join_completed":
            self._current_activity = f"证据检索准备 · {detail}"
            self._append_event("证据检索准备", detail, occurred_at)
            return

        if stage.startswith("embedding_") or stage == "vector_store_indexed":
            embedding = self._background_tasks["embedding"]
            if stage in {"embedding_started", "embedding_cache_hit"}:
                embedding["state"] = "running"
                embedding["progress"] = max(_as_int(embedding["progress"]), 5)
            elif stage == "embedding_progress":
                embedding["state"] = "running"
                match = _PROGRESS_PATTERN.search(detail)
                if match is not None:
                    current = int(match.group("current"))
                    total = max(1, int(match.group("total")))
                    embedding["progress"] = max(
                        _as_int(embedding["progress"]),
                        round(current / total * 95),
                    )
            elif stage in {"embedding_completed", "vector_store_indexed"}:
                embedding["state"] = "complete"
                embedding["progress"] = 100
            elif stage == "embedding_fallback":
                embedding["state"] = "degraded"
                embedding["progress"] = 100
            embedding["message"] = detail
            if stage not in {"embedding_progress"}:
                self._append_event(
                    _STAGE_LABELS.get(stage, "向量索引"),
                    detail,
                    occurred_at,
                )
            return
        if len(parts) == 3 and parts[0] == "agent":
            role, action = parts[1], parts[2]
            if action == "risk_catalog":
                self._update_agent(role, action, detail)
                return
            label = _AGENT_SPECS.get(role, (role, ""))[0]
            activity_detail = (
                "模型正在流式生成"
                if action == "delta"
                else ("本批结构化风险卡片已更新" if action == "structured" else detail)
            )
            self._current_activity = f"{label} · {activity_detail}"
            self._update_agent(role, action, detail)
            is_dynamic_scan = (
                role == "discovery"
                and action in {"prompt", "batch_completed"}
                and _PROGRESS_PATTERN.search(detail) is not None
            )
            if action not in {"delta", "structured"} and not is_dynamic_scan:
                self._append_event(label, detail, occurred_at)
            progress_range = _AGENT_OVERALL_RANGES.get(role)
            if role in _PARALLEL_AGENT_ROLES:
                percent = _parallel_agent_overall(
                    {
                        item_role: _as_int(self._agents[item_role]["progress"])
                        for item_role in _PARALLEL_AGENT_ROLES
                    }
                )
                step = _AGENT_STEP_LABELS.get(action, action)
                self._set_overall(percent, f"{label} · {step}")
            elif progress_range is not None:
                start, end = progress_range
                if action == "started":
                    percent = start
                elif action in {"completed", "failed", "degraded", "skipped"}:
                    percent = end
                else:
                    match = _PROGRESS_PATTERN.search(detail)
                    if match is not None:
                        fraction = int(match.group("current")) / max(
                            1,
                            int(match.group("total")),
                        )
                        percent = round(start + (end - start) * fraction)
                    else:
                        percent = self._max_percent
                step = _AGENT_STEP_LABELS.get(action, action)
                if action == "progress":
                    step_match = _PROGRESS_PATTERN.search(detail)
                    if step_match is not None:
                        step = f"风险任务 {step_match.group('current')}/{step_match.group('total')}"
                self._set_overall(percent, f"{label} · {step}")
        elif stage in {"parsing_progress", "features_progress"}:
            fallback_label = "PDF解析" if stage == "parsing_progress" else "上市前特征"
            label = _STAGE_LABELS.get(stage, fallback_label)
            match = _PROGRESS_PATTERN.search(detail)
            if match is not None:
                fraction = int(match.group("current")) / max(1, int(match.group("total")))
                if stage == "parsing_progress":
                    self._current_activity = (
                        f"{label} · {match.group('current')}/{match.group('total')}页"
                    )
                    percent = round(12 + 13 * fraction)
                else:
                    progress_detail = detail[match.end() :].strip()
                    self._current_activity = f"{label} · {progress_detail}"
                    percent = round(93 + fraction)
                self._set_overall(percent, self._current_activity)
        else:
            label = _STAGE_LABELS.get(stage, stage)
            self._current_activity = f"{label} · {detail}"
            self._append_event(label, detail, occurred_at)

        overall_progress = {
            "upload_started": 3,
            "upload_completed": 8,
            "parsing_started": 12,
            "parsing_completed": 25,
            "agents_started": 30,
            "risk_scoping_started": 54,
            "risk_scoping_completed": 58,
            "risk_scoping_fallback": 58,
            "risk_scope_review_requested": 58,
            "risk_scope_review_completed": 58,
            "llm_recovery_requested": 45,
            "llm_recovery_retry_selected": 45,
            "llm_recovery_bm25_selected": 45,
            "agents_completed": 88,
            "evidence_started": 92,
            "features_started": 93,
            "features_completed": 94,
            "features_failed": 94,
            "prediction_started": 96,
            "prediction_completed": 99,
            "prediction_failed": 99,
            "completed": 100,
        }.get(stage)
        if overall_progress is not None:
            self._set_overall(overall_progress, detail)

    def complete(self, detail: str) -> None:
        self._set_overall(100, detail)
        self._current_activity = f"分析完成 · {detail}"

    def fail(self, detail: str, *, occurred_at: datetime) -> None:
        self._append_event("执行失败", detail, occurred_at)
        self._set_overall(100, "分析失败，请查看错误信息")
        self._current_activity = f"执行失败 · {detail}"
        for role, agent in self._agents.items():
            if agent["state"] != "running":
                continue
            agent["state"] = "failed"
            agent["message"] = detail
            self._set_agent_progress(role, 100, "执行失败")

    def snapshot(
        self,
        *,
        now: datetime | None = None,
        status: AnalysisStatus = "running",
    ) -> dict[str, Any]:
        assignments = self._funnel["role_assignments"]
        effective_now = now or datetime.now().astimezone()
        end = self._last_event_at if status in {"completed", "failed"} else effective_now
        elapsed_seconds = 0
        review_wait_seconds = self._review_wait_seconds
        if self._review_pause_started_at is not None and end is not None:
            review_wait_seconds += max(
                0,
                round((end - self._review_pause_started_at).total_seconds()),
            )
        if self._started_at is not None and end is not None:
            elapsed_seconds = max(
                0,
                round((end - self._started_at).total_seconds()) - review_wait_seconds,
            )
        return {
            "events": list(self._events),
            "max_percent": self._max_percent,
            "overall_text": self._overall_text,
            "current_activity": self._current_activity,
            "agents": {role: dict(agent) for role, agent in self._agents.items()},
            "outputs": dict(self._outputs),
            "risk_titles": {role: dict(titles) for role, titles in self._risk_titles.items()},
            "discovery_failures": list(self._discovery_failures),
            "background_tasks": {key: dict(value) for key, value in self._background_tasks.items()},
            "elapsed_seconds": elapsed_seconds,
            "review_wait_seconds": review_wait_seconds,
            "funnel": {
                **self._funnel,
                "role_assignments": (dict(assignments) if isinstance(assignments, dict) else {}),
            },
        }


def build_analysis_progress_snapshot(
    events: Iterable[ProgressEvent],
    *,
    status: AnalysisStatus,
    error: str | None = None,
    now: datetime | None = None,
    include_outputs: bool = True,
) -> dict[str, Any]:
    """先在内存中归约完整历史，再由页面一次性渲染最终状态。"""
    state = AnalysisProgressState()
    last_occurred_at: datetime | None = None
    for event in events:
        last_occurred_at = event.occurred_at
        state.emit(event.stage, event.detail, occurred_at=event.occurred_at)
    if status == "completed":
        state.complete("分析完成，候选风险已经进入人工复核队列。")
    elif status == "failed":
        state.fail(
            error or "未知错误",
            occurred_at=last_occurred_at or datetime.now().astimezone(),
        )
    elif status == "paused":
        state.emit(
            "model_budget_paused",
            error or "模型预算达到上限，分析已安全暂停。",
            occurred_at=last_occurred_at or datetime.now().astimezone(),
        )
    snapshot = state.snapshot(now=now, status=status)
    if not include_outputs:
        snapshot.pop("outputs", None)
    return snapshot


__all__ = [
    "AnalysisProgressState",
    "build_analysis_progress_snapshot",
]
