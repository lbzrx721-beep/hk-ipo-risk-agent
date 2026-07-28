"""Streamlit实时分析运行台。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import json
import re
from typing import Any, Literal, Protocol

import streamlit as st

from ipo_risk.ui import theme
from ipo_risk.ui.agent_output import render_agent_model_output


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
    "vector_persistence_started": "知识库后台同步",
    "vector_persistence_completed": "知识库后台同步",
    "vector_persistence_failed": "知识库同步失败",
    "risk_scoping_started": "公司画像与风险范围",
    "risk_scoping_completed": "公司画像与风险范围",
    "risk_scoping_fallback": "风险范围识别降级",
    "risk_scope_review_requested": "人工确认风险范围",
    "risk_scope_review_completed": "人工确认风险范围",
    "agents_completed": "风险整合",
    "calculation_skipped": "数值复算",
    "evidence_started": "证据报告",
    "features_completed": "上市前特征",
    "features_failed": "上市前特征",
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
_SCOPING_PATTERN = re.compile(
    r"保留(?P<included>\d+)条.+排除(?P<excluded>\d+)条"
)
_HUMAN_SCOPE_PATTERN = re.compile(r"保留及新增(?P<selected>\d+)条")
_ROLE_ASSIGNMENT_PATTERN = re.compile(r"开始分析(?P<count>\d+)类风险")
_FINAL_ACCEPTED_PATTERN = re.compile(
    r"(?:接受|形成|进入人工复核，?)(?P<count>\d+)条"
)
_AGENT_OVERALL_RANGES = {
    "discovery": (30, 53),
    "decision": (83, 88),
}
_PARALLEL_AGENT_ROLES = ("finance", "legal", "business", "market")
_PARALLEL_AGENT_OVERALL_RANGE = (58, 83)


def _parallel_agent_overall(progress_by_role: dict[str, int]) -> int:
    """按四条并行分支的平均完成度映射专业 Agent 总进度。"""
    start, end = _PARALLEL_AGENT_OVERALL_RANGE
    total = sum(
        max(0, min(progress_by_role.get(role, 0), 100))
        for role in _PARALLEL_AGENT_ROLES
    )
    fraction = total / (len(_PARALLEL_AGENT_ROLES) * 100)
    return round(start + (end - start) * fraction)


class AnalysisProgressView:
    """把分析流水线事件实时投射为Agent卡片和活动流。"""

    def __init__(self) -> None:
        theme.section_label("实时分析控制台")
        with st.container(border=True):
            progress_header = st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            )
            progress_header.markdown("**端到端分析进度**")
            progress_header.badge(
                "LIVE",
                color="blue",
                icon=":material/radio_button_checked:",
            )
            self._overall = st.progress(0, text="等待开始")
            st.caption("所有状态均来自真实流水线事件，进度只前进、不回跳。")
        self._max_percent: int = 0  # 单调递增守卫：记录已达到的最大百分比
        self._cards: dict[str, dict[str, Any]] = {}

        execution_column, activity_column = st.columns([1.85, 1], gap="large")
        with execution_column:
            st.subheader("Agent协同矩阵")
            st.caption("上层看状态，下层切换角色查看该 Agent 的实时结构化输出。")
            agent_roles = list(_AGENT_SPECS)
            for start in range(0, len(agent_roles), 3):
                row_roles = agent_roles[start : start + 3]
                row = st.columns(3)
                for column, role in zip(row, row_roles, strict=False):
                    name, icon = _AGENT_SPECS[role]
                    card = column.container(border=True, height=165)
                    card.subheader(name)
                    badge = card.empty()
                    badge.badge(
                        "等待任务",
                        color="gray",
                        icon=":material/schedule:",
                    )
                    progress = card.progress(0, text="尚未开始")
                    message = card.empty()
                    message.caption("等待编排器分发任务。")
                    self._cards[role] = {
                        "badge": badge,
                        "progress": progress,
                        "message": message,
                        "output_text": "",
                        "last_message": "",
                        "max_progress": 0,
                        "state": "pending",
                    }

            output_tabs = st.tabs(
                [f"{name}" for name, _icon in _AGENT_SPECS.values()]
            )
            for tab, role in zip(output_tabs, _AGENT_SPECS, strict=True):
                name, icon = _AGENT_SPECS[role]
                with tab:
                    with st.container(border=True):
                        output_heading = st.container(horizontal=True)
                        output_heading.badge(name, color="blue", icon=icon)
                        output_heading.caption("实时模型输出 · 仅展示结构化响应")
                        output = st.empty()
                        output.code(
                            "等待模型输出…",
                            language="json",
                            wrap_lines=True,
                            height=230,
                        )
                        self._cards[role]["output"] = output

        with activity_column:
            stream = st.container(border=True, height=710)
            stream.subheader("实时事件流")
            stream.caption("文件、检索、模型、裁决和报告事件按时间连续记录。")
            self._stream = stream.empty()
        self._events: list[str] = []
        self._render_stream()

    def _set_overall(self, percent: int, text: str) -> None:
        """更新总进度条，保证百分比严格单调递增（永不回跳）。"""
        self._max_percent = max(self._max_percent, percent)
        self._overall.progress(self._max_percent, text=text)

    def _render_stream(self) -> None:
        content = "\n".join(self._events) or "[等待] 尚未收到分析事件"
        self._stream.code(
            content,
            language=None,
            wrap_lines=True,
            height=610,
        )

    def _append_event(
        self,
        label: str,
        detail: str,
        *,
        occurred_at: datetime | None = None,
    ) -> None:
        timestamp = (occurred_at or datetime.now().astimezone()).strftime("%H:%M:%S")
        self._events.append(f"[{timestamp}] {label:<8} | {detail}")
        self._render_stream()

    def _set_card_progress(self, card: dict[str, Any], percent: int, text: str) -> None:
        """更新卡片内进度条，保证百分比严格单调递增。"""
        card["max_progress"] = max(card["max_progress"], percent)
        card["progress"].progress(card["max_progress"], text=text)

    def _update_agent(self, role: str, action: str, detail: str) -> None:
        card = self._cards.get(role)
        if card is None:
            return
        if action == "started":
            card["state"] = "running"
            card["badge"].badge(
                "分析中",
                color="blue",
                icon=":material/progress_activity:",
            )
            self._set_card_progress(card, 5, "正在接收任务")
        elif action == "search_planned":
            card["state"] = "running"
            card["badge"].badge(
                "语义检索",
                color="blue",
                icon=":material/manage_search:",
            )
            self._set_card_progress(card, 25, "Agent 已生成语义搜索问题")
        elif action == "search_fallback":
            card["state"] = "running"
            card["badge"].badge(
                "检索兜底",
                color="orange",
                icon=":material/sync_problem:",
            )
            self._set_card_progress(card, 25, "固定查询兜底运行中")
        elif action == "retrieval_planning":
            card["state"] = "running"
            card["badge"].badge(
                "规划检索",
                color="blue",
                icon=":material/route:",
            )
            self._set_card_progress(card, 12, "正在生成公司特异的原文检索问题")
        elif action == "retrieval_plan_fallback":
            card["state"] = "running"
            card["badge"].badge(
                "规划兜底",
                color="orange",
                icon=":material/sync_problem:",
            )
            self._set_card_progress(card, 15, "使用文档目录与样本构造检索问题")
        elif action == "retrieval_ready":
            card["state"] = "running"
            card["badge"].badge(
                "原文就绪",
                color="blue",
                icon=":material/library_books:",
            )
            self._set_card_progress(card, 22, "向量命中原文与章节覆盖样本已就绪")
        elif action == "prompt":
            card["state"] = "running"
            card["badge"].badge(
                "生成中",
                color="blue",
                icon=":material/stream:",
            )
            match = _PROGRESS_PATTERN.search(detail)
            if match is not None and role == "discovery":
                current = int(match.group("current"))
                total = max(1, int(match.group("total")))
                percent = min(90, max(15, round(current / total * 90)))
                progress_text = f"风险原文包 {current}/{total}"
            else:
                percent = 30
                progress_text = "正在检索证据并请求模型"
            self._set_card_progress(card, percent, progress_text)
            if card["output_text"]:
                card["output_text"] += "\n\n--- 新的风险任务 ---\n"
                card["output"].code(
                    card["output_text"], language="json", wrap_lines=True, height=145
                )
            else:
                card["output"].code(
                    "模型已接收任务，等待首段输出…",
                    language="json",
                    wrap_lines=True,
                    height=145,
                )
        elif action == "delta":
            card["state"] = "running"
            self._set_card_progress(card, 50, "模型正在流式生成")
            card["output_text"] += detail
            card["output"].code(
                card["output_text"],
                language="json",
                wrap_lines=True,
                height=145,
            )
        elif action == "structured":
            card["state"] = "running"
            structured_match = _DISCOVERY_DELTA_PATTERN.match(detail)
            structured_output = (
                structured_match.group("delta")
                if structured_match is not None
                else detail
            )
            if structured_output not in card["output_text"]:
                if card["output_text"]:
                    card["output_text"] += _OUTPUT_SEPARATOR
                card["output_text"] += structured_output
            self._set_card_progress(card, 55, "本批风险卡片已生成")
            card["output"].code(
                card["output_text"],
                language="json",
                wrap_lines=True,
                height=145,
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
            self._set_card_progress(card, percent, progress_text)
        elif action == "batch_completed":
            match = _PROGRESS_PATTERN.search(detail)
            percent = 50
            progress_text = "风险原文包分析完成"
            if match is not None:
                current = int(match.group("current"))
                total = max(1, int(match.group("total")))
                percent = min(92, max(15, round(current / total * 90)))
                progress_text = f"风险原文包 {current}/{total} · 已形成候选"
            self._set_card_progress(card, percent, progress_text)
        elif action == "completed":
            card["state"] = "complete"
            card["badge"].badge(
                "已完成",
                color="green",
                icon=":material/check_circle:",
            )
            self._set_card_progress(card, 100, "任务完成")
        elif action == "failed":
            card["state"] = "failed"
            card["badge"].badge(
                "失败后降级",
                color="red",
                icon=":material/error:",
            )
            self._set_card_progress(card, 100, "已降级继续分析")
        elif action == "degraded":
            card["state"] = "degraded"
            card["badge"].badge(
                "部分批次降级",
                color="orange",
                icon=":material/warning:",
            )
            self._set_card_progress(card, 100, "扫描完成，部分批次失败")
        elif action == "skipped":
            card["state"] = "skipped"
            card["badge"].badge(
                "未启用",
                color="gray",
                icon=":material/pause_circle:",
            )
            self._set_card_progress(card, 100, "跳过全文动态发现")
        if action not in {"delta", "structured"}:
            card["message"].caption(detail)
            card["last_message"] = detail
        elif action == "structured":
            card["message"].caption("本批结构化风险卡片已更新。")
            card["last_message"] = "本批结构化风险卡片已更新。"

    def emit(
        self,
        stage: str,
        detail: str,
        *,
        occurred_at: datetime | None = None,
    ) -> None:
        """接收一次流水线回调并立即刷新页面。"""
        # 向量化是独立后台任务，不推动主进度条，也不刷入高频事件流。
        if stage == "embedding_progress":
            return

        parts = stage.split(":")
        if len(parts) == 3 and parts[0] == "agent":
            role, action = parts[1], parts[2]
            self._update_agent(role, action, detail)
            label = _AGENT_SPECS.get(role, (role, ""))[0]
            is_dynamic_scan = (
                role == "discovery"
                and action in {"prompt", "batch_completed"}
                and _PROGRESS_PATTERN.search(detail) is not None
            )
            if action not in {"delta", "structured"} and not is_dynamic_scan:
                self._append_event(label, detail, occurred_at=occurred_at)
            progress_range = _AGENT_OVERALL_RANGES.get(role)
            if role in _PARALLEL_AGENT_ROLES:
                percent = _parallel_agent_overall(
                    {
                        item_role: int(self._cards[item_role]["max_progress"])
                        for item_role in _PARALLEL_AGENT_ROLES
                    }
                )
                step = _AGENT_STEP_LABELS.get(action, action)
                self._set_overall(percent, text=f"{label} · {step}")
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
                        # 无 [current/total] 的中间事件（search_planned、prompt 等）
                        # 不改变百分比，仅更新文字标签，避免回跳。
                        percent = self._max_percent
                # 总进度条只标注"哪个 Agent · 哪一步"，不展示模型输出内容
                step = _AGENT_STEP_LABELS.get(action, action)
                if action == "progress":
                    step_match = _PROGRESS_PATTERN.search(detail)
                    if step_match is not None:
                        step = (
                            f"风险任务 {step_match.group('current')}"
                            f"/{step_match.group('total')}"
                        )
                self._set_overall(percent, text=f"{label} · {step}")
        elif stage == "parsing_progress":
            label = _STAGE_LABELS.get(stage, "PDF解析")
            match = _PROGRESS_PATTERN.search(detail)
            if match is not None:
                fraction = int(match.group("current")) / max(
                    1, int(match.group("total"))
                )
                self._set_overall(
                    round(12 + 13 * fraction),
                    text=f"{label} · {match.group('current')}/{match.group('total')}页",
                )
        else:
            label = _STAGE_LABELS.get(stage, stage)
            self._append_event(label, detail, occurred_at=occurred_at)

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
            "agents_completed": 88,
            "evidence_started": 92,
            "completed": 100,
        }.get(stage)
        if overall_progress is not None:
            self._set_overall(overall_progress, text=detail)

    def snapshot(self) -> dict[str, Any]:
        """导出当前分析状态快照，供页面切换后只读回顾。"""
        return {
            "events": list(self._events),
            "max_percent": self._max_percent,
            "agents": {
                role: {
                    "state": card["state"],
                    "message": card.get("last_message", ""),
                }
                for role, card in self._cards.items()
            },
            "outputs": {
                role: card.get("output_text", "")
                for role, card in self._cards.items()
            },
        }

    def complete(self, detail: str) -> None:
        """保证成功结束时进度条处于完成状态。"""
        self._set_overall(100, text=detail)

    def fail(self, detail: str) -> None:
        """把当前运行中的Agent和总进度切换为错误状态。"""
        self._append_event("执行失败", detail)
        self._set_overall(100, text="分析失败，请查看错误信息")
        for card in self._cards.values():
            if card["state"] != "running":
                continue
            card["badge"].badge(
                "执行失败",
                color="red",
                icon=":material/error:",
            )
            card["message"].caption(detail)


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
        self._background_tasks: dict[str, dict[str, object]] = {
            "embedding": {
                "state": "pending",
                "progress": 0,
                "message": "等待PDF解析完成后启动。",
            },
            "persistence": {
                "state": "pending",
                "progress": 0,
                "message": "内存索引完成后异步写入Chroma。",
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
        self._risk_titles: dict[str, dict[str, str]] = {
            role: {} for role in _AGENT_SPECS
        }
        self._discovery_failures: list[str] = []
        self._funnel: dict[str, object] = {
            "document_batches": 0,
            "raw_candidates": 0,
            "consolidated_candidates": 0,
            "scoped_in": 0,
            "scoped_out": 0,
            "human_selected": 0,
            "role_assignments": {
                role: 0 for role in _PARALLEL_AGENT_ROLES
            },
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
                stream_key = (
                    f"{delta_match.group('kind')}:"
                    f"{delta_match.group('current')}"
                )
                self._discovery_outputs[stream_key] = (
                    self._discovery_outputs.get(stream_key, "")
                    + delta_match.group("delta")
                )
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
                stream_key = (
                    f"{delta_match.group('kind')}:"
                    f"{delta_match.group('current')}"
                )
                role_outputs = self._role_batch_outputs[role]
                role_outputs[stream_key] = (
                    role_outputs.get(stream_key, "")
                    + delta_match.group("delta")
                )
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _key, content in sorted(
                        role_outputs.items(),
                        key=lambda item: int(
                            item[0].split(":", maxsplit=1)[1]
                        ),
                    )
                )
        elif action == "structured":
            agent["state"] = "running"
            self._set_agent_progress(role, 55, "本批风险卡片已生成")
            structured_match = _DISCOVERY_DELTA_PATTERN.match(detail)
            if structured_match is None:
                self._outputs[role] = detail
            elif role == "discovery":
                stream_key = (
                    f"{structured_match.group('kind')}:"
                    f"{structured_match.group('current')}"
                )
                self._discovery_outputs[stream_key] = (
                    structured_match.group("delta")
                )
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
                stream_key = (
                    f"{structured_match.group('kind')}:"
                    f"{structured_match.group('current')}"
                )
                role_outputs = self._role_batch_outputs[role]
                role_outputs[stream_key] = structured_match.group("delta")
                self._outputs[role] = _OUTPUT_SEPARATOR.join(
                    content
                    for _key, content in sorted(
                        role_outputs.items(),
                        key=lambda item: int(
                            item[0].split(":", maxsplit=1)[1]
                        ),
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
                (
                    "跳过全文动态发现"
                    if role == "discovery"
                    else "本次未分配风险"
                ),
            )
        if action not in {"delta", "structured"}:
            agent["message"] = detail
        elif action == "structured":
            agent["message"] = "本批结构化风险卡片已更新。"

    def _update_funnel(self, stage: str, detail: str) -> None:
        raw_match = _DISCOVERY_RAW_PATTERN.search(detail)
        if raw_match is not None:
            batch_count = (
                raw_match.group("full_batches")
                or raw_match.group("retrieved_batches")
                or "0"
            )
            self._funnel["document_batches"] = int(batch_count)
            self._funnel["raw_candidates"] = int(raw_match.group("raw"))

        batch_match = _DISCOVERY_BATCH_PATTERN.search(detail)
        if batch_match is not None:
            self._funnel["document_batches"] = int(batch_match.group("total"))
            self._funnel["raw_candidates"] = int(batch_match.group("raw"))

        consolidated_match = _DISCOVERY_CONSOLIDATED_PATTERN.search(detail)
        if consolidated_match is not None:
            self._funnel["raw_candidates"] = int(
                consolidated_match.group("raw")
            )
            self._funnel["consolidated_candidates"] = int(
                consolidated_match.group("consolidated")
            )

        if stage == "risk_scoping_completed":
            scope_match = _SCOPING_PATTERN.search(detail)
            if scope_match is not None:
                self._funnel["scoped_in"] = int(scope_match.group("included"))
                self._funnel["scoped_out"] = int(scope_match.group("excluded"))

        if stage == "risk_scope_review_completed":
            human_match = _HUMAN_SCOPE_PATTERN.search(detail)
            if human_match is not None:
                self._funnel["human_selected"] = int(
                    human_match.group("selected")
                )

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
                self._funnel["final_accepted"] = int(
                    decision_match.group("count")
                )

    def emit(self, stage: str, detail: str, *, occurred_at: datetime) -> None:
        """把单个事件折叠进当前状态，不产生Streamlit增量指令。"""
        if self._started_at is None:
            self._started_at = occurred_at
        self._last_event_at = occurred_at
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
        if stage.startswith("vector_persistence_"):
            persistence = self._background_tasks["persistence"]
            if stage == "vector_persistence_started":
                persistence["state"] = "running"
                persistence["progress"] = 15
            elif stage == "vector_persistence_completed":
                persistence["state"] = "complete"
                persistence["progress"] = 100
            elif stage == "vector_persistence_failed":
                persistence["state"] = "degraded"
                persistence["progress"] = 100
            persistence["message"] = detail
            self._append_event(
                _STAGE_LABELS.get(stage, "知识库后台同步"),
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
                else (
                    "本批结构化风险卡片已更新"
                    if action == "structured"
                    else detail
                )
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
                        item_role: _as_int(
                            self._agents[item_role]["progress"]
                        )
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
                        step = (
                            f"风险任务 {step_match.group('current')}"
                            f"/{step_match.group('total')}"
                        )
                self._set_overall(percent, f"{label} · {step}")
        elif stage == "parsing_progress":
            label = _STAGE_LABELS.get(stage, "PDF解析")
            match = _PROGRESS_PATTERN.search(detail)
            if match is not None:
                fraction = int(match.group("current")) / max(
                    1, int(match.group("total"))
                )
                self._current_activity = (
                    f"{label} · {match.group('current')}/{match.group('total')}页"
                )
                self._set_overall(
                    round(12 + 13 * fraction),
                    self._current_activity,
                )
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
            "agents_completed": 88,
            "evidence_started": 92,
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
        end = (
            self._last_event_at
            if status in {"completed", "failed"}
            else effective_now
        )
        elapsed_seconds = 0
        if self._started_at is not None and end is not None:
            elapsed_seconds = max(
                0,
                round((end - self._started_at).total_seconds()),
            )
        return {
            "events": list(self._events),
            "max_percent": self._max_percent,
            "overall_text": self._overall_text,
            "current_activity": self._current_activity,
            "agents": {
                role: dict(agent) for role, agent in self._agents.items()
            },
            "outputs": dict(self._outputs),
            "risk_titles": {
                role: dict(titles)
                for role, titles in self._risk_titles.items()
            },
            "discovery_failures": list(self._discovery_failures),
            "background_tasks": {
                key: dict(value)
                for key, value in self._background_tasks.items()
            },
            "elapsed_seconds": elapsed_seconds,
            "funnel": {
                **self._funnel,
                "role_assignments": (
                    dict(assignments) if isinstance(assignments, dict) else {}
                ),
            },
        }


def build_analysis_progress_snapshot(
    events: Iterable[ProgressEvent],
    *,
    status: AnalysisStatus,
    error: str | None = None,
    now: datetime | None = None,
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
    return state.snapshot(now=now, status=status)


def _agent_badge(
    state: str,
    *,
    role: str | None = None,
) -> tuple[str, theme.BadgeColor, str]:
    if state == "complete":
        return "已完成", "green", ":material/check_circle:"
    if state == "failed":
        return "执行失败", "red", ":material/error:"
    if state == "degraded":
        return "部分批次降级", "orange", ":material/warning:"
    if state == "skipped":
        return (
            ("未启用" if role == "discovery" else "本次无任务"),
            "gray",
            ":material/pause_circle:",
        )
    if state == "running":
        return "分析中", "blue", ":material/progress_activity:"
    return "等待任务", "gray", ":material/schedule:"


def _format_duration(seconds: int) -> str:
    seconds = max(0, seconds)
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分"
    if minutes:
        return f"{minutes}分{remaining_seconds:02d}秒"
    return f"{remaining_seconds}秒"


def render_analysis_progress_snapshot(
    snapshot: dict[str, object],
    *,
    status: AnalysisStatus,
) -> None:
    """一次性绘制当前快照，避免把0→当前值的历史更新发送给浏览器。"""
    st.html(
        """
        <style>
        .ipo-running-badge {
          display: inline-flex; align-items: center; gap: .45rem;
          padding: .22rem .58rem; border-radius: .45rem;
          color: #38bdf8; background: rgba(14,165,233,.14);
          font-size: .82rem; font-weight: 650;
        }
        .ipo-running-ring {
          width: .78rem; height: .78rem; border-radius: 50%;
          border: 2px solid rgba(56,189,248,.28);
          border-top-color: #38bdf8;
          animation: ipo-spin .8s linear infinite;
        }
        @keyframes ipo-spin { to { transform: rotate(360deg); } }
        </style>
        """
    )
    theme.section_label("实时分析控制台")
    with st.container(border=True):
        progress_header = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        progress_header.markdown("**端到端分析进度**")
        if status == "completed":
            progress_header.badge(
                "DONE", color="green", icon=":material/check_circle:"
            )
        elif status == "failed":
            progress_header.badge("FAILED", color="red", icon=":material/error:")
        elif status == "paused":
            progress_header.badge(
                "已暂停",
                color="orange",
                icon=":material/pause_circle:",
            )
        elif status == "awaiting_review":
            progress_header.badge(
                "待确认",
                color="orange",
                icon=":material/fact_check:",
            )
        else:
            progress_header.badge(
                "LIVE", color="blue", icon=":material/radio_button_checked:"
            )
        percent = max(0, min(_as_int(snapshot.get("max_percent", 0)), 100))
        overall_text = str(snapshot.get("overall_text", "等待开始"))
        st.progress(percent, text=overall_text)
        elapsed = _as_int(snapshot.get("elapsed_seconds", 0))
        st.caption(f"已用时 {_format_duration(elapsed)}")

    background_value = snapshot.get("background_tasks")
    background_tasks = (
        background_value if isinstance(background_value, dict) else {}
    )
    embedding_value = background_tasks.get("embedding", {})
    embedding = embedding_value if isinstance(embedding_value, dict) else {}
    embedding_state = str(embedding.get("state", "pending"))
    with st.container(border=True):
        background_header = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        background_header.markdown("**后台语义索引**")
        embedding_progress = max(
            0,
            min(_as_int(embedding.get("progress", 0)), 100),
        )
        if embedding_state == "running" and embedding_progress < 100:
            background_header.html(
                '<span class="ipo-running-badge">'
                '<span class="ipo-running-ring"></span>'
                "并行处理中</span>"
            )
        elif embedding_state == "complete" or embedding_progress >= 100:
            background_header.badge(
                "内存索引已就绪",
                color="green",
                icon=":material/check_circle:",
            )
        elif embedding_state == "degraded":
            background_header.badge(
                "已降级",
                color="orange",
                icon=":material/warning:",
            )
        else:
            background_header.badge(
                "等待启动",
                color="gray",
                icon=":material/schedule:",
            )
        st.progress(
            embedding_progress,
            text=str(embedding.get("message", "等待PDF解析完成后启动。")),
        )
        st.caption(
            "内存向量索引在PDF解析后后台构建，不单独计入主进度；"
            "风险发现会等待它就绪，再检索原文、相邻上下文和章节覆盖样本。"
        )
        persistence_value = background_tasks.get("persistence", {})
        persistence = (
            persistence_value
            if isinstance(persistence_value, dict)
            else {}
        )
        persistence_state = str(persistence.get("state", "pending"))
        persistence_header = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        persistence_header.markdown("**Chroma知识库同步**")
        if persistence_state == "running":
            persistence_header.html(
                '<span class="ipo-running-badge">'
                '<span class="ipo-running-ring"></span>后台写入中</span>'
            )
        elif persistence_state == "complete":
            persistence_header.badge(
                "已持久化",
                color="green",
                icon=":material/database:",
            )
        elif persistence_state == "degraded":
            persistence_header.badge(
                "写入失败",
                color="orange",
                icon=":material/warning:",
            )
        else:
            persistence_header.badge(
                "等待内存索引",
                color="gray",
                icon=":material/schedule:",
            )
        st.progress(
            max(0, min(_as_int(persistence.get("progress", 0)), 100)),
            text=str(
                persistence.get(
                    "message",
                    "内存索引完成后异步写入Chroma。",
                )
            ),
        )

    funnel_value = snapshot.get("funnel")
    funnel = funnel_value if isinstance(funnel_value, dict) else {}
    raw_candidates = _as_int(funnel.get("raw_candidates", 0))
    consolidated_candidates = _as_int(
        funnel.get("consolidated_candidates", 0)
    )
    if raw_candidates or consolidated_candidates:
        with st.container(border=True):
            st.markdown("**风险候选漏斗**")
            st.caption(
                "每一步口径独立：批次原始发现不是最终风险数，任何合并、排除和裁决"
                "都有对应阶段记录。"
            )
            metrics = st.columns(4)
            metrics[0].metric(
                "批次原始发现",
                raw_candidates,
                help="全文各批次直接输出的候选总数，尚未跨批次去重。",
            )
            metrics[1].metric(
                "全局语义合并",
                (
                    consolidated_candidates
                    if consolidated_candidates
                    else "处理中"
                ),
                help="大模型从全局视角合并同一底层风险后的数量。",
            )
            scoped_in = _as_int(funnel.get("scoped_in", 0))
            human_selected = _as_int(funnel.get("human_selected", 0))
            metrics[2].metric(
                "进入专业分析",
                human_selected or scoped_in or "待确认",
                help="业务适用性判断及人工范围确认后交给专业 Agent 的数量。",
            )
            final_accepted = _as_int(funnel.get("final_accepted", 0))
            metrics[3].metric(
                "决策最终采纳",
                (
                    final_accepted
                    if final_accepted or status == "completed"
                    else "待裁决"
                ),
                help="专业 Agent 核验后，由决策 Agent 最终采纳的风险数量。",
            )
            scoped_out = _as_int(funnel.get("scoped_out", 0))
            assignments_value = funnel.get("role_assignments")
            assignments = (
                assignments_value
                if isinstance(assignments_value, dict)
                else {}
            )
            role_text = " · ".join(
                f"{_AGENT_SPECS[role][0].replace(' Agent', '')} "
                f"{_as_int(assignments.get(role, 0))}"
                for role in _PARALLEL_AGENT_ROLES
                if _as_int(assignments.get(role, 0))
            )
            detail_parts = []
            batches = _as_int(funnel.get("document_batches", 0))
            if batches:
                detail_parts.append(f"风险原文包 {batches} 个")
            if scoped_out:
                detail_parts.append(f"业务适用性排除 {scoped_out} 条")
            if role_text:
                detail_parts.append(f"专业分派：{role_text}")
            if detail_parts:
                st.caption("；".join(detail_parts))

    agents_value = snapshot.get("agents")
    agents = agents_value if isinstance(agents_value, dict) else {}
    outputs_value = snapshot.get("outputs")
    outputs = outputs_value if isinstance(outputs_value, dict) else {}
    events_value = snapshot.get("events")
    events = events_value if isinstance(events_value, list) else []
    failures_value = snapshot.get("discovery_failures")
    discovery_failures = (
        failures_value if isinstance(failures_value, list) else []
    )
    if discovery_failures:
        st.warning(
            f"风险发现有 {len(discovery_failures)} 个原文包失败；"
            "成功批次的风险与证据已保留，失败范围可能存在漏检。",
            icon=":material/warning:",
        )
        with st.expander(
            "查看失败批次、重试次数与原因",
            icon=":material/troubleshoot:",
        ):
            for failure in discovery_failures:
                st.caption(str(failure))

    execution_column, activity_column = st.columns([1.85, 1], gap="large")
    with execution_column:
        st.subheader("Agent协同矩阵")
        st.caption("每个角色只呈现当前状态；模型输出按最新快照整体替换。")
        agent_roles = list(_AGENT_SPECS)
        for start in range(0, len(agent_roles), 3):
            row_roles = agent_roles[start : start + 3]
            row = st.columns(3)
            for column, role in zip(row, row_roles, strict=False):
                name, icon = _AGENT_SPECS[role]
                info_value = agents.get(role, {})
                info = info_value if isinstance(info_value, dict) else {}
                state = str(info.get("state", "pending"))
                badge_label, badge_color, badge_icon = _agent_badge(
                    state,
                    role=role,
                )
                with column.container(border=True, height=165):
                    st.subheader(name)
                    if state == "running":
                        st.html(
                            '<span class="ipo-running-badge">'
                            '<span class="ipo-running-ring"></span>分析中</span>'
                        )
                    else:
                        st.badge(
                            badge_label,
                            color=badge_color,
                            icon=badge_icon,
                        )
                    agent_percent = max(
                        0,
                        min(_as_int(info.get("progress", 0)), 100),
                    )
                    st.progress(
                        agent_percent,
                        text=str(info.get("progress_text", "尚未开始")),
                    )
                    st.caption(str(info.get("message", "等待编排器分发任务。")))

        output_tabs = st.tabs(
            [name for name, _icon in _AGENT_SPECS.values()],
        )
        risk_titles_value = snapshot.get("risk_titles")
        risk_titles = (
            risk_titles_value if isinstance(risk_titles_value, dict) else {}
        )
        for tab, (role, (name, icon)) in zip(
            output_tabs,
            _AGENT_SPECS.items(),
            strict=True,
        ):
            with tab:
                with st.container(border=True):
                    output_heading = st.container(horizontal=True)
                    output_heading.badge(name, color="blue", icon=icon)
                    output_heading.caption("实时模型输出 · 已转换为业务可读结论")
                    output = str(outputs.get(role, ""))
                    info_value = agents.get(role, {})
                    info = info_value if isinstance(info_value, dict) else {}
                    render_agent_model_output(
                        role,
                        output,
                        state=str(info.get("state", "pending")),
                        risk_titles=(
                            risk_titles.get(role)
                            if isinstance(risk_titles.get(role), dict)
                            else None
                        ),
                    )

    with activity_column:
        stream = st.container(border=True, height=710)
        stream.subheader("实时事件流")
        stream.caption("高频扫描只刷新下方动态，不重复写入历史。")
        stream.info(
            str(snapshot.get("current_activity", "等待分析事件")),
            icon=":material/sensors:",
        )
        # 限制每次片段重绘的文本量，避免高频刷新时整块控制台长时间变暗。
        visible_events = [str(event) for event in events[-50:]]
        if len(events) > len(visible_events):
            visible_events.insert(0, f"… 已折叠更早的 {len(events) - len(visible_events)} 条事件")
        stream.code(
            "\n".join(visible_events) or "[等待] 尚未收到分析事件",
            language=None,
            wrap_lines=True,
            height=525,
        )


__all__ = [
    "AnalysisProgressState",
    "AnalysisProgressView",
    "build_analysis_progress_snapshot",
    "render_analysis_progress_snapshot",
]
