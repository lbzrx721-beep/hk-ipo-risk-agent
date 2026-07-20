"""Streamlit实时分析运行台。"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import streamlit as st


_AGENT_SPECS = {
    "finance": ("财务 Agent", ":material/account_balance:"),
    "legal": ("法律 Agent", ":material/gavel:"),
    "business": ("业务 Agent", ":material/domain:"),
    "market": ("市场 Agent", ":material/candlestick_chart:"),
    "decision": ("决策 Agent", ":material/hub:"),
}
_STAGE_LABELS = {
    "upload_started": "资料登记",
    "upload_completed": "资料登记",
    "parsing_started": "PDF解析",
    "parsing_completed": "PDF解析",
    "agents_started": "任务编排",
    "agents_completed": "综合裁决",
    "evidence_started": "证据报告",
    "completed": "分析完成",
}
_PROGRESS_PATTERN = re.compile(r"\[(?P<current>\d+)/(?P<total>\d+)]")
_AGENT_OVERALL_RANGES = {
    "finance": (30, 48),
    "legal": (48, 64),
    "business": (64, 72),
    "market": (72, 77),
    "decision": (77, 82),
}


class AnalysisProgressView:
    """把分析流水线事件实时投射为Agent卡片和活动流。"""

    def __init__(self) -> None:
        st.subheader("实时分析运行台")
        st.caption("以下内容来自真实流水线事件；每个 Agent 的状态会随任务执行实时更新。")
        self._overall = st.progress(0, text="等待开始")
        self._cards: dict[str, dict[str, Any]] = {}

        first_row = st.columns(3)
        second_row = st.columns(2)
        for column, role in zip(
            (*first_row, *second_row),
            _AGENT_SPECS,
            strict=True,
        ):
            name, _icon = _AGENT_SPECS[role]
            card = column.container(border=True, height=330)
            card.subheader(name)
            badge = card.empty()
            badge.badge("等待任务", color="gray", icon=":material/schedule:")
            progress = card.progress(0, text="尚未开始")
            message = card.empty()
            message.caption("等待编排器分发风险任务。")
            card.caption("模型内容流（结构化答案）")
            output = card.empty()
            output.code(
                "[等待模型输出]",
                language=None,
                wrap_lines=True,
                height=145,
            )
            self._cards[role] = {
                "badge": badge,
                "progress": progress,
                "message": message,
                "output": output,
                "output_text": "",
                "state": "pending",
            }

        stream = st.container(border=True)
        stream.subheader("实时活动流")
        stream.caption("逐条显示文件解析、证据检索、模型分析、Agent 裁决和报告生成事件。")
        self._stream = stream.empty()
        self._events: list[str] = []
        self._render_stream()

    def _render_stream(self) -> None:
        content = "\n".join(self._events) or "[等待] 尚未收到分析事件"
        self._stream.code(
            content,
            language=None,
            wrap_lines=True,
            height=260,
        )

    def _append_event(self, label: str, detail: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self._events.append(f"[{timestamp}] {label:<8} | {detail}")
        self._render_stream()

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
            card["progress"].progress(5, text="正在接收任务")
        elif action == "search_planned":
            card["state"] = "running"
            card["badge"].badge(
                "语义检索",
                color="blue",
                icon=":material/manage_search:",
            )
            card["progress"].progress(25, text="Agent 已生成语义搜索问题")
        elif action == "search_fallback":
            card["state"] = "running"
            card["badge"].badge(
                "检索兜底",
                color="orange",
                icon=":material/sync_problem:",
            )
            card["progress"].progress(25, text="固定查询兜底运行中")
        elif action == "prompt":
            card["state"] = "running"
            card["badge"].badge(
                "生成中",
                color="blue",
                icon=":material/stream:",
            )
            card["progress"].progress(10, text="正在检索证据并请求模型")
            card["output_text"] += f"\n\n--- {detail} ---\n"
            card["output"].code(
                card["output_text"].strip(),
                language=None,
                wrap_lines=True,
                height=145,
            )
        elif action == "delta":
            card["state"] = "running"
            card["progress"].progress(50, text="模型正在流式生成")
            card["output_text"] += detail
            card["output"].code(
                card["output_text"][-8000:].strip(),
                language=None,
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
            card["progress"].progress(percent, text=progress_text)
        elif action == "completed":
            card["state"] = "complete"
            card["badge"].badge(
                "已完成",
                color="green",
                icon=":material/check_circle:",
            )
            card["progress"].progress(100, text="任务完成")
        if action != "delta":
            card["message"].caption(detail)

    def emit(self, stage: str, detail: str) -> None:
        """接收一次流水线回调并立即刷新页面。"""
        parts = stage.split(":")
        if len(parts) == 3 and parts[0] == "agent":
            role, action = parts[1], parts[2]
            self._update_agent(role, action, detail)
            label = _AGENT_SPECS.get(role, (role, ""))[0]
            if action != "delta":
                self._append_event(label, detail)
            progress_range = _AGENT_OVERALL_RANGES.get(role)
            if progress_range is not None:
                start, end = progress_range
                if action == "started":
                    percent = start
                elif action == "completed":
                    percent = end
                else:
                    match = _PROGRESS_PATTERN.search(detail)
                    fraction = 0.5
                    if match is not None:
                        fraction = int(match.group("current")) / max(
                            1,
                            int(match.group("total")),
                        )
                    percent = round(start + (end - start) * fraction)
                self._overall.progress(percent, text=detail)
        else:
            label = _STAGE_LABELS.get(stage, stage)
            self._append_event(label, detail)

        overall_progress = {
            "upload_started": 3,
            "upload_completed": 8,
            "parsing_started": 12,
            "parsing_completed": 25,
            "agents_started": 30,
            "agents_completed": 82,
            "evidence_started": 90,
            "completed": 100,
        }.get(stage)
        if overall_progress is not None:
            self._overall.progress(overall_progress, text=detail)

    def complete(self, detail: str) -> None:
        """保证成功结束时进度条处于完成状态。"""
        self._overall.progress(100, text=detail)

    def fail(self, detail: str) -> None:
        """把当前运行中的Agent和总进度切换为错误状态。"""
        self._append_event("执行失败", detail)
        self._overall.progress(100, text="分析失败，请查看错误信息")
        for card in self._cards.values():
            if card["state"] != "running":
                continue
            card["badge"].badge(
                "执行失败",
                color="red",
                icon=":material/error:",
            )
            card["message"].caption(detail)


__all__ = ["AnalysisProgressView"]
