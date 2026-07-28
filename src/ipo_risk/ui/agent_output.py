"""把Agent原始JSON响应转换为投研人员可读的结构化卡片。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

import streamlit as st

from ipo_risk.ui import theme


_OUTPUT_SEPARATOR = "--- 新的风险任务 ---"
_ROLE_LABELS = {
    "discovery": "风险发现 Agent",
    "finance": "财务 Agent",
    "legal": "法律 Agent",
    "business": "业务 Agent",
    "market": "市场 Agent",
    "decision": "决策 Agent",
}
_OWNER_LABELS = {
    "finance": "财务 Agent",
    "legal": "法律 Agent",
    "business": "业务 Agent",
    "market": "市场 Agent",
}

AgentOutputKind = Literal["discovery", "consolidated", "analysis", "decision"]


@dataclass(frozen=True)
class AgentOutputCard:
    """一条适合在界面直接阅读的Agent结构化结论。"""

    kind: AgentOutputKind
    title: str
    risk_type: str | None
    claim: str
    severity: str
    confidence: float | None
    evidence_ids: tuple[str, ...]
    owner: str | None = None
    supported: bool | None = None
    accepted: bool | None = None
    uncertainty: str = ""
    rationale: str = ""
    machine_status: str | None = None
    source_proposal_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentOutputPresentation:
    cards: tuple[AgentOutputCard, ...]
    has_incomplete_segment: bool
    parsed_segment_count: int
    raw_discovery_count: int = 0
    consolidated_discovery_count: int = 0


@dataclass(frozen=True)
class AgentOutputPage:
    """结构化卡片的一页，编号始终从完整列表的第一页开始。"""

    cards: tuple[AgentOutputCard, ...]
    page: int
    page_count: int
    total_count: int
    first_index: int


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(float(value), 1.0))


def _as_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _card_from_item(
    role: str,
    item: dict[str, Any],
    *,
    kind: AgentOutputKind,
    risk_titles: dict[str, str] | None = None,
) -> AgentOutputCard:
    supported = _as_optional_bool(item.get("supported"))
    accepted = _as_optional_bool(item.get("accepted"))
    title = _as_text(item.get("title"))
    risk_type = _as_text(item.get("risk_type"))
    if not title and risk_type and risk_titles:
        title = risk_titles.get(risk_type, "")
    if not title and risk_type:
        title = (
            f"专业核验 · {risk_type}"
            if kind == "analysis"
            else f"已有风险裁决 · {risk_type}"
        )
    if not title:
        role_label = _ROLE_LABELS.get(role, role)
        if supported is False:
            title = f"{role_label} · 证据不足"
        elif kind == "decision" and accepted is False:
            title = f"{role_label} · 未采纳风险"
        else:
            title = f"{role_label} · 结构化判断"
    return AgentOutputCard(
        kind=kind,
        title=title,
        risk_type=risk_type or None,
        claim=_as_text(item.get("claim")),
        severity=_as_text(item.get("severity")) or "unknown",
        confidence=_as_confidence(item.get("confidence")),
        evidence_ids=_as_ids(item.get("evidence_chunk_ids")),
        owner=_as_text(item.get("owner")) or None,
        supported=supported,
        accepted=accepted,
        uncertainty=_as_text(item.get("uncertainty")),
        rationale=_as_text(item.get("rationale")),
        machine_status=_as_text(item.get("status")) or None,
        source_proposal_ids=_as_ids(item.get("source_proposal_ids")),
    )


def format_raw_model_output(raw_output: str) -> str:
    """Pretty-print every completed JSON snapshot without discarding batches."""
    formatted: list[str] = []
    for raw_segment in raw_output.split(_OUTPUT_SEPARATOR):
        segment = _strip_code_fence(raw_segment)
        segment = re.sub(
            r"^\[\[(?:batch|consolidation):\d+/\d+\]\]",
            "",
            segment,
        ).strip()
        if not segment:
            continue
        try:
            payload = json.loads(segment)
        except json.JSONDecodeError:
            formatted.append(segment)
        else:
            formatted.append(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
    return f"\n\n{_OUTPUT_SEPARATOR}\n\n".join(formatted)


def empty_agent_output_message(role: str, state: str) -> str:
    """Return an accurate empty-state message for an Agent output panel."""
    if state == "skipped":
        role_name = _ROLE_LABELS.get(role, role).replace(" Agent", "")
        return f"本次未分配{role_name}类风险，未调用模型。"
    if state == "complete":
        return "本轮任务已完成，没有额外的模型输出。"
    return "等待模型输出…"


def _cards_from_payload(
    role: str,
    payload: dict[str, Any],
    *,
    risk_titles: dict[str, str] | None = None,
) -> list[AgentOutputCard]:
    consolidated = payload.get("consolidated_risks")
    if isinstance(consolidated, list):
        return [
            _card_from_item(
                role,
                item,
                kind="consolidated",
                risk_titles=risk_titles,
            )
            for item in consolidated
            if isinstance(item, dict)
        ]
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [
            _card_from_item(
                role,
                item,
                kind="discovery",
                risk_titles=risk_titles,
            )
            for item in candidates
            if isinstance(item, dict)
        ]
    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        return [
            _card_from_item(
                role,
                item,
                kind="decision",
                risk_titles=risk_titles,
            )
            for item in decisions
            if isinstance(item, dict)
        ]
    assessments = payload.get("assessments")
    if isinstance(assessments, list):
        return [
            _card_from_item(
                role,
                item,
                kind="analysis",
                risk_titles=risk_titles,
            )
            for item in assessments
            if isinstance(item, dict)
        ]
    if not any(
        key in payload
        for key in (
            "risk_type",
            "title",
            "claim",
            "status",
            "supported",
            "accepted",
        )
    ):
        return []
    kind: AgentOutputKind = (
        "decision" if "accepted" in payload or role == "decision" else "analysis"
    )
    return [
        _card_from_item(
            role,
            payload,
            kind=kind,
            risk_titles=risk_titles,
        )
    ]


_PARTIAL_ARRAY_KINDS: tuple[tuple[str, AgentOutputKind], ...] = (
    ("consolidated_risks", "consolidated"),
    ("candidates", "discovery"),
    ("decisions", "decision"),
    ("assessments", "analysis"),
)


def _partial_array_cards(
    role: str,
    segment: str,
    *,
    risk_titles: dict[str, str] | None,
) -> list[AgentOutputCard]:
    """从尚未闭合的JSON数组中提取已经完整生成的对象。"""
    cards: list[AgentOutputCard] = []
    for key, kind in _PARTIAL_ARRAY_KINDS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', segment)
        if match is None:
            continue
        for raw_item in _complete_json_objects(segment, start=match.end()):
            try:
                item = json.loads(raw_item)
            except json.JSONDecodeError:
                # 单个模型对象损坏时继续寻找后续已经闭合的对象，避免卡片永久停住。
                continue
            if not isinstance(item, dict):
                continue
            cards.append(
                _card_from_item(
                    role,
                    item,
                    kind=kind,
                    risk_titles=risk_titles,
                )
            )
        break
    return cards


def _complete_json_objects(value: str, *, start: int) -> list[str]:
    """提取数组中已经闭合的顶层对象，并正确处理字符串内的大括号。"""
    objects: list[str] = []
    object_start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char != "}" or depth == 0:
            continue
        depth -= 1
        if depth == 0 and object_start is not None:
            objects.append(value[object_start : index + 1])
            object_start = None
    return objects


@st.cache_data(max_entries=128, show_spinner=False)
def parse_agent_output(
    role: str,
    raw_output: str,
    *,
    risk_titles: dict[str, str] | None = None,
) -> AgentOutputPresentation:
    """逐段解析流式JSON；完整段立即成卡，最后的半截响应保持等待状态。"""
    if not raw_output.strip():
        return AgentOutputPresentation((), False, 0)
    cards: list[AgentOutputCard] = []
    parsed_segment_count = 0
    has_incomplete_segment = False
    for raw_segment in raw_output.split(_OUTPUT_SEPARATOR):
        segment = _strip_code_fence(raw_segment)
        if not segment:
            continue
        try:
            payload = json.loads(segment)
        except json.JSONDecodeError:
            has_incomplete_segment = True
            cards.extend(
                _partial_array_cards(
                    role,
                    segment,
                    risk_titles=risk_titles,
                )
            )
            continue
        if not isinstance(payload, dict):
            has_incomplete_segment = True
            continue
        parsed_segment_count += 1
        cards.extend(
            _cards_from_payload(
                role,
                payload,
                risk_titles=risk_titles,
            )
        )
    cards = [
        card
        for card in cards
        if (
            card.risk_type
            or card.title
            and "结构化判断" not in card.title
            or card.claim
            or card.evidence_ids
            or card.rationale
            or card.uncertainty
        )
    ]
    deduplicated: list[AgentOutputCard] = []
    index_by_identity: dict[tuple[AgentOutputKind, str], int] = {}
    for card in cards:
        if card.risk_type is None:
            deduplicated.append(card)
            continue
        identity = (card.kind, card.risk_type)
        existing_index = index_by_identity.get(identity)
        if existing_index is None:
            index_by_identity[identity] = len(deduplicated)
            deduplicated.append(card)
        else:
            deduplicated[existing_index] = card
    cards = deduplicated
    raw_discovery_count = sum(card.kind == "discovery" for card in cards)
    consolidated_discovery_count = sum(card.kind == "consolidated" for card in cards)
    return AgentOutputPresentation(
        cards=tuple(cards),
        has_incomplete_segment=has_incomplete_segment,
        parsed_segment_count=parsed_segment_count,
        raw_discovery_count=raw_discovery_count,
        consolidated_discovery_count=consolidated_discovery_count,
    )


def paginate_agent_output(
    presentation: AgentOutputPresentation,
    *,
    role: str,
    page: int = 1,
    page_size: int = 10,
    discovery_view: Literal["raw", "consolidated"] = "consolidated",
) -> AgentOutputPage:
    """选择业务可见卡片并分页；实时原始发现和最终合并结果口径分离。"""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if (
        role == "discovery"
        and presentation.consolidated_discovery_count
        and discovery_view == "consolidated"
    ):
        all_cards = tuple(card for card in presentation.cards if card.kind == "consolidated")
    elif role == "discovery":
        all_cards = tuple(card for card in presentation.cards if card.kind == "discovery")
    else:
        all_cards = presentation.cards

    total_count = len(all_cards)
    page_count = max(1, (total_count + page_size - 1) // page_size)
    normalized_page = max(1, min(page, page_count))
    start = (normalized_page - 1) * page_size
    return AgentOutputPage(
        cards=all_cards[start : start + page_size],
        page=normalized_page,
        page_count=page_count,
        total_count=total_count,
        first_index=start + 1,
    )


def _render_card_badges(card: AgentOutputCard) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        if card.accepted is True:
            st.badge("裁决采纳", color="green", icon=":material/check_circle:")
        elif card.accepted is False:
            st.badge("等待人工", color="orange", icon=":material/fact_check:")
        elif card.machine_status:
            status_meta: dict[
                str,
                tuple[str, theme.BadgeColor, str],
            ] = {
                "supported": ("证据支持", "green", ":material/verified:"),
                "needs_more_evidence": (
                    "证据不足 · 待人工",
                    "orange",
                    ":material/fact_check:",
                ),
                "evidence_insufficient": (
                    "证据不足",
                    "orange",
                    ":material/help:",
                ),
                "subject_or_period_mismatch": (
                    "主体/期间不匹配",
                    "orange",
                    ":material/event_busy:",
                ),
                "duplicate_merged": (
                    "已并入同类风险",
                    "gray",
                    ":material/merge:",
                ),
                "conflict": (
                    "存在冲突",
                    "orange",
                    ":material/compare_arrows:",
                ),
            }
            label, color, icon = status_meta.get(
                card.machine_status,
                (card.machine_status, "gray", ":material/info:"),
            )
            st.badge(label, color=color, icon=icon)
        elif card.supported is True:
            st.badge("证据支持", color="green", icon=":material/verified:")
        elif card.supported is False:
            st.badge("证据不足", color="orange", icon=":material/help:")

        severity_label, severity_color = theme.SEVERITY_META.get(
            card.severity,
            ("待判断", "gray"),
        )
        st.badge(
            f"{severity_label}风险",
            color=severity_color,
            icon=":material/crisis_alert:",
        )
        if card.confidence is not None:
            st.badge(
                f"置信度 {card.confidence:.0%}",
                color="blue",
                icon=":material/analytics:",
            )
        if card.owner:
            owner_label = _OWNER_LABELS.get(card.owner, card.owner)
            st.badge(
                f"建议核验：{owner_label}",
                color="violet",
                icon=":material/forward_to_inbox:",
            )


def _render_agent_card(card: AgentOutputCard, index: int) -> None:
    with st.container(border=True):
        st.markdown(f"**{index:02d} · {card.title}**")
        _render_card_badges(card)
        if card.machine_status == "needs_more_evidence":
            st.write("当前证据尚不足以形成风险结论，已转入人工核验。")
            st.caption("本轮专业分析已经结束；系统未开启二次模型补证。")
        elif card.claim:
            st.write(card.claim)
        elif card.rationale:
            st.write(card.rationale)
        else:
            st.caption("本轮没有形成可展示的风险结论。")

        if card.evidence_ids:
            evidence_preview = " · ".join(card.evidence_ids)
            st.caption(f"原文证据 {len(card.evidence_ids)} 条 · {evidence_preview}")
        else:
            st.caption("本轮未引用原文证据片段。")
        if card.rationale and card.claim:
            st.caption(f"裁决依据：{card.rationale}")
        if card.source_proposal_ids:
            st.caption(
                "来源：已有专业提案 · "
                + " · ".join(card.source_proposal_ids)
            )
        if card.uncertainty:
            st.caption(f"待核实：{card.uncertainty}")


def _change_page(page_key: str, delta: int) -> None:
    current = int(st.session_state.get(page_key, 1))
    st.session_state[page_key] = max(1, current + delta)


def render_agent_model_output(
    role: str,
    raw_output: str,
    *,
    state: str = "pending",
    risk_titles: dict[str, str] | None = None,
) -> None:
    """默认展示业务卡片，将原始JSON降为折叠的审计详情。"""
    presentation = parse_agent_output(
        role,
        raw_output,
        risk_titles=risk_titles,
    )
    if presentation.cards:
        discovery_view: Literal["raw", "consolidated"] = "consolidated"
        if (
            role == "discovery"
            and presentation.raw_discovery_count
            and presentation.consolidated_discovery_count
        ):
            view_key = f"agent_output_view_{role}"
            selected_view = st.segmented_control(
                "风险发现口径",
                options=["合并后风险", "原始发现"],
                default="合并后风险",
                key=view_key,
                help=(
                    "合并不会删除原始发现。可在这里随时切换查看全部原始风险，"
                    "以及合并后进入专业核验的风险主题。"
                ),
            )
            discovery_view = "raw" if selected_view == "原始发现" else "consolidated"
        page_key = f"agent_output_page_{role}_{discovery_view if role == 'discovery' else 'live'}"
        requested_page = int(st.session_state.get(page_key, 1))
        page_view = paginate_agent_output(
            presentation,
            role=role,
            page=requested_page,
            discovery_view=discovery_view,
        )
        st.session_state[page_key] = page_view.page
        with st.container(horizontal=True, vertical_alignment="center"):
            if role == "discovery":
                st.badge(
                    f"批次原始发现 {presentation.raw_discovery_count} 条",
                    color="blue",
                    icon=":material/find_in_page:",
                )
                if presentation.consolidated_discovery_count:
                    st.badge(
                        (f"全局语义合并 {presentation.consolidated_discovery_count} 条"),
                        color="green",
                        icon=":material/merge:",
                    )
                else:
                    st.badge(
                        "尚未完成全局合并",
                        color="orange",
                        icon=":material/progress_activity:",
                    )
            else:
                st.badge(
                    (
                        f"已完成 {len(presentation.cards)} 项裁决"
                        if role == "decision"
                        else f"已形成 {len(presentation.cards)} 条结构化结论"
                    ),
                    color="green",
                    icon=":material/checklist:",
                )
            if presentation.has_incomplete_segment:
                st.html(
                    '<span class="ipo-running-badge">'
                    '<span class="ipo-running-ring"></span>'
                    "下一条生成中</span>"
                )
            if page_view.total_count > 10:
                st.badge(
                    f"共 {page_view.total_count} 条 · 每页 10 条",
                    color="gray",
                    icon=":material/view_list:",
                )
        if role == "discovery" and presentation.consolidated_discovery_count:
            st.caption(
                (
                    "当前展示全部批次原始发现；这些记录不会被全局合并覆盖。"
                    if discovery_view == "raw"
                    else "当前展示跨原文包语义合并后的风险主题；原始发现仍完整保留，可切换查看。"
                )
            )
        if page_view.page_count > 1:
            navigation = st.container(
                horizontal=True,
                horizontal_alignment="center",
                vertical_alignment="center",
            )
            navigation.button(
                "上一页",
                key=f"{page_key}_previous",
                icon=":material/chevron_left:",
                disabled=page_view.page <= 1,
                on_click=_change_page,
                args=(page_key, -1),
            )
            navigation.caption(f"第 {page_view.page} / {page_view.page_count} 页")
            navigation.button(
                "下一页",
                key=f"{page_key}_next",
                icon=":material/chevron_right:",
                icon_position="right",
                disabled=page_view.page >= page_view.page_count,
                on_click=_change_page,
                args=(page_key, 1),
            )
        for index, card in enumerate(
            page_view.cards,
            start=page_view.first_index,
        ):
            _render_agent_card(card, index)
    elif raw_output and presentation.parsed_segment_count:
        st.caption("本轮模型已完成结构化响应，未发现可展示的风险结论。")
    elif raw_output and state == "running":
        st.info(
            "模型正在生成结构化结果，完成后会自动转换为风险卡片。",
            icon=":material/progress_activity:",
        )
    elif raw_output:
        st.warning(
            "本次响应未能解析为结构化卡片，请在下方查看原始模型响应。",
            icon=":material/data_alert:",
        )
    else:
        st.caption(empty_agent_output_message(role, state))

    if raw_output:
        with st.expander(
            "查看原始模型响应",
            icon=":material/code:",
        ):
            st.caption("用于调试、审计和模型输出追踪；不作为默认业务结论。")
            st.code(
                format_raw_model_output(raw_output),
                language="json",
                wrap_lines=True,
                height=260,
            )


__all__ = [
    "AgentOutputCard",
    "AgentOutputPage",
    "AgentOutputPresentation",
    "empty_agent_output_message",
    "format_raw_model_output",
    "paginate_agent_output",
    "parse_agent_output",
    "render_agent_model_output",
]
