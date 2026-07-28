"""分析完成后的只读结果工作区。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ipo_risk.domain import AgentTrace
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.reviewing import ReviewService
from ipo_risk.storage.files import read_json
from ipo_risk.ui import theme
from ipo_risk.ui.agent_output import render_agent_model_output
from ipo_risk.ui.routes import switch_to_review


_SPECIALIST_LABELS = {
    "finance": "财务 Agent",
    "legal": "法律 Agent",
    "business": "业务 Agent",
    "market": "市场 Agent",
}


def _clear_deleted_run_state(run_id: str) -> None:
    """在控件创建前清除指向已删除运行的浏览器状态。"""
    if st.session_state.get("results_selected_run") == run_id:
        st.session_state.pop("results_selected_run", None)
    if st.session_state.get("last_analysis_run_id") == run_id:
        for key in (
            "last_analysis_run_id",
            "last_analysis_report_path",
            "last_analysis_result_summary",
            "last_analysis_snapshot",
            "last_analysis_snapshot_run_id",
        ):
            st.session_state.pop(key, None)
    if st.session_state.get("current_analysis_run_id") == run_id:
        for key in (
            "current_analysis_job",
            "current_analysis_job_synced",
            "current_analysis_file_name",
            "current_analysis_run_id",
            "current_analysis_snapshot",
            "current_analysis_result_summary",
            "current_analysis_status",
            "current_analysis_error",
        ):
            st.session_state.pop(key, None)


@st.dialog("删除分析运行", width="small")
def _render_delete_dialog(service: ReviewService, run_id: str, company_name: str) -> None:
    st.warning(
        f"将删除 **{company_name}** 的本次分析、风险结论、证据关联和人工复核记录。"
        "原始招股书及公司资料会保留。"
    )
    st.code(run_id, language=None)
    confirmation = st.text_input(
        "输入 run_id 确认",
        placeholder=run_id,
        key=f"delete_confirmation_{run_id}",
    )
    left, right = st.columns(2)
    if left.button("取消", width="stretch"):
        st.session_state.pop("analysis_delete_target", None)
        st.rerun()
    if right.button(
        "确认删除",
        type="primary",
        icon=":material/delete:",
        width="stretch",
        disabled=confirmation.strip() != run_id,
    ):
        try:
            result = service.delete_analysis_run(run_id)
        except (KeyError, OSError, ValueError) as exc:
            st.error(f"删除失败：{exc}")
            return
        st.session_state.pop("analysis_delete_target", None)
        st.session_state["analysis_deleted_run"] = run_id
        recovery = str(result.recovery_path) if result.recovery_path else "无派生文件"
        st.session_state["analysis_delete_flash"] = (
            f"已删除分析 {run_id}。原始招股书仍保留；派生文件回收位置：{recovery}"
        )
        st.rerun()


def _load_report_reference(
    service: ReviewService,
    reference: object,
) -> RiskAnalysisReport | None:
    if not reference:
        return None
    path = Path(str(reference))
    if not path.is_absolute():
        path = service.project_root / path
    if not path.is_file():
        return None
    try:
        return RiskAnalysisReport.model_validate(read_json(path))
    except (OSError, ValueError):
        return None


def _load_selected_report(service: ReviewService) -> RiskAnalysisReport | None:
    return _load_report_reference(
        service,
        st.session_state.get("last_analysis_report_path"),
    )


def analysis_snapshot_from_report(report: RiskAnalysisReport) -> dict[str, object]:
    """从持久化AgentTrace重建只读回放，供新会话恢复页面。"""
    roles = ("discovery", "finance", "legal", "business", "market", "decision")
    agents: dict[str, dict[str, str]] = {
        role: {"state": "pending", "message": "该运行没有保存此角色节点。"}
        for role in roles
    }
    outputs: dict[str, str] = {role: "" for role in roles}
    events: list[str] = []
    for trace in report.agent_traces:
        role = trace.agent.value
        if role not in agents:
            continue
        state = "complete" if trace.status.value == "succeeded" else "failed"
        agents[role] = {
            "state": state,
            "message": trace.error or trace.decision,
        }
        events.append(f"[历史回放] {role} Agent | {trace.decision}")
    discovery = _discovery_output(report)
    if discovery:
        outputs["discovery"] = json.dumps(
            {"consolidated_risks": discovery.get("discovered_risks", [])},
            ensure_ascii=False,
            indent=2,
        )
    for role in _SPECIALIST_LABELS:
        assessments = [
            {
                "risk_type": finding.risk_type,
                "title": finding.title,
                "supported": True,
                "claim": finding.claim,
                "severity": finding.severity.value,
                "confidence": finding.confidence,
                "evidence_chunk_ids": list(finding.evidence_ids),
                "uncertainty": finding.uncertainty,
            }
            for finding in report.risk_findings
            if finding.created_by.value == role
        ]
        if assessments:
            outputs[role] = json.dumps(
                {"assessments": assessments},
                ensure_ascii=False,
                indent=2,
            )
    if report.risk_findings:
        outputs["decision"] = json.dumps(
            {
                "decisions": [
                    {
                        "risk_type": finding.risk_type,
                        "title": finding.title,
                        "accepted": True,
                        "claim": finding.claim,
                        "severity": finding.severity.value,
                        "confidence": finding.confidence,
                        "evidence_chunk_ids": list(finding.evidence_ids),
                        "uncertainty": finding.uncertainty,
                        "rationale": "决策 Agent 保留该风险并提交人工复核。",
                    }
                    for finding in report.risk_findings
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    return {
        "events": events,
        "max_percent": 100,
        "agents": agents,
        "outputs": outputs,
    }


def restore_latest_analysis_state(
    service: ReviewService,
) -> RiskAnalysisReport | None:
    """从SQLite登记的最新成功运行恢复会话状态，不依赖浏览器临时状态。"""
    runs = service.list_runs()
    if not runs:
        return _load_selected_report(service)
    run_by_id = {run.run_id: run for run in runs}
    preferred = str(st.session_state.get("last_analysis_run_id", ""))
    selected = run_by_id.get(preferred, runs[0])
    report = _load_report_reference(service, selected.report_path)
    if report is None:
        return _load_selected_report(service)
    st.session_state["last_analysis_run_id"] = report.run_id
    st.session_state["last_analysis_report_path"] = str(selected.report_path)
    st.session_state["last_analysis_result_summary"] = {
        "run_id": report.run_id,
        "finding_count": report.summary.finding_count,
        "evidence_count": report.summary.evidence_count,
        "trace_count": len(report.agent_traces),
    }
    if st.session_state.get("last_analysis_snapshot_run_id") != report.run_id:
        st.session_state["last_analysis_snapshot"] = analysis_snapshot_from_report(
            report
        )
        st.session_state["last_analysis_snapshot_run_id"] = report.run_id
    return report


def render_agent_snapshot(
    snapshot: dict[str, object],
    *,
    title: str = "Agent协同回放",
    caption: str = "查看各角色最终状态，并切换下方页签回放模型结构化输出。",
    event_label: str = "查看完整分析事件流",
) -> None:
    agents = snapshot.get("agents")
    specs = {
        "discovery": "风险发现 Agent",
        "finance": "财务 Agent",
        "legal": "法律 Agent",
        "business": "业务 Agent",
        "market": "市场 Agent",
        "decision": "决策 Agent",
    }
    if isinstance(agents, dict) and agents:
        st.subheader(title)
        st.caption(caption)
        spec_items = list(specs.items())
        for start in range(0, len(spec_items), 3):
            row_items = spec_items[start : start + 3]
            columns = st.columns(3)
            for column, (role, title) in zip(columns, row_items, strict=False):
                with column.container(border=True, height=125):
                    st.subheader(title)
                    info = agents.get(role, {})
                    state = (
                        info.get("state", "pending")
                        if isinstance(info, dict)
                        else "pending"
                    )
                    color: theme.BadgeColor = (
                        "green"
                        if state == "complete"
                        else "red"
                        if state == "failed"
                        else "orange"
                        if state == "degraded"
                        else "blue"
                        if state == "running"
                        else "gray"
                    )
                    label = (
                        "已完成"
                        if state == "complete"
                        else "已降级"
                        if state == "failed"
                        else "部分降级"
                        if state == "degraded"
                        else (
                            "未启用"
                            if role == "discovery"
                            else "本次无任务"
                        )
                        if state == "skipped"
                        else "分析中"
                        if state == "running"
                        else "等待任务"
                    )
                    st.badge(label, color=color)
                    if isinstance(info, dict) and info.get("message"):
                        st.caption(str(info["message"]))

    events = snapshot.get("events")
    outputs = snapshot.get("outputs")
    risk_titles_value = snapshot.get("risk_titles")
    risk_titles = (
        risk_titles_value if isinstance(risk_titles_value, dict) else {}
    )
    if isinstance(outputs, dict) and any(outputs.values()):
        tabs = st.tabs(list(specs.values()))
        for tab, (role, _title) in zip(tabs, specs.items(), strict=True):
            with tab:
                output = outputs.get(role)
                info = agents.get(role, {}) if isinstance(agents, dict) else {}
                state = (
                    str(info.get("state", "pending"))
                    if isinstance(info, dict)
                    else "pending"
                )
                render_agent_model_output(
                    role,
                    str(output) if output else "",
                    state=state,
                    risk_titles=(
                        risk_titles.get(role)
                        if isinstance(risk_titles.get(role), dict)
                        else None
                    ),
                )
    if isinstance(events, list) and events:
        with st.expander(
            event_label,
            icon=":material/receipt_long:",
        ):
            st.code(
                "\n".join(str(line) for line in events),
                language=None,
                wrap_lines=True,
                height=300,
            )


def _trace_by_role(report: RiskAnalysisReport, role: str) -> AgentTrace | None:
    return next(
        (trace for trace in report.agent_traces if trace.agent.value == role),
        None,
    )


def _decision_output(report: RiskAnalysisReport) -> dict[str, object]:
    trace = _trace_by_role(report, "decision")
    if trace is None:
        return {}
    call = next(
        (item for item in trace.tool_calls if item.tool_name == "risk_adjudication"),
        None,
    )
    if call is None or not isinstance(call.output, dict):
        return {}
    return dict(call.output)


def _discovery_output(report: RiskAnalysisReport) -> dict[str, object]:
    trace = _trace_by_role(report, "discovery")
    if trace is None:
        return {}
    call = next(
        (item for item in trace.tool_calls if item.tool_name == "open_risk_discovery"),
        None,
    )
    if call is None or not isinstance(call.output, dict):
        return {}
    return dict(call.output)


def _render_result_overview(
    report: RiskAnalysisReport,
    *,
    reviewed_findings: int,
    pending_findings: int,
) -> None:
    with st.container(horizontal=True):
        st.metric("候选风险", report.summary.finding_count, border=True)
        st.metric("原文证据", report.summary.evidence_count, border=True)
        st.metric(
            "人工复核",
            f"{reviewed_findings} / {report.summary.finding_count}",
            border=True,
        )
        st.metric("待处理", pending_findings, border=True)
        st.metric("Agent轨迹", len(report.agent_traces), border=True)


def _render_decision_summary(
    report: RiskAnalysisReport,
    *,
    reviewed_findings: int,
    pending_findings: int,
) -> None:
    trace = _trace_by_role(report, "decision")
    output = _decision_output(report)
    mode = str(output.get("mode", ""))
    conflicts = int(output.get("conflict_count", 0) or 0)
    proposals = int(
        output.get(
            "proposal_count",
            output.get("direct_proposal_count", report.summary.finding_count),
        )
        or 0
    )

    with st.container(border=True):
        heading = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        heading.subheader("决策 Agent 结论")
        heading.badge(
            "等待人工确认" if pending_findings else "人工复核完成",
            color="orange" if pending_findings else "green",
            icon=(
                ":material/pending_actions:"
                if pending_findings
                else ":material/verified:"
            ),
        )
        st.write(
            trace.decision
            if trace is not None
            else "本次运行没有保存决策 Agent 的最终结论。"
        )
        badges = st.container(horizontal=True, vertical_alignment="center")
        badges.badge(
            f"汇总提案 {proposals} 条",
            color="blue",
            icon=":material/account_tree:",
        )
        badges.badge(
            f"角色冲突 {conflicts} 项",
            color="green" if conflicts == 0 else "orange",
            icon=":material/compare_arrows:",
        )
        badges.badge(
            f"已复核 {reviewed_findings} 条",
            color="green" if pending_findings == 0 else "gray",
            icon=":material/fact_check:",
        )
        if mode == "deterministic_integration":
            st.caption(
                "本次采用确定性规则整合专业 Agent 提案，没有调用裁决模型；"
                "最终确认或驳回仍由人工复核完成。"
            )
        elif output.get("model"):
            st.caption(f"裁决模型：{output['model']}")


def _render_findings(report: RiskAnalysisReport) -> None:
    st.subheader("待复核风险")
    st.caption("这里展示决策阶段保留下来的风险结论；证据原文和截图在人工复核页核对。")
    severity_options = ("全部", "严重", "高", "中", "低")
    selected_label = st.segmented_control(
        "风险等级",
        severity_options,
        default="全部",
        key=f"results_severity_{report.run_id}",
        label_visibility="collapsed",
    )
    severity_values = {
        "严重": "critical",
        "高": "high",
        "中": "medium",
        "低": "low",
    }
    selected_value = severity_values.get(str(selected_label))
    findings = [
        finding
        for finding in report.risk_findings
        if selected_value is None or finding.severity.value == selected_value
    ]
    st.caption(f"当前显示 {len(findings)} / {len(report.risk_findings)} 条")
    if not findings:
        st.info("当前等级下没有风险结论。", icon=":material/filter_alt_off:")
        return

    for index, finding in enumerate(findings, start=1):
        label, color = theme.SEVERITY_META.get(
            finding.severity.value,
            (finding.severity.value, "gray"),
        )
        with st.container(border=True):
            heading = st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            )
            heading.badge(f"{index:02d}", color="gray")
            heading.subheader(finding.title)
            heading.badge(f"{label}风险", color=color)
            st.write(finding.claim)
            metadata = st.container(horizontal=True, vertical_alignment="center")
            metadata.badge(
                f"置信度 {finding.confidence:.0%}",
                color="blue",
                icon=":material/analytics:",
            )
            metadata.badge(
                f"证据 {len(finding.evidence_ids)} 条",
                color="gray",
                icon=":material/find_in_page:",
            )
            metadata.caption(finding.finding_id)
            if finding.uncertainty:
                st.caption(f"待核实：{finding.uncertainty}")


def _render_review_panel(
    *,
    run_id: str,
    total_findings: int,
    pending_findings: int,
    verified_findings: int,
    rejected_findings: int,
) -> None:
    with st.container(border=True):
        st.subheader("人工复核")
        if pending_findings:
            st.write(
                f"还有 **{pending_findings}** 条风险需要核对原文、页码、"
                "高亮截图和结论措辞。"
            )
        else:
            st.write("本次风险已经全部完成复核，可以继续导出正式报告。")
        st.caption(
            f"已确认 {verified_findings} · 已驳回 {rejected_findings} · "
            f"总计 {total_findings}"
        )
        if st.button(
            "进入人工复核",
            type="primary",
            icon=":material/fact_check:",
            width="stretch",
            key=f"results_review_{run_id}",
        ):
            st.session_state["review_selected_run"] = run_id
            switch_to_review()


def _render_agent_audit(report: RiskAnalysisReport) -> None:
    with st.expander(
        "分析审计与 Agent 协作摘要",
        icon=":material/account_tree:",
        type="compact",
    ):
        st.caption(
            "这里保留结果追踪信息，不重复播放新建分析页中的实时进度、事件流和模型输出。"
        )
        discovery = _discovery_output(report)
        if discovery:
            st.markdown("**风险发现覆盖与合并映射**")
            metrics = st.container(horizontal=True)
            metrics.metric(
                "全文切片",
                int(discovery.get("source_chunk_count", 0) or 0),
                border=True,
            )
            metrics.metric(
                "送入发现",
                int(discovery.get("selected_chunk_count", 0) or 0),
                border=True,
            )
            metrics.metric(
                "原始风险",
                int(discovery.get("raw_candidate_count", 0) or 0),
                border=True,
            )
            metrics.metric(
                "合并主题",
                int(discovery.get("consolidated_candidate_count", 0) or 0),
                border=True,
            )
            raw_candidates = discovery.get("raw_candidates", [])
            discovered_risks = discovery.get("discovered_risks", [])
            mappings = discovery.get("consolidation_mappings", [])
            raw_by_id = {
                str(item.get("candidate_id")): item
                for item in raw_candidates
                if isinstance(item, dict)
            }
            risk_by_type = {
                str(item.get("risk_type")): item
                for item in discovered_risks
                if isinstance(item, dict)
            }
            mapping_rows = []
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    continue
                risk_type = str(mapping.get("risk_type", ""))
                source_ids = [
                    str(item) for item in mapping.get("source_candidate_ids", [])
                ]
                risk = risk_by_type.get(risk_type, {})
                mapping_rows.append(
                    {
                        "合并后风险": str(risk.get("title", risk_type)),
                        "专业角色": _SPECIALIST_LABELS.get(
                            str(risk.get("owner", "")),
                            str(risk.get("owner", "")),
                        ),
                        "原始发现数": len(source_ids),
                        "原始风险": "；".join(
                            str(raw_by_id.get(item, {}).get("title", item))
                            for item in source_ids
                        ),
                    }
                )
            if mapping_rows:
                st.dataframe(
                    mapping_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "原始风险": st.column_config.TextColumn(width="large"),
                    },
                )
            st.caption(
                "全局合并只建立主题映射，不删除原始发现；每条原始风险的去向均可追踪。"
            )
        for role, label in _SPECIALIST_LABELS.items():
            trace = _trace_by_role(report, role)
            with st.container(border=True):
                row = st.container(
                    horizontal=True,
                    horizontal_alignment="distribute",
                    vertical_alignment="center",
                )
                row.markdown(f"**{label}**")
                row.badge(
                    "已完成" if trace and trace.status.value == "succeeded" else "未完成",
                    color=(
                        "green"
                        if trace and trace.status.value == "succeeded"
                        else "orange"
                    ),
                )
                st.caption(
                    trace.decision
                    if trace is not None
                    else "本次运行没有保存该专业 Agent 的轨迹。"
                )
        st.markdown("**完整审计轨迹**")
        for trace in report.agent_traces:
            st.markdown(f"**{trace.agent.value} Agent**")
            st.caption(
                f"{trace.status.value} · 工具调用 {len(trace.tool_calls)} 次"
            )
            st.write(trace.decision)


def render_results_view(service: ReviewService) -> None:
    """展示可消费的最终风险结果，并把运行过程收进审计区。"""
    deleted_run = st.session_state.pop("analysis_deleted_run", None)
    if deleted_run:
        _clear_deleted_run_state(str(deleted_run))
    theme.page_header(
        "02 / ANALYSIS RESULT",
        "招股书风险分析结果",
        "先查看机器分析与证据链，再进入人工复核确认正式结论。",
    )
    delete_flash = st.session_state.pop("analysis_delete_flash", None)
    if delete_flash:
        st.success(str(delete_flash), icon=":material/delete_sweep:")
    report = restore_latest_analysis_state(service)
    runs = service.list_runs()
    if runs:
        run_by_id = {run.run_id: run for run in runs}
        selected_run = str(
            st.session_state.get("results_selected_run")
            or st.session_state.get("last_analysis_run_id")
            or ""
        )
        if selected_run not in run_by_id:
            selected_run = runs[0].run_id
            st.session_state["results_selected_run"] = selected_run
        selector_column, delete_column = st.columns(
            [4.6, 1.15],
            gap="small",
            vertical_alignment="bottom",
        )
        with selector_column:
            selected_run = st.selectbox(
                "分析运行",
                options=tuple(run_by_id),
                format_func=lambda run_id: (
                    f"{run_by_id[run_id].company_name} · "
                    f"{run_by_id[run_id].as_of_date} · {run_id}"
                ),
                key="results_selected_run",
            )
        with delete_column:
            if st.button(
                "删除本次分析",
                icon=":material/delete:",
                width="stretch",
            ):
                st.session_state["analysis_delete_target"] = selected_run
        selected_summary = run_by_id[selected_run]
        selected_report = _load_report_reference(
            service,
            selected_summary.report_path,
        )
        if selected_report is not None and (
            report is None or selected_report.run_id != report.run_id
        ):
            report = selected_report
            st.session_state["last_analysis_run_id"] = report.run_id
            st.session_state["last_analysis_report_path"] = str(
                selected_summary.report_path
            )
            st.session_state["last_analysis_result_summary"] = {
                "run_id": report.run_id,
                "finding_count": report.summary.finding_count,
                "evidence_count": report.summary.evidence_count,
                "trace_count": len(report.agent_traces),
            }
            st.session_state["last_analysis_snapshot"] = (
                analysis_snapshot_from_report(report)
            )
            st.session_state["last_analysis_snapshot_run_id"] = report.run_id
        delete_target = st.session_state.get("analysis_delete_target")
        if delete_target in run_by_id:
            target_summary = run_by_id[str(delete_target)]
            _render_delete_dialog(
                service,
                target_summary.run_id,
                target_summary.company_name,
            )
        elif delete_target:
            st.session_state.pop("analysis_delete_target", None)
    fallback = st.session_state.get("last_analysis_result_summary", {})
    run_id = report.run_id if report else str(fallback.get("run_id", ""))
    if not run_id:
        st.info("还没有可展示的分析结果，请先上传一份招股书。")
        return

    if report is not None:
        selected_summary = next(
            (run for run in runs if run.run_id == report.run_id),
            None,
        )
        pending_findings = (
            selected_summary.pending_findings
            if selected_summary is not None
            else report.summary.finding_count
        )
        reviewed_findings = (
            selected_summary.reviewed_findings
            if selected_summary is not None
            else 0
        )
        verified_findings = (
            selected_summary.verified_findings
            if selected_summary is not None
            else 0
        )
        rejected_findings = (
            selected_summary.rejected_findings
            if selected_summary is not None
            else 0
        )
        st.caption(
            f"运行 {report.run_id} · {report.company_name} · "
            f"报告生成于 {report.generated_at:%Y-%m-%d %H:%M}"
        )
        _render_result_overview(
            report,
            reviewed_findings=reviewed_findings,
            pending_findings=pending_findings,
        )
        _render_decision_summary(
            report,
            reviewed_findings=reviewed_findings,
            pending_findings=pending_findings,
        )
        findings_column, action_column = st.columns([2.25, 1], gap="large")
        with findings_column:
            if report.risk_findings:
                _render_findings(report)
            else:
                st.info(
                    "本次没有形成满足证据约束的候选风险。",
                    icon=":material/info:",
                )
        with action_column:
            _render_review_panel(
                run_id=report.run_id,
                total_findings=report.summary.finding_count,
                pending_findings=pending_findings,
                verified_findings=verified_findings,
                rejected_findings=rejected_findings,
            )
            with st.container(border=True):
                st.subheader("数据口径")
                st.caption(
                    "风险卡片是决策阶段保留的候选结论；人工确认前不作为正式报告结论。"
                )
                st.caption(
                    f"源文件：{Path(report.source_document.source_file).name}"
                )
                st.caption(
                    f"PDF {report.source_document.page_count} 页 · "
                    f"全文切片 {report.source_document.chunk_count} 个"
                )
        _render_agent_audit(report)
    else:
        finding_count = int(fallback.get("finding_count", 0))
        evidence_count = int(fallback.get("evidence_count", 0))
        trace_count = int(fallback.get("trace_count", 0))
        with st.container(horizontal=True):
            st.metric("候选风险", finding_count, border=True)
            st.metric("原文证据", evidence_count, border=True)
            st.metric("Agent轨迹", trace_count, border=True)
        st.warning(
            "结果摘要仍在，但完整报告文件暂时不可读取。",
            icon=":material/warning:",
        )


__all__ = [
    "analysis_snapshot_from_report",
    "render_agent_snapshot",
    "render_results_view",
    "restore_latest_analysis_state",
]
