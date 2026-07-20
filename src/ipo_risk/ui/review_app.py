"""Streamlit人工复核工作台。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ipo_risk.domain import (
    ReviewDecision,
    RiskFinding,
    RiskSeverity,
    VerificationStatus,
)
from ipo_risk.reviewing import (
    IncompleteReviewError,
    ReviewBlockedError,
    ReviewFindingDetail,
    ReviewRunSummary,
    ReviewService,
)


_STATUS_LABELS = {
    VerificationStatus.PROPOSED: "待核验",
    VerificationStatus.PENDING_VERIFICATION: "待人工复核",
    VerificationStatus.VERIFIED: "已确认",
    VerificationStatus.REJECTED: "已驳回",
}
_SEVERITY_LABELS = {
    RiskSeverity.LOW: "低",
    RiskSeverity.MEDIUM: "中",
    RiskSeverity.HIGH: "高",
    RiskSeverity.CRITICAL: "严重",
}


def _resolve_project_path(service: ReviewService, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else service.project_root / path


def _run_label(run: ReviewRunSummary) -> str:
    return f"{run.company_name} · {run.run_id} · {run.reviewed_findings}/{run.total_findings}"


def _finding_label(finding: RiskFinding) -> str:
    return (
        f"{finding.finding_id} · {_STATUS_LABELS[finding.status]} · "
        f"{finding.title}"
    )


def _render_evidence(service: ReviewService, detail: ReviewFindingDetail) -> None:
    st.subheader("证据与PDF定位")
    for evidence in detail.evidences:
        with st.expander(
            f"{evidence.evidence_id} · PDF物理页 {evidence.page}",
            expanded=True,
        ):
            page_label = evidence.printed_page_label or "未识别"
            st.caption(
                f"印刷页码：{page_label} ｜ 坐标：{tuple(evidence.bbox)} ｜ "
                f"证据置信度：{evidence.confidence:.0%}"
            )
            st.markdown(f"> {evidence.quote}")
            st.caption(f"源文件：{evidence.source_file}")
            if evidence.screenshot_path:
                screenshot = _resolve_project_path(service, evidence.screenshot_path)
                if screenshot.is_file():
                    st.image(str(screenshot), caption=f"高亮证据 · {evidence.evidence_id}")
                else:
                    st.warning(f"证据截图不存在：{screenshot}")
            else:
                st.caption("该证据未声明高亮截图。")


def _verification_rows(detail: ReviewFindingDetail) -> list[dict[str, str]]:
    rows = [
        {
            "对象": detail.finding.finding_id,
            "检查": check.name,
            "结果": "通过" if check.passed else "失败",
            "详情": check.detail,
        }
        for check in detail.verification.checks
    ]
    for evidence_result in detail.verification.evidence_verifications:
        rows.extend(
            {
                "对象": evidence_result.evidence_id,
                "检查": check.name,
                "结果": "通过" if check.passed else "失败",
                "详情": check.detail,
            }
            for check in evidence_result.checks
        )
    return rows


def _render_verification(detail: ReviewFindingDetail) -> None:
    st.subheader("确定性核验")
    if detail.verification.passed:
        st.success("文件、页码、坐标、原文、证据引用及公式核验全部通过。")
    else:
        st.error("存在未通过的核验项；系统会阻止确认或编辑为正式结论。")
    st.dataframe(_verification_rows(detail), hide_index=True, width="stretch")

    calculation = detail.finding.calculation
    if calculation is not None:
        st.markdown("**可复算数值结论**")
        st.json(calculation.model_dump(mode="json"), expanded=True)


def _render_history(detail: ReviewFindingDetail) -> None:
    st.subheader("历史审计记录")
    if not detail.reviews:
        st.info("这条风险尚无人工复核记录。")
        return
    rows = [
        {
            "复核时间": review.created_at.isoformat(),
            "决定": review.decision.value,
            "复核人": review.reviewer,
            "意见": review.comment or "",
            "结果状态": review.resulting_finding.status.value,
            "复核ID": review.review_id,
        }
        for review in reversed(detail.reviews)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_agent_traces(detail: ReviewFindingDetail) -> None:
    st.subheader("Agent推理轨迹")
    if not detail.agent_traces:
        st.info("这个历史运行没有保存 Agent 轨迹。")
        return

    st.caption("每个节点展示角色分工、输入输出引用、工具调用、模型模式和失败降级记录。")
    for trace in detail.agent_traces:
        with st.expander(
            f"{trace.agent.value} · {trace.status.value} · {trace.trace_id}",
            expanded=trace.agent.value in {"decision", "orchestrator"},
        ):
            parent = trace.parent_trace_id or "顶层节点"
            st.caption(
                f"父轨迹：{parent} ｜ 开始：{trace.started_at.isoformat()} ｜ "
                f"结束：{trace.ended_at.isoformat()}"
            )
            st.markdown(f"**节点结论**\n\n{trace.decision}")
            st.caption(
                f"输入引用：{', '.join(trace.input_refs) or '无'} ｜ "
                f"输出引用：{', '.join(trace.output_refs) or '无'}"
            )
            if not trace.tool_calls:
                st.caption("该节点没有独立工具调用。")
                continue
            rows: list[dict[str, object]] = []
            for call in trace.tool_calls:
                output = call.output if isinstance(call.output, dict) else {}
                rows.append(
                    {
                        "工具": call.tool_name,
                        "状态": call.status.value,
                        "模式": output.get("mode")
                        or output.get("analysis_mode")
                        or output.get("retrieval_mode")
                        or "",
                        "模型": output.get("model") or "",
                        "尝试次数": output.get("attempts") or 1,
                        "错误/降级": output.get("fallback_error")
                        or output.get("analysis_error")
                        or output.get("retrieval_error")
                        or call.error
                        or "",
                        "调用ID": call.tool_call_id,
                    }
                )
                conflicts = output.get("conflicts")
                if isinstance(conflicts, list):
                    for conflict in conflicts:
                        if not isinstance(conflict, dict):
                            continue
                        st.warning(
                            f"分歧：{conflict.get('detail', '')}\n\n"
                            f"裁决：{conflict.get('resolution', '')}"
                        )
            st.dataframe(rows, hide_index=True, width="stretch")


def _render_review_form(
    service: ReviewService,
    run_id: str,
    detail: ReviewFindingDetail,
) -> None:
    st.subheader("人工复核决定")
    terminal = detail.finding.status in {
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
    }
    if terminal:
        st.info("这条风险已进入终态；历史决定保留在审计记录中。")

    finding = detail.finding
    field_key = f"{run_id}-{finding.finding_id}"
    with st.form(f"review-{field_key}"):
        reviewer = st.text_input("复核人", key="reviewer", placeholder="姓名或账号")
        comment = st.text_area("复核意见", key=f"review-comment-{field_key}")
        with st.expander("编辑风险内容（仅点击“编辑”时生效）"):
            edited_title = st.text_input(
                "标题",
                value=finding.title,
                key=f"edited-title-{field_key}",
            )
            edited_severity = st.selectbox(
                "严重程度",
                options=tuple(RiskSeverity),
                index=tuple(RiskSeverity).index(finding.severity),
                format_func=lambda severity: _SEVERITY_LABELS[severity],
                key=f"edited-severity-{field_key}",
            )
            edited_claim = st.text_area(
                "风险结论",
                value=finding.claim,
                key=f"edited-claim-{field_key}",
            )
            edited_confidence = st.slider(
                "置信度",
                min_value=0.0,
                max_value=1.0,
                value=float(finding.confidence),
                step=0.01,
                key=f"edited-confidence-{field_key}",
            )
        columns = st.columns(3)
        confirmed = columns[0].form_submit_button(
            "确认",
            type="primary",
            disabled=terminal,
            width="stretch",
        )
        rejected = columns[1].form_submit_button(
            "驳回",
            disabled=terminal,
            width="stretch",
        )
        edited = columns[2].form_submit_button(
            "编辑",
            disabled=terminal,
            width="stretch",
        )

    if not (confirmed or rejected or edited):
        return
    if not reviewer.strip():
        st.error("请填写复核人姓名或账号。")
        return

    decision = (
        ReviewDecision.CONFIRMED
        if confirmed
        else ReviewDecision.REJECTED
        if rejected
        else ReviewDecision.EDITED
    )
    edited_finding = None
    if decision is ReviewDecision.EDITED:
        edited_finding = finding.model_copy(
            update={
                "title": edited_title,
                "severity": edited_severity,
                "claim": edited_claim,
                "confidence": edited_confidence,
            }
        )
    try:
        review = service.review_finding(
            run_id=run_id,
            finding_id=finding.finding_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment or None,
            edited_finding=edited_finding,
        )
    except ReviewBlockedError as exc:
        st.error(f"核验未通过，无法确认或编辑：{exc}")
        return
    except (KeyError, ValueError) as exc:
        st.error(str(exc))
        return

    decision_label = {
        ReviewDecision.CONFIRMED: "已确认",
        ReviewDecision.REJECTED: "已驳回",
        ReviewDecision.EDITED: "已编辑并确认",
    }[review.decision]
    st.session_state["review_flash"] = (
        f"{review.finding_id} {decision_label}，审计记录 {review.review_id} 已保存。"
    )
    st.rerun()


def _render_exports(service: ReviewService, run: ReviewRunSummary) -> None:
    st.divider()
    st.subheader("导出")
    st.caption("标注快照保存当前复核进度；正式报告仅包含已确认风险。")
    columns = st.columns(2)
    export_annotations = columns[0].button(
        "导出版本化标注",
        disabled=run.reviewed_findings == 0,
        width="stretch",
        key=f"export-annotations-{run.run_id}",
    )
    export_report = columns[1].button(
        "导出正式报告",
        type="primary",
        disabled=not run.complete,
        width="stretch",
        key=f"export-report-{run.run_id}",
    )

    if not run.complete:
        st.info(f"还有 {run.pending_findings} 条风险待复核，正式报告暂不可导出。")
    elif run.verified_report_path:
        st.caption(f"最近正式报告：{run.verified_report_path}")

    if export_annotations:
        try:
            path = service.export_review_annotations(run.run_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            st.error(str(exc))
            return
        st.session_state["review_flash"] = f"标注快照已导出：{path}"
        st.session_state["last_annotation_export"] = str(path)
        st.rerun()

    if export_report:
        try:
            path = service.export_verified_report(run.run_id)
        except (FileNotFoundError, IncompleteReviewError, KeyError, ValueError) as exc:
            st.error(str(exc))
            return
        st.session_state["review_flash"] = f"正式报告已导出：{path}"
        st.session_state["last_report_export"] = str(path)
        st.rerun()

    for state_key, label in (
        ("last_annotation_export", "下载最近标注快照"),
        ("last_report_export", "下载最近正式报告"),
    ):
        exported = st.session_state.get(state_key)
        if not exported:
            continue
        exported_path = Path(exported)
        if exported_path.is_file():
            st.download_button(
                label,
                data=exported_path.read_bytes(),
                file_name=exported_path.name,
                mime="application/json",
                key=f"download-{state_key}",
            )


def render_review_app(
    service: ReviewService,
    *,
    configure_page: bool = True,
) -> None:
    """渲染一个只通过ReviewService读写业务状态的复核页面。"""
    if configure_page:
        st.set_page_config(
            page_title="IPO风险人工复核",
            page_icon=":material/fact_check:",
            layout="wide",
        )
    st.title("人工复核工作台")
    st.caption("核对机器候选风险的原文证据，作出可审计的人工决定。")
    flash = st.session_state.pop("review_flash", None)
    if flash:
        st.success(flash)

    runs = service.list_runs()
    if not runs:
        st.warning("没有可复核的成功分析运行，请先执行招股书分析。")
        return

    run_by_id = {run.run_id: run for run in runs}
    selected_run_id = st.selectbox(
        "分析 run_id",
        options=tuple(run_by_id),
        format_func=lambda run_id: _run_label(run_by_id[run_id]),
    )
    run = run_by_id[selected_run_id]

    metrics = st.columns(4)
    metrics[0].metric("复核进度", f"{run.reviewed_findings} / {run.total_findings}")
    metrics[1].metric("待复核", run.pending_findings)
    metrics[2].metric("已确认", run.verified_findings)
    metrics[3].metric("已驳回", run.rejected_findings)
    progress = run.reviewed_findings / run.total_findings if run.total_findings else 0.0
    st.progress(progress)

    findings = service.list_findings(selected_run_id)
    finding_by_id = {finding.finding_id: finding for finding in findings}
    selected_finding_id = st.selectbox(
        "风险结论",
        options=tuple(finding_by_id),
        format_func=lambda finding_id: _finding_label(finding_by_id[finding_id]),
    )
    detail = service.get_finding_detail(selected_run_id, selected_finding_id)
    finding = detail.finding

    st.header(finding.title)
    finding_metrics = st.columns(4)
    finding_metrics[0].metric("状态", _STATUS_LABELS[finding.status])
    finding_metrics[1].metric("严重程度", _SEVERITY_LABELS[finding.severity])
    finding_metrics[2].metric("置信度", f"{finding.confidence:.0%}")
    finding_metrics[3].metric("证据数量", len(detail.evidences))
    st.markdown(f"**风险结论**\n\n{finding.claim}")
    if finding.uncertainty:
        st.warning(f"剩余不确定性：{finding.uncertainty}")

    evidence_tab, verification_tab, agent_tab, history_tab = st.tabs(
        ["证据原文", "自动核验", "Agent轨迹", "审计记录"]
    )
    with evidence_tab:
        _render_evidence(service, detail)
    with verification_tab:
        _render_verification(detail)
    with agent_tab:
        _render_agent_traces(detail)
    with history_tab:
        _render_history(detail)

    _render_review_form(service, selected_run_id, detail)
    _render_exports(service, run)


__all__ = ["render_review_app"]
