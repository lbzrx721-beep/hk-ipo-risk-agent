"""招股书上传分析与人工复核的一体化Streamlit工作台。"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

import streamlit as st

from ipo_risk.agents.intervention import (
    HumanAddedRisk,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
    RiskScopeSelection,
)
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.reviewing import ReviewService
from ipo_risk.ui import theme
from ipo_risk.ui.analysis_job import AnalysisJob, AnalysisJobSnapshot
from ipo_risk.ui.analysis_progress import (
    build_analysis_progress_snapshot,
    render_analysis_progress_snapshot,
)
from ipo_risk.ui.routes import (
    ANALYSIS_PAGE_PATH,
    ANALYSIS_URL_PATH,
    RESULTS_PAGE_PATH,
    RESULTS_URL_PATH,
    REVIEW_PAGE_PATH,
    REVIEW_URL_PATH,
    bind_page_routes,
    switch_to_results,
    switch_to_review,
)
from ipo_risk.uploads import ProspectusUploadService

if TYPE_CHECKING:
    from ipo_risk.pipeline import AnalysisArtifacts

_ROLE_LABELS = {
    AgentRole.FINANCE: "财务 Agent",
    AgentRole.LEGAL: "法律 Agent",
    AgentRole.BUSINESS: "业务 Agent",
    AgentRole.MARKET: "市场 Agent",
}
_ROLE_BY_LABEL = {label: role for role, label in _ROLE_LABELS.items()}
_SEVERITY_LABELS = {
    RiskSeverity.CRITICAL: "极高",
    RiskSeverity.HIGH: "高",
    RiskSeverity.MEDIUM: "中",
    RiskSeverity.LOW: "低",
}
_SEVERITY_BY_LABEL = {label: severity for severity, label in _SEVERITY_LABELS.items()}


AnalysisRunner = Callable[..., "AnalysisArtifacts"]

_CURRENT_ANALYSIS_KEYS = (
    "current_analysis_job",
    "current_analysis_job_synced",
    "current_analysis_file_name",
    "current_analysis_run_id",
    "current_analysis_snapshot",
    "current_analysis_result_summary",
    "current_analysis_status",
    "current_analysis_error",
)
_UPLOAD_METADATA_KEYS = (
    "inspected_upload_hash",
    "upload_company_id",
    "upload_company_name",
    "upload_document_date",
    "upload_metadata_source",
    "upload_page_count",
    "analysis_field_error",
    "analysis_pending",
)
_UPLOAD_SERVICE_STATE = "_workspace_upload_service"
_REVIEW_SERVICE_STATE = "_workspace_review_service"
_ANALYSIS_RUNNER_STATE = "_workspace_analysis_runner"


def _workspace_context() -> tuple[
    ProspectusUploadService,
    ReviewService,
    AnalysisRunner,
]:
    """读取入口在当前会话中注入的共享服务。"""
    upload_service = st.session_state.get(_UPLOAD_SERVICE_STATE)
    review_service = st.session_state.get(_REVIEW_SERVICE_STATE)
    analyze = st.session_state.get(_ANALYSIS_RUNNER_STATE)
    if not isinstance(upload_service, ProspectusUploadService):
        raise RuntimeError("workspace upload service is unavailable")
    if not isinstance(review_service, ReviewService):
        raise RuntimeError("workspace review service is unavailable")
    if not callable(analyze):
        raise RuntimeError("workspace analysis runner is unavailable")
    return upload_service, review_service, cast(AnalysisRunner, analyze)


def _default_analysis_runner(*args: Any, **kwargs: Any) -> AnalysisArtifacts:
    """用户真正开始分析时才加载完整流水线。"""
    from ipo_risk.pipeline import analyze_company

    return analyze_company(*args, **kwargs)


def _reset_current_analysis_for_upload(upload_hash: str) -> None:
    """仅在上传内容改变时，把新建分析页切换到新的当前任务。"""
    active_hash = st.session_state.get("current_analysis_upload_hash")
    if active_hash == upload_hash:
        return
    for key in (*_CURRENT_ANALYSIS_KEYS, *_UPLOAD_METADATA_KEYS):
        st.session_state.pop(key, None)
    st.session_state["current_analysis_upload_hash"] = upload_hash


def _render_current_analysis() -> None:
    """持续展示当前会话中的本次分析，不加载历史运行。"""
    snapshot = st.session_state.get("current_analysis_snapshot")
    if not isinstance(snapshot, dict):
        return
    from ipo_risk.ui.results_view import render_agent_snapshot

    run_id = str(st.session_state.get("current_analysis_run_id", ""))
    status = str(st.session_state.get("current_analysis_status", "completed"))
    with st.container(border=True):
        heading = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        heading.subheader("本次分析进度")
        badge_options: dict[str, tuple[str, theme.BadgeColor, str]] = {
            "completed": ("分析完成", "green", ":material/check_circle:"),
            "paused": ("安全暂停", "orange", ":material/pause_circle:"),
        }
        badge = badge_options.get(
            status,
            ("分析失败", "red", ":material/error:"),
        )
        heading.badge(badge[0], color=badge[1], icon=badge[2])
        percent = int(snapshot.get("max_percent", 100 if status == "completed" else 0))
        st.progress(
            max(0, min(percent, 100)),
            text="本次任务已完成" if status == "completed" else "本次任务未完成",
        )
        if run_id:
            st.caption(f"运行编号：{run_id}")
        error = st.session_state.get("current_analysis_error")
        if error and status == "paused":
            st.warning(f"分析已安全暂停：{error}")
        elif error:
            st.error(f"分析未完成：{error}")

    _render_analysis_summary(run_id)
    render_agent_snapshot(
        snapshot,
        title="本次 Agent 协同状态",
        caption="当前任务的角色状态与模型结构化输出会保留到上传另一份PDF。",
        event_label="查看本次分析事件流",
    )


def _render_analysis_summary(run_id: str) -> None:
    summary = st.session_state.get("current_analysis_result_summary")
    if not isinstance(summary, dict):
        return
    st.success(f"本次分析已完成：{run_id}")
    persistence_status = summary.get("vector_persistence_status")
    if persistence_status == "building":
        st.caption("知识库正在后台同步；不影响查看和复核本次报告。")
    elif persistence_status == "ready":
        st.caption("知识库已同步，可供后续跨文档检索复用。")
    elif persistence_status == "failed":
        st.warning("本次报告和内存向量检索均可正常使用；仅Chroma知识库后台同步失败，可稍后重试。")
    with st.container(horizontal=True):
        st.metric("候选风险", int(summary.get("finding_count", 0)), border=True)
        st.metric("原文证据", int(summary.get("evidence_count", 0)), border=True)
        st.metric("Agent轨迹", int(summary.get("trace_count", 0)), border=True)
    usage = summary.get("model_usage")
    if isinstance(usage, dict):
        estimated_cost = usage.get("estimated_cost_rmb")
        cost_text = (
            f"¥{float(estimated_cost):.4f}"
            if isinstance(estimated_cost, (int, float))
            else "未配置单价"
        )
        st.caption(
            "模型用量 · "
            f"{int(usage.get('calls') or 0)}次调用 · "
            f"输入{int(usage.get('input_tokens') or 0):,} Token · "
            f"输出{int(usage.get('output_tokens') or 0):,} Token · "
            f"估算费用{cost_text}"
        )
    with st.container(horizontal=True):
        if st.button(
            "查看分析结果",
            type="primary",
            icon=":material/assessment:",
        ):
            switch_to_results()
        if st.button(
            "进入人工复核",
            icon=":material/fact_check:",
        ):
            switch_to_review()


def _sync_analysis_job(snapshot: AnalysisJobSnapshot, progress_snapshot: dict[str, object]) -> None:
    """在页面线程中把后台任务结果同步到当前会话和历史结果入口。"""
    if st.session_state.get("current_analysis_job_synced"):
        if snapshot.status == "completed" and snapshot.vector_persistence is not None:
            summary = st.session_state.get("current_analysis_result_summary")
            if isinstance(summary, dict):
                summary["vector_persistence_status"] = snapshot.vector_persistence.status
        return
    if snapshot.status in {"failed", "paused"}:
        st.session_state["current_analysis_snapshot"] = progress_snapshot
        st.session_state["current_analysis_status"] = snapshot.status
        st.session_state["current_analysis_error"] = snapshot.error or "未知错误"
        st.session_state["current_analysis_job_synced"] = True
        return
    if snapshot.status != "completed" or snapshot.artifacts is None:
        return

    artifacts = snapshot.artifacts
    model_usage = getattr(artifacts, "model_usage", None)
    summary = {
        "run_id": artifacts.report.run_id,
        "finding_count": artifacts.report.summary.finding_count,
        "evidence_count": artifacts.report.summary.evidence_count,
        "trace_count": len(artifacts.report.agent_traces),
        "vector_persistence_status": (
            snapshot.vector_persistence.status if snapshot.vector_persistence is not None else None
        ),
        "model_usage": (model_usage.as_dict() if model_usage is not None else None),
    }
    st.session_state["last_analysis_run_id"] = artifacts.report.run_id
    st.session_state["last_analysis_report_path"] = str(artifacts.report_path)
    st.session_state["last_analysis_result_summary"] = summary
    st.session_state["last_analysis_snapshot"] = progress_snapshot
    st.session_state["last_analysis_snapshot_run_id"] = artifacts.report.run_id
    st.session_state["current_analysis_run_id"] = artifacts.report.run_id
    st.session_state["current_analysis_snapshot"] = progress_snapshot
    st.session_state["current_analysis_result_summary"] = dict(summary)
    st.session_state["current_analysis_status"] = "completed"
    st.session_state.pop("current_analysis_error", None)
    st.session_state["results_selected_run"] = artifacts.report.run_id
    st.session_state["current_analysis_job_synced"] = True


def _render_analysis_job_state(
    job: AnalysisJob,
    snapshot: AnalysisJobSnapshot | None = None,
) -> AnalysisJobSnapshot:
    """从线程安全事件构建一次性快照并绘制当前实时控制台。"""
    snapshot = snapshot or job.snapshot()
    file_name = st.session_state.get("current_analysis_file_name")
    if file_name:
        st.caption(f"当前任务：{file_name} · 切换工作区不会停止分析")
    progress_snapshot = build_analysis_progress_snapshot(
        snapshot.events,
        status=snapshot.status,
        error=snapshot.error,
    )
    render_analysis_progress_snapshot(progress_snapshot, status=snapshot.status)
    _sync_analysis_job(snapshot, progress_snapshot)
    if snapshot.status == "completed" and snapshot.artifacts is not None:
        _render_analysis_summary(snapshot.artifacts.report.run_id)
    elif snapshot.status == "failed":
        st.error(f"分析未完成：{snapshot.error or '未知错误'}")
    elif snapshot.status == "paused":
        st.warning(
            f"分析已按预算安全暂停，已发现的风险与证据不会被删除。\n\n{snapshot.error or ''}"
        )
    return snapshot


def _render_risk_scope_review(
    job: AnalysisJob,
    request: RiskScopeReviewRequest,
) -> None:
    """渲染AI发现完成后的风险范围确认门。"""
    import pandas as pd

    def cell_text(value: object) -> str:
        return "" if pd.isna(value) else str(value).strip()

    theme.section_label("人工干预节点")
    with st.container(border=True):
        heading = st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        )
        heading.subheader("确认本次分析范围")
        heading.badge(
            "等待人工确认",
            color="orange",
            icon=":material/fact_check:",
        )
        st.caption(
            "AI已完成向量检索原文、风险发现与业务适用性判断。请确认分析边界；"
            "提交前，财务、法律、业务和市场 Agent 均不会启动。"
        )
        profile_columns = st.columns(3)
        profile_columns[0].metric("所属行业", request.industry, border=True)
        profile_columns[1].metric("业务模式", request.business_model, border=True)
        profile_columns[2].metric("收入模式", request.revenue_model, border=True)

    funnel = request.funnel
    if funnel.raw_candidates or funnel.consolidated_candidates or funnel.catalog_added:
        with st.container(border=True):
            st.markdown("**这批候选是怎样形成的**")
            st.caption(
                "先保留各风险原文包的广泛发现，再由大模型跨包合并同一底层风险；"
                "业务适用性排除不会删除审计记录。"
            )
            funnel_columns = st.columns(4)
            funnel_columns[0].metric(
                "批次原始发现",
                funnel.raw_candidates,
                help="全文各批次输出之和，包含跨章节重复表达。",
            )
            funnel_columns[1].metric(
                "全局语义合并",
                funnel.consolidated_candidates,
                help="由大模型合并同一底层风险后的动态候选。",
            )
            funnel_columns[2].metric(
                "目录兜底补充",
                funnel.catalog_added,
                help="仅在风险原文包失败或没有动态发现时补充，已匹配项不会重复加入。",
            )
            funnel_columns[3].metric(
                "AI建议纳入",
                funnel.scoped_in,
                delta=(f"排除 {funnel.scoped_out}" if funnel.scoped_out else None),
                delta_color="off",
            )
            detail_parts = [f"风险原文包 {funnel.document_batches} 个"]
            if funnel.catalog_merged:
                detail_parts.append(f"{funnel.catalog_merged} 条动态风险已与目录项匹配去重")
            detail_parts.append(
                (
                    "全局合并：大模型语义判断"
                    if funnel.consolidation_mode == "semantic_model"
                    else "全局合并：精确标题兜底（没有截断候选）"
                )
            )
            st.caption("；".join(detail_parts))
            if funnel.consolidation_error:
                st.warning(
                    "本次全局语义合并调用失败，系统已保留全部候选并使用精确标题"
                    "兜底；建议在表格中重点检查同义重复项。",
                    icon=":material/warning:",
                )

    role_options = list(_ROLE_BY_LABEL)
    severity_options = list(_SEVERITY_BY_LABEL)
    catalog_fallback_count = sum(
        candidate.source == "catalog_fallback" for candidate in request.candidates
    )
    if catalog_fallback_count:
        st.warning(
            f"其中 {catalog_fallback_count} 项来自风险目录兜底，不是 AI 从本次"
            "招股书中发现的结论，也尚无原文证据。系统默认不纳入；如需核验，"
            "请人工勾选后交给专业 Agent 检索原文。",
            icon=":material/library_books:",
        )

    candidate_rows = [
        {
            "风险编号": candidate.risk_type,
            "纳入分析": candidate.ai_recommended,
            "风险名称": candidate.title,
            "候选来源": (
                f"全文发现 · 合并{candidate.source_candidate_count}条"
                if candidate.source == "document_discovery"
                else "目录兜底 · 待人工"
            ),
            "置信度": candidate.confidence,
            "证据线索": candidate.evidence_count,
            "负责 Agent": _ROLE_LABELS[candidate.owner],
            "严重程度": _SEVERITY_LABELS[candidate.severity],
            "AI建议": (
                "建议纳入"
                if candidate.ai_recommended
                else ("待人工选择" if candidate.source == "catalog_fallback" else "建议排除")
            ),
            "AI判断理由": candidate.rationale,
            "核验重点": candidate.review_question,
            "补充分析要求": "",
        }
        for candidate in request.candidates
    ]
    form_key = f"risk_scope_review_form_{request.run_id}"
    with st.form(form_key, border=True):
        st.subheader("候选风险清单")
        st.caption("直接勾选取舍，也可以调整负责 Agent、严重程度，并给单条风险补充核验要求。")
        edited_candidates = st.data_editor(
            pd.DataFrame(candidate_rows),
            key=f"risk_scope_candidates_{request.run_id}",
            hide_index=True,
            width="stretch",
            height=min(620, max(250, 82 + len(candidate_rows) * 38)),
            disabled=(
                "风险编号",
                "风险名称",
                "候选来源",
                "置信度",
                "证据线索",
                "AI建议",
                "AI判断理由",
                "核验重点",
            ),
            column_config={
                "风险编号": None,
                "纳入分析": st.column_config.CheckboxColumn(
                    "纳入",
                    help="取消勾选后，该风险不会进入专业 Agent 分析。",
                    width="small",
                ),
                "风险名称": st.column_config.TextColumn("风险名称", width="medium"),
                "候选来源": st.column_config.TextColumn(
                    "候选来源",
                    width="small",
                    help="动态风险会显示由多少条跨批次原始候选合并而来。",
                ),
                "置信度": st.column_config.NumberColumn(
                    "发现置信度",
                    format="percent",
                    width="small",
                ),
                "证据线索": st.column_config.NumberColumn(
                    "种子证据",
                    format="%d 条",
                    width="small",
                ),
                "负责 Agent": st.column_config.SelectboxColumn(
                    "负责 Agent",
                    options=role_options,
                    required=True,
                    width="small",
                ),
                "严重程度": st.column_config.SelectboxColumn(
                    "严重度",
                    options=severity_options,
                    required=True,
                    width="small",
                ),
                "AI建议": st.column_config.TextColumn("AI建议", width="small"),
                "AI判断理由": st.column_config.TextColumn(
                    "AI判断理由",
                    width="large",
                ),
                "核验重点": st.column_config.TextColumn(
                    "核验重点",
                    width="large",
                ),
                "补充分析要求": st.column_config.TextColumn(
                    "补充分析要求",
                    help="例如：重点核验2024年后客户集中度是否继续上升。",
                    width="large",
                ),
            },
        )

        with st.expander("新增 AI 未覆盖的风险", icon=":material/add_circle:"):
            st.caption(
                "如AI漏掉了重要方向，可直接补一行。分析要求要写清楚希望 Agent 在原文中核验什么。"
            )
            added_rows = st.data_editor(
                pd.DataFrame(
                    [
                        {
                            "风险名称": "",
                            "负责 Agent": _ROLE_LABELS[AgentRole.BUSINESS],
                            "严重程度": _SEVERITY_LABELS[RiskSeverity.MEDIUM],
                            "分析要求": "",
                        }
                    ]
                ),
                key=f"risk_scope_added_{request.run_id}",
                hide_index=True,
                width="stretch",
                num_rows="dynamic",
                column_config={
                    "风险名称": st.column_config.TextColumn(
                        "风险名称",
                        required=False,
                        width="medium",
                    ),
                    "负责 Agent": st.column_config.SelectboxColumn(
                        "负责 Agent",
                        options=role_options,
                        required=True,
                        width="small",
                    ),
                    "严重程度": st.column_config.SelectboxColumn(
                        "严重度",
                        options=severity_options,
                        required=True,
                        width="small",
                    ),
                    "分析要求": st.column_config.TextColumn(
                        "分析要求",
                        width="large",
                    ),
                },
            )

        requirement_columns = st.columns([1, 2])
        with requirement_columns[0]:
            reviewer = st.text_input(
                "确认人",
                key=f"risk_scope_reviewer_{request.run_id}",
                placeholder="填写姓名或工号",
            )
        with requirement_columns[1]:
            global_requirement = st.text_area(
                "本次统一分析要求（可选）",
                key=f"risk_scope_global_requirement_{request.run_id}",
                placeholder="例如：所有结论必须优先引用最近两个报告期的数据，并说明变化趋势。",
                height=90,
            )
        submitted = st.form_submit_button(
            "确认范围并启动专业 Agent",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    if not submitted:
        return
    reviewer_name = reviewer.strip()
    if not reviewer_name:
        st.error("请填写确认人，便于后续追踪这次人工干预。")
        return

    try:
        selections = tuple(
            RiskScopeSelection(
                risk_type=str(row["风险编号"]),
                included=bool(row["纳入分析"]),
                owner=_ROLE_BY_LABEL[str(row["负责 Agent"])],
                severity=_SEVERITY_BY_LABEL[str(row["严重程度"])],
                analysis_requirement=cell_text(row["补充分析要求"]),
            )
            for row in edited_candidates.to_dict(orient="records")
        )
        added_risks: list[HumanAddedRisk] = []
        for row in added_rows.to_dict(orient="records"):
            title = cell_text(row["风险名称"])
            requirement = cell_text(row["分析要求"])
            if not title and not requirement:
                continue
            if not title or not requirement:
                raise ValueError("新增风险必须同时填写风险名称和分析要求。")
            added_risks.append(
                HumanAddedRisk(
                    title=title,
                    owner=_ROLE_BY_LABEL[str(row["负责 Agent"])],
                    severity=_SEVERITY_BY_LABEL[str(row["严重程度"])],
                    analysis_requirement=requirement,
                )
            )
        decision = RiskScopeReviewDecision(
            run_id=request.run_id,
            reviewer=reviewer_name,
            selections=selections,
            added_risks=tuple(added_risks),
            global_requirement=global_requirement.strip(),
        )
        job.submit_risk_scope_review(decision)
        st.session_state["current_analysis_status"] = "running"
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"无法提交风险范围：{exc}")
        return

    st.toast("风险范围已确认，专业 Agent 正在并行启动。", icon=":material/hub:")
    st.rerun()


@st.fragment(run_every="1s")
def _render_running_analysis_job(job: AnalysisJob) -> None:
    snapshot = job.snapshot()
    persistence_is_running = (
        snapshot.vector_persistence is not None and snapshot.vector_persistence.status == "building"
    )
    if snapshot.status in {"awaiting_review", "failed", "paused"} or (
        snapshot.status == "completed" and not persistence_is_running
    ):
        st.rerun()
        return
    _render_analysis_job_state(job, snapshot)


def _render_analysis_job(job: AnalysisJob) -> None:
    snapshot = job.snapshot()
    persistence_is_running = (
        snapshot.vector_persistence is not None and snapshot.vector_persistence.status == "building"
    )
    if snapshot.status in {"pending", "running"} or persistence_is_running:
        _render_running_analysis_job(job)
    elif snapshot.status == "awaiting_review" and snapshot.risk_scope_review is not None:
        # 人工表单脱离自动刷新 fragment 渲染，编辑期间不会被定时重建。
        st.session_state["current_analysis_status"] = "awaiting_review"
        _render_analysis_job_state(job)
        _render_risk_scope_review(job, snapshot.risk_scope_review)
    else:
        _render_analysis_job_state(job)


def _render_upload_page(
    *,
    upload_service: ProspectusUploadService,
    analyze: AnalysisRunner,
) -> None:
    theme.page_header(
        "01 / PROSPECTUS INTAKE",
        "拖入招股书，开始风险分析",
        "系统会依次完成文件校验、资料登记、PDF解析、语义检索、专业Agent分析、"
        "决策Agent裁决和证据报告生成。",
    )

    config = upload_service.config
    current_job = st.session_state.get("current_analysis_job")
    if not isinstance(current_job, AnalysisJob):
        current_job = None
    current_job_status = current_job.snapshot().status if current_job is not None else None
    job_is_running = current_job is not None and current_job_status in {
        "pending",
        "running",
        "awaiting_review",
    }
    intake_column, engine_column = st.columns([1.7, 1], gap="large")
    with intake_column:
        with st.container(border=True, height=285):
            st.subheader("招股书资料")
            st.caption("上传一份正式招股说明书，系统会先读取封面并自动补全公司信息。")
            uploaded = st.file_uploader(
                "拖入招股说明书PDF",
                type=("pdf",),
                max_upload_size=250,
                help="单个PDF最大250MB；带密码或损坏文件会被拒绝。",
                disabled=job_is_running,
            )
    with engine_column:
        with st.container(border=True, height=285):
            st.subheader("分析引擎")
            model_status = st.container(horizontal=True)
            model_status.badge(
                "向量模型就绪" if config.embedding.is_configured else "向量模型未配置",
                color="green" if config.embedding.is_configured else "orange",
                icon=":material/database:",
            )
            model_status.badge(
                "DeepSeek就绪" if config.llm.is_configured else "DeepSeek未配置",
                color="green" if config.llm.is_configured else "orange",
                icon=":material/psychology:",
            )
            st.markdown("**分析链路**")
            st.caption("向量检索原文 → 风险发现 → 专业Agent核验 → 决策裁决 → 证据报告")
            if not config.embedding.is_configured or not config.llm.is_configured:
                st.caption("未配置的模型会重试或降级，但仍禁止生成无原文证据的结论。")
    if uploaded is None:
        if current_job is not None:
            _render_analysis_job(current_job)
        else:
            st.caption("等待资料输入 · 股票代码、公司名称和招股书日期将在上传后自动识别。")
            _render_current_analysis()
        return

    content = uploaded.getvalue()
    upload_hash = sha256(content).hexdigest()
    _reset_current_analysis_for_upload(upload_hash)
    if st.session_state.get("current_analysis_job") is not current_job:
        current_job = None
        job_is_running = False
    if st.session_state.get("inspected_upload_hash") != upload_hash:
        try:
            suggestion = upload_service.inspect(
                file_name=uploaded.name,
                content=content,
            )
        except Exception as exc:
            st.error(f"无法读取这份招股书：{exc}")
            return
        st.session_state["inspected_upload_hash"] = suggestion.source_hash
        st.session_state["upload_company_id"] = suggestion.company_id or ""
        st.session_state["upload_company_name"] = suggestion.company_name or ""
        st.session_state["upload_document_date"] = suggestion.document_date
        st.session_state["upload_metadata_source"] = suggestion.source
        st.session_state["upload_page_count"] = suggestion.page_count

    metadata_source = st.session_state.get("upload_metadata_source")
    with st.container(border=True):
        document_status = st.container(horizontal=True)
        if metadata_source == "registered_document":
            document_status.badge(
                "已匹配登记资料",
                color="green",
                icon=":material/verified:",
            )
            st.caption("已根据文件指纹自动填写基础信息。")
        elif metadata_source == "cover_text":
            document_status.badge(
                "封面识别完成",
                color="green",
                icon=":material/document_scanner:",
            )
            st.caption("请确认自动识别字段后开始分析。")
        else:
            document_status.badge(
                "需要补充信息",
                color="orange",
                icon=":material/edit_note:",
            )
            st.caption("部分字段未能自动识别，请补充或修正。")
        document_status.badge(
            f"{st.session_state.get('upload_page_count', 0)} 页",
            color="blue",
            icon=":material/description:",
        )

    with st.container(border=True):
        st.subheader("确认公司与文档信息")
        st.caption("这些字段会进入分析报告和后续人工复核记录。")
        fields = st.columns(3)
        fields[0].text_input(
            "股票代码",
            key="upload_company_id",
            placeholder="例如：03696",
            disabled=job_is_running,
        )
        fields[1].text_input(
            "公司名称",
            key="upload_company_name",
            placeholder="例如：英矽智能",
            disabled=job_is_running,
        )
        fields[2].date_input(
            "招股书日期",
            key="upload_document_date",
            disabled=job_is_running,
        )

        if job_is_running:
            st.button(
                (
                    "请在下方确认风险范围"
                    if current_job_status == "awaiting_review"
                    else "正在分析，请稍候..."
                ),
                type="primary",
                icon=(
                    ":material/fact_check:"
                    if current_job_status == "awaiting_review"
                    else ":material/hourglass_top:"
                ),
                width="stretch",
                disabled=True,
            )
        else:
            clicked = st.button(
                "开始分析",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )
            if clicked:
                # 校验字段后再触发 rerun，避免空字段进入分析
                if (
                    not st.session_state.get("upload_company_id", "").strip()
                    or not st.session_state.get("upload_company_name", "").strip()
                    or st.session_state.get("upload_document_date") is None
                ):
                    st.session_state["analysis_field_error"] = True
                    st.rerun()
                st.session_state.pop("analysis_field_error", None)
                job = AnalysisJob(
                    upload_service=upload_service,
                    analyze=analyze,
                    file_name=uploaded.name,
                    content=content,
                    company_id=str(st.session_state["upload_company_id"]),
                    company_name=str(st.session_state["upload_company_name"]),
                    document_date=st.session_state["upload_document_date"],
                )
                st.session_state["current_analysis_job"] = job
                st.session_state["current_analysis_job_synced"] = False
                st.session_state["current_analysis_file_name"] = uploaded.name
                st.session_state["current_analysis_status"] = "running"
                job.start()
                st.rerun()

    # 字段校验提示（在 rerun 后渲染）
    if st.session_state.get("analysis_field_error"):
        st.error("请补全股票代码、公司名称和招股书日期。")

    if current_job is not None:
        _render_analysis_job(current_job)
    else:
        _render_current_analysis()


def render_workspace_app(
    *,
    upload_service: ProspectusUploadService,
    review_service: ReviewService,
    analyze: AnalysisRunner = _default_analysis_runner,
) -> None:
    """注册三个独立URL路由并渲染统一工作台外框。"""
    st.set_page_config(
        page_title="IPO风险分析工作台",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    theme.inject_theme()
    st.session_state[_UPLOAD_SERVICE_STATE] = upload_service
    st.session_state[_REVIEW_SERVICE_STATE] = review_service
    st.session_state[_ANALYSIS_RUNNER_STATE] = analyze
    analysis_page = st.Page(
        ANALYSIS_PAGE_PATH,
        title="新建分析",
        icon=":material/upload_file:",
        url_path=ANALYSIS_URL_PATH,
        default=True,
    )
    results_page = st.Page(
        RESULTS_PAGE_PATH,
        title="分析结果",
        icon=":material/assessment:",
        url_path=RESULTS_URL_PATH,
    )
    review_page = st.Page(
        REVIEW_PAGE_PATH,
        title="人工复核",
        icon=":material/fact_check:",
        url_path=REVIEW_URL_PATH,
    )
    bind_page_routes(
        analysis=analysis_page,
        results=results_page,
        review=review_page,
    )
    page = st.navigation(
        [analysis_page, results_page, review_page],
        position="top",
    )
    st.caption("HK IPO · AGENTIC RISK INTELLIGENCE")
    st.title("IPO风险分析工作台")
    st.caption("招股书全文发现、证据核验、协同裁决与人工复核的一体化工作区")
    page.run()


def render_analysis_route() -> None:
    upload_service, _review_service, analyze = _workspace_context()
    _render_upload_page(upload_service=upload_service, analyze=analyze)


def render_results_route() -> None:
    from ipo_risk.ui.results_view import render_results_view

    _upload_service, review_service, _analyze = _workspace_context()
    render_results_view(review_service)


def render_review_route() -> None:
    from ipo_risk.ui.review_app import render_review_app

    _upload_service, review_service, _analyze = _workspace_context()
    render_review_app(review_service, configure_page=False)


__all__ = [
    "AnalysisRunner",
    "render_analysis_route",
    "render_results_route",
    "render_review_route",
    "render_workspace_app",
]
