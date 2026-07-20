"""招股书上传分析与人工复核的一体化Streamlit工作台。"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import streamlit as st

from ipo_risk.pipeline import AnalysisArtifacts, analyze_company
from ipo_risk.reviewing import ReviewService
from ipo_risk.ui.analysis_progress import AnalysisProgressView
from ipo_risk.ui.review_app import render_review_app
from ipo_risk.uploads import ProspectusUploadService


AnalysisRunner = Callable[..., AnalysisArtifacts]


def _open_review() -> None:
    st.session_state["workspace_view"] = "人工复核"


def _render_upload_page(
    *,
    upload_service: ProspectusUploadService,
    analyze: AnalysisRunner,
) -> None:
    st.header("拖入招股书，开始风险分析")
    st.caption(
        "系统会依次完成文件校验、资料登记、PDF解析、语义检索、专业Agent分析、"
        "决策Agent裁决和证据报告生成。"
    )

    config = upload_service.config
    with st.container(border=True):
        model_status = st.container(horizontal=True)
        model_status.badge(
            "向量模型已配置" if config.embedding.is_configured else "向量模型未配置",
            color="green" if config.embedding.is_configured else "orange",
            icon=":material/database:",
        )
        model_status.badge(
            "DeepSeek已配置" if config.llm.is_configured else "DeepSeek未配置",
            color="green" if config.llm.is_configured else "orange",
            icon=":material/psychology:",
        )
        if not config.embedding.is_configured or not config.llm.is_configured:
            st.caption("未配置的模型会按既定策略重试或降级，系统仍禁止生成无原文证据的结论。")

    uploaded = st.file_uploader(
        "拖入招股说明书PDF",
        type=("pdf",),
        max_upload_size=250,
        help="单个PDF最大250MB；带密码或损坏文件会被拒绝。",
    )
    if uploaded is None:
        st.info("请先拖入PDF，系统会自动识别股票代码、公司名称和招股书日期。")
        last_run_id = st.session_state.get("last_analysis_run_id")
        if last_run_id:
            st.success(f"最近完成的分析：{last_run_id}")
            st.button(
                "进入人工复核",
                type="primary",
                icon=":material/fact_check:",
                on_click=_open_review,
            )
        return

    content = uploaded.getvalue()
    upload_hash = sha256(content).hexdigest()
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
    if metadata_source == "registered_document":
        st.success("已根据文件指纹匹配到已登记资料，并自动填写基础信息。")
    elif metadata_source == "cover_text":
        st.success("已从招股书封面自动识别基础信息，请确认后开始分析。")
    else:
        st.warning("部分字段未能自动识别，请补充或修正后开始分析。")
    st.caption(f"已校验PDF：{st.session_state.get('upload_page_count', 0)}页")

    with st.form("new-prospectus-analysis"):
        fields = st.columns(3)
        company_id = fields[0].text_input(
            "股票代码",
            key="upload_company_id",
            placeholder="例如：03696",
        )
        company_name = fields[1].text_input(
            "公司名称",
            key="upload_company_name",
            placeholder="例如：英矽智能",
        )
        document_date = fields[2].date_input(
            "招股书日期",
            key="upload_document_date",
        )
        submitted = st.form_submit_button(
            "开始分析",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    if not submitted:
        last_run_id = st.session_state.get("last_analysis_run_id")
        if last_run_id:
            st.success(f"最近完成的分析：{last_run_id}")
            st.button(
                "进入人工复核",
                type="primary",
                icon=":material/fact_check:",
                on_click=_open_review,
            )
        return
    if not company_id.strip() or not company_name.strip() or document_date is None:
        st.error("请补全股票代码、公司名称和招股书日期。")
        return

    progress_view = AnalysisProgressView()
    try:
        progress_view.emit("upload_started", "正在校验PDF并登记不可变文档版本。")
        registered = upload_service.register(
            file_name=uploaded.name,
            content=content,
            company_id=company_id,
            company_name=company_name,
            document_date=document_date,
        )
        source_action = "复用已登记的同一文件" if registered.reused else "保存新的原始文件"
        progress_view.emit(
            "upload_completed",
            f"已{source_action}：{registered.page_count}页，文档ID {registered.document_id}。",
        )

        def show_progress(stage: str, detail: str) -> None:
            progress_view.emit(stage, detail)

        artifacts = analyze(
            registered.company_folder,
            project_root=upload_service.project_root,
            document_id=registered.document_id,
            progress_callback=show_progress,
        )
    except Exception as exc:
        progress_view.fail(str(exc))
        st.error(f"分析未完成：{exc}")
        return

    progress_view.complete("分析完成，候选风险已经进入人工复核队列。")
    st.session_state["last_analysis_run_id"] = artifacts.report.run_id
    st.session_state["last_analysis_report_path"] = str(artifacts.report_path)
    metrics = st.columns(3)
    metrics[0].metric("候选风险", artifacts.report.summary.finding_count)
    metrics[1].metric("原文证据", artifacts.report.summary.evidence_count)
    metrics[2].metric("Agent轨迹", len(artifacts.report.agent_traces))
    st.success(f"分析运行 {artifacts.report.run_id} 已完成，全部风险等待人工复核。")
    st.caption(f"候选报告：{artifacts.report_path}")
    st.button(
        "进入人工复核",
        type="primary",
        icon=":material/fact_check:",
        on_click=_open_review,
    )


def render_workspace_app(
    *,
    upload_service: ProspectusUploadService,
    review_service: ReviewService,
    analyze: AnalysisRunner = analyze_company,
) -> None:
    """渲染上传分析和人工复核两个相互独立的工作区。"""
    st.set_page_config(
        page_title="IPO风险分析工作台",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    st.title("IPO风险分析工作台")
    st.session_state.setdefault("workspace_view", "新建分析")
    view = st.segmented_control(
        "工作区",
        options=("新建分析", "人工复核"),
        key="workspace_view",
        width="stretch",
    )
    if view == "人工复核":
        render_review_app(review_service, configure_page=False)
    else:
        _render_upload_page(upload_service=upload_service, analyze=analyze)


__all__ = ["AnalysisRunner", "render_workspace_app"]
