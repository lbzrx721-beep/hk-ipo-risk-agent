from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import fitz

from ipo_risk.ui.routes import (
    ANALYSIS_PAGE_PATH,
    RESULTS_PAGE_PATH,
    REVIEW_PAGE_PATH,
)


def test_workspace_import_does_not_load_analysis_pipeline_eagerly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ipo_risk.ui.workspace_app; "
                "print('ipo_risk.pipeline' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def _pdf_bytes(extra_text: str = "") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        (50, 50, 550, 300),
        "英矽智能 InSilico Medicine Cayman TopCo\n股份代號：3696\n"
        f"2025年12月18日\n{extra_text}",
        fontname="china-s",
        fontsize=12,
    )
    payload = document.tobytes()
    document.close()
    return payload


def _workspace_app(
    project_root: Path,
    *,
    pause_analysis: bool = False,
    require_scope_review: bool = False,
):
    from streamlit.testing.v1 import AppTest

    root_literal = repr(str(project_root))
    pause_literal = repr(pause_analysis)
    scope_review_literal = repr(require_scope_review)
    return AppTest.from_string(
        f"""
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import streamlit as st
from ipo_risk.agents.intervention import (
    RiskScopeCandidate,
    RiskScopeFunnel,
    RiskScopeReviewController,
    RiskScopeReviewRequest,
)
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.reviewing import ReviewService
from ipo_risk.storage.repositories import CatalogRepository
from ipo_risk.ui.workspace_app import render_workspace_app
from ipo_risk.uploads import ProspectusUploadService

root = Path({root_literal})
repository = CatalogRepository(root / "data" / "app.db")
repository.initialize()
analysis_gate = st.session_state.setdefault("analysis_gate", Event())
analysis_waiting = st.session_state.setdefault("analysis_waiting", Event())
pause_analysis = {pause_literal}
require_scope_review = {scope_review_literal}

def fake_analyze(company_folder, **kwargs):
    progress = kwargs["progress_callback"]
    for stage, detail in (
        ("parsing_started", "开始解析PDF页面与坐标。"),
        ("parsing_completed", "PDF解析完成：1页，3个文本切片。"),
        ("agents_started", "开始运行专业Agent与决策Agent。"),
        ("agent:discovery:started", "风险发现Agent开始扫描全文。"),
        ("agent:discovery:prompt", "开始扫描第1批全文切片。"),
        (
            "agent:discovery:delta",
            '{{"candidates":[{{"title":"独家数据授权终止","owner":"legal",'
            '"severity":"high","claim":"核心数据授权可能提前终止。",'
            '"confidence":0.88,"evidence_chunk_ids":["C-1"],'
            '"uncertainty":"仍需人工确认替代安排。"}}]}}',
        ),
        ("agent:discovery:completed", "全文扫描发现1条开放风险。"),
        ("agent:finance:started", "财务Agent开始分析2类风险。"),
        ("agent:finance:prompt", "开始分析现金消耗风险。"),
        ("agent:finance:delta", '{{"claim":"现金'),
        ("agent:finance:delta", '消耗风险"}}'),
        ("agent:finance:progress", "[1/2] 现金消耗与融资依赖：找到3个候选证据。"),
        ("agent:finance:prompt", "开始分析融资依赖风险。"),
        ("agent:finance:delta", '{{"claim":"融资依赖风险"}}'),
        ("agent:finance:completed", "财务Agent完成分析，形成1条有证据提案。"),
        ("agent:legal:started", "法律Agent开始分析1类风险。"),
        ("agent:legal:completed", "法律Agent完成分析，形成1条有证据提案。"),
        ("agent:decision:started", "决策Agent开始交叉核验证据与提案。"),
        ("agent:decision:completed", "决策Agent接受2条有证据风险。"),
        ("agents_completed", "Agent裁决完成：接受2条有证据风险。"),
        ("evidence_started", "开始生成高亮截图与候选报告。"),
        ("completed", "分析产物已经写入并登记。"),
    ):
        if require_scope_review and stage == "agent:finance:started":
            controller = kwargs["risk_scope_review_controller"]
            assert isinstance(controller, RiskScopeReviewController)
            progress(
                "risk_scope_review_requested",
                "AI已生成1条候选风险，等待人工确认分析范围。",
            )
            controller.request_review(
                RiskScopeReviewRequest(
                    run_id="RUN-UPLOAD-TEST",
                    industry="AI制药",
                    business_model="研发合作与管线授权",
                    revenue_model="里程碑付款与授权收入",
                    candidates=(
                        RiskScopeCandidate(
                            risk_type="cash_burn",
                            title="现金消耗与融资依赖",
                            owner=AgentRole.FINANCE,
                            severity=RiskSeverity.HIGH,
                            ai_recommended=True,
                            rationale="持续研发投入导致现金消耗压力。",
                            review_question="核验现金跑道和后续融资依赖。",
                            source="document_discovery",
                            source_candidate_count=4,
                            evidence_count=3,
                            confidence=0.91,
                        ),
                    ),
                    funnel=RiskScopeFunnel(
                        document_batches=12,
                        raw_candidates=27,
                        consolidated_candidates=9,
                        catalog_added=1,
                        catalog_merged=2,
                        scoped_in=8,
                        scoped_out=2,
                        consolidation_mode="semantic_model",
                    ),
                )
            )
            progress(
                "risk_scope_review_completed",
                "风险范围已确认，专业Agent继续运行。",
            )
        progress(stage, detail)
        if pause_analysis and stage == "agent:finance:progress":
            analysis_waiting.set()
            analysis_gate.wait(timeout=5)
    report_path = root / "artifacts" / "analysis" / "risk_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{{}}", encoding="utf-8")
    return SimpleNamespace(
        report=SimpleNamespace(
            run_id="RUN-UPLOAD-TEST",
            summary=SimpleNamespace(
                finding_count=6,
                evidence_count=8,
            ),
            agent_traces=(1, 2, 3, 4, 5),
        ),
        report_path=report_path,
    )

render_workspace_app(
    upload_service=ProspectusUploadService(project_root=root, repository=repository),
    review_service=ReviewService(repository=repository, project_root=root),
    analyze=fake_analyze,
)
""",
        default_timeout=5,
    ).run()


def _wait_for_analysis(app):
    job = app.session_state["current_analysis_job"]
    assert job.wait(timeout=2)
    return app.run()


def test_workspace_defaults_to_drag_and_drop_analysis_page(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path)

    assert not app.exception
    assert app.title[0].value == "IPO风险分析工作台"
    assert not app.segmented_control
    assert app.file_uploader[0].label == "拖入招股说明书PDF"
    assert not app.text_input
    assert not any(button.label == "开始分析" for button in app.button)

    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()

    assert not app.exception
    assert app.text_input[0].label == "股票代码"
    assert app.text_input[0].value == "03696"
    assert app.text_input[1].label == "公司名称"
    assert app.text_input[1].value == "英矽智能"
    assert app.date_input[0].label == "招股书日期"
    assert str(app.date_input[0].value) == "2025-12-18"
    assert any(button.label == "开始分析" for button in app.button)


def test_workspace_exposes_three_distinct_page_routes(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path)
    assert app.file_uploader

    app = app.switch_page(str(RESULTS_PAGE_PATH)).run()
    assert any(item.value == "招股书风险分析结果" for item in app.header)
    assert not app.file_uploader

    app = app.switch_page(str(REVIEW_PAGE_PATH)).run()
    assert any(item.value == "人工复核工作台" for item in app.header)

    app = app.switch_page(str(ANALYSIS_PAGE_PATH)).run()
    assert app.file_uploader[0].label == "拖入招股说明书PDF"


def test_new_analysis_page_does_not_render_a_previous_analysis_snapshot(
    tmp_path: Path,
) -> None:
    app = _workspace_app(tmp_path)
    app.session_state["last_analysis_run_id"] = "RUN-PREVIOUS"
    app.session_state["last_analysis_snapshot"] = {
        "max_percent": 100,
        "events": ["不应出现在新建分析页"],
        "agents": {},
        "outputs": {},
    }

    app = app.run()

    assert not app.exception
    assert not any(item.value == "上次分析回顾" for item in app.subheader)
    assert not any("RUN-PREVIOUS" in item.value for item in app.markdown)


def test_dragged_pdf_runs_the_visible_analysis_flow(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path)
    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()
    next(button for button in app.button if button.label == "开始分析").click()
    app = app.run()
    app = _wait_for_analysis(app)

    assert not app.exception
    assert app.file_uploader[0].label == "拖入招股说明书PDF"
    assert any("RUN-UPLOAD-TEST" in message.value for message in app.success)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {"候选风险": "6", "原文证据": "8", "Agent轨迹": "5"}
    agent_headings = {heading.value for heading in app.subheader}
    assert {
        "风险发现 Agent",
        "财务 Agent",
        "法律 Agent",
        "业务 Agent",
        "市场 Agent",
        "决策 Agent",
    } <= agent_headings
    activity_stream = "\n".join(block.value for block in app.code)
    assert "现金消耗与融资依赖" in activity_stream
    assert "决策Agent接受2条有证据风险" in activity_stream
    assert any(
        "独家数据授权终止" in item.value for item in app.markdown
    )
    assert any(button.label == "进入人工复核" for button in app.button)
    uploaded = list(
        (tmp_path / "data" / "raw" / "prospectus" / "03696_英矽智能").glob(
            "*.pdf"
        )
    )
    assert len(uploaded) == 1

    app = app.switch_page(str(RESULTS_PAGE_PATH)).run()
    assert any(item.value == "招股书风险分析结果" for item in app.header)
    app = app.switch_page(str(ANALYSIS_PAGE_PATH)).run()

    assert not app.exception
    assert any(item.value == "实时分析控制台" for item in app.subheader)
    assert not any(item.value == "上次分析回顾" for item in app.subheader)
    restored_stream = "\n".join(block.value for block in app.code)
    assert "决策Agent接受2条有证据风险" in restored_stream
    assert any(
        "独家数据授权终止" in item.value for item in app.markdown
    )


def test_uploading_a_different_pdf_replaces_the_current_analysis_view(
    tmp_path: Path,
) -> None:
    app = _workspace_app(tmp_path)
    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()
    next(button for button in app.button if button.label == "开始分析").click()
    app = app.run()
    app = _wait_for_analysis(app)
    assert any(item.value == "实时分析控制台" for item in app.subheader)

    app.file_uploader[0].set_value(
        (
            "prospectus-v2.pdf",
            _pdf_bytes("第二版招股书"),
            "application/pdf",
        )
    )
    app = app.run()

    assert not app.exception
    assert not any(item.value == "实时分析控制台" for item in app.subheader)
    assert any(button.label == "开始分析" for button in app.button)
    assert app.text_input[0].value == "03696"


def test_running_analysis_survives_switching_workspaces(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path, pause_analysis=True)
    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()
    next(button for button in app.button if button.label == "开始分析").click()
    app = app.run()

    assert app.session_state["analysis_waiting"].wait(timeout=2)
    app = app.run()
    assert any(item.value == "实时分析控制台" for item in app.subheader)
    assert any("现金消耗与融资依赖" in block.value for block in app.code)

    app = app.switch_page(str(RESULTS_PAGE_PATH)).run()
    app = app.switch_page(str(ANALYSIS_PAGE_PATH)).run()

    assert not app.exception
    assert any(item.value == "实时分析控制台" for item in app.subheader)
    assert any("现金消耗与融资依赖" in block.value for block in app.code)
    assert any("切换工作区不会停止分析" in item.value for item in app.caption)

    app.session_state["analysis_gate"].set()
    job = app.session_state["current_analysis_job"]
    assert job.wait(timeout=2)
    app = app.run()
    assert any("RUN-UPLOAD-TEST" in message.value for message in app.success)


def test_scope_review_is_interactive_and_survives_route_switches(
    tmp_path: Path,
) -> None:
    app = _workspace_app(tmp_path, require_scope_review=True)
    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()
    next(button for button in app.button if button.label == "开始分析").click()
    app = app.run()

    job = app.session_state["current_analysis_job"]
    assert job.wait_for_risk_scope_review(timeout=2)
    app = app.run()

    assert not app.exception
    assert job.snapshot().status == "awaiting_review"
    assert any(item.value == "确认本次分析范围" for item in app.subheader)
    assert any(item.label == "确认人" for item in app.text_input)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["批次原始发现"] == "27"
    assert metrics["全局语义合并"] == "9"
    assert metrics["目录兜底补充"] == "1"
    assert metrics["AI建议纳入"] == "8"
    assert any(
        button.label == "确认范围并启动专业 Agent" for button in app.button
    )
    assert app.file_uploader[0].disabled

    app = app.switch_page(str(RESULTS_PAGE_PATH)).run()
    app = app.switch_page(str(ANALYSIS_PAGE_PATH)).run()
    assert any(item.value == "确认本次分析范围" for item in app.subheader)

    reviewer = next(item for item in app.text_input if item.label == "确认人")
    reviewer.set_value("投资经理A")
    app = app.run()
    next(
        button
        for button in app.button
        if button.label == "确认范围并启动专业 Agent"
    ).click()
    app = app.run()

    assert job.wait(timeout=2)
    assert not app.exception
    assert job.snapshot().status == "completed"
