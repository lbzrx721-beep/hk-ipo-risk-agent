from __future__ import annotations

from pathlib import Path

import fitz


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        (50, 50, 550, 300),
        "英矽智能 InSilico Medicine Cayman TopCo\n股份代號：3696\n2025年12月18日",
        fontname="china-s",
        fontsize=12,
    )
    payload = document.tobytes()
    document.close()
    return payload


def _workspace_app(project_root: Path):
    from streamlit.testing.v1 import AppTest

    root_literal = repr(str(project_root))
    return AppTest.from_string(
        f"""
from pathlib import Path
from types import SimpleNamespace
from ipo_risk.reviewing import ReviewService
from ipo_risk.storage.repositories import CatalogRepository
from ipo_risk.ui.workspace_app import render_workspace_app
from ipo_risk.uploads import ProspectusUploadService

root = Path({root_literal})
repository = CatalogRepository(root / "data" / "app.db")
repository.initialize()

def fake_analyze(company_folder, **kwargs):
    progress = kwargs["progress_callback"]
    for stage, detail in (
        ("parsing_started", "开始解析PDF页面与坐标。"),
        ("parsing_completed", "PDF解析完成：1页，3个文本切片。"),
        ("agents_started", "开始运行专业Agent与决策Agent。"),
        ("agent:finance:started", "财务Agent开始分析2类风险。"),
        ("agent:finance:delta", '{{"claim":"现金'),
        ("agent:finance:delta", '消耗风险"}}'),
        ("agent:finance:progress", "[1/2] 现金消耗与融资依赖：找到3个候选证据。"),
        ("agent:finance:completed", "财务Agent完成分析，形成1条有证据提案。"),
        ("agent:legal:started", "法律Agent开始分析1类风险。"),
        ("agent:legal:completed", "法律Agent完成分析，形成1条有证据提案。"),
        ("agent:decision:started", "决策Agent开始交叉核验证据与提案。"),
        ("agent:decision:completed", "决策Agent接受2条有证据风险。"),
        ("agents_completed", "Agent裁决完成：接受2条有证据风险。"),
        ("evidence_started", "开始生成高亮截图与候选报告。"),
        ("completed", "分析产物已经写入并登记。"),
    ):
        progress(stage, detail)
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


def test_workspace_defaults_to_drag_and_drop_analysis_page(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path)

    assert not app.exception
    assert app.title[0].value == "IPO风险分析工作台"
    assert app.segmented_control[0].value == "新建分析"
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


def test_dragged_pdf_runs_the_visible_analysis_flow(tmp_path: Path) -> None:
    app = _workspace_app(tmp_path)
    app.file_uploader[0].set_value(
        ("prospectus.pdf", _pdf_bytes(), "application/pdf")
    )
    app = app.run()
    next(button for button in app.button if button.label == "开始分析").click()
    app = app.run()

    assert not app.exception
    assert any("RUN-UPLOAD-TEST" in message.value for message in app.success)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {"候选风险": "6", "原文证据": "8", "Agent轨迹": "5"}
    agent_headings = {heading.value for heading in app.subheader}
    assert {
        "财务 Agent",
        "法律 Agent",
        "业务 Agent",
        "市场 Agent",
        "决策 Agent",
    } <= agent_headings
    activity_stream = "\n".join(block.value for block in app.code)
    assert "现金消耗与融资依赖" in activity_stream
    assert "决策Agent接受2条有证据风险" in activity_stream
    assert '{"claim":"现金消耗风险"}' in activity_stream
    assert any(button.label == "进入人工复核" for button in app.button)
    uploaded = list(
        (tmp_path / "data" / "raw" / "prospectus" / "03696_英矽智能").glob(
            "*.pdf"
        )
    )
    assert len(uploaded) == 1
