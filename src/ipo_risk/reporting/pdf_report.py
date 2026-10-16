"""Human-readable verified IPO risk report in PDF format."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from opencc import OpenCC
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ipo_risk.domain import (
    AgentTrace,
    AgentRole,
    Evidence,
    EvidenceType,
    FindingReview,
    RiskFinding,
    RiskSeverity,
)
from ipo_risk.reporting.json_report import RiskAnalysisReport


_FONT_REGULAR = "IPORiskChinese"
_FONT_BOLD = "IPORiskChineseBold"
_OPENCC = OpenCC("t2s")
_SEVERITY_LABELS = {
    RiskSeverity.CRITICAL: "极高风险",
    RiskSeverity.HIGH: "高风险",
    RiskSeverity.MEDIUM: "中风险",
    RiskSeverity.LOW: "低风险",
}
_SEVERITY_ORDER = {
    RiskSeverity.CRITICAL: 4,
    RiskSeverity.HIGH: 3,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.LOW: 1,
}
_ROLE_LABELS = {
    AgentRole.DOCUMENT_PARSER: "文档解析",
    AgentRole.FINANCE: "财务",
    AgentRole.LEGAL: "法务与合规",
    AgentRole.BUSINESS: "业务",
    AgentRole.MARKET: "市场",
    AgentRole.DISCOVERY: "风险发现",
    AgentRole.VERIFIER: "证据核验",
    AgentRole.DECISION: "决策",
    AgentRole.REPORTER: "报告生成",
    AgentRole.ORCHESTRATOR: "流程编排",
}
_FEATURE_LABELS = {
    "offering_offer_price_low_hkd": "发售价下限",
    "offering_offer_price_high_hkd": "发售价上限",
    "offering_share_count": "发行股数",
    "offering_net_proceeds_hkd": "预计净募资额",
    "offering_market_cap_low_hkd": "预计市值下限",
    "offering_market_cap_high_hkd": "预计市值上限",
    "offering_share_ratio_pct": "发行比例",
    "fundamental_revenue_latest_native": "最近完整年度收入",
    "fundamental_revenue_cagr_3y_pct": "三年营收复合增长率",
    "fundamental_gross_margin_latest_pct": "最近完整年度毛利率",
    "fundamental_operating_cash_flow_latest_native": "经营活动现金流净额",
}
_BEIJING_TZ = timezone(timedelta(hours=8))
_REVIEW_LABELS = {
    "confirmed": "确认",
    "rejected": "驳回",
    "edited": "编辑后确认",
}
_FINDING_STATUS_LABELS = {
    "proposed": "候选",
    "pending_verification": "待复核",
    "verified": "已确认",
    "rejected": "已驳回",
}
_EXECUTION_STATUS_LABELS = {
    "succeeded": "成功",
    "failed": "失败",
}
_EVIDENCE_TYPE_LABELS = {
    "text": "文本",
    "table": "表格",
    "ocr_text": "OCR文本",
    "image": "图片",
}


def _register_fonts() -> None:
    global _FONT_REGULAR, _FONT_BOLD
    if _FONT_REGULAR in pdfmetrics.getRegisteredFontNames():
        return
    candidates = (
        (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\msyhbd.ttc")),
        (Path(r"C:\Windows\Fonts\simsun.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")),
        # Linux / Docker 环境（Noto CJK）
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")),
        (Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"), Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc")),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            try:
                pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(regular)))
                pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(bold)))
                return
            except TTFError:
                # 部分Linux Noto CJK TTC使用PostScript轮廓，ReportLab无法嵌入。
                continue
    # ReportLab内置简体中文CID字体作为跨平台可靠兜底。
    fallback = "STSong-Light"
    if fallback not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(fallback))
    _FONT_REGULAR = fallback
    _FONT_BOLD = fallback


def _zh(value: object) -> str:
    return _OPENCC.convert(str(value))


def _markup(value: object) -> str:
    return escape(_zh(value)).replace("\n", "<br/>")


def _format_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return "未记录"
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text


def _format_datetime(value: object) -> str:
    """Render user-facing audit times consistently in Beijing time."""
    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return "未记录"
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _report_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_label": ParagraphStyle(
            "CoverLabel", parent=sample["Normal"], fontName=_FONT_REGULAR,
            fontSize=10, leading=16, textColor=colors.HexColor("#64748B"),
            alignment=TA_CENTER,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=sample["Title"], fontName=_FONT_BOLD,
            fontSize=25, leading=36, textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
        ),
        "cover_company": ParagraphStyle(
            "CoverCompany", parent=sample["Heading2"], fontName=_FONT_BOLD,
            fontSize=17, leading=26, textColor=colors.HexColor("#0369A1"),
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "ReportH1", parent=sample["Heading1"], fontName=_FONT_BOLD,
            fontSize=17, leading=25, spaceAfter=9, textColor=colors.HexColor("#0F172A"),
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=sample["Heading2"], fontName=_FONT_BOLD,
            fontSize=12.5, leading=19, spaceBefore=7, spaceAfter=5,
            textColor=colors.HexColor("#0F172A"), wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "ReportH3", parent=sample["Heading3"], fontName=_FONT_BOLD,
            fontSize=10, leading=16, spaceBefore=5, spaceAfter=4,
            textColor=colors.HexColor("#0F172A"), wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=sample["BodyText"], fontName=_FONT_REGULAR,
            fontSize=9.5, leading=16, spaceAfter=6, textColor=colors.HexColor("#243247"),
            alignment=TA_LEFT, wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "ReportSmall", parent=sample["BodyText"], fontName=_FONT_REGULAR,
            fontSize=8, leading=13, spaceAfter=4, textColor=colors.HexColor("#64748B"),
            wordWrap="CJK",
        ),
        "cell": ParagraphStyle(
            "ReportCell", parent=sample["BodyText"], fontName=_FONT_REGULAR,
            fontSize=7.4, leading=11, textColor=colors.HexColor("#243247"), wordWrap="CJK",
        ),
        "cell_head": ParagraphStyle(
            "ReportCellHead", parent=sample["BodyText"], fontName=_FONT_BOLD,
            fontSize=7.4, leading=11, textColor=colors.white, wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "ReportQuote", parent=sample["BodyText"], fontName=_FONT_REGULAR,
            fontSize=8.5, leading=14, leftIndent=8, rightIndent=8,
            borderColor=colors.HexColor("#CBD5E1"), borderWidth=0.7,
            borderPadding=8, backColor=colors.HexColor("#F8FAFC"),
            textColor=colors.HexColor("#334155"), wordWrap="CJK",
        ),
        "disclaimer": ParagraphStyle(
            "ReportDisclaimer", parent=sample["BodyText"], fontName=_FONT_REGULAR,
            fontSize=8.5, leading=15, borderColor=colors.HexColor("#F59E0B"),
            borderWidth=0.8, borderPadding=9, backColor=colors.HexColor("#FFFBEB"),
            textColor=colors.HexColor("#78350F"), wordWrap="CJK",
        ),
    }


def _page_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont(_FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 9 * mm, "港股 IPO 风险预警系统 · 人工复核正式报告")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def _audit_page_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont(_FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 9 * mm, "港股 IPO 风险预警系统 · 完整分析审计报告")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def _evidence_image(evidence: Evidence, *, project_root: Path) -> Image | None:
    path = _resolve_path(evidence.screenshot_path, project_root)
    if path is None or not path.is_file():
        return None
    payload = path.read_bytes()
    width, height = ImageReader(BytesIO(payload)).getSize()
    max_width, max_height = 165 * mm, 86 * mm
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(BytesIO(payload), width=width * scale, height=height * scale)


def _styled_table(
    rows: Sequence[Sequence[object]],
    *,
    col_widths: Sequence[float] | None = None,
    header: bool = False,
    repeat_rows: int = 0,
) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#075985")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _review_by_finding(reviews: tuple[FindingReview, ...]) -> dict[str, FindingReview]:
    latest: dict[str, FindingReview] = {}
    for review in sorted(reviews, key=lambda item: item.created_at):
        latest[review.finding_id] = review
    return latest


def _ranked_findings(findings: Iterable[RiskFinding]) -> list[RiskFinding]:
    return sorted(
        findings,
        key=lambda finding: (
            _SEVERITY_ORDER[finding.severity], finding.confidence, len(finding.evidence_ids)
        ),
        reverse=True,
    )


def _prediction_direction_table(
    prediction: Mapping[str, Any],
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    """Render categorical prediction conclusions without exposing probabilities."""
    horizon_probs = prediction.get("horizon_predictions", {}) or {}
    horizon_thresholds = prediction.get("horizon_thresholds", {}) or {}
    horizon_names = [
        ("h1", "上市首日"),
        ("h5", "5个交易日"),
        ("h20", "20个交易日"),
        ("h60", "60个交易日"),
    ]
    rows: list[tuple[str, str]] = [("预测目标", "上市后多时间窗显著下跌风险")]
    for key, label in horizon_names:
        value = horizon_probs.get(key)
        if value is None and key == "h5":
            value = prediction.get("probability")
        threshold = horizon_thresholds.get(key)
        if value is None:
            result = "暂不可用"
        elif threshold is not None:
            result = "触发显著下跌预警" if float(value) >= float(threshold) else "未触发显著下跌预警"
        elif key == "h5":
            decision = str(prediction.get("decision", "uncertain"))
            result = {
                "high_risk": "触发显著下跌预警",
                "low_risk": "未触发显著下跌预警",
                "uncertain": "未触发显著下跌预警",
            }.get(decision, "未触发显著下跌预警")
        else:
            result = "结果可用"
        rows.append((label, result))
    return _styled_table(
        [
            [Paragraph(_markup(key), styles["small"]), Paragraph(_markup(value), styles["body"])]
            for key, value in rows
        ],
        col_widths=(42 * mm, 108 * mm),
    )


def _feature_value(item: object) -> tuple[object | None, str | None]:
    if isinstance(item, Mapping):
        return item.get("value"), str(item.get("unit")) if item.get("unit") else None
    return item, None


def _format_feature(value: object | None, unit: str | None) -> str:
    if value is None or value == "":
        return "未披露"
    if isinstance(value, bool):
        rendered = "是" if value else "否"
    elif isinstance(value, (int, float)):
        rendered = f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        rendered = _zh(value)
    units = {
        "HKD": "港元",
        "count": "股",
        "percent": "%",
        "reporting_currency_million": "（报告币种，百万元）",
    }
    return f"{rendered}{units.get(unit or '', unit or '')}"


_FEATURE_GROUP_LABELS = {
    "offering": "发行与估值",
    "fundamental": "财务基本面",
    "concentration": "客户与供应商集中度",
    "market": "上市前市场",
    "recent_ipo": "近期 IPO 市场",
    "agent": "Agent 风险语义",
    "engineered": "确定性衍生计算",
    "quality": "数据质量与缺失状态",
    "data_quality": "数据质量与缺失状态",
    "document": "文档属性",
}


def _feature_audit_flowables(
    feature_payload: Mapping[str, Any] | None,
    styles: Mapping[str, ParagraphStyle],
) -> list[object]:
    """Render business-facing feature lineage without exposing model internals."""
    if not feature_payload:
        return [Paragraph("本次历史运行未保存结构化特征审计快照。", styles["body"])]

    features = {
        str(name): value
        for name, value in feature_payload.items()
        if isinstance(value, Mapping)
    }
    group_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"total": 0, "available": 0, "formulas": set(), "sources": set()}
    )
    derived_count = 0
    source_feature_count = 0
    currency_feature_count = 0
    fx_conversion_count = 0
    reporting_currencies: set[str] = set()
    for name, item in features.items():
        group = str(item.get("group") or name.split("_", 1)[0] or "other")
        stats = group_stats[group]
        stats["total"] = int(stats["total"]) + 1
        if item.get("value") is not None and item.get("status") != "missing":
            stats["available"] = int(stats["available"]) + 1
        formula = str(item.get("formula_version") or "").strip()
        source = str(item.get("source_ref") or "").strip()
        if formula:
            cast_formulas = stats["formulas"]
            if isinstance(cast_formulas, set):
                cast_formulas.add(formula)
        if source:
            cast_sources = stats["sources"]
            if isinstance(cast_sources, set):
                cast_sources.add(source.split("#", 1)[0])
        if source.startswith("derived:") or "derived" in formula:
            derived_count += 1
        if any(marker in source for marker in ("chunks.jsonl", "tables.jsonl", "rag_")):
            source_feature_count += 1
        unit = str(item.get("unit") or "")
        if "currency" in unit.lower() or "hkd" in unit.lower() or name.endswith("_hkd"):
            currency_feature_count += 1
        if "fx_to_hkd" in name or "currency_conversion" in formula or "fx_conversion" in formula:
            fx_conversion_count += 1
        if name == "fundamental_reporting_currency" and item.get("value"):
            reporting_currencies.add(str(item["value"]))

    operation_rows: list[list[Paragraph]] = [[
        Paragraph("关键业务工具链", styles["cell_head"]),
        Paragraph("处理结果", styles["cell_head"]),
        Paragraph("可追溯依据", styles["cell_head"]),
    ]]
    operation_rows.extend([
        [
            Paragraph("招股书事实与表格特征提取", styles["cell"]),
            Paragraph(f"形成 {source_feature_count} 项带原文来源的候选特征", styles["cell"]),
            Paragraph("切片或表格编号、页码、字段来源", styles["cell"]),
        ],
        [
            Paragraph("确定性公式与衍生特征计算", styles["cell"]),
            Paragraph(f"完成 {derived_count} 项确定性衍生结果", styles["cell"]),
            Paragraph("公式版本、输入来源、单位与可用日期", styles["cell"]),
        ],
        [
            Paragraph("币种识别、汇率换算与金额单位标准化", styles["cell"]),
            Paragraph(
                f"核验 {currency_feature_count} 项币种或金额字段；报告币种 "
                f"{_markup('、'.join(sorted(reporting_currencies)) or '按字段契约记录')}；"
                + (
                    f"记录 {fx_conversion_count} 项汇率换算字段"
                    if fx_conversion_count
                    else "本次未发现需单列的汇率换算字段"
                ),
                styles["cell"],
            ),
            Paragraph("原始币种、目标单位、字段公式版本", styles["cell"]),
        ],
        [
            Paragraph("上市前市场特征构建", styles["cell"]),
            Paragraph(
                f"处理 {int(group_stats.get('market', {}).get('total', 0))} 项市场候选特征",
                styles["cell"],
            ),
            Paragraph("上市日期、数据时点和市场数据来源", styles["cell"]),
        ],
        [
            Paragraph("Agent 风险语义结构化", styles["cell"]),
            Paragraph(
                f"处理 {int(group_stats.get('agent', {}).get('total', 0))} 项风险语义候选特征",
                styles["cell"],
            ),
            Paragraph("风险编号、证据引用和语义规则版本", styles["cell"]),
        ],
    ])
    result: list[object] = [
        Paragraph(
            "以下记录展示从招股书证据到模型特征之间的关键业务处理。"
            "它们属于确定性工具链，与大模型文本分析共同构成完整分析轨迹。",
            styles["small"],
        ),
        _styled_table(
            operation_rows,
            col_widths=(47 * mm, 62 * mm, 54 * mm),
            header=True,
            repeat_rows=1,
        ),
        Spacer(1, 7 * mm),
        Paragraph("特征组与数据血缘", styles["h2"]),
    ]
    group_rows: list[list[Paragraph]] = [[
        Paragraph("特征组", styles["cell_head"]),
        Paragraph("候选项", styles["cell_head"]),
        Paragraph("已取得", styles["cell_head"]),
        Paragraph("公式 / 规则版本", styles["cell_head"]),
        Paragraph("来源数", styles["cell_head"]),
    ]]
    for group, stats in sorted(group_stats.items()):
        formulas = sorted(str(value) for value in stats["formulas"])
        formula_text = "、".join(formulas[:3])
        if len(formulas) > 3:
            formula_text += f" 等 {len(formulas)} 类"
        group_rows.append([
            Paragraph(_markup(_FEATURE_GROUP_LABELS.get(group, group)), styles["cell"]),
            Paragraph(str(stats["total"]), styles["cell"]),
            Paragraph(str(stats["available"]), styles["cell"]),
            Paragraph(_markup(formula_text or "原始字段"), styles["cell"]),
            Paragraph(str(len(stats["sources"])), styles["cell"]),
        ])
    result.append(_styled_table(
        group_rows,
        col_widths=(40 * mm, 18 * mm, 18 * mm, 67 * mm, 20 * mm),
        header=True,
        repeat_rows=1,
    ))
    return result


def _formal_technology_flowables(
    report: RiskAnalysisReport,
    *,
    traces: Sequence[AgentTrace],
    feature_payload: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    reviews: Sequence[FindingReview],
    styles: Mapping[str, ParagraphStyle],
) -> list[object]:
    """Compact business-facing summary of the completed technical workflow."""
    calls = [call for trace in traces for call in trace.tool_calls]
    tool_names = {call.tool_name for call in calls}
    roles = {
        _ROLE_LABELS.get(trace.agent, trace.agent.value)
        for trace in traces
        if trace.agent not in {AgentRole.ORCHESTRATOR, AgentRole.DOCUMENT_PARSER}
    }
    roles.update(
        _ROLE_LABELS.get(finding.created_by, finding.created_by.value)
        for finding in report.risk_findings
    )
    feature_items = {
        str(name): item
        for name, item in (feature_payload or {}).items()
        if isinstance(item, Mapping)
    }
    derived_count = sum(
        str(item.get("source_ref") or "").startswith("derived:")
        or "derived" in str(item.get("formula_version") or "")
        for item in feature_items.values()
    )
    market_count = sum(item.get("group") == "market" for item in feature_items.values())
    agent_count = sum(item.get("group") == "agent" for item in feature_items.values())
    horizon_probs = prediction.get("horizon_predictions", {}) if prediction else {}
    horizon_count = sum(
        value is not None
        for value in (
            horizon_probs.get("h1") if isinstance(horizon_probs, Mapping) else None,
            horizon_probs.get("h5") if isinstance(horizon_probs, Mapping) else None,
            horizon_probs.get("h20") if isinstance(horizon_probs, Mapping) else None,
            horizon_probs.get("h60") if isinstance(horizon_probs, Mapping) else None,
        )
    )
    if prediction and horizon_count == 0 and prediction.get("probability") is not None:
        horizon_count = 1
    retrieval_used = bool(
        tool_names.intersection({"evidence_retrieval", "open_risk_discovery"})
        or report.evidences
    )
    debate_count = sum(call.tool_name == "risk_debate_verification" for call in calls)
    rows: list[list[Paragraph]] = [[
        Paragraph("技术环节", styles["cell_head"]),
        Paragraph("采用能力", styles["cell_head"]),
        Paragraph("本次处理结果", styles["cell_head"]),
        Paragraph("状态", styles["cell_head"]),
    ]]
    rows.extend([
        [
            Paragraph("招股书 PDF 解析", styles["cell"]),
            Paragraph("原生文本解析、表格结构恢复、页码与坐标定位", styles["cell"]),
            Paragraph(
                f"解析 {report.source_document.page_count} 页，形成 "
                f"{report.source_document.chunk_count} 个可检索内容单元",
                styles["cell"],
            ),
            Paragraph("已完成", styles["cell"]),
        ],
        [
            Paragraph("证据检索与定位", styles["cell"]),
            Paragraph("语义向量检索、关键词检索与原文截图", styles["cell"]),
            Paragraph(f"定位并保留 {len(report.evidences)} 条原文或表格证据", styles["cell"]),
            Paragraph("已完成" if retrieval_used else "未记录", styles["cell"]),
        ],
        [
            Paragraph("多 Agent 专业分析", styles["cell"]),
            Paragraph("风险发现、财务、法务、业务、市场与决策协同", styles["cell"]),
            Paragraph(
                f"参与角色：{_markup('、'.join(sorted(roles)) or '按风险结论记录')}；"
                f"形成 {len(report.risk_findings)} 条复核后风险；"
                f"完成 {debate_count} 项质疑、补证与再裁决",
                styles["cell"],
            ),
            Paragraph("已完成", styles["cell"]),
        ],
        [
            Paragraph("特征提取与确定性计算", styles["cell"]),
            Paragraph("发行、财务、市场、Agent 语义、公式计算与单位标准化", styles["cell"]),
            Paragraph(
                f"保存 {len(feature_items)} 项候选特征，其中确定性衍生 {derived_count} 项、"
                f"上市前市场 {market_count} 项、Agent 语义 {agent_count} 项",
                styles["cell"],
            ),
            Paragraph("已完成" if feature_items else "历史记录未保存", styles["cell"]),
        ],
        [
            Paragraph("多时间窗口风险预测", styles["cell"]),
            Paragraph("结构化特征融合、集成学习与时点约束", styles["cell"]),
            Paragraph(
                f"已生成 {horizon_count} 个时间窗口的方向性预警"
                if horizon_count
                else "本次未生成可用预测",
                styles["cell"],
            ),
            Paragraph("已完成" if horizon_count else "未完成", styles["cell"]),
        ],
        [
            Paragraph("人工复核与正式交付", styles["cell"]),
            Paragraph("证据核对、确认或驳回、意见留痕与报告生成", styles["cell"]),
            Paragraph(f"保存 {len(reviews)} 条人工复核记录并生成正式报告", styles["cell"]),
            Paragraph("已完成", styles["cell"]),
        ],
    ])
    return [
        Paragraph(
            "本节概括本报告实际经过的技术处理。为保证可读性，仅展示业务级链路；"
            "逐次 Agent 轨迹、工具输入输出和异常记录见完整审计报告。",
            styles["small"],
        ),
        _styled_table(
            rows,
            col_widths=(37 * mm, 53 * mm, 58 * mm, 17 * mm),
            header=True,
            repeat_rows=1,
        ),
    ]


def _company_table(
    report: RiskAnalysisReport,
    company_profile: Mapping[str, Any] | None,
    feature_payload: Mapping[str, Any] | None,
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    profile = company_profile or {}
    features = feature_payload or {}
    documents = profile.get("documents")
    prospectus = documents.get("prospectus", {}) if isinstance(documents, Mapping) else {}
    offering = profile.get("offering")
    offering = offering if isinstance(offering, Mapping) else {}
    rows: list[tuple[str, str]] = [
        ("公司名称", report.company_name),
        ("股票代码", report.company_id),
        ("上市日期", str(profile.get("listing_date") or "未披露")),
        ("招股书日期", _format_date(report.source_document.available_at)),
        ("招股书页数", str(prospectus.get("page_count") or "未披露")),
        ("发行价", _format_feature(offering.get("offer_price_hkd"), "HKD")),
    ]
    for key in _FEATURE_LABELS:
        if key in features:
            value, unit = _feature_value(features[key])
            rows.append((_FEATURE_LABELS[key], _format_feature(value, unit)))
    paired: list[list[Paragraph]] = []
    for offset in range(0, len(rows), 2):
        left = rows[offset]
        right = rows[offset + 1] if offset + 1 < len(rows) else ("", "")
        paired.append(
            [
                Paragraph(_markup(left[0]), styles["small"]),
                Paragraph(_markup(left[1]), styles["body"]),
                Paragraph(_markup(right[0]), styles["small"]),
                Paragraph(_markup(right[1]), styles["body"]),
            ]
        )
    table = _styled_table(paired, col_widths=(27 * mm, 48 * mm, 27 * mm, 48 * mm))
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")), ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9"))]))
    return table


def _risk_summary_table(findings: Sequence[RiskFinding], styles: Mapping[str, ParagraphStyle]) -> Table:
    rows: list[list[Paragraph]] = [
        [Paragraph("风险结论", styles["cell_head"]), Paragraph("等级", styles["cell_head"]), Paragraph("置信度", styles["cell_head"]), Paragraph("证据", styles["cell_head"])]
    ]
    for finding in findings:
        rows.append(
            [
                Paragraph(_markup(finding.title), styles["cell"]),
                Paragraph(_markup(_SEVERITY_LABELS[finding.severity]), styles["cell"]),
                Paragraph(f"{finding.confidence:.0%}", styles["cell"]),
                Paragraph(str(len(finding.evidence_ids)), styles["cell"]),
            ]
        )
    return _styled_table(rows, col_widths=(104 * mm, 22 * mm, 18 * mm, 16 * mm), header=True, repeat_rows=1)


def _markdown_headers(quote: str) -> list[str]:
    lines = [line.strip() for line in quote.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    for index, line in enumerate(lines[:-1]):
        separator = [cell.strip() for cell in lines[index + 1].strip("|").split("|")]
        if separator and all(re.fullmatch(r":?-{2,}:?", cell) for cell in separator):
            return [cell.strip() for cell in line.strip("|").split("|")]
    return []


def _table_matrix(evidence: Evidence) -> tuple[list[str], list[list[str]]]:
    cell_map: dict[tuple[int, int], str] = {}
    for reference, value in zip(evidence.table_cell_refs, evidence.table_cell_values, strict=False):
        match = re.search(r":R(\d+):C(\d+)$", reference)
        if match:
            cell_map[(int(match.group(1)), int(match.group(2)))] = value
    if not cell_map:
        return [], []
    row_count = max(row for row, _ in cell_map)
    col_count = max(col for _, col in cell_map)
    headers = _markdown_headers(evidence.quote)
    if len(headers) != col_count:
        headers = ["项目"] + [f"列 {index}" for index in range(2, col_count + 1)]
    rows = [[cell_map.get((row, col), "") for col in range(1, col_count + 1)] for row in range(1, row_count + 1)]
    return headers, rows


def _table_flowables(evidence: Evidence, styles: Mapping[str, ParagraphStyle]) -> list[object]:
    headers, rows = _table_matrix(evidence)
    if not headers or not rows:
        return [Paragraph(_markup(evidence.quote), styles["quote"])]
    result: list[object] = [
        Paragraph("结构化表格（简体中文）", styles["h3"]),
        Paragraph("数值、币种、单位、日期及正负号保持原样；宽表按列分段展示。", styles["small"]),
    ]
    data_columns = list(range(1, len(headers)))
    blocks = [data_columns[index:index + 4] for index in range(0, len(data_columns), 4)] or [[]]
    for block_number, block in enumerate(blocks, start=1):
        indices = [0, *block]
        matrix: list[list[Paragraph]] = [
            [Paragraph(_markup(headers[index]), styles["cell_head"]) for index in indices]
        ]
        matrix.extend(
            [Paragraph(_markup(row[index]), styles["cell"]) for index in indices]
            for row in rows
        )
        width = 160 * mm / len(indices)
        if len(blocks) > 1:
            result.append(Paragraph(f"列组 {block_number}/{len(blocks)}", styles["small"]))
        result.append(_styled_table(matrix, col_widths=[width] * len(indices), header=True, repeat_rows=1))
        result.append(Spacer(1, 2 * mm))
    return result


def _finding_section(
    story: list[object],
    title: str,
    findings: Sequence[RiskFinding],
    styles: Mapping[str, ParagraphStyle],
) -> None:
    story.append(Paragraph(title, styles["h1"]))
    if not findings:
        story.append(Paragraph("本次人工确认结论中未形成该领域风险。", styles["body"]))
        return
    story.append(_risk_summary_table(findings, styles))
    story.append(Spacer(1, 5 * mm))
    for finding in findings[:5]:
        story.append(Paragraph(_markup(finding.title), styles["h3"]))
        story.append(Paragraph(_markup(finding.claim), styles["body"]))


def write_verified_pdf_report(
    report: RiskAnalysisReport,
    path: Path,
    *,
    reviews: tuple[FindingReview, ...] = (),
    project_root: Path,
    prediction: Mapping[str, Any] | None = None,
    company_profile: Mapping[str, Any] | None = None,
    feature_payload: Mapping[str, Any] | None = None,
    execution_traces: Sequence[AgentTrace] = (),
) -> Path:
    """Write a compact formal PDF without using an LLM to rewrite source facts."""
    _register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _report_styles()
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title=f"{report.company_name} IPO风险正式报告",
        author="港股 IPO 风险预警系统",
        subject="人工复核后的证据化风险结论",
    )
    ranked = _ranked_findings(report.risk_findings)
    top_findings = ranked[:5]
    evidence_by_id = {evidence.evidence_id: evidence for evidence in report.evidences}
    reviews_by_finding = _review_by_finding(reviews)
    profile = company_profile or {}
    story: list[object] = [
        Spacer(1, 30 * mm),
        Paragraph("港股 IPO 风险预警", styles["cover_label"]),
        Spacer(1, 5 * mm),
        Paragraph("风险穿透预警报告", styles["cover_title"]),
        Spacer(1, 8 * mm),
        Paragraph(f"{_markup(report.company_name)}（{_markup(report.company_id)}）", styles["cover_company"]),
        Spacer(1, 13 * mm),
    ]
    cover_rows: list[tuple[str, object]] = [
        ("股票代码", report.company_id),
        ("上市日期", profile.get("listing_date") or "未披露"),
        ("招股书日期", _format_date(report.source_document.available_at)),
        ("人工确认风险", report.summary.finding_count),
        ("引用原文证据", report.summary.evidence_count),
        ("报告生成时间（北京时间）", _format_datetime(report.generated_at)),
    ]
    cover_table = _styled_table(
        [[Paragraph(_markup(k), styles["small"]), Paragraph(_markup(v), styles["body"])] for k, v in cover_rows],
        col_widths=(38 * mm, 112 * mm),
    )
    cover_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9"))]))
    story.extend(
        [
            cover_table,
            Spacer(1, 14 * mm),
            Paragraph("本报告呈现人工确认后的风险结论、多时间窗方向性预警与可回溯原文证据，不构成投资建议。", styles["disclaimer"]),
            PageBreak(),
            Paragraph("执行摘要", styles["h1"]),
        ]
    )
    if prediction is not None:
        story.extend([_prediction_direction_table(prediction, styles), Spacer(1, 5 * mm)])
    if report.text_embellishment is not None:
        embellishment = report.text_embellishment
        story.extend(
            [
                Paragraph("招股书文本粉饰分析", styles["h2"]),
                Paragraph(
                    f"粉饰程度：{_markup({'low': '低', 'medium': '中', 'high': '高', 'critical': '严重'}[embellishment.level])}"
                    f"｜原文证据 {len(embellishment.evidence_ids)} 条"
                    f"{'｜已完成补证辩论' if embellishment.debate_used else ''}<br/>"
                    f"{_markup(embellishment.summary)}",
                    styles["body"],
                ),
                Spacer(1, 4 * mm),
            ]
        )
    story.append(Paragraph(f"系统共形成 {len(report.risk_findings)} 条人工确认风险；以下列示严重度、置信度和证据支撑最突出的核心风险。", styles["body"]))
    for index, finding in enumerate(top_findings, start=1):
        story.append(Paragraph(f"{index}. {_markup(finding.title)} - {_SEVERITY_LABELS[finding.severity]}，置信度 {finding.confidence:.0%}", styles["body"]))
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("公司与发行信息", styles["h1"]),
            _company_table(report, company_profile, feature_payload, styles),
            PageBreak(),
        ]
    )

    finance = [finding for finding in ranked if finding.created_by is AgentRole.FINANCE]
    legal = [finding for finding in ranked if finding.created_by is AgentRole.LEGAL]
    business_market = [
        finding for finding in ranked
        if finding.created_by in {AgentRole.BUSINESS, AgentRole.MARKET}
        or finding not in finance + legal
    ]
    _finding_section(story, "财务穿透分析", finance, styles)
    story.append(Spacer(1, 7 * mm))
    _finding_section(story, "法务与合规风险", legal, styles)
    story.append(PageBreak())
    _finding_section(story, "业务与市场风险", business_market, styles)
    story.extend([PageBreak(), Paragraph("核心风险证据链", styles["h1"])])
    story.append(Paragraph("本节展开前5项核心风险，每项展示最多2条关键证据；全部证据均保留在附录索引和审计文件中。", styles["small"]))

    for finding_index, finding in enumerate(top_findings, start=1):
        story.extend(
            [
                HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#94A3B8"), spaceBefore=7, spaceAfter=7),
                Paragraph(f"风险 {finding_index}：{_markup(finding.title)}", styles["h2"]),
                Paragraph(f"{_SEVERITY_LABELS[finding.severity]} ｜ 置信度 {finding.confidence:.0%} ｜ 负责领域 {_ROLE_LABELS.get(finding.created_by, finding.created_by.value)}", styles["small"]),
                Paragraph(_markup(finding.claim), styles["body"]),
            ]
        )
        for evidence_number, evidence_id in enumerate(finding.evidence_ids[:2], start=1):
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            story.extend(
                [
                    Paragraph(f"证据 {evidence_number} · {_markup(evidence.evidence_id)}", styles["h3"]),
                    Paragraph(
                        f"PDF物理页 {evidence.page} ｜ 印刷页码 {_markup(evidence.printed_page_label or '未标注')} ｜ "
                        f"坐标 {_markup(evidence.bbox)} ｜ 证据置信度 {evidence.confidence:.0%}",
                        styles["small"],
                    ),
                ]
            )
            if evidence.evidence_type is EvidenceType.TABLE:
                story.extend(_table_flowables(evidence, styles))
            else:
                display = report.evidence_display_zh_hans.get(evidence.evidence_id, evidence.quote)
                story.extend([Paragraph("原文简体转写", styles["h3"]), Paragraph(_markup(display), styles["quote"])])
            image = _evidence_image(evidence, project_root=project_root)
            if image is not None:
                story.append(
                    KeepTogether(
                        [
                            Paragraph("原始 PDF 截图（保持原貌）", styles["h3"]),
                            image,
                            Spacer(1, 2 * mm),
                        ]
                    )
                )

    story.extend([PageBreak(), Paragraph("多时间窗风险预警", styles["h1"])])
    if prediction is None:
        story.append(Paragraph("本次导出未生成可用的多时间窗风险预测。", styles["body"]))
    else:
        story.extend(
            [
                _prediction_direction_table(prediction, styles),
                Spacer(1, 4 * mm),
                Paragraph(_markup(prediction.get("disclaimer", "方向性风险预警，不构成投资建议。")), styles["disclaimer"]),
            ]
        )

    story.extend([Spacer(1, 9 * mm), Paragraph("人工复核结论", styles["h1"])])
    review_rows: list[list[Paragraph]] = [
        [Paragraph("风险", styles["cell_head"]), Paragraph("复核人", styles["cell_head"]), Paragraph("结论", styles["cell_head"]), Paragraph("意见", styles["cell_head"])]
    ]
    for finding in ranked:
        review = reviews_by_finding.get(finding.finding_id)
        review_rows.append(
            [
                Paragraph(_markup(finding.title), styles["cell"]),
                Paragraph(_markup(review.reviewer if review else "未记录"), styles["cell"]),
                Paragraph(_REVIEW_LABELS.get(review.decision.value, review.decision.value) if review else "未记录", styles["cell"]),
                Paragraph(_markup(review.comment if review and review.comment else "无"), styles["cell"]),
            ]
        )
    story.append(_styled_table(review_rows, col_widths=(67 * mm, 28 * mm, 24 * mm, 41 * mm), header=True, repeat_rows=1))

    story.extend([PageBreak(), Paragraph("技术分析链路", styles["h1"])])
    story.extend(
        _formal_technology_flowables(
            report,
            traces=execution_traces,
            feature_payload=feature_payload,
            prediction=prediction,
            reviews=reviews,
            styles=styles,
        )
    )

    story.extend([PageBreak(), Paragraph("附录：证据索引与追溯信息", styles["h1"])])
    index_rows: list[list[Paragraph]] = [
        [Paragraph("证据ID", styles["cell_head"]), Paragraph("对应风险", styles["cell_head"]), Paragraph("页码", styles["cell_head"]), Paragraph("类型", styles["cell_head"]), Paragraph("置信度", styles["cell_head"])]
    ]
    finding_title_by_evidence = {evidence_id: finding.title for finding in ranked for evidence_id in finding.evidence_ids}
    for evidence in report.evidences:
        index_rows.append(
            [
                Paragraph(_markup(evidence.evidence_id), styles["cell"]),
                Paragraph(_markup(finding_title_by_evidence.get(evidence.evidence_id, "-")), styles["cell"]),
                Paragraph(f"物理 {evidence.page}<br/>印刷 {_markup(evidence.printed_page_label or '-')}", styles["cell"]),
                Paragraph(_markup(evidence.evidence_type.value), styles["cell"]),
                Paragraph(f"{evidence.confidence:.0%}", styles["cell"]),
            ]
        )
    story.append(_styled_table(index_rows, col_widths=(32 * mm, 75 * mm, 23 * mm, 16 * mm, 14 * mm), header=True, repeat_rows=1))
    story.extend(
        [
            Spacer(1, 7 * mm),
            Paragraph("模型、来源与免责声明", styles["h2"]),
            Paragraph(
                "预测技术：结构化特征融合、集成学习与上市前时点约束<br/>"
                f"源文件：{_markup(report.source_document.source_file)}<br/>"
                f"文档ID：{_markup(report.source_document.document_id)}<br/>"
                f"SHA-256：{_markup(report.source_document.source_hash)}<br/>"
                f"分析运行：{_markup(report.run_id)}",
                styles["body"],
            ),
            Paragraph("完整 Agent 轨迹、工具调用、被驳回风险和复核快照保存在 analysis_audit_bundle.json。报告仅供研究和风险提示，不构成投资建议。", styles["disclaimer"]),
        ]
    )
    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return path


def write_audit_pdf_report(
    report: RiskAnalysisReport,
    path: Path,
    *,
    final_findings: Sequence[RiskFinding],
    reviews: Sequence[FindingReview],
    project_root: Path,
    prediction: Mapping[str, Any] | None = None,
    company_profile: Mapping[str, Any] | None = None,
    feature_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Write a review-first audit PDF; raw payloads remain in the audit JSON."""
    _register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _report_styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"{report.company_name} IPO分析完整审计报告",
        author="港股 IPO 风险预警系统",
        subject="全部风险、原文证据、人工复核与执行轨迹审计",
    )
    traces = tuple(report.agent_traces)
    tool_calls = tuple(call for trace in traces for call in trace.tool_calls)
    profile = company_profile or {}
    ranked_findings = _ranked_findings(final_findings)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in report.evidences}
    reviews_by_finding: dict[str, list[FindingReview]] = defaultdict(list)
    for review in sorted(reviews, key=lambda item: item.created_at):
        reviews_by_finding[review.finding_id].append(review)
    verified_count = sum(finding.status.value == "verified" for finding in final_findings)
    rejected_count = sum(finding.status.value == "rejected" for finding in final_findings)
    story: list[object] = [
        Spacer(1, 28 * mm),
        Paragraph("港股 IPO 风险预警系统", styles["cover_label"]),
        Spacer(1, 5 * mm),
        Paragraph("完整分析审计报告", styles["cover_title"]),
        Spacer(1, 8 * mm),
        Paragraph(
            f"{_markup(report.company_name)}（{_markup(report.company_id)}）",
            styles["cover_company"],
        ),
        Spacer(1, 13 * mm),
        _styled_table(
            [
                [Paragraph("运行编号", styles["small"]), Paragraph(_markup(report.run_id), styles["body"])],
                [Paragraph("上市日期", styles["small"]), Paragraph(_markup(profile.get("listing_date") or "未披露"), styles["body"])],
                [Paragraph("招股书日期", styles["small"]), Paragraph(_format_date(report.source_document.available_at), styles["body"])],
                [Paragraph("风险记录", styles["small"]), Paragraph(f"{len(final_findings)}（确认 {verified_count} / 驳回 {rejected_count}）", styles["body"])],
                [Paragraph("原文证据", styles["small"]), Paragraph(str(len(report.evidences)), styles["body"])],
                [Paragraph("Agent轨迹 / 工具调用", styles["small"]), Paragraph(f"{len(traces)} / {len(tool_calls)}", styles["body"])],
                [Paragraph("人工复核记录", styles["small"]), Paragraph(str(len(reviews)), styles["body"])],
                [Paragraph("报告时间（北京时间）", styles["small"]), Paragraph(_format_datetime(report.generated_at), styles["body"])],
            ],
            col_widths=(43 * mm, 112 * mm),
        ),
        Spacer(1, 12 * mm),
        Paragraph(
            "本PDF按风险集中呈现全部原文证据和人工复核记录；模型原始输出、提示词及工具输入输出仅保存在机器可读审计JSON中。",
            styles["disclaimer"],
        ),
        PageBreak(),
        Paragraph("一、审计摘要", styles["h1"]),
    ]
    story.extend([
        Paragraph(
            f"本次审计覆盖 {len(final_findings)} 条风险、{len(report.evidences)} 条原文证据、"
            f"{len(reviews)} 条人工复核记录、{len(traces)} 条 Agent 轨迹和 {len(tool_calls)} 次工具调用。",
            styles["body"],
        ),
        *(
            [
                Paragraph("招股书文本粉饰分析", styles["h2"]),
                Paragraph(
                    f"粉饰程度：{_markup({'low': '低', 'medium': '中', 'high': '高', 'critical': '严重'}[report.text_embellishment.level])}"
                    f"｜原文证据 {len(report.text_embellishment.evidence_ids)} 条"
                    f"{'｜已完成补证辩论' if report.text_embellishment.debate_used else ''}<br/>"
                    f"{_markup(report.text_embellishment.summary)}",
                    styles["body"],
                ),
            ]
            if report.text_embellishment is not None
            else []
        ),
        _styled_table(
            [
                [Paragraph("项目", styles["cell_head"]), Paragraph("数量", styles["cell_head"]), Paragraph("说明", styles["cell_head"])],
                [Paragraph("已确认风险", styles["cell"]), Paragraph(str(verified_count), styles["cell"]), Paragraph("人工确认或编辑后确认", styles["cell"])],
                [Paragraph("已驳回风险", styles["cell"]), Paragraph(str(rejected_count), styles["cell"]), Paragraph("保留在审计报告中，不进入正式风险结论", styles["cell"])],
                [Paragraph("原文证据", styles["cell"]), Paragraph(str(len(report.evidences)), styles["cell"]), Paragraph("按对应风险完整展开", styles["cell"])],
                [Paragraph("人工复核", styles["cell"]), Paragraph(str(len(reviews)), styles["cell"]), Paragraph("包含复核人、意见、决定和时间", styles["cell"])],
            ],
            col_widths=(42 * mm, 22 * mm, 99 * mm),
            header=True,
            repeat_rows=1,
        ),
        Spacer(1, 6 * mm),
        Paragraph("二、全部风险、证据与人工复核", styles["h1"]),
        Paragraph("以下按风险逐项展开；已驳回风险也完整保留，以便复盘人工判断。", styles["small"]),
    ])

    displayed_evidence_ids: set[str] = set()
    for finding_index, finding in enumerate(ranked_findings, start=1):
        if finding_index > 1:
            story.append(PageBreak())
        story.extend([
            HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#0EA5E9"), spaceBefore=4, spaceAfter=8),
            Paragraph(f"风险 {finding_index}：{_markup(finding.title)}", styles["h1"]),
            _styled_table(
                [
                    [Paragraph("风险编号", styles["small"]), Paragraph(_markup(finding.finding_id), styles["body"])],
                    [Paragraph("负责 Agent", styles["small"]), Paragraph(_markup(_ROLE_LABELS.get(finding.created_by, finding.created_by.value)), styles["body"])],
                    [Paragraph("风险等级", styles["small"]), Paragraph(_SEVERITY_LABELS[finding.severity], styles["body"])],
                    [Paragraph("复核状态", styles["small"]), Paragraph(_FINDING_STATUS_LABELS.get(finding.status.value, _markup(finding.status.value)), styles["body"])],
                ],
                col_widths=(34 * mm, 129 * mm),
            ),
            Spacer(1, 4 * mm),
            Paragraph("风险结论", styles["h2"]),
            Paragraph(_markup(finding.claim), styles["body"]),
            Paragraph("人工复核记录", styles["h2"]),
        ])
        finding_reviews = reviews_by_finding.get(finding.finding_id, [])
        if not finding_reviews:
            story.append(Paragraph("未记录人工复核。", styles["body"]))
        for review_index, review in enumerate(finding_reviews, start=1):
            verification = review.verification
            all_checks = list(verification.checks) + [
                check
                for evidence_verification in verification.evidence_verifications
                for check in evidence_verification.checks
            ]
            failed_checks = sum(not check.passed for check in all_checks)
            story.append(
                Paragraph(
                    f"第 {review_index} 次复核｜复核人：{_markup(review.reviewer)}｜"
                    f"决定：{_REVIEW_LABELS.get(review.decision.value, _markup(review.decision.value))}｜"
                    f"时间（北京时间）：{_format_datetime(review.created_at)}<br/>"
                    f"复核意见：{_markup(review.comment or '无')}<br/>"
                    f"自动核验：{'通过' if verification.passed else '未通过'}"
                    f"（检查 {len(all_checks)} 项，未通过 {failed_checks} 项）"
                    f"{'｜人工覆盖确认' if review.manual_override else ''}",
                    styles["quote"],
                )
            )

        story.append(Paragraph(f"原文证据（{len(finding.evidence_ids)} 条）", styles["h2"]))
        for evidence_index, evidence_id in enumerate(finding.evidence_ids, start=1):
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                story.append(Paragraph(f"证据 {evidence_index}：{_markup(evidence_id)}（记录缺失）", styles["disclaimer"]))
                continue
            displayed_evidence_ids.add(evidence.evidence_id)
            display = report.evidence_display_zh_hans.get(evidence.evidence_id, evidence.quote)
            story.extend([
                Paragraph(f"证据 {evidence_index}：{_markup(evidence.evidence_id)}", styles["h3"]),
                Paragraph(
                    f"PDF物理页 {evidence.page}｜印刷页 {_markup(evidence.printed_page_label or '未标注')}｜"
                    f"类型 {_EVIDENCE_TYPE_LABELS.get(evidence.evidence_type.value, _markup(evidence.evidence_type.value))}｜"
                    f"提取 Agent {_markup(_ROLE_LABELS.get(evidence.extractor, evidence.extractor.value))}",
                    styles["small"],
                ),
            ])
            if evidence.evidence_type is EvidenceType.TABLE:
                story.extend(_table_flowables(evidence, styles))
            else:
                story.extend([Paragraph("原文简体转写", styles["h3"]), Paragraph(_markup(display), styles["quote"])])
            image = _evidence_image(evidence, project_root=project_root)
            if image is not None:
                story.extend([Paragraph("原始 PDF 证据截图", styles["h3"]), image, Spacer(1, 3 * mm)])

    unlinked_evidences = [
        evidence for evidence in report.evidences if evidence.evidence_id not in displayed_evidence_ids
    ]
    if unlinked_evidences:
        story.extend([PageBreak(), Paragraph("三、未关联风险的原文证据", styles["h1"])])
        for evidence in unlinked_evidences:
            display = report.evidence_display_zh_hans.get(evidence.evidence_id, evidence.quote)
            story.extend([
                Paragraph(f"证据：{_markup(evidence.evidence_id)}", styles["h2"]),
                Paragraph(f"PDF物理页 {evidence.page}｜印刷页 {_markup(evidence.printed_page_label or '未标注')}", styles["small"]),
            ])
            if evidence.evidence_type is EvidenceType.TABLE:
                story.extend(_table_flowables(evidence, styles))
            else:
                story.append(Paragraph(_markup(display), styles["quote"]))

    execution_section_number = "四" if unlinked_evidences else "三"
    feature_section_number = "五" if unlinked_evidences else "四"
    source_section_number = "六" if unlinked_evidences else "五"
    story.extend([PageBreak(), Paragraph(f"{execution_section_number}、Agent与关键工具链", styles["h1"])] )
    story.append(Paragraph(
        "本节展示角色执行轨迹和关键工具类别。重复截图、检索与核验调用按类别汇总；"
        "每次调用的编号、输入、输出、时间和异常仍完整保存在原始审计JSON中。",
        styles["small"],
    ))
    if not traces:
        story.append(Paragraph("本次运行没有保存 Agent 执行轨迹。", styles["body"]))
    else:
        trace_rows: list[list[Paragraph]] = [[
            Paragraph("角色 / 阶段", styles["cell_head"]),
            Paragraph("状态", styles["cell_head"]),
            Paragraph("开始（北京时间）", styles["cell_head"]),
            Paragraph("结束（北京时间）", styles["cell_head"]),
            Paragraph("耗时 / 工具", styles["cell_head"]),
        ]]
        for trace in traces:
            duration = max(0.0, (trace.ended_at - trace.started_at).total_seconds())
            trace_rows.append([
                Paragraph(_markup(_ROLE_LABELS.get(trace.agent, trace.agent.value)), styles["cell"]),
                Paragraph(_EXECUTION_STATUS_LABELS.get(trace.status.value, _markup(trace.status.value)), styles["cell"]),
                Paragraph(_format_datetime(trace.started_at), styles["cell"]),
                Paragraph(_format_datetime(trace.ended_at), styles["cell"]),
                Paragraph(f"{duration:.2f} 秒<br/>{len(trace.tool_calls)} 次", styles["cell"]),
            ])
        story.append(_styled_table(trace_rows, col_widths=(46 * mm, 18 * mm, 37 * mm, 37 * mm, 25 * mm), header=True, repeat_rows=1))

        tool_labels = {
            "evidence_retrieval": ("证据检索", "从招股书索引定位候选原文与表格"),
            "evidence_analysis": ("证据核验", "核对结论与证据是否一致"),
            "render_evidence_screenshot": ("证据截图", "生成带页码和坐标的原文截图"),
            "open_risk_discovery": ("开放风险发现", "阅读招股书并识别非标准风险主题"),
            "grounded_risk_analysis": ("专业证据分析", "依据已召回证据形成专业风险判断"),
            "risk_adjudication": ("风险裁决", "归并冲突意见并形成最终风险结论"),
            "risk_debate_verification": ("争议风险补证辩论", "由决策Agent质疑、专业Agent定向查证并再次裁决"),
            "agent_model_assignment": ("Agent模型编排锁定", "分析开始前固化六个Agent的模型分配并写入审计记录"),
            "company_risk_scoping": ("分析范围识别", "识别行业、业务模式与适用风险范围"),
            "langgraph_state_graph": ("多 Agent 流程编排", "调度专业角色并维护分析状态"),
            "model_usage_budget": ("模型调用控制", "控制模型调用范围与运行资源"),
            "parse_document": ("招股书解析", "提取页面、正文、表格与页面坐标"),
            "llm_risk_discovery": ("风险发现", "识别招股书中的候选风险主题"),
            "llm_batch_analysis": ("专业分析", "按角色批量分析候选风险"),
            "llm_decision": ("决策归并", "处理重复、冲突并形成最终风险"),
        }
        grouped_calls: dict[tuple[AgentRole, str], dict[str, object]] = defaultdict(
            lambda: {"count": 0, "succeeded": 0, "failed": 0, "duration": 0.0, "errors": set()}
        )
        for trace in traces:
            for call in trace.tool_calls:
                duration = max(0.0, (call.ended_at - call.started_at).total_seconds())
                stats = grouped_calls[(trace.agent, call.tool_name)]
                stats["count"] = int(stats["count"]) + 1
                stats["duration"] = float(stats["duration"]) + duration
                outcome = "failed" if call.status.value == "failed" else "succeeded"
                stats[outcome] = int(stats[outcome]) + 1
                if call.error:
                    errors = stats["errors"]
                    if isinstance(errors, set):
                        errors.add(call.error)
        if grouped_calls:
            tool_rows: list[list[Paragraph]] = [[
                Paragraph("角色 / 关键工具", styles["cell_head"]),
                Paragraph("业务作用", styles["cell_head"]),
                Paragraph("调用", styles["cell_head"]),
                Paragraph("结果", styles["cell_head"]),
                Paragraph("总耗时 / 异常", styles["cell_head"]),
            ]]
            for (agent, tool_name), stats in sorted(
                grouped_calls.items(),
                key=lambda item: (_ROLE_LABELS.get(item[0][0], item[0][0].value), item[0][1]),
            ):
                label, purpose = tool_labels.get(
                    tool_name,
                    (_zh(tool_name).replace("_", " "), "执行分析流程中的确定性工具操作"),
                )
                errors = stats["errors"]
                error_count = len(errors) if isinstance(errors, set) else 0
                tool_rows.append([
                    Paragraph(f"{_markup(_ROLE_LABELS.get(agent, agent.value))}<br/>{_markup(label)}", styles["cell"]),
                    Paragraph(_markup(purpose), styles["cell"]),
                    Paragraph(f"{stats['count']} 次", styles["cell"]),
                    Paragraph(f"成功 {stats['succeeded']}<br/>失败 {stats['failed']}", styles["cell"]),
                    Paragraph(f"{float(stats['duration']):.2f} 秒" + (f"<br/>{error_count} 类异常" if error_count else ""), styles["cell"]),
                ])
            story.extend([Spacer(1, 7 * mm), Paragraph("关键工具调用汇总", styles["h2"]), _styled_table(
                tool_rows,
                col_widths=(38 * mm, 55 * mm, 15 * mm, 25 * mm, 30 * mm),
                header=True,
                repeat_rows=1,
            )])

    story.extend([
        PageBreak(),
        Paragraph(f"{feature_section_number}、特征构建与确定性计算", styles["h1"]),
    ])
    story.extend(_feature_audit_flowables(feature_payload, styles))

    story.extend([
        PageBreak(),
        Paragraph(f"{source_section_number}、来源与审计说明", styles["h1"]),
        Paragraph(
            f"源文件：{_markup(report.source_document.source_file)}<br/>"
            f"文档ID：{_markup(report.source_document.document_id)}<br/>"
            f"SHA-256：{_markup(report.source_document.source_hash)}<br/>"
            f"分析运行：{_markup(report.run_id)}<br/>"
            f"招股书日期：{_format_date(report.source_document.available_at)}<br/>"
            f"报告时间（北京时间）：{_format_datetime(report.generated_at)}",
            styles["body"],
        ),
        Paragraph(
            "本PDF完整呈现风险、证据、复核意见及执行记录摘要。模型原始输出、提示词、工具输入输出、嵌套字段原值及机器重放数据以同目录的 analysis_audit_bundle.json 为准。",
            styles["disclaimer"],
        ),
    ])
    document.build(
        story,
        onFirstPage=_audit_page_footer,
        onLaterPages=_audit_page_footer,
    )
    return path


__all__ = ["write_audit_pdf_report", "write_verified_pdf_report"]
