"""Human-readable verified IPO risk report in PDF format."""

from __future__ import annotations

from collections import Counter
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ipo_risk.domain import Evidence, FindingReview, RiskSeverity
from ipo_risk.reporting.json_report import RiskAnalysisReport


_FONT_REGULAR = "IPORiskChinese"
_FONT_BOLD = "IPORiskChineseBold"
_SEVERITY_LABELS = {
    RiskSeverity.CRITICAL: "极高风险",
    RiskSeverity.HIGH: "高风险",
    RiskSeverity.MEDIUM: "中风险",
    RiskSeverity.LOW: "低风险",
}
_SEVERITY_COLORS = {
    RiskSeverity.CRITICAL: colors.HexColor("#B91C1C"),
    RiskSeverity.HIGH: colors.HexColor("#DC2626"),
    RiskSeverity.MEDIUM: colors.HexColor("#D97706"),
    RiskSeverity.LOW: colors.HexColor("#047857"),
}


def _register_fonts() -> None:
    if _FONT_REGULAR in pdfmetrics.getRegisteredFontNames():
        return
    candidates = (
        (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        ),
        (
            Path(r"C:\Windows\Fonts\simsun.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(regular)))
            pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(bold)))
            return
    raise FileNotFoundError("Chinese report fonts are not available")


def _markup(value: object) -> str:
    return escape(str(value)).replace("\n", "<br/>")


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _report_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_label": ParagraphStyle(
            "CoverLabel",
            parent=sample["Normal"],
            fontName=_FONT_REGULAR,
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#64748B"),
            alignment=TA_CENTER,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName=_FONT_BOLD,
            fontSize=25,
            leading=36,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
        ),
        "cover_company": ParagraphStyle(
            "CoverCompany",
            parent=sample["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=17,
            leading=26,
            textColor=colors.HexColor("#0369A1"),
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=sample["Heading1"],
            fontName=_FONT_BOLD,
            fontSize=18,
            leading=27,
            spaceAfter=10,
            textColor=colors.HexColor("#0F172A"),
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=sample["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=13,
            leading=21,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#0F172A"),
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=10,
            leading=17,
            spaceAfter=7,
            textColor=colors.HexColor("#243247"),
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=8.5,
            leading=14,
            spaceAfter=5,
            textColor=colors.HexColor("#64748B"),
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "ReportQuote",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=9,
            leading=15,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.7,
            borderPadding=8,
            backColor=colors.HexColor("#F8FAFC"),
            textColor=colors.HexColor("#334155"),
            wordWrap="CJK",
        ),
        "disclaimer": ParagraphStyle(
            "ReportDisclaimer",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=9,
            leading=16,
            borderColor=colors.HexColor("#F59E0B"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=colors.HexColor("#FFFBEB"),
            textColor=colors.HexColor("#78350F"),
            wordWrap="CJK",
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


def _evidence_image(
    evidence: Evidence,
    *,
    project_root: Path,
) -> Image | None:
    path = _resolve_path(evidence.screenshot_path, project_root)
    if path is None or not path.is_file():
        return None
    payload = path.read_bytes()
    reader = ImageReader(BytesIO(payload))
    width, height = reader.getSize()
    max_width = 165 * mm
    max_height = 92 * mm
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(
        BytesIO(payload),
        width=width * scale,
        height=height * scale,
    )


def _review_by_finding(
    reviews: tuple[FindingReview, ...],
) -> dict[str, FindingReview]:
    latest: dict[str, FindingReview] = {}
    for review in sorted(reviews, key=lambda item: item.created_at):
        latest[review.finding_id] = review
    return latest


def write_verified_pdf_report(
    report: RiskAnalysisReport,
    path: Path,
    *,
    reviews: tuple[FindingReview, ...] = (),
    project_root: Path,
    prediction: Mapping[str, Any] | None = None,
) -> Path:
    """Write a concise formal PDF without rewriting findings or evidence."""
    _register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _report_styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"{report.company_name} IPO风险正式报告",
        author="港股 IPO 风险预警系统",
        subject="人工复核后的证据化风险结论",
    )
    story: list[object] = []
    story.extend(
        [
            Spacer(1, 34 * mm),
            Paragraph("港股 IPO 风险预警", styles["cover_label"]),
            Spacer(1, 5 * mm),
            Paragraph("人工复核正式报告", styles["cover_title"]),
            Spacer(1, 8 * mm),
            Paragraph(
                f"{_markup(report.company_name)}（{_markup(report.company_id)}）",
                styles["cover_company"],
            ),
            Spacer(1, 15 * mm),
        ]
    )
    cover_rows = [
        ["分析运行", report.run_id],
        ["招股书日期", report.source_document.available_at.isoformat()],
        ["人工确认风险", str(report.summary.finding_count)],
        ["引用原文证据", str(report.summary.evidence_count)],
        ["生成时间", report.generated_at.isoformat()],
    ]
    if prediction is not None:
        cover_rows.insert(
            4,
            [
                "五日显著下跌预警",
                f"{float(prediction.get('probability', 0.0)):.1%}",
            ],
        )
    cover_table = Table(
        [
            [
                Paragraph(_markup(label), styles["small"]),
                Paragraph(_markup(value), styles["body"]),
            ]
            for label, value in cover_rows
        ],
        colWidths=(35 * mm, 115 * mm),
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(cover_table)
    story.extend(
        [
            Spacer(1, 16 * mm),
            Paragraph(
                "本报告仅包含通过确定性核验并经人工确认的风险结论。"
                "风险结论与证据原文均来自系统保存的结构化对象，排版过程未使用大模型改写。"
                "本报告不构成投资建议。",
                styles["disclaimer"],
            ),
            PageBreak(),
        ]
    )

    if prediction is not None:
        decision_labels = {
            "high_risk": "高风险预警",
            "low_risk": "低风险",
            "uncertain": "边界信号，暂不作高风险预警",
        }
        probability = float(prediction.get("probability", 0.0))
        confidence = float(prediction.get("confidence", 0.0))
        decision = str(prediction.get("decision", "uncertain"))
        recall_guard = prediction.get("recall_guard")
        if not isinstance(recall_guard, Mapping):
            recall_guard = {}
        watchlist = bool(recall_guard.get("watchlist", False))
        guard_reasons = recall_guard.get("reasons")
        if not isinstance(guard_reasons, list):
            guard_reasons = []
        prediction_rows = [
            ["预测目标", "上市后5个交易日内显著下跌风险"],
            ["风险概率", f"{probability:.1%}"],
            ["方向判断", decision_labels.get(decision, decision)],
            ["模型置信度", f"{confidence:.1%}"],
        ]
        if watchlist:
            prediction_rows.extend(
                [
                    ["召回保护层", "风险观察名单"],
                    [
                        "保护层原因",
                        "、".join(str(item) for item in guard_reasons)
                        or "多项稳定风险因子叠加",
                    ],
                ]
            )
        prediction_table = Table(
            [
                [
                    Paragraph(_markup(label), styles["small"]),
                    Paragraph(_markup(value), styles["body"]),
                ]
                for label, value in prediction_rows
            ],
            colWidths=(42 * mm, 108 * mm),
        )
        prediction_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E0F2FE")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend(
            [
                Paragraph("上市后五日风险预警", styles["h1"]),
                prediction_table,
                Spacer(1, 4 * mm),
                Paragraph(
                    _markup(str(prediction.get("disclaimer", "方向性风险预警，不构成投资建议。"))),
                    styles["disclaimer"],
                ),
                Spacer(1, 8 * mm),
            ]
        )

    story.append(Paragraph("风险总览", styles["h1"]))

    severity_counts = Counter(finding.severity for finding in report.risk_findings)
    summary_data = [
        [
            Paragraph("风险等级", styles["small"]),
            Paragraph("数量", styles["small"]),
        ]
    ]
    for severity in (
        RiskSeverity.CRITICAL,
        RiskSeverity.HIGH,
        RiskSeverity.MEDIUM,
        RiskSeverity.LOW,
    ):
        summary_data.append(
            [
                Paragraph(_SEVERITY_LABELS[severity], styles["body"]),
                Paragraph(str(severity_counts.get(severity, 0)), styles["body"]),
            ]
        )
    summary_table = Table(summary_data, colWidths=(65 * mm, 30 * mm))
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2FE")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 8 * mm))
    for index, finding in enumerate(report.risk_findings, start=1):
        story.append(
            Paragraph(
                f"{index:02d} · {_markup(finding.title)}",
                styles["h2"],
            )
        )
        story.append(
            Paragraph(
                f"{_SEVERITY_LABELS[finding.severity]} ｜ "
                f"置信度 {finding.confidence:.0%} ｜ "
                f"证据 {len(finding.evidence_ids)} 条",
                styles["small"],
            )
        )
        story.append(Paragraph(_markup(finding.claim), styles["body"]))
        story.append(Spacer(1, 2 * mm))
    reviews_by_finding = _review_by_finding(reviews)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in report.evidences}
    for index, finding in enumerate(report.risk_findings, start=1):
        story.append(PageBreak())
        story.append(
            Paragraph(
                f"{index:02d} · {_markup(finding.title)}",
                styles["h1"],
            )
        )
        severity_color = _SEVERITY_COLORS[finding.severity]
        risk_meta = Table(
            [
                [
                    Paragraph(_SEVERITY_LABELS[finding.severity], styles["small"]),
                    Paragraph(f"置信度 {finding.confidence:.0%}", styles["small"]),
                    Paragraph(_markup(finding.created_by.value), styles["small"]),
                ]
            ],
            colWidths=(38 * mm, 45 * mm, 55 * mm),
        )
        risk_meta.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), severity_color),
                    ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                    ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend(
            [
                risk_meta,
                Spacer(1, 5 * mm),
                Paragraph("风险结论", styles["h2"]),
                Paragraph(_markup(finding.claim), styles["body"]),
            ]
        )
        if finding.uncertainty:
            story.extend(
                [
                    Paragraph("证据边界", styles["h2"]),
                    Paragraph(_markup(finding.uncertainty), styles["body"]),
                ]
            )
        review = reviews_by_finding.get(finding.finding_id)
        if review is not None:
            review_text = f"复核人：{review.reviewer} ｜ 复核时间：{review.created_at.isoformat()}"
            if review.comment:
                review_text += f"<br/>复核意见：{_markup(review.comment)}"
            story.extend(
                [
                    Paragraph("人工复核", styles["h2"]),
                    Paragraph(review_text, styles["body"]),
                ]
            )
        story.append(Paragraph("原文证据", styles["h2"]))
        for evidence_number, evidence_id in enumerate(finding.evidence_ids, start=1):
            evidence = evidence_by_id[evidence_id]
            printed_page = evidence.printed_page_label or "未标注"
            story.extend(
                [
                    HRFlowable(
                        width="100%",
                        thickness=0.6,
                        color=colors.HexColor("#CBD5E1"),
                        spaceBefore=5,
                        spaceAfter=6,
                    ),
                    Paragraph(
                        f"证据 {evidence_number} · {_markup(evidence.evidence_id)}",
                        styles["h2"],
                    ),
                    Paragraph(
                        f"PDF物理页 {evidence.page} ｜ 印刷页码 {_markup(printed_page)} ｜ "
                        f"类型 {_markup(evidence.evidence_type.value)} ｜ "
                        f"证据置信度 {evidence.confidence:.0%}",
                        styles["small"],
                    ),
                    Paragraph(_markup(evidence.quote), styles["quote"]),
                    Spacer(1, 3 * mm),
                ]
            )
            image = _evidence_image(evidence, project_root=project_root)
            if image is not None:
                story.append(image)
                story.append(Spacer(1, 3 * mm))
    story.extend(
        [
            PageBreak(),
            Paragraph("文档与追溯信息", styles["h1"]),
            Paragraph(
                f"源文件：{_markup(report.source_document.source_file)}<br/>"
                f"文档ID：{_markup(report.source_document.document_id)}<br/>"
                f"SHA-256：{_markup(report.source_document.source_hash)}<br/>"
                f"分析运行：{_markup(report.run_id)}",
                styles["body"],
            ),
            Paragraph(
                "完整 Agent 轨迹、工具调用、被驳回风险与人工复核快照保存在独立的"
                " analysis_audit_bundle.json 中，不属于本正式结论。",
                styles["disclaimer"],
            ),
        ]
    )
    document.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer,
    )
    return path


__all__ = ["write_verified_pdf_report"]
