"""报告结构与导出接口。"""

from .json_report import (
    AnalysisSummary,
    RiskAnalysisReport,
    SourceDocument,
    TextEmbellishmentAssessment,
    write_risk_report,
)
from .pdf_report import write_audit_pdf_report, write_verified_pdf_report

__all__ = [
    "AnalysisSummary",
    "RiskAnalysisReport",
    "SourceDocument",
    "TextEmbellishmentAssessment",
    "write_risk_report",
    "write_audit_pdf_report",
    "write_verified_pdf_report",
]
