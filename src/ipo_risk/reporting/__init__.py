"""报告结构与导出接口。"""

from .json_report import (
    AnalysisSummary,
    RiskAnalysisReport,
    SourceDocument,
    write_risk_report,
)

__all__ = [
    "AnalysisSummary",
    "RiskAnalysisReport",
    "SourceDocument",
    "write_risk_report",
]
