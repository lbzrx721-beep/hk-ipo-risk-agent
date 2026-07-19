"""单份招股书风险 JSON 报告的结构与写出函数。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.domain import AgentTrace, Evidence, RiskFinding
from ipo_risk.storage.files import write_json


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceDocument(ReportModel):
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    available_at: date
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)


class AnalysisSummary(ReportModel):
    evidence_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)
    screenshot_count: int = Field(ge=0)
    requires_human_review: bool = True


class RiskAnalysisReport(ReportModel):
    """可直接交给 API、界面和报告模块的第一阶段产物。"""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    company_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    generated_at: datetime
    source_document: SourceDocument
    summary: AnalysisSummary
    evidences: tuple[Evidence, ...]
    risk_findings: tuple[RiskFinding, ...]
    agent_traces: tuple[AgentTrace, ...]
    evidence_display_zh_hans: dict[str, str]

    @model_validator(mode="after")
    def validate_references_and_counts(self) -> RiskAnalysisReport:
        evidence_ids = [evidence.evidence_id for evidence in self.evidences]
        finding_ids = [finding.finding_id for finding in self.risk_findings]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("report evidence IDs must be unique")
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("report finding IDs must be unique")

        known_evidence = set(evidence_ids)
        for evidence in self.evidences:
            if evidence.company_id != self.company_id:
                raise ValueError("report evidence belongs to another company")
        for finding in self.risk_findings:
            if finding.company_id != self.company_id:
                raise ValueError("report finding belongs to another company")
            missing = set(finding.evidence_ids) - known_evidence
            if missing:
                raise ValueError(f"finding references missing evidence: {sorted(missing)}")

        if set(self.evidence_display_zh_hans) != known_evidence:
            raise ValueError("simplified display quotes must cover every evidence")
        if self.summary.evidence_count != len(self.evidences):
            raise ValueError("summary evidence_count does not match report")
        if self.summary.finding_count != len(self.risk_findings):
            raise ValueError("summary finding_count does not match report")
        screenshot_count = sum(evidence.screenshot_path is not None for evidence in self.evidences)
        if self.summary.screenshot_count != screenshot_count:
            raise ValueError("summary screenshot_count does not match report")
        return self


def write_risk_report(report: RiskAnalysisReport, path: Path) -> Path:
    """以 UTF-8 格式写出稳定字段名的风险 JSON。"""
    return write_json(report.model_dump(mode="json"), path, pretty=True)


__all__ = [
    "AnalysisSummary",
    "RiskAnalysisReport",
    "SourceDocument",
    "write_risk_report",
]

