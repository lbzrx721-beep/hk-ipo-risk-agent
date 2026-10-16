"""Deterministic verification and immutable human-review state transitions."""

from __future__ import annotations

import json
import csv
import re
import shutil
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz

from ipo_risk.agents.verifier import verify_finding as run_verification
from ipo_risk.domain import (
    AgentTrace,
    Evidence,
    FindingReview,
    FindingVerification,
    ReviewDecision,
    RiskFinding,
    VerificationStatus,
)
from ipo_risk.reporting import AnalysisSummary, RiskAnalysisReport, write_risk_report
from ipo_risk.reporting import write_audit_pdf_report, write_verified_pdf_report
from ipo_risk.modeling.prelisting import collect_prelisting_candidates
from ipo_risk.parser.document_parser import parse_page
from ipo_risk.parser.tables import TableArtifact
from ipo_risk.runtime.prediction import (
    RuntimePredictionUnavailable,
    enrich_horizon_prediction_status,
    generate_runtime_prediction_from_feature_file,
)
from ipo_risk.storage.files import read_json, write_json
from ipo_risk.storage.paths import safe_path_segment
from ipo_risk.storage.repositories import AnalysisRunRecord, CatalogRepository


class ReviewBlockedError(ValueError):
    """Raised when confirmation or editing fails deterministic verification."""

    def __init__(self, verification: FindingVerification) -> None:
        self.verification = verification
        failed = [check.name for check in verification.checks if not check.passed]
        failed.extend(
            f"evidence:{result.evidence_id}"
            for result in verification.evidence_verifications
            if not result.passed
        )
        super().__init__(f"finding verification failed: {', '.join(failed)}")


class IncompleteReviewError(ValueError):
    """Raised when a final report is requested before every finding is reviewed."""

    def __init__(self, finding_ids: tuple[str, ...]) -> None:
        self.finding_ids = finding_ids
        super().__init__(f"findings still require review: {', '.join(finding_ids)}")


def _bbox_intersection_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = fitz.Rect(first) & fitz.Rect(second)
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(1.0, fitz.Rect(first).get_area())


def _looks_like_unstructured_table(evidence: Evidence) -> bool:
    if evidence.evidence_type.value == "table" or evidence.table_id:
        return False
    lines = tuple(line.strip() for line in evidence.quote.splitlines() if line.strip())
    numeric_lines = sum(
        bool(re.fullmatch(r"(?:[-–—]|\(?[+-]?[\d,.]+\)?%?)", line))
        for line in lines
    )
    return numeric_lines >= 8 and len(lines) >= 12


@lru_cache(maxsize=256)
def _recover_page_tables(
    source_path: str,
    source_mtime_ns: int,
    page_number: int,
) -> tuple[TableArtifact, ...]:
    """按页恢复旧记录缺失的表格元数据；mtime 参与缓存键避免文件替换后复用。"""
    del source_mtime_ns
    document = fitz.open(source_path)
    try:
        page_result = parse_page(document, page_number - 1, min_chars=1)
        return tuple(page_result.tables)
    finally:
        document.close()


def build_evidence_table_preview(
    evidence: Evidence,
    *,
    project_root: Path,
) -> dict[str, object] | None:
    """为历史普通文本证据恢复二维表格，仅影响展示，不篡改审计证据。"""
    if not _looks_like_unstructured_table(evidence):
        return None
    source_path = Path(evidence.source_file)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    if not source_path.is_file():
        return None
    try:
        tables = _recover_page_tables(
            str(source_path.resolve()),
            source_path.stat().st_mtime_ns,
            evidence.page,
        )
    except Exception:  # noqa: BLE001 - 展示增强失败不能阻断人工复核
        return None
    matches = [
        table
        for table in tables
        if _bbox_intersection_ratio(evidence.bbox, table.bbox) >= 0.45
    ]
    if not matches:
        return None
    table = max(matches, key=lambda item: _bbox_intersection_ratio(evidence.bbox, item.bbox))
    metadata = [
        "[表格证据]",
        f"表格ID: {table.table_id}",
        f"标题: {table.title or '未识别'}",
    ]
    if table.unit:
        metadata.append(f"单位: {table.unit}")
    if table.currency:
        metadata.append(f"币种: {table.currency}")
    if table.period:
        metadata.append(f"期间: {table.period}")
    metadata.append(table.markdown)
    return {
        "quote": "\n".join(metadata),
        "table_id": table.table_id,
        "table_cell_refs": [cell.cell_id for cell in table.cells],
        "table_cell_values": [cell.value for cell in table.cells],
    }


def _report_file_stem(company_id: str, company_name: str, report_type: str) -> str:
    """生成适用于 Windows 和浏览器下载的正式报告文件名。"""
    safe_company = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", company_name).strip(" ._")
    safe_company = safe_company or company_id
    return f"{company_id}_{safe_company}_{report_type}_{date.today():%Y%m%d}"


@dataclass(frozen=True)
class ReviewRunSummary:
    """人工复核工作台中的一条运行及进度摘要。"""

    run_id: str
    company_id: str
    company_name: str
    as_of_date: str
    total_findings: int
    pending_findings: int
    verified_findings: int
    rejected_findings: int
    reviewed_findings: int
    complete: bool
    report_path: str | None
    verified_report_path: str | None
    base_feature_available: bool
    agent_feature_available: bool
    base_prediction: dict[str, Any] | None
    prediction: dict[str, Any] | None


@dataclass(frozen=True)
class ReviewFindingDetail:
    """页面展示一条风险所需的结论、证据、核验和审计记录。"""

    finding: RiskFinding
    evidences: tuple[Evidence, ...]
    verification: FindingVerification
    reviews: tuple[FindingReview, ...]
    agent_traces: tuple[AgentTrace, ...]


@dataclass(frozen=True)
class AnalysisDeletionResult:
    """一次分析删除后的可恢复位置和路径处理结果。"""

    run_id: str
    recovery_path: Path | None
    moved_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerifiedExportBundle:
    """Human PDF, compact verified JSON and the separate complete audit."""

    pdf_path: Path
    audit_pdf_path: Path
    json_path: Path
    audit_path: Path
    prediction_path: Path
    trace_path: Path
    evidence_path: Path


def _prediction_export_rows(
    prediction: dict[str, Any] | None,
    *,
    company_id: str,
    company_name: str,
) -> list[dict[str, Any]]:
    """固定导出赛题要求的四个窗口，不暴露内部概率。"""
    payload = prediction or {}
    horizon_values = payload.get("horizon_predictions")
    horizon_values = horizon_values if isinstance(horizon_values, dict) else {}
    horizon_thresholds = payload.get("horizon_thresholds")
    horizon_thresholds = horizon_thresholds if isinstance(horizon_thresholds, dict) else {}
    horizon_alerts = payload.get("horizon_alerts")
    horizon_alerts = horizon_alerts if isinstance(horizon_alerts, dict) else {}
    legacy_windows = payload.get("windows")
    legacy_windows = legacy_windows if isinstance(legacy_windows, dict) else {}
    rows: list[dict[str, Any]] = []
    for key, window, legacy_key in (
        ("h1", "1d", "1d"),
        ("h5", "5d", "5d"),
        ("h20", "20d", "20d"),
        ("h60", "60d", "60d"),
    ):
        legacy = legacy_windows.get(key) or legacy_windows.get(legacy_key)
        legacy = legacy if isinstance(legacy, dict) else {}
        value = horizon_values.get(key)
        if value is None and key == "h5":
            value = payload.get("probability")
        threshold = horizon_thresholds.get(key)
        alert = horizon_alerts.get(key)
        if isinstance(alert, bool):
            warning: bool | str = alert
            risk_level = "触发显著下跌预警" if warning else "未触发显著下跌预警"
        elif value is not None and threshold is not None:
            warning = float(value) >= float(threshold)
            risk_level = "触发显著下跌预警" if warning else "未触发显著下跌预警"
        elif legacy:
            warning = legacy.get(
                "warning",
                legacy.get("predicted_significant_drop", ""),
            )
            risk_level = str(legacy.get("risk_level") or legacy.get("decision") or "")
        elif key == "h5" and value is not None:
            decision = str(payload.get("decision", "uncertain"))
            warning = decision == "high_risk"
            risk_level = "触发显著下跌预警" if warning else "未触发显著下跌预警"
        else:
            warning = ""
            risk_level = "暂不可用"
        rows.append(
            {
                "company_id": company_id,
                "company_name": company_name,
                "window": window,
                "risk_level": risk_level,
                "warning": warning,
            }
        )
    return rows


def _business_toolchain(
    report: RiskAnalysisReport,
    feature_payload: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a concise, user-facing chain from durable execution evidence."""
    calls = [call for trace in report.agent_traces for call in trace.tool_calls]
    names = {call.tool_name for call in calls}
    debate_count = sum(call.tool_name == "risk_debate_verification" for call in calls)
    feature_names = sorted((feature_payload or {}).keys())
    groups = sorted({
        "市场" if name.startswith("market_") else
        "Agent语义" if name.startswith("agent_") else
        "发行" if name.startswith("offering_") else
        "财务" if name.startswith("fundamental_") else "其他"
        for name in feature_names
    })
    chain: list[dict[str, Any]] = [
        {"stage": "PDF解析", "status": "completed", "tools": sorted(name for name in names if "parse" in name), "summary": f"形成{len(report.evidences)}条可追溯证据对象。"},
        {"stage": "证据检索与核验", "status": "completed", "tools": sorted(name for name in names if "retrieval" in name or "analysis" in name), "summary": f"形成{len(report.risk_findings)}条结构化风险结论。"},
        {"stage": "多Agent协同", "status": "completed", "tools": sorted(name for name in names if "agent" in name or "decision" in name or "debate" in name), "summary": f"保存{len(report.agent_traces)}条Agent轨迹，完成{debate_count}项争议风险补证辩论。"},
    ]
    if feature_names:
        chain.append({"stage": "特征提取与计算", "status": "completed", "tools": ["feature_extraction", "deterministic_calculation", "currency_unit_normalization"], "summary": f"汇总{len(feature_names)}项特征，覆盖{'、'.join(groups)}特征组。"})
    if any(name.startswith("market_") for name in feature_names):
        chain.append({"stage": "上市前市场数据", "status": "completed", "tools": ["prelisting_market_lookup", "point_in_time_validation"], "summary": "按上市日前窗口构建市场特征并完成时点校验。"})
    if prediction:
        windows = prediction.get("windows")
        window_count = len(windows) if isinstance(windows, dict) else 1
        chain.append({"stage": "风险模型预测", "status": "completed", "tools": ["final_risk_prediction"], "summary": f"使用最终模型形成{window_count}个时间窗口的方向性风险结果。"})
    chain.append({"stage": "人工复核与报告", "status": "completed", "tools": ["finding_review", "report_generation"], "summary": "保存复核结论并生成正式报告与审计材料。"})
    return chain


def _debate_timeline(report: RiskAnalysisReport) -> list[dict[str, Any]]:
    """Expose durable debate records as a concise business timeline."""
    records: list[dict[str, Any]] = []
    for trace in report.agent_traces:
        for call in trace.tool_calls:
            if call.tool_name != "risk_debate_verification":
                continue
            inputs = call.inputs if isinstance(call.inputs, dict) else {}
            output = call.output if isinstance(call.output, dict) else {}
            risk_type = str(output.get("risk_type") or inputs.get("risk_type") or "")
            records.append(
                {
                    "debate_id": str(
                        output.get("debate_id") or call.tool_call_id
                    ),
                    "risk_type": risk_type,
                    "risk_name": str(output.get("risk_name") or risk_type or "争议风险"),
                    "challenge": str(
                        output.get("challenge") or inputs.get("challenge") or ""
                    ),
                    "challenge_role": str(
                        output.get("challenge_role")
                        or inputs.get("challenge_role")
                        or trace.agent.value
                    ),
                    "challenge_queries": list(
                        output.get("challenge_queries")
                        or inputs.get("challenge_queries")
                        or []
                    ),
                    "first_round_accepted": bool(
                        output.get("first_round_accepted", False)
                    ),
                    "final_accepted": bool(output.get("final_accepted", False)),
                    "response_status": str(
                        output.get("response_status") or "completed"
                    ),
                    "recalled_evidence_count": int(
                        output.get("recalled_evidence_count") or 0
                    ),
                    "new_evidence_count": int(output.get("new_evidence_count") or 0),
                    "final_rationale": str(output.get("final_rationale") or ""),
                    "status": str(output.get("status") or call.status.value),
                    "error": output.get("error") or call.error,
                    "started_at": call.started_at.isoformat(),
                    "ended_at": call.ended_at.isoformat(),
                }
            )
    return records


def _agent_model_assignments(report: RiskAnalysisReport) -> dict[str, str]:
    """Read the immutable six-Agent model routing snapshot from the run trace."""
    for trace in report.agent_traces:
        for call in trace.tool_calls:
            if call.tool_name != "agent_model_assignment":
                continue
            output = call.output if isinstance(call.output, dict) else {}
            assignments = output.get("assignments")
            if isinstance(assignments, dict):
                return {str(role): str(model) for role, model in assignments.items()}
    return {}



def _rag_feature_unit(name: str) -> str:
    """Map RAG feature name to its unit."""
    if name.endswith("_ratio_pct"):
        return "percent"
    if name.endswith("_lockup_months") or name.endswith("_runway_months"):
        return "months"
    if name.endswith("_stage_score"):
        return "score"
    if name in ("industry_primary", "pipeline_stage_name"):
        return "category"
    if "_share_count" in name:
        return "shares"
    return "native_currency_million"


def _rag_feature_group(name: str) -> str:
    """Map RAG feature name to its group."""
    if "customer" in name or "supplier" in name:
        return "business"
    if "cornerstone" in name or "lockup" in name or "share_count" in name:
        return "offering"
    if "industry" in name or "pipeline" in name or "cash_runway" in name:
        return "business"
    return "fundamental"


def _merge_rag_supplements(
    *,
    project_root: Path,
    feature_path: Path,
    company_id: str,
) -> None:
    """Merge RAG-extracted fundamental features into prelisting_features.json."""
    supplement_path = (
        project_root / "data" / "processed" / "modeling" / "rag_fundamental_feature_audit.jsonl"
    )
    if not supplement_path.is_file():
        return
    supplements: dict[str, Any] = {}
    for line in supplement_path.read_text(encoding="utf-8-sig").strip().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = str(record.get("company_id", "")).removesuffix(".0").strip()
        if cid.zfill(5) == company_id.zfill(5):
            supplements = record.get("features") or {}
            break
    if not supplements:
        return
    payload = read_json(feature_path)
    features = payload.get("features")
    if not isinstance(features, dict):
        return
    merged = 0
    for name, record in supplements.items():
        value = record.get("value") if isinstance(record, dict) else None
        if value is None:
            continue
        if name in features and features[name].get("value") is not None:
            continue
        unit = _rag_feature_unit(name)
        features[name] = {
            "value": value,
            "available_at": payload.get("prediction_as_of", ""),
            "source_ref": "rag_supplement",
            "formula_version": "prelisting_base_v1",
            "group": _rag_feature_group(name),
            "unit": unit,
        }
        merged += 1
    if merged:
        write_json(payload, feature_path, pretty=True)

class ReviewService:

    """Small public interface for checking and reviewing stored findings."""

    def __init__(
        self,
        *,
        repository: CatalogRepository,
        project_root: Path,
    ) -> None:
        self.repository = repository
        self.project_root = project_root.resolve()

    def _prediction_for_report(
        self,
        report_reference: str | None,
        *,
        variant: str = "agent_enhanced",
    ) -> dict[str, Any] | None:
        prediction_path = self._artifact_for_report(
            report_reference,
            "prediction_5d_base.json" if variant == "base" else "prediction_5d.json",
        )
        if prediction_path is None or not prediction_path.is_file():
            return None
        payload = read_json(prediction_path)
        return (
            enrich_horizon_prediction_status(
                project_root=self.project_root,
                prediction=payload,
            )
            if isinstance(payload, dict)
            else None
        )

    def _stored_prediction_for_run(self, run_id: str) -> dict[str, Any] | None:
        """Load the latest persisted prediction when artifact files are unavailable."""

        records = self.repository.list_predictions(run_id=run_id)
        return (
            enrich_horizon_prediction_status(
                project_root=self.project_root,
                prediction=records[0].payload,
            )
            if records
            else None
        )

    def _artifact_for_report(
        self,
        report_reference: str | None,
        artifact_name: str,
    ) -> Path | None:
        if not report_reference:
            return None
        report_path = Path(report_reference)
        if not report_path.is_absolute():
            report_path = self.project_root / report_path
        resolved = report_path.resolve().with_name(artifact_name)
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            return None
        return resolved

    def _company_profile_for_report(self, company_id: str) -> dict[str, Any] | None:
        """Load the official company metadata used by the formal PDF."""
        root = self.project_root / "data" / "raw" / "company_info"
        matches = sorted(root.glob(f"{company_id}_*/公司信息.json"))
        if not matches:
            return None
        payload = read_json(matches[0])
        return payload if isinstance(payload, dict) else None

    def _feature_payload_for_report(
        self,
        report_reference: str | None,
    ) -> dict[str, Any] | None:
        """Load preserved feature objects without recomputing PDF analysis."""
        merged: dict[str, Any] = {}
        for name in ("prelisting_features_base.json", "prelisting_features.json"):
            path = self._artifact_for_report(report_reference, name)
            if path is None or not path.is_file():
                continue
            payload = read_json(path)
            features = payload.get("features") if isinstance(payload, dict) else None
            if isinstance(features, dict):
                merged.update(features)
        return merged or None

    def audit_materials(self, run_id: str) -> dict[str, Any]:
        """Read the trace and evidence views from durable database records."""
        reference = self.repository.get_analysis_report_path(run_id)
        if reference is None:
            raise KeyError(f"analysis run does not exist: {run_id}")
        report_path = Path(reference)
        if not report_path.is_absolute():
            report_path = self.project_root / report_path
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = RiskAnalysisReport.model_validate(read_json(report_path))
        traces = self.repository.list_agent_traces(run_id)
        evidences = self.repository.list_evidence(run_id)
        findings = self.repository.list_findings(run_id)
        durable_report = report.model_copy(
            update={"agent_traces": traces, "evidences": evidences, "risk_findings": findings}
        )
        feature_payload = self._feature_payload_for_report(reference)
        prediction = self._prediction_for_report(reference) or self._stored_prediction_for_run(run_id)
        evidence_findings: dict[str, list[dict[str, str]]] = {}
        for finding in findings:
            for evidence_id in finding.evidence_ids:
                evidence_findings.setdefault(evidence_id, []).append(
                    {
                        "finding_id": finding.finding_id,
                        "title": finding.title,
                        "risk_type": finding.risk_type,
                        "severity": finding.severity.value,
                    }
                )
        return {
            "run_id": run_id,
            "debates": _debate_timeline(durable_report),
            "agent_models": _agent_model_assignments(durable_report),
            "model_assignment_locked": True,
            "text_embellishment": (
                durable_report.text_embellishment.model_dump(mode="json")
                if durable_report.text_embellishment is not None
                else None
            ),
            "business_toolchain": _business_toolchain(
                durable_report, feature_payload, prediction
            ),
            "agent_traces": [trace.model_dump(mode="json") for trace in traces],
            "evidences": [
                {
                    **evidence.model_dump(mode="json"),
                    "findings": evidence_findings.get(evidence.evidence_id, []),
                }
                for evidence in evidences
            ],
        }

    def _repair_legacy_prediction_features(
        self,
        *,
        run: AnalysisRunRecord,
        feature_path: Path,
        report_path: Path,
        listing_date: date | None,
    ) -> None:
        """按现有报告重建缺少上市时点的旧候选池，不重跑PDF或Agent。"""

        payload = read_json(feature_path)
        if payload.get("prediction_as_of") or listing_date is None:
            return
        company_info = self._company_profile_for_report(run.company_id)
        if company_info is None:
            return
        report = RiskAnalysisReport.model_validate(read_json(report_path))
        company = self.repository.get_company(run.company_id)
        company_folder = company.company_folder if company else report_path.parent.parent.name
        document_id = report.source_document.document_id
        tables_path = (
            self.project_root
            / "data"
            / "interim"
            / "tables"
            / company_folder
            / document_id
            / "tables.jsonl"
        )
        chunks_path = (
            self.project_root
            / "data"
            / "interim"
            / "chunks"
            / company_folder
            / document_id
            / "chunks.jsonl"
        )
        snapshot = collect_prelisting_candidates(
            report,
            project_root=self.project_root,
            company_info=company_info,
            prediction_as_of=listing_date,
            tables_path=tables_path if tables_path.is_file() else None,
            chunks_path=chunks_path if chunks_path.is_file() else None,
            listing_date=listing_date,
        )
        write_json(
            snapshot.to_payload(run_id=run.run_id, company_id=run.company_id),
            feature_path,
            pretty=True,
        )

    def _summarize_run(self, run: AnalysisRunRecord) -> ReviewRunSummary | None:
        findings = self.repository.list_findings(run.run_id)
        if not findings:
            return None
        company = self.repository.get_company(run.company_id)
        pending = sum(
            finding.status
            in {
                VerificationStatus.PROPOSED,
                VerificationStatus.PENDING_VERIFICATION,
            }
            for finding in findings
        )
        verified = sum(finding.status is VerificationStatus.VERIFIED for finding in findings)
        rejected = sum(finding.status is VerificationStatus.REJECTED for finding in findings)
        base_feature = self._artifact_for_report(
            run.report_path,
            "prelisting_features_base.json",
        )
        agent_feature = self._artifact_for_report(
            run.report_path,
            "prelisting_features.json",
        )
        return ReviewRunSummary(
            run_id=run.run_id,
            company_id=run.company_id,
            company_name=(company.company_name_zh if company else run.company_id),
            as_of_date=run.as_of_date.isoformat(),
            total_findings=len(findings),
            pending_findings=pending,
            verified_findings=verified,
            rejected_findings=rejected,
            reviewed_findings=verified + rejected,
            complete=pending == 0,
            report_path=run.report_path,
            verified_report_path=run.verified_report_path,
            base_feature_available=bool(base_feature and base_feature.is_file()),
            agent_feature_available=bool(agent_feature and agent_feature.is_file()),
            base_prediction=self._prediction_for_report(run.report_path, variant="base"),
            prediction=(
                self._prediction_for_report(run.report_path)
                or self._stored_prediction_for_run(run.run_id)
            ),
        )

    def list_runs(self) -> tuple[ReviewRunSummary, ...]:
        summaries: list[ReviewRunSummary] = []
        for run in self.repository.list_analysis_runs(status="succeeded"):
            summary = self._summarize_run(run)
            if summary is not None:
                summaries.append(summary)
        return tuple(summaries)

    def rerun_prediction(
        self,
        run_id: str,
        *,
        variant: str = "auto",
    ) -> tuple[dict[str, Any], ReviewRunSummary]:
        """基于历史运行的已保存特征独立重算预测。"""

        run = self.repository.get_analysis_run(run_id)
        if run is None or run.status != "succeeded":
            raise KeyError(f"analysis run does not exist: {run_id}")
        company = self.repository.get_company(run.company_id)
        company_name = company.company_name_zh if company else run.company_id
        agent_feature = self._artifact_for_report(run.report_path, "prelisting_features.json")
        if agent_feature is None or not agent_feature.is_file():
            raise RuntimePredictionUnavailable("Agent增强特征尚未生成。")
        report_path = self._artifact_for_report(run.report_path, "risk_analysis.json")
        if report_path is None:
            raise RuntimePredictionUnavailable("分析结果路径无效。")
        self._repair_legacy_prediction_features(
            run=run,
            feature_path=agent_feature,
            report_path=report_path,
            listing_date=(company.listing_date if company else None),
        )
        # Merge RAG supplemental features for missing fields
        _merge_rag_supplements(
            project_root=self.project_root,
            feature_path=agent_feature,
            company_id=run.company_id,
        )
        destination = report_path.with_name("prediction_5d.json")
        prediction, _ = generate_runtime_prediction_from_feature_file(
            project_root=self.project_root,
            company_id=run.company_id,
            company_name=company_name,
            feature_path=agent_feature,
            report_path=report_path,
            prediction_variant="agent_enhanced",
            prediction_as_of_fallback=(company.listing_date if company else None),
            model_path=None,
            destination_path=destination,
        )
        self.repository.save_prediction(
            run_id=run.run_id,
            company_id=run.company_id,
            as_of_date=run.as_of_date,
            payload=prediction,
        )
        summary = self._summarize_run(run)
        if summary is None:
            raise RuntimePredictionUnavailable("该运行没有可展示的风险结论。")
        return prediction, summary

    def list_findings(self, run_id: str) -> tuple[RiskFinding, ...]:
        return self.repository.list_findings(run_id)

    def delete_analysis_run(self, run_id: str) -> AnalysisDeletionResult:
        """删除指定运行，派生文件移入项目回收区并保留原始文档。"""
        run = self.repository.get_analysis_run(run_id)
        if run is None:
            raise KeyError(f"analysis run does not exist: {run_id}")

        run_segment = safe_path_segment(run_id, "run_id")
        candidates: list[tuple[Path, str, Path]] = []
        missing: list[Path] = []
        skipped: list[Path] = []

        def resolve_project_reference(reference: str) -> Path:
            path = Path(reference)
            if not path.is_absolute():
                path = self.project_root / path
            return path.resolve()

        def related_export_paths(reference: str) -> tuple[Path, ...]:
            primary = resolve_project_reference(reference)
            if primary.name != "verified_risk_analysis.json":
                return (primary,)
            return (
                primary,
                primary.with_name("verified_risk_report.pdf"),
                primary.with_name("analysis_audit_report.pdf"),
                primary.with_name("analysis_audit_bundle.json"),
                *primary.parent.glob("*_IPO风险穿透预警报告_*.pdf"),
                *primary.parent.glob("*_IPO分析完整审计报告_*.pdf"),
            )

        # A document or derived asset may be reused by multiple analysis runs.
        # Such files remain active; only the selected run's database links are removed.
        protected_paths = {
            resolve_project_reference(document.local_file)
            for document in self.repository.list_documents(run.company_id)
        }
        for other_run in self.repository.list_analysis_runs():
            if other_run.run_id == run_id:
                continue
            protected_paths.update(
                related_path
                for reference in (
                    other_run.report_path,
                    other_run.verified_report_path,
                )
                if reference
                for related_path in related_export_paths(reference)
            )
            protected_paths.update(
                resolve_project_reference(evidence.screenshot_path)
                for evidence in self.repository.list_evidence(other_run.run_id)
                if evidence.screenshot_path
            )

        def add_project_candidate(reference: str) -> None:
            source = resolve_project_reference(reference)
            if not source.is_relative_to(self.project_root) or source in protected_paths:
                skipped.append(source)
            elif source.exists():
                candidates.append((source, "project", source.relative_to(self.project_root)))
            else:
                missing.append(source)

        if run.report_path:
            add_project_candidate(run.report_path)
        if run.verified_report_path:
            for related_path in related_export_paths(run.verified_report_path):
                add_project_candidate(str(related_path))

        for evidence in self.repository.list_evidence(run_id):
            if evidence.screenshot_path:
                add_project_candidate(evidence.screenshot_path)

        unique_candidates: list[tuple[Path, str, Path]] = []
        seen_sources: set[Path] = set()
        for candidate in candidates:
            if candidate[0] not in seen_sources:
                unique_candidates.append(candidate)
                seen_sources.add(candidate[0])

        recovery_path: Path | None = None
        moved: list[tuple[Path, Path]] = []
        if unique_candidates:
            version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
            recovery_path = self.project_root / ".trash" / "analysis_runs" / run_segment / version
            try:
                for source, category, relative_path in unique_candidates:
                    destination = recovery_path / category / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    moved.append((source, destination))
                if not self.repository.delete_analysis_run(run_id):
                    raise KeyError(f"analysis run does not exist: {run_id}")
            except Exception:
                for source, destination in reversed(moved):
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        shutil.move(str(destination), str(source))
                if recovery_path.exists():
                    shutil.rmtree(recovery_path)
                raise
        elif not self.repository.delete_analysis_run(run_id):
            raise KeyError(f"analysis run does not exist: {run_id}")

        return AnalysisDeletionResult(
            run_id=run_id,
            recovery_path=recovery_path,
            moved_paths=tuple(source for source, _ in moved),
            missing_paths=tuple(dict.fromkeys(missing)),
            skipped_paths=tuple(dict.fromkeys(skipped)),
        )

    def get_agent_traces(self, run_id: str) -> tuple[AgentTrace, ...]:
        """从候选报告读取不可变 Agent 轨迹，供复核和审计展示。"""
        report_reference = self.repository.get_analysis_report_path(run_id)
        if report_reference is None:
            return ()
        report_path = Path(report_reference)
        if not report_path.is_absolute():
            report_path = self.project_root / report_path
        if not report_path.is_file():
            return ()
        report = RiskAnalysisReport.model_validate(read_json(report_path))
        return report.agent_traces

    def get_finding_detail(
        self,
        run_id: str,
        finding_id: str,
    ) -> ReviewFindingDetail:
        finding = self.repository.get_finding(run_id, finding_id)
        if finding is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")
        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=finding.evidence_ids,
        )
        return ReviewFindingDetail(
            finding=finding,
            evidences=evidences,
            verification=run_verification(
                finding,
                evidences,
                project_root=self.project_root,
            ),
            reviews=self.repository.list_reviews(run_id, finding_id=finding_id),
            agent_traces=self.get_agent_traces(run_id),
        )

    def verify_finding(self, run_id: str, finding_id: str) -> FindingVerification:
        finding = self.repository.get_finding(run_id, finding_id)
        if finding is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")
        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=finding.evidence_ids,
        )
        return run_verification(finding, evidences, project_root=self.project_root)

    def review_finding(
        self,
        *,
        run_id: str,
        finding_id: str,
        decision: ReviewDecision,
        reviewer: str,
        comment: str | None = None,
        edited_finding: RiskFinding | None = None,
        manual_override: bool = False,
    ) -> FindingReview:
        original = self.repository.get_finding(run_id, finding_id)
        if original is None:
            raise KeyError(f"finding does not exist: {run_id}/{finding_id}")

        if decision is ReviewDecision.EDITED:
            if edited_finding is None:
                raise ValueError("edited review requires edited_finding")
            if edited_finding.finding_id != original.finding_id:
                raise ValueError("edited finding must keep finding_id")
            if edited_finding.company_id != original.company_id:
                raise ValueError("edited finding must keep company_id")
            candidate = edited_finding.model_copy(
                update={"status": VerificationStatus.PENDING_VERIFICATION}
            )
        else:
            if edited_finding is not None:
                raise ValueError("edited_finding is only valid for an edited review")
            candidate = original.model_copy(
                update={"status": VerificationStatus.PENDING_VERIFICATION}
            )

        evidences = self.repository.list_evidence(
            run_id,
            evidence_ids=candidate.evidence_ids,
        )
        verification = run_verification(
            candidate,
            evidences,
            project_root=self.project_root,
        )
        if (
            decision is not ReviewDecision.REJECTED
            and not verification.passed
            and not manual_override
        ):
            raise ReviewBlockedError(verification)
        status = (
            VerificationStatus.REJECTED
            if decision is ReviewDecision.REJECTED
            else VerificationStatus.VERIFIED
        )
        resulting_finding = candidate.model_copy(update={"status": status})
        review = FindingReview(
            review_id=f"REV-{uuid4().hex}",
            run_id=run_id,
            finding_id=finding_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
            manual_override=manual_override,
            verification=verification,
            resulting_finding=resulting_finding,
        )
        self.repository.save_review(review)
        return review

    def export_verified_report(
        self,
        run_id: str,
        *,
        output_path: Path | None = None,
    ) -> Path:
        """Export a final report only after every finding reaches a terminal state."""
        candidate_reference = self.repository.get_analysis_report_path(run_id)
        if candidate_reference is None:
            raise KeyError(f"analysis run does not exist or has no report: {run_id}")
        candidate_path = Path(candidate_reference)
        if not candidate_path.is_absolute():
            candidate_path = self.project_root / candidate_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        candidate = RiskAnalysisReport.model_validate(read_json(candidate_path))

        stored_findings = self.repository.list_findings(run_id)
        incomplete = tuple(
            finding.finding_id
            for finding in stored_findings
            if finding.status
            in {
                VerificationStatus.PROPOSED,
                VerificationStatus.PENDING_VERIFICATION,
            }
        )
        if incomplete:
            raise IncompleteReviewError(incomplete)

        verified_findings = tuple(
            finding for finding in stored_findings if finding.status is VerificationStatus.VERIFIED
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id for finding in verified_findings for evidence_id in finding.evidence_ids
            )
        )
        evidences = self.repository.list_evidence(run_id, evidence_ids=evidence_ids)
        evidence_display = {
            evidence.evidence_id: candidate.evidence_display_zh_hans.get(
                evidence.evidence_id,
                evidence.quote,
            )
            for evidence in evidences
        }
        final_report = RiskAnalysisReport(
            run_id=candidate.run_id,
            company_id=candidate.company_id,
            company_name=candidate.company_name,
            generated_at=datetime.now(timezone.utc),
            source_document=candidate.source_document,
            summary=AnalysisSummary(
                evidence_count=len(evidences),
                finding_count=len(verified_findings),
                screenshot_count=sum(
                    evidence.screenshot_path is not None for evidence in evidences
                ),
                requires_human_review=False,
            ),
            evidences=evidences,
            risk_findings=verified_findings,
            agent_traces=(),
            evidence_display_zh_hans=evidence_display,
        )
        destination = output_path or candidate_path.with_name("verified_risk_analysis.json")
        if not destination.is_absolute():
            destination = self.project_root / destination
        write_risk_report(final_report, destination)
        try:
            stored_path = destination.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            stored_path = str(destination.resolve())
        self.repository.set_verified_report_path(run_id, stored_path)
        return destination

    def export_verified_bundle(
        self,
        run_id: str,
        *,
        output_directory: Path | None = None,
    ) -> VerifiedExportBundle:
        """Export a concise PDF/JSON plus a separately labelled full audit."""
        candidate_reference = self.repository.get_analysis_report_path(run_id)
        if candidate_reference is None:
            raise KeyError(f"analysis run does not exist or has no report: {run_id}")
        candidate_path = Path(candidate_reference)
        if not candidate_path.is_absolute():
            candidate_path = self.project_root / candidate_path
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)

        directory = output_directory or candidate_path.parent
        if not directory.is_absolute():
            directory = self.project_root / directory
        directory.mkdir(parents=True, exist_ok=True)
        compact_path = self.export_verified_report(
            run_id,
            output_path=directory / "verified_risk_analysis.json",
        )
        compact_report = RiskAnalysisReport.model_validate(read_json(compact_path))
        candidate_report = RiskAnalysisReport.model_validate(read_json(candidate_path))
        stored_findings = self.repository.list_findings(run_id)
        reviews = self.repository.list_reviews(run_id)
        prediction = self._prediction_for_report(candidate_reference)
        if prediction is None:
            prediction = self._stored_prediction_for_run(run_id)
        company_profile = self._company_profile_for_report(compact_report.company_id)
        feature_payload = self._feature_payload_for_report(candidate_reference)
        business_toolchain = _business_toolchain(
            candidate_report, feature_payload, prediction
        )

        prediction_path = directory / "final_prediction_results.csv"
        prediction_rows = _prediction_export_rows(
            prediction,
            company_id=compact_report.company_id,
            company_name=compact_report.company_name,
        )
        with prediction_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["company_id", "company_name", "window", "risk_level", "warning"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

        trace_path = directory / "agent_reasoning_log.json"
        write_json(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "business_toolchain": business_toolchain,
                "agent_traces": [trace.model_dump(mode="json") for trace in candidate_report.agent_traces],
            },
            trace_path,
        )
        evidence_path = directory / "key_evidence_snippets.json"
        write_json(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "evidences": [evidence.model_dump(mode="json") for evidence in candidate_report.evidences],
            },
            evidence_path,
        )

        audit_path = directory / "analysis_audit_bundle.json"
        write_json(
            {
                "schema_version": "1.0",
                "export_type": "analysis_audit_bundle",
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "candidate_report": candidate_report.model_dump(mode="json"),
                "final_findings": [finding.model_dump(mode="json") for finding in stored_findings],
                "reviews": [review.model_dump(mode="json") for review in reviews],
                "prediction": prediction,
                "company_profile": company_profile,
                "feature_audit": feature_payload,
                "business_toolchain": business_toolchain,
            },
            audit_path,
        )

        pdf_path = directory / (
            _report_file_stem(
                compact_report.company_id,
                compact_report.company_name,
                "IPO风险穿透预警报告",
            )
            + ".pdf"
        )
        write_verified_pdf_report(
            compact_report,
            pdf_path,
            reviews=reviews,
            project_root=self.project_root,
            prediction=prediction,
            company_profile=company_profile,
            feature_payload=feature_payload,
            execution_traces=candidate_report.agent_traces,
        )
        audit_pdf_path = directory / (
            _report_file_stem(
                compact_report.company_id,
                compact_report.company_name,
                "IPO分析完整审计报告",
            )
            + ".pdf"
        )
        write_audit_pdf_report(
            candidate_report,
            audit_pdf_path,
            final_findings=stored_findings,
            reviews=reviews,
            project_root=self.project_root,
            prediction=prediction,
            company_profile=company_profile,
            feature_payload=feature_payload,
        )
        return VerifiedExportBundle(
            pdf_path=pdf_path,
            audit_pdf_path=audit_pdf_path,
            json_path=compact_path,
            audit_path=audit_path,
            prediction_path=prediction_path,
            trace_path=trace_path,
            evidence_path=evidence_path,
        )


__all__ = [
    "AnalysisDeletionResult",
    "IncompleteReviewError",
    "ReviewBlockedError",
    "ReviewFindingDetail",
    "ReviewRunSummary",
    "ReviewService",
    "VerifiedExportBundle",
]
