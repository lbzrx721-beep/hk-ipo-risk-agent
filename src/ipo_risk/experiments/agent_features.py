"""历史招股书Agent特征的可恢复批量实验。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
import traceback
from typing import Any, cast

import pandas as pd

from ipo_risk.agents.intervention import RiskScopeReviewController
from ipo_risk.config import AppConfig, load_config
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.modeling.features import build_agent_feature_snapshot
from ipo_risk.modeling.agent_precision import AGENT_PRECISION_FEATURE_NAMES
from ipo_risk.modeling.risk_semantics import (
    RISK_SEMANTIC_SCHEMA_VERSION,
    build_risk_semantic_feature_snapshot,
    risk_semantic_catalog,
)
from ipo_risk.pipeline import (
    AnalysisArtifacts,
    analyze_company,
    prepare_company_semantic_index,
)
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.storage.files import read_json, write_json


Analyzer = Callable[..., AnalysisArtifacts]
CompanyPreparer = Callable[..., object]
PreparationProgress = Callable[[int, int, str, str], None]
AGENT_FEATURE_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class AgentFeatureExperimentResult:
    experiment_dir: Path
    selection_path: Path
    status_path: Path
    summary_path: Path
    agent_feature_matrix_path: Path
    augmented_feature_matrix_path: Path | None
    selected_count: int
    completed_count: int
    failed_count: int


def select_pilot_companies(
    manifest_rows: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    splits: tuple[str, ...] = ("train", "validation"),
    seed: str = "agent-feature-pilot-v1",
) -> list[dict[str, Any]]:
    """按数据量比例抽样，排序只依赖文档哈希而不读取市场标签值。"""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not splits:
        raise ValueError("splits must not be empty")
    eligible = [
        row
        for row in manifest_rows
        if row.get("split") in splits
        and row.get("label_status") == "complete"
        and row.get("company_folder")
        and row.get("source_hash")
    ]
    if sample_size > len(eligible):
        raise ValueError("sample_size exceeds eligible company count")

    by_split = {
        split: [row for row in eligible if row.get("split") == split]
        for split in splits
    }
    by_split = {split: rows for split, rows in by_split.items() if rows}
    if not by_split:
        raise ValueError("no eligible companies for requested splits")

    total = sum(len(rows) for rows in by_split.values())
    exact = {
        split: sample_size * len(rows) / total
        for split, rows in by_split.items()
    }
    quotas = {split: int(value) for split, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    residual_order = sorted(
        by_split,
        key=lambda split: (exact[split] - quotas[split], len(by_split[split])),
        reverse=True,
    )
    for split in residual_order[:remaining]:
        quotas[split] += 1

    selected: list[dict[str, Any]] = []
    for split, rows in by_split.items():
        ranked = sorted(
            rows,
            key=lambda row: sha256(
                f"{seed}:{row['source_hash']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[split]])
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("split")),
            str(row.get("prospectus_date")),
            str(row.get("company_id")),
        ),
    )


def normalize_historical_agent_features(
    project_root: Path | str,
    *,
    experiment_name: str = "agent_feature_history_normalized_v1",
) -> AgentFeatureExperimentResult:
    """把既有风险报告统一为最新版Agent特征，不重新调用模型。"""
    root = Path(project_root).resolve()
    experiment_dir = root / "artifacts" / "experiments" / experiment_name
    company_dir = experiment_dir / "companies"
    company_dir.mkdir(parents=True, exist_ok=True)
    selection_path = experiment_dir / "selection.json"
    status_path = experiment_dir / "status.json"
    summary_path = experiment_dir / "summary.json"
    agent_matrix_path = experiment_dir / "agent_feature_matrix.csv"
    semantic_audit_path = experiment_dir / "risk_semantic_finding_audit.jsonl"

    reports_by_company: dict[str, tuple[RiskAnalysisReport, Path]] = {}
    report_root = root / "artifacts" / "analysis"
    for report_path in sorted(report_root.rglob("risk_analysis.json")):
        report = RiskAnalysisReport.model_validate(read_json(report_path))
        company_id = report.company_id.zfill(5)
        current = reports_by_company.get(company_id)
        if current is None or report.generated_at > current[0].generated_at:
            reports_by_company[company_id] = (report, report_path)
    if not reports_by_company:
        raise ValueError("no historical risk reports found")

    base_path = (
        root
        / "data"
        / "processed"
        / "modeling"
        / "prelisting_feature_matrix.csv"
    )
    if not base_path.is_file():
        raise FileNotFoundError(f"missing prelisting feature matrix: {base_path}")
    base = pd.read_csv(base_path, dtype={"company_id": str})
    base["company_id"] = base["company_id"].astype(str).str.zfill(5)
    base_by_id = {
        str(row["company_id"]): row
        for row in base.to_dict(orient="records")
    }

    existing_payloads: dict[str, dict[str, Any]] = {}
    experiments_root = root / "artifacts" / "experiments"
    for feature_path in sorted(experiments_root.glob("*/companies/*.json")):
        if feature_path.parent == company_dir:
            continue
        payload = read_json(feature_path)
        run_id = str(payload.get("run_id") or "")
        if run_id:
            existing_payloads[run_id] = payload

    selected: list[dict[str, Any]] = []
    status_companies: dict[str, dict[str, Any]] = {}
    semantic_audits: list[dict[str, Any]] = []
    normalized_at = _now()
    for company_id, (report, report_path) in sorted(reports_by_company.items()):
        base_row = base_by_id.get(company_id, {})
        existing = existing_payloads.get(report.run_id, {})
        company_folder = str(
            existing.get("company_folder")
            or report_path.parents[1].name
        )
        split = str(base_row.get("split") or existing.get("split") or "unassigned")
        prospectus_date = str(
            base_row.get("prospectus_date")
            or report.source_document.available_at.isoformat()
        )
        row = {
            "company_id": company_id,
            "company_folder": company_folder,
            "company_name": report.company_name,
            "prospectus_date": prospectus_date,
            "source_hash": report.source_document.source_hash,
            "split": split,
            "label_status": "complete",
        }
        selected.append(row)

        snapshot = build_agent_feature_snapshot(
            report,
            as_of_date=report.source_document.available_at,
        )
        _, classifications = build_risk_semantic_feature_snapshot(
            report.risk_findings,
            report.evidences,
            available_at=report.source_document.available_at,
            source_ref=f"analysis:{report.run_id}",
        )
        findings_by_id = {
            finding.finding_id: finding for finding in report.risk_findings
        }
        evidences_by_id = {
            evidence.evidence_id: evidence for evidence in report.evidences
        }
        for classification in classifications:
            finding = findings_by_id[classification.finding_id]
            referenced_evidence = [
                evidences_by_id[evidence_id]
                for evidence_id in finding.evidence_ids
                if evidence_id in evidences_by_id
            ]
            semantic_audits.append(
                {
                    "semantic_schema_version": RISK_SEMANTIC_SCHEMA_VERSION,
                    "company_id": company_id,
                    "company_name": report.company_name,
                    "run_id": report.run_id,
                    "finding_id": finding.finding_id,
                    "original_risk_type": finding.risk_type,
                    "title": finding.title,
                    "claim": finding.claim,
                    "owner": finding.created_by.value,
                    "severity": finding.severity.value,
                    "finding_confidence": finding.confidence,
                    "specialist_status": finding.specialist_status,
                    "primary_family": classification.primary_family,
                    "primary_subtype": classification.primary_subtype,
                    "primary_label_zh": classification.primary_label_zh,
                    "secondary_subtypes": list(classification.secondary_subtypes),
                    "mapping_method": classification.method,
                    "mapping_confidence": classification.mapping_confidence,
                    "matched_keywords": list(classification.matched_keywords),
                    "evidence_ids": [item.evidence_id for item in referenced_evidence],
                    "evidence_pages": sorted({item.page for item in referenced_evidence}),
                    "evidence_count": len(referenced_evidence),
                    "report_path": _portable(report_path, root),
                }
            )
        feature_path = company_dir / f"{company_id}.json"
        payload = {
            "schema_version": 1,
            "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
            "experiment_name": experiment_name,
            **{key: row[key] for key in (
                "company_id",
                "company_folder",
                "company_name",
                "split",
                "prospectus_date",
                "source_hash",
            )},
            "run_id": report.run_id,
            "started_at": existing.get("started_at"),
            "ended_at": existing.get("ended_at") or report.generated_at.isoformat(),
            "duration_seconds": existing.get("duration_seconds"),
            "finding_count": report.summary.finding_count,
            "evidence_count": report.summary.evidence_count,
            "trace_count": len(report.agent_traces),
            "report_path": _portable(report_path, root),
            "embedding_dir": existing.get("embedding_dir"),
            "features": {
                name: feature.model_dump(mode="json")
                for name, feature in sorted(snapshot.items())
            },
            "model_fingerprint": existing.get("model_fingerprint"),
            "normalization": {
                "normalized_at": normalized_at,
                "source": "historical_risk_report",
                "model_reinvoked": False,
            },
        }
        write_json(payload, feature_path)
        status_companies[company_folder] = {
            "company_id": company_id,
            "company_folder": company_folder,
            "split": split,
            "state": "completed",
            "ended_at": payload["ended_at"],
            "duration_seconds": payload["duration_seconds"],
            "run_id": report.run_id,
            "finding_count": report.summary.finding_count,
            "evidence_count": report.summary.evidence_count,
            "feature_path": _portable(feature_path, root),
        }

    selected.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["prospectus_date"]),
            str(row["company_id"]),
        )
    )
    normalization_fingerprint = {
        "kind": "historical_report_normalization",
        "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
        "model_reinvoked": False,
    }
    write_json(
        {
            "schema_version": 1,
            "created_at": normalized_at,
            "selection_policy": "all_unique_historical_risk_reports",
            "label_values_used_for_selection": False,
            "model_fingerprint": normalization_fingerprint,
            "companies": selected,
        },
        selection_path,
    )
    write_json(
        {
            "schema_version": 1,
            "created_at": normalized_at,
            "companies": status_companies,
            "queue_state": "completed",
            "pause_reason": None,
            "normalization": normalization_fingerprint,
        },
        status_path,
    )
    augmented_matrix_path = _write_feature_matrices(
        root,
        company_dir=company_dir,
        agent_matrix_path=agent_matrix_path,
        selected=selected,
    )
    agent_matrix = pd.read_csv(agent_matrix_path, dtype={"company_id": str})
    identity_columns = {
        "company_id",
        "company_folder",
        "company_name",
        "split",
        "prospectus_date",
        "run_id",
        "duration_seconds",
    }
    feature_columns = sorted(
        column for column in agent_matrix.columns if column not in identity_columns
    )
    split_counts = {
        str(split): int(count)
        for split, count in agent_matrix["split"].value_counts().sort_index().items()
    }
    semantic_audit_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in semantic_audits
        ),
        encoding="utf-8",
    )
    semantic_method_counts: dict[str, int] = {}
    semantic_family_counts: dict[str, int] = {}
    semantic_subtype_counts: dict[str, int] = {}
    for row in semantic_audits:
        method = str(row["mapping_method"])
        family = str(row["primary_family"])
        subtype = str(row["primary_subtype"])
        semantic_method_counts[method] = semantic_method_counts.get(method, 0) + 1
        semantic_family_counts[family] = semantic_family_counts.get(family, 0) + 1
        semantic_subtype_counts[subtype] = semantic_subtype_counts.get(subtype, 0) + 1
    specific_count = sum(
        count
        for method, count in semantic_method_counts.items()
        if method != "agent_role_fallback"
    )
    summary = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "updated_at": normalized_at,
        "normalization_source": "historical_risk_reports",
        "model_reinvoked": False,
        "selected_count": len(selected),
        "completed_count": len(selected),
        "failed_count": 0,
        "pending_count": 0,
        "total_findings": sum(
            report.summary.finding_count
            for report, _path in reports_by_company.values()
        ),
        "total_evidences": sum(
            report.summary.evidence_count
            for report, _path in reports_by_company.values()
        ),
        "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
        "agent_feature_count": len(feature_columns),
        "agent_feature_columns": feature_columns,
        "risk_semantic_schema_version": RISK_SEMANTIC_SCHEMA_VERSION,
        "risk_semantic_finding_count": len(semantic_audits),
        "risk_semantic_specific_classified_count": specific_count,
        "risk_semantic_specific_classified_share": (
            round(specific_count / len(semantic_audits), 8)
            if semantic_audits
            else 0.0
        ),
        "risk_semantic_mapping_method_counts": dict(
            sorted(semantic_method_counts.items())
        ),
        "risk_semantic_family_counts": dict(sorted(semantic_family_counts.items())),
        "risk_semantic_subtype_counts": dict(sorted(semantic_subtype_counts.items())),
        "risk_semantic_catalog": list(risk_semantic_catalog()),
        "risk_semantic_audit_path": _portable(semantic_audit_path, root),
        "split_counts": split_counts,
        "models": normalization_fingerprint,
        "selection_path": _portable(selection_path, root),
        "status_path": _portable(status_path, root),
        "agent_feature_matrix_path": _portable(agent_matrix_path, root),
        "augmented_feature_matrix_path": (
            _portable(augmented_matrix_path, root)
            if augmented_matrix_path is not None
            else None
        ),
        "failed_companies": [],
    }
    write_json(summary, summary_path)
    return AgentFeatureExperimentResult(
        experiment_dir=experiment_dir,
        selection_path=selection_path,
        status_path=status_path,
        summary_path=summary_path,
        agent_feature_matrix_path=agent_matrix_path,
        augmented_feature_matrix_path=augmented_matrix_path,
        selected_count=len(selected),
        completed_count=len(selected),
        failed_count=0,
    )


def run_agent_feature_experiment(
    project_root: Path,
    *,
    experiment_name: str = "agent_feature_pilot_v1",
    sample_size: int = 20,
    splits: tuple[str, ...] = ("train", "validation"),
    company_folders: tuple[str, ...] = (),
    company_rows: tuple[dict[str, Any], ...] = (),
    min_findings: int = 1,
    continue_on_error: bool = True,
    pause_requested: Callable[[], bool] | None = None,
    analyzer: Analyzer = analyze_company,
    preparer: CompanyPreparer | None = None,
    preparation_progress: PreparationProgress | None = None,
    risk_scope_review_controllers: (
        Mapping[str, RiskScopeReviewController] | None
    ) = None,
    agent_company_concurrency: int | None = None,
) -> AgentFeatureExperimentResult:
    """并行预处理文档并受控并行运行公司Agent；每家公司独立落盘。"""
    root = project_root.resolve()
    config = load_config()
    experiment_dir = root / "artifacts" / "experiments" / experiment_name
    company_dir = experiment_dir / "companies"
    event_dir = experiment_dir / "events"
    company_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    selection_path = experiment_dir / "selection.json"
    status_path = experiment_dir / "status.json"
    summary_path = experiment_dir / "summary.json"

    manifest_path = (
        root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    )
    manifest_rows = (
        [dict(row) for row in company_rows]
        if company_rows
        else _read_jsonl(manifest_path)
    )
    fingerprint = _experiment_fingerprint(root, config)
    selected = _load_or_create_selection(
        selection_path,
        manifest_rows=manifest_rows,
        sample_size=sample_size,
        splits=splits,
        company_folders=company_folders,
        fingerprint=fingerprint,
    )
    status = _load_status(status_path, selected)
    status["queue_state"] = "running"
    status["pause_reason"] = None
    status_lock = Lock()

    selected_by_folder = {
        str(row["company_folder"]): row for row in selected
    }
    effective_preparer = (
        prepare_company_semantic_index
        if preparer is None and analyzer is analyze_company
        else preparer
    )
    if effective_preparer is not None:
        pending_folders = [
            company_folder
            for company_folder in selected_by_folder
            if status["companies"][company_folder].get("state") != "completed"
        ]
        if pending_folders:
            worker_count = min(
                config.embedding.document_concurrency,
                (
                    config.embedding.request_concurrency
                    if config.embedding.is_configured
                    else config.embedding.document_concurrency
                ),
                len(pending_folders),
            )
            per_document_request_concurrency = max(
                1,
                config.embedding.request_concurrency // worker_count,
            )
            status["preparation"] = {
                "state": "running",
                "completed": 0,
                "total": len(pending_folders),
                "document_concurrency": worker_count,
                "request_concurrency_per_document": (
                    per_document_request_concurrency
                ),
            }
            write_json(status, status_path)
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="prospectus-preparation",
            ) as executor:
                futures = {}
                for company_folder in pending_folders:
                    company_id = str(
                        selected_by_folder[company_folder]["company_id"]
                    ).zfill(5)
                    events = _EventLogger(event_dir / f"{company_id}.jsonl")
                    current = status["companies"][company_folder]
                    current["preparation_state"] = "running"
                    futures[
                        executor.submit(
                            effective_preparer,
                            company_folder,
                            project_root=root,
                            request_concurrency=(
                                per_document_request_concurrency
                            ),
                            progress_callback=events,
                        )
                    ] = company_folder
                write_json(status, status_path)
                completed_preparations = 0
                for future in as_completed(futures):
                    company_folder = futures[future]
                    current = status["companies"][company_folder]
                    try:
                        future.result()
                        current["preparation_state"] = "completed"
                        preparation_state = "completed"
                    except Exception as exc:  # noqa: BLE001
                        # 预处理失败不直接淘汰公司；正式分析仍会重试并可降级BM25。
                        current["preparation_state"] = "failed"
                        current["preparation_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        preparation_state = "failed"
                    completed_preparations += 1
                    status["preparation"]["completed"] = completed_preparations
                    write_json(status, status_path)
                    if preparation_progress is not None:
                        preparation_progress(
                            completed_preparations,
                            len(pending_folders),
                            company_folder,
                            preparation_state,
                        )
            status["preparation"]["state"] = "completed"
            write_json(status, status_path)

    def analyze_one_company(
        company_folder: str,
        manifest_row: dict[str, Any],
    ) -> tuple[str | None, bool]:
        """执行一家公司的完整流水线，返回失败指纹与预算暂停标记。"""
        company_id = str(manifest_row["company_id"]).zfill(5)
        feature_path = company_dir / f"{company_id}.json"
        current = status["companies"][company_folder]
        if current.get("state") == "completed" and feature_path.is_file():
            payload = read_json(feature_path)
            if payload.get("feature_schema_version") == AGENT_FEATURE_SCHEMA_VERSION:
                return None, False
            report_path = root / str(payload["report_path"])
            report = RiskAnalysisReport.model_validate(read_json(report_path))
            feature_snapshot = build_agent_feature_snapshot(
                report,
                as_of_date=report.source_document.available_at,
            )
            payload["feature_schema_version"] = AGENT_FEATURE_SCHEMA_VERSION
            payload["features"] = {
                name: feature.model_dump(mode="json")
                for name, feature in sorted(feature_snapshot.items())
            }
            write_json(payload, feature_path)
            return None, False

        started_at = _now()
        started_clock = perf_counter()
        with status_lock:
            current.update(
                {
                    "state": "running",
                    "started_at": started_at,
                    "ended_at": None,
                    "error": None,
                }
            )
            write_json(status, status_path)
        events = _EventLogger(event_dir / f"{company_id}.jsonl")
        controller = (
            risk_scope_review_controllers.get(company_folder)
            if risk_scope_review_controllers is not None
            else None
        )
        review_wait_before = (
            controller.wait_seconds() if controller is not None else 0.0
        )
        try:
            analyzer_kwargs: dict[str, Any] = {
                "project_root": root,
                "min_findings": min_findings,
                "progress_callback": events,
            }
            if controller is not None:
                analyzer_kwargs["risk_scope_review_controller"] = controller
            artifacts = analyzer(company_folder, **analyzer_kwargs)
            report = artifacts.report
            as_of_date = report.source_document.available_at
            feature_snapshot = build_agent_feature_snapshot(
                report,
                as_of_date=as_of_date,
            )
            ended_at = _now()
            review_wait_seconds = (
                max(0.0, controller.wait_seconds() - review_wait_before)
                if controller is not None
                else 0.0
            )
            duration_seconds = round(
                max(0.0, perf_counter() - started_clock - review_wait_seconds),
                3,
            )
            payload = {
                "schema_version": 1,
                "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
                "experiment_name": experiment_name,
                "company_id": company_id,
                "company_folder": company_folder,
                "company_name": manifest_row.get("company_name"),
                "split": manifest_row.get("split"),
                "prospectus_date": manifest_row.get("prospectus_date"),
                "source_hash": manifest_row.get("source_hash"),
                "run_id": report.run_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration_seconds,
                "finding_count": report.summary.finding_count,
                "evidence_count": report.summary.evidence_count,
                "trace_count": len(report.agent_traces),
                "report_path": _portable(artifacts.report_path, root),
                "embedding_dir": (
                    _portable(artifacts.embedding_dir, root)
                    if artifacts.embedding_dir is not None
                    else None
                ),
                "features": {
                    name: feature.model_dump(mode="json")
                    for name, feature in sorted(feature_snapshot.items())
                },
                "model_fingerprint": fingerprint,
            }
            write_json(payload, feature_path)
            with status_lock:
                current.update(
                    {
                        "state": "completed",
                        "ended_at": ended_at,
                        "duration_seconds": duration_seconds,
                        "run_id": report.run_id,
                        "finding_count": report.summary.finding_count,
                        "evidence_count": report.summary.evidence_count,
                        "feature_path": _portable(feature_path, root),
                    }
                )
                write_json(status, status_path)
            return None, False
        except Exception as exc:  # noqa: BLE001 - 批量任务必须记录单公司失败后续跑
            ended_at = _now()
            budget_pause = isinstance(exc, ModelBudgetExceededError)
            review_wait_seconds = (
                max(0.0, controller.wait_seconds() - review_wait_before)
                if controller is not None
                else 0.0
            )
            with status_lock:
                current.update(
                    {
                        "state": "paused" if budget_pause else "failed",
                        "ended_at": ended_at,
                        "duration_seconds": round(
                            max(
                                0.0,
                                perf_counter()
                                - started_clock
                                - review_wait_seconds,
                            ),
                            3,
                        ),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                write_json(status, status_path)
            failure_key = f"{type(exc).__name__}:{str(exc).strip()[:300]}"
            if not continue_on_error:
                raise
            return failure_key, budget_pause

    pending_items = [
        (company_folder, manifest_row)
        for company_folder, manifest_row in selected_by_folder.items()
    ]
    requested_concurrency = (
        agent_company_concurrency
        if agent_company_concurrency is not None
        else config.agents.company_concurrency
    )
    company_worker_count = min(
        max(1, requested_concurrency),
        max(1, len(pending_items)),
    )
    status["agent_execution"] = {
        "state": "running",
        "company_concurrency": company_worker_count,
        "active": 0,
    }
    write_json(status, status_path)
    failure_counts: dict[str, int] = {}
    circuit_reason: str | None = None
    next_item = 0
    active_futures: dict[Future[tuple[str | None, bool]], str] = {}

    def can_submit() -> bool:
        return (
            next_item < len(pending_items)
            and circuit_reason is None
            and not (pause_requested is not None and pause_requested())
        )

    with ThreadPoolExecutor(
        max_workers=company_worker_count,
        thread_name_prefix="company-agent",
    ) as executor:
        while len(active_futures) < company_worker_count and can_submit():
            company_folder, manifest_row = pending_items[next_item]
            next_item += 1
            active_futures[
                executor.submit(
                    analyze_one_company,
                    company_folder,
                    manifest_row,
                )
            ] = company_folder
        with status_lock:
            status["agent_execution"]["active"] = len(active_futures)
            write_json(status, status_path)

        while active_futures:
            completed_futures = cast(
                set[Future[tuple[str | None, bool]]],
                wait(
                    active_futures,
                    return_when=FIRST_COMPLETED,
                ).done,
            )
            for company_future in completed_futures:
                active_futures.pop(company_future)
                failure_key, budget_pause = company_future.result()
                if failure_key is not None:
                    failure_counts[failure_key] = (
                        failure_counts.get(failure_key, 0) + 1
                    )
                    if budget_pause:
                        circuit_reason = failure_key.split(":", maxsplit=1)[-1]
                    elif failure_counts[failure_key] >= 2:
                        circuit_reason = (
                            "两家公司出现相同失败，队列已安全暂停："
                            f"{failure_key}"
                        )
            while len(active_futures) < company_worker_count and can_submit():
                company_folder, manifest_row = pending_items[next_item]
                next_item += 1
                active_futures[
                    executor.submit(
                        analyze_one_company,
                        company_folder,
                        manifest_row,
                    )
                ] = company_folder
            with status_lock:
                status["agent_execution"]["active"] = len(active_futures)
                write_json(status, status_path)

    pause_was_requested = pause_requested is not None and pause_requested()
    has_unfinished_company = any(
        item.get("state") in {"pending", "running", "paused"}
        for item in status["companies"].values()
    )
    if circuit_reason is not None or (
        pause_was_requested and has_unfinished_company
    ):
        status["queue_state"] = "paused"
        status["pause_reason"] = (
            circuit_reason
            if circuit_reason is not None
            else "已按用户要求在当前运行公司完成后暂停。"
        )
    status["agent_execution"]["state"] = "completed"
    status["agent_execution"]["active"] = 0
    write_json(status, status_path)

    unfinished_states = {"pending", "running", "paused"}
    if status.get("queue_state") != "paused" and not any(
        item.get("state") in unfinished_states
        for item in status["companies"].values()
    ):
        status["queue_state"] = "completed"
        status["pause_reason"] = None
        write_json(status, status_path)

    agent_matrix_path = experiment_dir / "agent_feature_matrix.csv"
    augmented_matrix_path = _write_feature_matrices(
        root,
        company_dir=company_dir,
        agent_matrix_path=agent_matrix_path,
        selected=selected,
    )
    completed = [
        item
        for item in status["companies"].values()
        if item.get("state") == "completed"
    ]
    failed = [
        item
        for item in status["companies"].values()
        if item.get("state") == "failed"
    ]
    summary = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "updated_at": _now(),
        "selected_count": len(selected),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "pending_count": len(selected) - len(completed) - len(failed),
        "total_findings": sum(int(item.get("finding_count") or 0) for item in completed),
        "total_evidences": sum(int(item.get("evidence_count") or 0) for item in completed),
        "mean_duration_seconds": (
            round(
                sum(float(item.get("duration_seconds") or 0) for item in completed)
                / len(completed),
                3,
            )
            if completed
            else None
        ),
        "models": fingerprint,
        "selection_path": _portable(selection_path, root),
        "status_path": _portable(status_path, root),
        "agent_feature_matrix_path": _portable(agent_matrix_path, root),
        "augmented_feature_matrix_path": (
            _portable(augmented_matrix_path, root)
            if augmented_matrix_path is not None
            else None
        ),
        "failed_companies": [
            {
                "company_folder": item["company_folder"],
                "error": item.get("error"),
            }
            for item in failed
        ],
    }
    write_json(summary, summary_path)
    return AgentFeatureExperimentResult(
        experiment_dir=experiment_dir,
        selection_path=selection_path,
        status_path=status_path,
        summary_path=summary_path,
        agent_feature_matrix_path=agent_matrix_path,
        augmented_feature_matrix_path=augmented_matrix_path,
        selected_count=len(selected),
        completed_count=len(completed),
        failed_count=len(failed),
    )


class _EventLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    def __call__(self, stage: str, detail: str) -> None:
        record = {
            "timestamp": _now(),
            "stage": stage,
            "detail": detail,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")


def _load_or_create_selection(
    path: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    sample_size: int,
    splits: tuple[str, ...],
    company_folders: tuple[str, ...],
    fingerprint: dict[str, Any],
) -> list[dict[str, Any]]:
    if path.is_file():
        existing = read_json(path)
        if existing.get("model_fingerprint") != fingerprint:
            raise ValueError(
                "experiment model or prompts changed; use a new experiment_name"
            )
        return list(existing["companies"])

    if company_folders:
        by_folder = {
            str(row.get("company_folder")): row for row in manifest_rows
        }
        missing = sorted(set(company_folders).difference(by_folder))
        if missing:
            raise ValueError(f"unknown company folders: {', '.join(missing)}")
        selected = [by_folder[folder] for folder in company_folders]
    else:
        selected = select_pilot_companies(
            manifest_rows,
            sample_size=sample_size,
            splits=splits,
        )
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "selection_policy": (
            "explicit_company_folders"
            if company_folders
            else "split_proportional_document_hash"
        ),
        "requested_sample_size": sample_size,
        "requested_splits": list(splits),
        "label_values_used_for_selection": False,
        "model_fingerprint": fingerprint,
        "companies": selected,
    }
    write_json(payload, path)
    return selected


def _load_status(
    path: Path,
    selected: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if path.is_file():
        status = read_json(path)
    else:
        status = {
            "schema_version": 1,
            "created_at": _now(),
            "companies": {},
        }
    companies = status.setdefault("companies", {})
    for row in selected:
        folder = str(row["company_folder"])
        companies.setdefault(
            folder,
            {
                "company_id": str(row["company_id"]).zfill(5),
                "company_folder": folder,
                "split": row.get("split"),
                "state": "pending",
            },
        )
        if companies[folder].get("state") == "running":
            companies[folder]["state"] = "pending"
    write_json(status, path)
    return status


def _write_feature_matrices(
    project_root: Path,
    *,
    company_dir: Path,
    agent_matrix_path: Path,
    selected: Sequence[dict[str, Any]],
) -> Path | None:
    records: list[dict[str, Any]] = []
    selected_by_id = {
        str(row["company_id"]).zfill(5): row for row in selected
    }
    for feature_path in sorted(company_dir.glob("*.json")):
        payload = read_json(feature_path)
        company_id = str(payload["company_id"]).zfill(5)
        if company_id not in selected_by_id:
            continue
        record: dict[str, Any] = {
            "company_id": company_id,
            "company_folder": payload["company_folder"],
            "company_name": payload.get("company_name"),
            "split": payload.get("split"),
            "prospectus_date": payload.get("prospectus_date"),
            "run_id": payload.get("run_id"),
            "duration_seconds": payload.get("duration_seconds"),
        }
        record.update(
            {
                name: feature["value"]
                for name, feature in payload["features"].items()
            }
        )
        records.append(record)
    identity_columns = {
        "company_id",
        "company_folder",
        "company_name",
        "split",
        "prospectus_date",
        "run_id",
        "duration_seconds",
    }
    if records:
        agent_frame = pd.DataFrame(records)
        agent_columns = [
            column for column in agent_frame.columns if column not in identity_columns
        ]
        zero_fill_columns = [
            column
            for column in agent_columns
            if column not in AGENT_PRECISION_FEATURE_NAMES
        ]
        agent_frame[zero_fill_columns] = agent_frame[zero_fill_columns].fillna(0)
        agent_frame.sort_values(["split", "prospectus_date", "company_id"], inplace=True)
    else:
        agent_frame = pd.DataFrame(columns=sorted(identity_columns))
        agent_columns = []
    agent_frame.to_csv(agent_matrix_path, index=False, encoding="utf-8-sig")

    base_path = (
        project_root
        / "data"
        / "processed"
        / "modeling"
        / "prelisting_feature_matrix.csv"
    )
    if not records or not base_path.is_file():
        return None
    base = pd.read_csv(base_path, dtype={"company_id": str})
    base["company_id"] = base["company_id"].astype(str).str.zfill(5)
    augmentation = agent_frame[["company_id", *agent_columns]]
    merged = base.merge(augmentation, on="company_id", how="inner", validate="one_to_one")
    output_path = agent_matrix_path.with_name("agent_augmented_feature_matrix.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _experiment_fingerprint(project_root: Path, config: AppConfig) -> dict[str, Any]:
    prompt_files = sorted((project_root / "prompts").rglob("*.md"))
    prompt_hashes = {
        path.relative_to(project_root).as_posix(): sha256(
            path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest()
        for path in prompt_files
    }
    return {
        "embedding_provider": config.embedding.provider,
        "embedding_model": config.embedding.model_name,
        "embedding_dimensions": config.embedding.dimensions,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model_name,
        "discovery_llm_provider": config.llm.provider,
        "discovery_llm_model": config.llm.model_name,
        "specialist_llm_provider": (
            config.specialist_llm.provider
            if config.specialist_llm is not None
            else config.llm.provider
        ),
        "specialist_llm_model": (
            config.specialist_llm.model_name
            if config.specialist_llm is not None
            else config.llm.model_name
        ),
        "discovery_batch_chars": config.agents.discovery_batch_chars,
        "specialist_batch_size": config.agents.specialist_batch_size,
        "specialist_followup_rounds": config.agents.specialist_followup_rounds,
        "decision_conflict_batch_size": (
            config.agents.decision_conflict_batch_size
        ),
        "prompt_hashes": prompt_hashes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _portable(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AgentFeatureExperimentResult",
    "normalize_historical_agent_features",
    "run_agent_feature_experiment",
    "select_pilot_companies",
]
