"""历史招股书Agent特征的可恢复批量实验。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
import traceback
from typing import Any

import pandas as pd

from ipo_risk.config import AppConfig, load_config
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.modeling.features import build_agent_feature_snapshot
from ipo_risk.pipeline import AnalysisArtifacts, analyze_company
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.storage.files import read_json, write_json


Analyzer = Callable[..., AnalysisArtifacts]
AGENT_FEATURE_SCHEMA_VERSION = 2


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


def run_agent_feature_experiment(
    project_root: Path,
    *,
    experiment_name: str = "agent_feature_pilot_v1",
    sample_size: int = 20,
    splits: tuple[str, ...] = ("train", "validation"),
    company_folders: tuple[str, ...] = (),
    min_findings: int = 1,
    continue_on_error: bool = True,
    analyzer: Analyzer = analyze_company,
) -> AgentFeatureExperimentResult:
    """顺序运行Agent分析；每家公司完成后立即落盘，重复运行自动续跑。"""
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
    manifest_rows = _read_jsonl(manifest_path)
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
    consecutive_failure_key: str | None = None
    consecutive_failure_count = 0

    selected_by_folder = {
        str(row["company_folder"]): row for row in selected
    }
    for company_folder, manifest_row in selected_by_folder.items():
        company_id = str(manifest_row["company_id"]).zfill(5)
        feature_path = company_dir / f"{company_id}.json"
        current = status["companies"][company_folder]
        if current.get("state") == "completed" and feature_path.is_file():
            payload = read_json(feature_path)
            if payload.get("feature_schema_version") == AGENT_FEATURE_SCHEMA_VERSION:
                continue
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
            continue

        started_at = _now()
        started_clock = perf_counter()
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
        try:
            artifacts = analyzer(
                company_folder,
                project_root=root,
                min_findings=min_findings,
                progress_callback=events,
            )
            if artifacts.vector_persistence_task is not None:
                artifacts.vector_persistence_task.wait(timeout=300)
            report = artifacts.report
            as_of_date = report.source_document.available_at
            feature_snapshot = build_agent_feature_snapshot(
                report,
                as_of_date=as_of_date,
            )
            ended_at = _now()
            duration_seconds = round(perf_counter() - started_clock, 3)
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
            consecutive_failure_key = None
            consecutive_failure_count = 0
        except Exception as exc:  # noqa: BLE001 - 批量任务必须记录单公司失败后续跑
            ended_at = _now()
            budget_pause = isinstance(exc, ModelBudgetExceededError)
            current.update(
                {
                    "state": "paused" if budget_pause else "failed",
                    "ended_at": ended_at,
                    "duration_seconds": round(perf_counter() - started_clock, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            failure_key = f"{type(exc).__name__}:{str(exc).strip()[:300]}"
            if failure_key == consecutive_failure_key:
                consecutive_failure_count += 1
            else:
                consecutive_failure_key = failure_key
                consecutive_failure_count = 1
            circuit_open = budget_pause or consecutive_failure_count >= 2
            if circuit_open:
                status["queue_state"] = "paused"
                status["pause_reason"] = (
                    str(exc)
                    if budget_pause
                    else (
                        "连续两家公司出现相同失败，队列已安全暂停："
                        f"{failure_key}"
                    )
                )
            write_json(status, status_path)
            if not continue_on_error:
                raise
            if circuit_open:
                break
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
        agent_frame[agent_columns] = agent_frame[agent_columns].fillna(0)
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
        path.relative_to(project_root).as_posix(): sha256(path.read_bytes()).hexdigest()
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
    "run_agent_feature_experiment",
    "select_pilot_companies",
]
