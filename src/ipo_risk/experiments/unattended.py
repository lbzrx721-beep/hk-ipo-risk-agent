"""无人值守Agent实验的样本选择与JSON汇总。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from ipo_risk.config import load_config
from ipo_risk.storage.files import read_json, write_json
from ipo_risk.storage.repositories import CatalogRepository


def _company_id(value: object) -> str:
    text = str(value or "").strip()
    return text.zfill(5) if text.isdigit() else text.upper()


def load_competition_manifest(project_root: Path) -> tuple[dict[str, Any], ...]:
    """读取比赛公司清单，不加载上市后标签值。"""
    path = (
        project_root.resolve()
        / "data"
        / "processed"
        / "competition"
        / "company_manifest.jsonl"
    )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            company_folder = str(payload.get("company_folder") or "").strip()
            company_id = _company_id(payload.get("company_id"))
            if not company_folder or not company_id:
                continue
            rows.append({**payload, "company_id": company_id})
    return tuple(rows)


def completed_agent_company_ids(project_root: Path) -> frozenset[str]:
    """汇总已成功生成报告或Agent特征的公司，避免重复模型调用。"""
    root = project_root.resolve()
    config = load_config(
        config_path=root / "configs" / "base.yaml",
        providers_path=root / "configs" / "providers.local.yaml",
    )
    repository = CatalogRepository(root / config.paths.database)
    repository.initialize()
    completed = {
        _company_id(run.company_id)
        for run in repository.list_analysis_runs(status="succeeded")
    }
    experiment_root = root / "artifacts" / "experiments"
    for feature_path in experiment_root.glob("*/companies/*.json"):
        try:
            payload = read_json(feature_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("run_id") and isinstance(payload.get("features"), dict):
            completed.add(_company_id(payload.get("company_id")))
    return frozenset(completed)


def active_agent_company_ids(
    project_root: Path,
    *,
    recent_hours: int = 12,
) -> frozenset[str]:
    """识别近期仍在运行的公司，防止网页和脚本重复调用同一家公司。"""
    root = project_root.resolve()
    experiment_root = root / "artifacts" / "experiments"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
    active: set[str] = set()
    for status_path in experiment_root.glob("*/status.json"):
        try:
            modified_at = datetime.fromtimestamp(
                status_path.stat().st_mtime,
                tz=timezone.utc,
            )
            if modified_at < cutoff:
                continue
            payload = read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("queue_state") not in {"pending", "running", "paused"}:
            continue
        companies = payload.get("companies")
        if not isinstance(companies, dict):
            continue
        for item in companies.values():
            if not isinstance(item, dict):
                continue
            if item.get("state") in {
                "pending",
                "preparing",
                "ready",
                "awaiting_review",
                "running",
                "paused",
            }:
                active.add(_company_id(item.get("company_id")))
    return frozenset(active)


def select_unattended_companies(
    project_root: Path,
    *,
    split: str = "train",
    limit: int = 10,
) -> tuple[dict[str, Any], ...]:
    """按日期和公司代码稳定选择尚未成功、也未在运行的公司。"""
    if limit < 1:
        raise ValueError("limit must be positive")
    completed = completed_agent_company_ids(project_root)
    active = active_agent_company_ids(project_root)
    eligible = [
        row
        for row in load_competition_manifest(project_root)
        if str(row.get("split")) == split
        and _company_id(row.get("company_id")) not in completed
        and _company_id(row.get("company_id")) not in active
    ]
    eligible.sort(
        key=lambda row: (
            str(row.get("prospectus_date") or ""),
            _company_id(row.get("company_id")),
        )
    )
    return tuple(eligible[:limit])


def write_agent_training_dataset(experiment_dir: Path) -> Path:
    """把逐公司Agent特征整理成一个可直接交给训练流程的JSON。"""
    records: list[dict[str, Any]] = []
    for feature_path in sorted((experiment_dir / "companies").glob("*.json")):
        payload = read_json(feature_path)
        if payload.get("run_id") and isinstance(payload.get("features"), dict):
            records.append(payload)
    output = experiment_dir / "agent_training_dataset.json"
    write_json(
        {
            "schema_version": 1,
            "experiment_name": experiment_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "company_count": len(records),
            "companies": records,
        },
        output,
    )
    return output


__all__ = [
    "active_agent_company_ids",
    "completed_agent_company_ids",
    "load_competition_manifest",
    "select_unattended_companies",
    "write_agent_training_dataset",
]
