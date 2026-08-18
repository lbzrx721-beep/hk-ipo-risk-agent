"""把分散的Agent实验结果整理为可训练、可审计的统一数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ipo_risk.modeling.agent_precision import (
    AGENT_PRECISION_FEATURE_NAMES,
    reconcile_precision_agent_features,
)
from ipo_risk.storage.files import write_json

from .agent_features import AGENT_FEATURE_SCHEMA_VERSION


LABEL_HORIZONS = (1, 5, 20, 60)
IDENTITY_COLUMNS = (
    "company_id",
    "company_folder",
    "company_name",
    "split",
    "prospectus_date",
    "source_hash",
    "run_id",
    "experiment_name",
    "ended_at",
    "duration_seconds",
    "report_path",
)


@dataclass(frozen=True)
class ConsolidatedAgentDatasetResult:
    """统一Agent增强数据集的落盘位置和核心统计。"""

    agent_feature_matrix_path: Path
    augmented_feature_matrix_path: Path
    audit_path: Path
    summary_path: Path
    company_count: int
    duplicate_record_count: int


def build_consolidated_agent_dataset(
    project_root: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> ConsolidatedAgentDatasetResult:
    """按公司保留最新成功run，并合并上市前特征与四个期限的市场标签。"""
    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "data" / "processed" / "modeling"
    )
    destination.mkdir(parents=True, exist_ok=True)

    base_path = destination / "prelisting_feature_matrix.csv"
    if not base_path.is_file():
        base_path = root / "data" / "processed" / "modeling" / base_path.name
    labels_path = root / "data" / "processed" / "competition" / "market_labels.csv"
    manifest_path = (
        root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    )
    for required in (base_path, labels_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing required modeling input: {required}")

    base = pd.read_csv(base_path, dtype={"company_id": str})
    base["company_id"] = base["company_id"].map(_company_id)
    _require_unique(base, "base prelisting feature matrix")
    base_by_id = base.set_index("company_id", drop=False)
    manifest = _load_manifest(manifest_path)

    candidates, skipped = _load_successful_agent_snapshots(root)
    selected, duplicate_count = _select_latest_by_company(candidates)
    selected_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for company_id, payload in sorted(selected.items()):
        manifest_row = manifest.get(company_id)
        if manifest_row is None:
            excluded.append({"company_id": company_id, "reason": "missing_manifest"})
            continue
        if company_id not in base_by_id.index:
            excluded.append({"company_id": company_id, "reason": "missing_base_features"})
            continue
        expected_hash = str(manifest_row.get("source_hash") or "")
        actual_hash = str(payload.get("source_hash") or "")
        if expected_hash and actual_hash and expected_hash != actual_hash:
            excluded.append({"company_id": company_id, "reason": "source_hash_mismatch"})
            continue

        features = payload["features"]
        feature_values = {
            str(name): feature.get("value")
            for name, feature in features.items()
            if isinstance(feature, dict) and "value" in feature
        }
        if not feature_values:
            excluded.append({"company_id": company_id, "reason": "empty_features"})
            continue
        identity = {
            "company_id": company_id,
            "company_folder": manifest_row.get("company_folder"),
            "company_name": manifest_row.get("company_name"),
            "split": manifest_row.get("split"),
            "prospectus_date": manifest_row.get("prospectus_date"),
            "source_hash": expected_hash or actual_hash,
            "run_id": payload.get("run_id"),
            "experiment_name": payload.get("experiment_name"),
            "ended_at": payload.get("ended_at"),
            "duration_seconds": payload.get("duration_seconds"),
            "report_path": payload.get("report_path"),
        }
        selected_rows.append({**identity, **feature_values})
        audits.append(
            {
                **identity,
                "feature_schema_version": payload.get("feature_schema_version"),
                "candidate_run_count": len(candidates[company_id]),
                "discarded_run_ids": [
                    item.get("run_id")
                    for item in candidates[company_id]
                    if item.get("run_id") != payload.get("run_id")
                ],
                "features": features,
            }
        )

    if not selected_rows:
        raise ValueError("no valid Agent feature snapshots found")
    agent_frame = pd.DataFrame(selected_rows)
    raw_feature_columns = sorted(
        column for column in agent_frame.columns if column not in IDENTITY_COLUMNS
    )
    zero_fill_columns = [
        column
        for column in raw_feature_columns
        if column not in AGENT_PRECISION_FEATURE_NAMES
    ]
    agent_frame[zero_fill_columns] = agent_frame[zero_fill_columns].fillna(0)
    agent_frame = reconcile_precision_agent_features(agent_frame)
    feature_columns = sorted(
        column for column in agent_frame.columns if column not in IDENTITY_COLUMNS
    )
    agent_frame.sort_values(["split", "prospectus_date", "company_id"], inplace=True)
    _require_unique(agent_frame, "consolidated Agent feature matrix")

    label_frame = _build_wide_market_labels(labels_path)
    legacy_targets = [
        column
        for column in base.columns
        if column.startswith("target_") or column.startswith("label_")
    ]
    prelisting = base.drop(columns=legacy_targets)
    augmented = prelisting.merge(
        agent_frame[["company_id", *feature_columns]],
        on="company_id",
        how="inner",
        validate="one_to_one",
    ).merge(label_frame, on="company_id", how="left", validate="one_to_one")
    augmented.sort_values(["split", "prospectus_date", "company_id"], inplace=True)
    _require_unique(augmented, "Agent augmented modeling dataset")

    agent_path = destination / "agent_feature_matrix.csv"
    augmented_path = destination / "agent_augmented_feature_matrix.csv"
    audit_path = destination / "agent_feature_audit.jsonl"
    summary_path = destination / "agent_augmented_feature_summary.json"
    agent_frame.to_csv(agent_path, index=False, encoding="utf-8-sig")
    augmented.to_csv(augmented_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits),
        encoding="utf-8",
    )

    label_coverage = {
        f"{horizon}d": {
            "complete": int(
                (augmented[f"target_label_status_{horizon}d"] == "complete").sum()
            ),
            "missing": int(
                augmented[f"target_label_status_{horizon}d"].isna().sum()
            ),
        }
        for horizon in LABEL_HORIZONS
    }
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "latest_successful_run_per_company",
        "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
        "candidate_record_count": sum(len(items) for items in candidates.values()),
        "candidate_company_count": len(candidates),
        "company_count": len(augmented),
        "duplicate_record_count": duplicate_count,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "skipped_input_count": len(skipped),
        "skipped_inputs_by_reason": _count_reasons(skipped),
        "split_counts": {
            str(key): int(value)
            for key, value in augmented["split"].value_counts().sort_index().items()
        },
        "agent_feature_count": len(feature_columns),
        "agent_feature_columns": feature_columns,
        "precision_agent_features": {
            name: int(agent_frame[name].notna().sum())
            for name in AGENT_PRECISION_FEATURE_NAMES
            if name in agent_frame
        },
        "label_horizons": list(LABEL_HORIZONS),
        "label_coverage": label_coverage,
        "target_columns": [
            column for column in augmented.columns if column.startswith("target_")
        ],
        "point_in_time_note": (
            "agent_* and prelisting columns are model inputs; target_* columns are "
            "post-listing labels and must never enter model features"
        ),
        "agent_feature_matrix_path": _portable(agent_path, root),
        "augmented_feature_matrix_path": _portable(augmented_path, root),
        "audit_path": _portable(audit_path, root),
    }
    write_json(summary, summary_path)
    return ConsolidatedAgentDatasetResult(
        agent_feature_matrix_path=agent_path,
        augmented_feature_matrix_path=augmented_path,
        audit_path=audit_path,
        summary_path=summary_path,
        company_count=len(augmented),
        duplicate_record_count=duplicate_count,
    )


def _load_successful_agent_snapshots(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    for path in sorted((root / "artifacts" / "experiments").glob("*/companies/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"path": _portable(path, root), "reason": type(exc).__name__})
            continue
        reason = _invalid_snapshot_reason(payload, root)
        if reason is not None:
            skipped.append({"path": _portable(path, root), "reason": reason})
            continue
        company_id = _company_id(payload["company_id"])
        payload["_source_path"] = _portable(path, root)
        payload["_sort_time"] = _snapshot_timestamp(payload, path)
        candidates.setdefault(company_id, []).append(payload)
    return candidates, skipped


def _invalid_snapshot_reason(payload: dict[str, Any], root: Path) -> str | None:
    if payload.get("feature_schema_version") != AGENT_FEATURE_SCHEMA_VERSION:
        return "outdated_feature_schema"
    if not payload.get("company_id") or not payload.get("run_id"):
        return "missing_identity"
    if not isinstance(payload.get("features"), dict) or not payload["features"]:
        return "missing_features"
    report_path = payload.get("report_path")
    if not report_path or not (root / str(report_path)).is_file():
        return "missing_report"
    return None


def _select_latest_by_company(
    candidates: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for company_id, items in candidates.items():
        by_run: dict[str, dict[str, Any]] = {}
        for item in items:
            run_id = str(item["run_id"])
            current = by_run.get(run_id)
            if current is None or _snapshot_key(item) > _snapshot_key(current):
                by_run[run_id] = item
        unique_runs = list(by_run.values())
        selected[company_id] = max(unique_runs, key=_snapshot_key)
        duplicate_count += len(items) - 1
    return selected, duplicate_count


def _snapshot_key(payload: dict[str, Any]) -> tuple[datetime, str, str]:
    return (
        payload["_sort_time"],
        str(payload.get("run_id") or ""),
        str(payload.get("_source_path") or ""),
    )


def _snapshot_timestamp(payload: dict[str, Any], path: Path) -> datetime:
    value = str(payload.get("ended_at") or "").strip()
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _build_wide_market_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"company_id": str})
    labels["company_id"] = labels["company_id"].map(_company_id)
    labels = labels[labels["horizon"].isin(LABEL_HORIZONS)].copy()
    duplicates = labels.duplicated(["company_id", "horizon"], keep=False)
    if duplicates.any():
        duplicate_ids = sorted(labels.loc[duplicates, "company_id"].unique())
        raise ValueError(f"duplicate market labels: {', '.join(duplicate_ids[:10])}")

    source_columns = {
        "label_status": "label_status",
        "observation_date": "observation_date",
        "return_vs_offer_pct": "return_vs_offer_pct",
        "return_vs_first_close_pct": "return_vs_first_close_pct",
        "broke_issue_price": "broke_issue_price",
        "significant_drop": "significant_drop",
        "max_drawdown_pct": "max_drawdown_pct",
    }
    rows: list[dict[str, Any]] = []
    for company_id, group in labels.groupby("company_id", sort=True):
        row: dict[str, Any] = {"company_id": company_id}
        by_horizon = {int(item["horizon"]): item for item in group.to_dict("records")}
        for horizon in LABEL_HORIZONS:
            item = by_horizon.get(horizon, {})
            for source, suffix in source_columns.items():
                row[f"target_{suffix}_{horizon}d"] = item.get(source)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _require_unique(frame, "wide market label matrix")
    return frame


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        company_id = _company_id(row["company_id"])
        if company_id in rows:
            raise ValueError(f"duplicate company in manifest: {company_id}")
        rows[company_id] = row
    return rows


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if frame["company_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate company_id")


def _company_id(value: Any) -> str:
    return str(value).strip().split(".", maxsplit=1)[0].zfill(5)


def _count_reasons(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


__all__ = ["ConsolidatedAgentDatasetResult", "build_consolidated_agent_dataset"]
