from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ipo_risk.experiments import build_consolidated_agent_dataset
from ipo_risk.experiments.agent_features import AGENT_FEATURE_SCHEMA_VERSION


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _feature_payload(
    *,
    company_id: str,
    run_id: str,
    ended_at: str,
    finding_count: int,
    split: str = "upload",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "feature_schema_version": AGENT_FEATURE_SCHEMA_VERSION,
        "experiment_name": "fixture",
        "company_id": company_id,
        "company_folder": f"{company_id}_公司",
        "company_name": f"公司{company_id}",
        "split": split,
        "prospectus_date": "2024-01-01",
        "source_hash": f"hash-{company_id}",
        "run_id": run_id,
        "ended_at": ended_at,
        "duration_seconds": 10,
        "report_path": f"artifacts/analysis/{company_id}/risk_analysis.json",
        "features": {
            "agent_finding_count": {
                "value": finding_count,
                "available_at": "2024-01-01",
                "source_ref": f"analysis:{run_id}",
            },
            "agent_severity_high_count": {
                "value": 2,
                "available_at": "2024-01-01",
                "source_ref": f"analysis:{run_id}",
            },
        },
    }


def test_consolidation_keeps_latest_run_and_builds_four_horizon_labels(
    tmp_path: Path,
) -> None:
    modeling = tmp_path / "data" / "processed" / "modeling"
    competition = tmp_path / "data" / "processed" / "competition"
    modeling.mkdir(parents=True)
    competition.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "company_id": "00001",
                "company_name": "公司1",
                "prospectus_date": "2024-01-01",
                "split": "validation",
                "document_page_count": 100,
                "target_significant_drop_5d": 0,
            },
            {
                "company_id": "00002",
                "company_name": "公司2",
                "prospectus_date": "2023-01-01",
                "split": "train",
                "document_page_count": 200,
                "target_significant_drop_5d": 1,
            },
        ]
    ).to_csv(modeling / "prelisting_feature_matrix.csv", index=False)
    manifest = [
        {
            "company_id": "00001",
            "company_name": "公司1",
            "company_folder": "00001_公司1",
            "prospectus_date": "2024-01-01",
            "source_hash": "hash-00001",
            "split": "validation",
        },
        {
            "company_id": "00002",
            "company_name": "公司2",
            "company_folder": "00002_公司2",
            "prospectus_date": "2023-01-01",
            "source_hash": "hash-00002",
            "split": "train",
        },
    ]
    (competition / "company_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    labels = []
    for company_id in ("00001", "00002"):
        for horizon in (1, 5, 20, 60):
            labels.append(
                {
                    "company_id": company_id,
                    "horizon": horizon,
                    "label_status": "complete",
                    "observation_date": f"2024-02-{min(horizon, 28):02d}",
                    "return_vs_offer_pct": -float(horizon),
                    "return_vs_first_close_pct": -float(horizon) / 2,
                    "broke_issue_price": horizon >= 5,
                    "significant_drop": horizon >= 20,
                    "max_drawdown_pct": -float(horizon + 1),
                }
            )
    pd.DataFrame(labels).to_csv(competition / "market_labels.csv", index=False)

    for company_id in ("00001", "00002"):
        report = tmp_path / "artifacts" / "analysis" / company_id / "risk_analysis.json"
        _write_json(report, {})
    old = _feature_payload(
        company_id="00001",
        run_id="RUN-OLD",
        ended_at="2026-01-01T00:00:00Z",
        finding_count=3,
    )
    latest = _feature_payload(
        company_id="00001",
        run_id="RUN-LATEST",
        ended_at="2026-01-02T00:00:00Z",
        finding_count=8,
    )
    second = _feature_payload(
        company_id="00002",
        run_id="RUN-SECOND",
        ended_at="2026-01-03T00:00:00Z",
        finding_count=5,
    )
    _write_json(tmp_path / "artifacts/experiments/old/companies/00001.json", old)
    _write_json(tmp_path / "artifacts/experiments/new/companies/00001.json", latest)
    _write_json(tmp_path / "artifacts/experiments/new/companies/00002.json", second)

    result = build_consolidated_agent_dataset(tmp_path)

    assert result.company_count == 2
    assert result.duplicate_record_count == 1
    agent = pd.read_csv(result.agent_feature_matrix_path, dtype={"company_id": str})
    company = agent.loc[agent["company_id"] == "00001"].iloc[0]
    assert company["run_id"] == "RUN-LATEST"
    assert company["split"] == "validation"
    assert company["agent_finding_count"] == 8
    augmented = pd.read_csv(
        result.augmented_feature_matrix_path,
        dtype={"company_id": str},
    )
    assert "target_significant_drop_1d" in augmented.columns
    assert "target_significant_drop_60d" in augmented.columns
    assert "target_observation_date_5d" in augmented.columns
    assert "target_observation_date" not in augmented.columns
    assert augmented.loc[augmented["company_id"] == "00001", "split"].item() == (
        "validation"
    )
    audit = [
        json.loads(line)
        for line in result.audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit) == 2
    assert audit[0]["candidate_run_count"] == 2
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["split_counts"] == {"train": 1, "validation": 1}
    assert summary["label_coverage"]["60d"]["complete"] == 2


def test_consolidation_skips_outdated_or_mismatched_snapshots(tmp_path: Path) -> None:
    modeling = tmp_path / "data" / "processed" / "modeling"
    competition = tmp_path / "data" / "processed" / "competition"
    modeling.mkdir(parents=True)
    competition.mkdir(parents=True)
    pd.DataFrame(
        [{"company_id": "00001", "split": "train", "prospectus_date": "2023-01-01"}]
    ).to_csv(modeling / "prelisting_feature_matrix.csv", index=False)
    manifest = {
        "company_id": "00001",
        "company_name": "公司1",
        "company_folder": "00001_公司1",
        "prospectus_date": "2023-01-01",
        "source_hash": "hash-00001",
        "split": "train",
    }
    (competition / "company_manifest.jsonl").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "company_id": "00001",
                "horizon": horizon,
                "label_status": "complete",
            }
            for horizon in (1, 5, 20, 60)
        ]
    ).to_csv(competition / "market_labels.csv", index=False)
    _write_json(tmp_path / "artifacts/analysis/00001/risk_analysis.json", {})
    valid = _feature_payload(
        company_id="00001",
        run_id="RUN-VALID",
        ended_at="2026-01-01T00:00:00Z",
        finding_count=2,
    )
    outdated = dict(valid)
    outdated["feature_schema_version"] = 1
    _write_json(tmp_path / "artifacts/experiments/valid/companies/00001.json", valid)
    _write_json(tmp_path / "artifacts/experiments/old/companies/00001.json", outdated)

    result = build_consolidated_agent_dataset(tmp_path)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["company_count"] == 1
    assert summary["skipped_inputs_by_reason"] == {"outdated_feature_schema": 1}
