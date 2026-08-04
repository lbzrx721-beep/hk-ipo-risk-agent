from __future__ import annotations

import json
from pathlib import Path

from ipo_risk.experiments.unattended import (
    select_unattended_companies,
    write_agent_training_dataset,
)


def _write_manifest(root: Path) -> None:
    path = root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "company_id": f"{index:05d}",
            "company_folder": f"{index:05d}_公司{index}",
            "company_name": f"公司{index}",
            "prospectus_date": f"202{index}-01-01",
            "split": "train",
        }
        for index in range(1, 5)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_unattended_selection_skips_completed_and_active_companies(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    completed_path = (
        tmp_path / "artifacts" / "experiments" / "old" / "companies" / "00001.json"
    )
    completed_path.parent.mkdir(parents=True)
    completed_path.write_text(
        json.dumps(
            {"company_id": "00001", "run_id": "RUN-1", "features": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    status_path = tmp_path / "artifacts" / "experiments" / "live" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "queue_state": "running",
                "companies": {
                    "00002_公司2": {
                        "company_id": "00002",
                        "state": "running",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected = select_unattended_companies(tmp_path, split="train", limit=10)

    assert [row["company_id"] for row in selected] == ["00003", "00004"]


def test_training_dataset_contains_only_completed_company_payloads(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "artifacts" / "experiments" / "auto-test"
    company_dir = experiment_dir / "companies"
    company_dir.mkdir(parents=True)
    (company_dir / "00001.json").write_text(
        json.dumps(
            {"company_id": "00001", "run_id": "RUN-1", "features": {"x": 1}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (company_dir / "00002.json").write_text(
        json.dumps(
            {"company_id": "00002", "run_id": None, "features": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output = write_agent_training_dataset(experiment_dir)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["company_count"] == 1
    assert payload["companies"][0]["company_id"] == "00001"
