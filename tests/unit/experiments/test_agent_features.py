from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd

from ipo_risk.experiments import (
    run_agent_feature_experiment,
    select_pilot_companies,
)
from ipo_risk.pipeline import AnalysisArtifacts
from ipo_risk.reporting import AnalysisSummary, RiskAnalysisReport, SourceDocument


def _manifest_row(index: int, split: str) -> dict[str, object]:
    return {
        "company_id": f"{index:05d}",
        "company_folder": f"{index:05d}_公司{index}",
        "company_name": f"公司{index}",
        "prospectus_date": f"202{index % 4}-01-01",
        "source_hash": f"{index:064x}",
        "label_status": "complete",
        "split": split,
        "significant_drop": bool(index % 2),
    }


def test_pilot_selection_is_deterministic_and_does_not_depend_on_label_values() -> None:
    rows = [
        *[_manifest_row(index, "train") for index in range(1, 9)],
        *[_manifest_row(index, "validation") for index in range(9, 13)],
    ]

    first = select_pilot_companies(rows, sample_size=6)
    for row in rows:
        row["significant_drop"] = not bool(row["significant_drop"])
    second = select_pilot_companies(rows, sample_size=6)

    assert [row["company_id"] for row in first] == [
        row["company_id"] for row in second
    ]
    assert {row["split"] for row in first} == {"train", "validation"}


def test_agent_experiment_resumes_completed_company_without_reanalysis(
    tmp_path: Path,
) -> None:
    root = tmp_path
    competition_dir = root / "data" / "processed" / "competition"
    modeling_dir = root / "data" / "processed" / "modeling"
    competition_dir.mkdir(parents=True)
    modeling_dir.mkdir(parents=True)
    row = _manifest_row(1, "train")
    (competition_dir / "company_manifest.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "company_id": "00001",
                "ticker": "00001.HK",
                "company_name": "公司1",
                "prospectus_date": "2021-01-01",
                "as_of_date": "2021-01-01",
                "split": "train",
                "target_significant_drop_5d": 0,
                "target_return_vs_offer_5d_pct": 1.0,
                "target_observation_date": "2021-01-08",
                "label_offer_price_source": "official",
                "document_page_count": 100,
            }
        ]
    ).to_csv(modeling_dir / "prelisting_feature_matrix.csv", index=False)
    calls: list[str] = []

    def analyzer(company_folder: str, **_: object) -> AnalysisArtifacts:
        calls.append(company_folder)
        report = RiskAnalysisReport(
            run_id="RUN-00001",
            company_id="00001",
            company_name="公司1",
            generated_at=datetime(2021, 1, 1, tzinfo=timezone.utc),
            source_document=SourceDocument(
                document_id="DOC-00001",
                source_file="data/raw/prospectus/00001_公司1/招股书.pdf",
                source_hash="1".zfill(64),
                available_at=date(2021, 1, 1),
                page_count=100,
                chunk_count=200,
            ),
            summary=AnalysisSummary(
                evidence_count=0,
                finding_count=0,
                screenshot_count=0,
            ),
            evidences=(),
            risk_findings=(),
            agent_traces=(),
            evidence_display_zh_hans={},
        )
        return AnalysisArtifacts(
            report=report,
            report_path=root / "artifacts" / "risk_analysis.json",
            pages_path=root / "data" / "interim" / "pages.jsonl",
            chunks_path=root / "data" / "interim" / "chunks.jsonl",
        )

    first = run_agent_feature_experiment(
        root,
        experiment_name="resume-test",
        company_folders=("00001_公司1",),
        analyzer=analyzer,
    )
    second = run_agent_feature_experiment(
        root,
        experiment_name="resume-test",
        company_folders=("00001_公司1",),
        analyzer=analyzer,
    )

    assert calls == ["00001_公司1"]
    assert first.completed_count == second.completed_count == 1
    assert second.failed_count == 0
    matrix = pd.read_csv(second.agent_feature_matrix_path)
    assert matrix.loc[0, "agent_finding_count"] == 0
