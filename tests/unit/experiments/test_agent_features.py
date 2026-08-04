from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from threading import Event, Lock
from time import sleep

import pandas as pd

from ipo_risk.experiments import (
    normalize_historical_agent_features,
    run_agent_feature_experiment,
    select_pilot_companies,
)
from ipo_risk.storage.files import write_json
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


def _analysis_artifacts(root: Path, company_id: str) -> AnalysisArtifacts:
    report = RiskAnalysisReport(
        run_id=f"RUN-{company_id}",
        company_id=company_id,
        company_name=f"公司{int(company_id)}",
        generated_at=datetime(2021, 1, 1, tzinfo=timezone.utc),
        source_document=SourceDocument(
            document_id=f"DOC-{company_id}",
            source_file=f"data/raw/prospectus/{company_id}_公司/招股书.pdf",
            source_hash=company_id.zfill(64),
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
        report_path=root / "artifacts" / company_id / "risk_analysis.json",
        pages_path=root / "data" / "interim" / company_id / "pages.jsonl",
        chunks_path=root / "data" / "interim" / company_id / "chunks.jsonl",
    )


def test_historical_reports_are_normalized_without_model_calls(
    tmp_path: Path,
) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    matrix_rows: list[dict[str, object]] = []
    for company_id, split in (("00001", "train"), ("00002", "validation")):
        artifacts = _analysis_artifacts(tmp_path, company_id)
        report_path = (
            tmp_path
            / "artifacts"
            / "analysis"
            / f"{company_id}_公司"
            / f"doc_{company_id}"
            / "risk_analysis.json"
        )
        write_json(artifacts.report.model_dump(mode="json"), report_path)
        matrix_rows.append(
            {
                "company_id": company_id,
                "ticker": f"{company_id}.HK",
                "company_name": f"公司{int(company_id)}",
                "prospectus_date": "2021-01-01",
                "as_of_date": "2021-01-01",
                "split": split,
                "target_significant_drop_5d": 0,
                "document_page_count": 100,
            }
        )
    pd.DataFrame(matrix_rows).to_csv(
        modeling_dir / "prelisting_feature_matrix.csv",
        index=False,
    )

    result = normalize_historical_agent_features(
        tmp_path,
        experiment_name="normalized-history-test",
    )

    assert result.selected_count == 2
    assert result.completed_count == 2
    assert result.failed_count == 0
    matrix = pd.read_csv(result.agent_feature_matrix_path, dtype={"company_id": str})
    assert matrix["company_id"].tolist() == ["00001", "00002"]
    assert matrix["agent_finding_count"].tolist() == [0, 0]
    assert set(matrix["split"]) == {"train", "validation"}
    assert result.augmented_feature_matrix_path is not None
    augmented = pd.read_csv(
        result.augmented_feature_matrix_path,
        dtype={"company_id": str},
    )
    assert len(augmented) == 2
    payload = json.loads(
        (
            result.experiment_dir
            / "companies"
            / "00001.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["feature_schema_version"] == 2
    assert payload["normalization"]["model_reinvoked"] is False


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
        return _analysis_artifacts(root, "00001")

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


def test_agent_experiment_pauses_between_companies_and_resumes(
    tmp_path: Path,
) -> None:
    competition_dir = tmp_path / "data" / "processed" / "competition"
    competition_dir.mkdir(parents=True)
    rows = [_manifest_row(1, "train"), _manifest_row(2, "validation")]
    (competition_dir / "company_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    calls: list[str] = []
    pause_requested = False

    def analyzer(company_folder: str, **_: object) -> AnalysisArtifacts:
        nonlocal pause_requested
        calls.append(company_folder)
        pause_requested = True
        company_id = company_folder.split("_", maxsplit=1)[0]
        return _analysis_artifacts(tmp_path, company_id)

    first = run_agent_feature_experiment(
        tmp_path,
        experiment_name="pause-test",
        company_folders=("00001_公司1", "00002_公司2"),
        pause_requested=lambda: pause_requested,
        analyzer=analyzer,
        agent_company_concurrency=1,
    )

    assert calls == ["00001_公司1"]
    status = json.loads(first.status_path.read_text(encoding="utf-8"))
    assert status["queue_state"] == "paused"
    assert status["companies"]["00001_公司1"]["state"] == "completed"
    assert status["companies"]["00002_公司2"]["state"] == "pending"

    pause_requested = False
    second = run_agent_feature_experiment(
        tmp_path,
        experiment_name="pause-test",
        company_folders=("00001_公司1", "00002_公司2"),
        pause_requested=lambda: pause_requested,
        analyzer=analyzer,
        agent_company_concurrency=1,
    )

    assert calls == ["00001_公司1", "00002_公司2"]
    resumed_status = json.loads(second.status_path.read_text(encoding="utf-8"))
    assert resumed_status["queue_state"] == "completed"
    assert second.completed_count == 2


def test_agent_experiment_runs_different_companies_with_controlled_concurrency(
    tmp_path: Path,
) -> None:
    rows = tuple(_manifest_row(index, "upload") for index in range(1, 4))
    active = 0
    max_active = 0
    lock = Lock()
    release = Event()

    def analyzer(company_folder: str, **_: object) -> AnalysisArtifacts:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if max_active >= 2:
                release.set()
        release.wait(timeout=1)
        sleep(0.01)
        with lock:
            active -= 1
        return _analysis_artifacts(
            tmp_path,
            company_folder.split("_", maxsplit=1)[0],
        )

    result = run_agent_feature_experiment(
        tmp_path,
        experiment_name="parallel-company-agent-test",
        company_folders=tuple(str(row["company_folder"]) for row in rows),
        company_rows=rows,
        analyzer=analyzer,
        agent_company_concurrency=2,
    )

    assert max_active == 2
    assert result.completed_count == 3


def test_agent_experiment_accepts_uploaded_company_rows_without_manifest(
    tmp_path: Path,
) -> None:
    row = _manifest_row(1, "upload")
    calls: list[str] = []

    def analyzer(company_folder: str, **_: object) -> AnalysisArtifacts:
        calls.append(company_folder)
        return _analysis_artifacts(tmp_path, "00001")

    result = run_agent_feature_experiment(
        tmp_path,
        experiment_name="uploaded-row-test",
        company_folders=("00001_公司1",),
        company_rows=(row,),
        analyzer=analyzer,
    )

    assert calls == ["00001_公司1"]
    assert result.completed_count == 1
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["companies"][0]["split"] == "upload"


def test_agent_experiment_prepares_documents_in_parallel_before_agents(
    tmp_path: Path,
) -> None:
    rows = (
        _manifest_row(1, "upload"),
        _manifest_row(2, "upload"),
        _manifest_row(3, "upload"),
    )
    active = 0
    max_active = 0
    lock = Lock()
    release = Event()
    sequence: list[str] = []

    def preparer(company_folder: str, **_: object) -> object:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            sequence.append(f"prepare:{company_folder}")
            if max_active >= 2:
                release.set()
        release.wait(timeout=1)
        sleep(0.01)
        with lock:
            active -= 1
        return object()

    def analyzer(company_folder: str, **_: object) -> AnalysisArtifacts:
        sequence.append(f"analyze:{company_folder}")
        return _analysis_artifacts(
            tmp_path,
            company_folder.split("_", maxsplit=1)[0],
        )

    run_agent_feature_experiment(
        tmp_path,
        experiment_name="parallel-preparation-test",
        company_folders=tuple(str(row["company_folder"]) for row in rows),
        company_rows=rows,
        analyzer=analyzer,
        preparer=preparer,
    )

    assert max_active >= 2
    first_analysis = next(
        index for index, item in enumerate(sequence) if item.startswith("analyze:")
    )
    assert all(
        item.startswith("prepare:") for item in sequence[:first_analysis]
    )
