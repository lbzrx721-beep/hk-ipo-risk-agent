from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import fitz

from ipo_risk.api.batch_service import BatchWorkspace
from ipo_risk.api.main import create_app
from ipo_risk.runtime.analysis_job import AnalysisProgressEvent
from ipo_risk.runtime.batch_analysis_job import BatchCompany, BatchCompanySnapshot


class FakeBatchJob:
    def __init__(self) -> None:
        self.companies = (
            BatchCompany(
                company_id="00001",
                company_folder="00001_测试公司",
                company_name="测试公司",
                prospectus_date="2025-01-01",
                split="upload",
            ),
        )
        self.started = False

    def start(self) -> None:
        self.started = True

    def snapshot(self):
        return SimpleNamespace(
            state="running",
            phase="analyzing",
            phase_detail="专业Agent正在分析。",
            active=True,
            error=None,
            companies=(
                BatchCompanySnapshot(
                    company_id="00001",
                    company_folder="00001_测试公司",
                    company_name="测试公司",
                    prospectus_date="2025-01-01",
                    split="upload",
                    state="running",
                    run_id="RUN-00001",
                ),
            ),
        )

    def events_for(self, _company_folder: str):
        return (
            AnalysisProgressEvent(
                stage="agent:discovery:structured",
                detail=(
                    '[[batch:1/2]]{"candidates":['
                    '{"title":"客户集中风险","owner":"business",'
                    '"severity":"high","claim":"客户集中。",'
                    '"confidence":0.9,"evidence_chunk_ids":["C-1"]}]}'
                ),
                occurred_at=datetime.now(timezone.utc),
            ),
            AnalysisProgressEvent(
                stage="agent:finance:completed",
                detail="财务Agent完成分析。",
                occurred_at=datetime.now(timezone.utc),
            ),
        )


def test_workspace_exposes_stable_company_key_and_events(tmp_path: Path) -> None:
    workspace = BatchWorkspace(workspace_id="web_batch_test", project_root=tmp_path)
    job = FakeBatchJob()

    workspace.add_job(job)  # type: ignore[arg-type]
    snapshot = workspace.snapshot_payload()
    events = workspace.events_payload("s01:00001")

    assert job.started is True
    assert snapshot["state"] == "running"
    assert snapshot["companies"][0]["company_key"] == "s01:00001"
    progress = snapshot["companies"][0]["progress"]
    assert progress["agents"]["discovery"]["state"] == "running"
    assert progress["agents"]["finance"]["state"] == "complete"
    assert "outputs" not in progress
    assert "events" not in progress
    assert events[1]["stage"] == "agent:finance:completed"


def test_workspace_keeps_structured_agent_events_outside_recent_window(
    tmp_path: Path,
) -> None:
    workspace = BatchWorkspace(workspace_id="web_batch_test", project_root=tmp_path)
    workspace.add_job(FakeBatchJob())  # type: ignore[arg-type]

    durable = workspace.structured_events_payload("s01:00001")

    assert [event["stage"] for event in durable] == [
        "agent:discovery:structured",
        "agent:finance:completed",
    ]


def test_workspace_keeps_base_prediction_events_for_reconnect(tmp_path: Path) -> None:
    workspace = BatchWorkspace(workspace_id="web_batch_test", project_root=tmp_path)
    job = FakeBatchJob()
    job.events_for = lambda _company_folder: (
        AnalysisProgressEvent(
            stage="base_prediction_completed",
            detail="基础预测已生成。",
            occurred_at=datetime.now(timezone.utc),
        ),
    )
    workspace.add_job(job)  # type: ignore[arg-type]

    durable = workspace.structured_events_payload("s01:00001")

    assert [event["stage"] for event in durable] == ["base_prediction_completed"]


def test_fastapi_health_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fastapi_reports_endpoint_starts_with_empty_registry(tmp_path: Path) -> None:
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/api/reports/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_upload_inspection_returns_editable_pdf_metadata(tmp_path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Stock Code: 1234")
    content = document.tobytes()
    document.close()
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/api/uploads/inspect",
        files={
            "files": (
                "prospectus_2025-01-02.pdf",
                content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()["files"][0]
    assert payload["company_id"] == "01234"
    assert payload["document_date"] == "2025-01-02"
    assert payload["page_count"] == 1
