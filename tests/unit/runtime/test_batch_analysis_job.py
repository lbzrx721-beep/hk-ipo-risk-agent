from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

from ipo_risk.agents.intervention import (
    RiskScopeCandidate,
    RiskScopeReviewRequest,
)
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.storage.files import write_json
from ipo_risk.runtime.batch_analysis_job import (
    BatchAnalysisJob,
    BatchCompany,
    BatchUploadSpec,
    load_batch_companies,
)


def _company(company_id: str = "00001") -> BatchCompany:
    return BatchCompany(
        company_id=company_id,
        company_folder=f"{company_id}_测试公司",
        company_name="测试公司",
        prospectus_date="2025-01-01",
        split="train",
    )


def test_load_batch_companies_uses_manifest_identity_without_market_label_values(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path
        / "data"
        / "processed"
        / "competition"
        / "company_manifest.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "company_id": "1",
                "company_folder": "00001_测试公司",
                "company_name": "测试公司",
                "prospectus_date": "2025-01-01",
                "split": "validation",
                "significant_drop": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    companies = load_batch_companies(tmp_path)

    assert companies == (
        BatchCompany(
            company_id="00001",
            company_folder="00001_测试公司",
            company_name="测试公司",
            prospectus_date="2025-01-01",
            split="validation",
        ),
    )


def test_batch_job_coalesces_incremental_model_stream_events(
    tmp_path: Path,
) -> None:
    job = BatchAnalysisJob(
        project_root=tmp_path,
        experiment_name="event-test",
        companies=(_company(),),
    )
    event_path = job.experiment_dir / "events" / "00001.jsonl"
    event_path.parent.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    event_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "stage": "agent:finance:delta",
                        "detail": '[[batch:1/1]]{"assessments":[',
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "stage": "agent:finance:delta",
                        "detail": '[[batch:1/1]]{"risk_type":"cash"}]}',
                    },
                    ensure_ascii=False,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    first = job.events_for("00001_测试公司")
    second = job.events_for("00001_测试公司")

    assert len(first) == 1
    assert first[0].detail == (
        '[[batch:1/1]]{"assessments":[{"risk_type":"cash"}]}'
    )
    assert second == first


def test_batch_job_retry_failed_resets_interrupted_companies(
    tmp_path: Path,
) -> None:
    first = _company("00001")
    second = _company("00002")
    job = BatchAnalysisJob(
        project_root=tmp_path,
        experiment_name="retry-test",
        companies=(first, second),
    )
    write_json(
        {
            "schema_version": 1,
            "companies": {
                first.company_folder: {
                    "company_id": first.company_id,
                    "company_folder": first.company_folder,
                    "split": first.split,
                    "state": "completed",
                },
                second.company_folder: {
                    "company_id": second.company_id,
                    "company_folder": second.company_folder,
                    "split": second.split,
                    "state": "failed",
                    "error": "temporary provider error",
                    "traceback": "trace",
                },
            },
            "queue_state": "completed",
            "pause_reason": None,
        },
        job.status_path,
    )

    reset_count = job.retry_failed()
    snapshot = job.snapshot()

    assert reset_count == 1
    assert snapshot.state == "paused"
    assert snapshot.companies[0].state == "completed"
    assert snapshot.companies[1].state == "pending"
    assert snapshot.companies[1].error is None


def test_batch_snapshot_exposes_document_preparation_as_distinct_state(
    tmp_path: Path,
) -> None:
    company = _company()
    job = BatchAnalysisJob(
        project_root=tmp_path,
        experiment_name="preparing-state-test",
        companies=(company,),
    )
    write_json(
        {
            "schema_version": 1,
            "companies": {
                company.company_folder: {
                    "company_id": company.company_id,
                    "company_folder": company.company_folder,
                    "split": company.split,
                    "state": "pending",
                    "preparation_state": "running",
                }
            },
            "queue_state": "running",
            "pause_reason": None,
        },
        job.status_path,
    )

    snapshot = job.snapshot()

    assert snapshot.companies[0].state == "preparing"
    assert snapshot.current_company_folder == company.company_folder


def test_batch_snapshot_keeps_prepared_company_visible_while_waiting_for_agent(
    tmp_path: Path,
) -> None:
    company = _company()
    job = BatchAnalysisJob(
        project_root=tmp_path,
        experiment_name="ready-state-test",
        companies=(company,),
    )
    write_json(
        {
            "schema_version": 1,
            "companies": {
                company.company_folder: {
                    "company_id": company.company_id,
                    "company_folder": company.company_folder,
                    "split": company.split,
                    "state": "pending",
                    "preparation_state": "completed",
                }
            },
            "queue_state": "running",
            "pause_reason": None,
        },
        job.status_path,
    )

    snapshot = job.snapshot()

    assert snapshot.companies[0].state == "ready"
    assert snapshot.companies[0].preparation_state == "completed"


def test_batch_company_enters_its_own_risk_scope_review_state(
    tmp_path: Path,
) -> None:
    company = _company()
    job = BatchAnalysisJob(
        project_root=tmp_path,
        experiment_name="company-review-test",
        companies=(company,),
    )
    write_json(
        {
            "schema_version": 1,
            "companies": {
                company.company_folder: {
                    "company_id": company.company_id,
                    "company_folder": company.company_folder,
                    "split": company.split,
                    "state": "running",
                }
            },
            "queue_state": "running",
            "pause_reason": None,
        },
        job.status_path,
    )
    request = RiskScopeReviewRequest(
        run_id="RUN-BATCH-REVIEW",
        industry="软件",
        business_model="订阅",
        revenue_model="服务费",
        candidates=(
            RiskScopeCandidate(
                risk_type="customer_concentration",
                title="客户集中度",
                owner=AgentRole.BUSINESS,
                severity=RiskSeverity.HIGH,
                ai_recommended=True,
                rationale="客户集中度较高。",
                review_question="核验最近期间占比。",
                source="document_discovery",
            ),
        ),
    )
    controller = job._risk_scope_review_controllers[company.company_folder]
    waiter = Thread(target=controller.request_review, args=(request,), daemon=True)
    waiter.start()
    assert controller.wait_for_request(timeout=1)

    snapshot = job.snapshot()

    assert snapshot.companies[0].state == "awaiting_review"
    assert snapshot.companies[0].risk_scope_review == request


def test_uploaded_batch_prepares_documents_before_running_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeUploadService:
        def __init__(self, *, project_root: Path) -> None:
            assert project_root == tmp_path

        def register(self, **kwargs):
            captured["registered"] = kwargs
            return SimpleNamespace(
                company_folder="00001_上传公司",
                source_hash="a" * 64,
            )

    def runner(project_root: Path, **kwargs):
        captured["project_root"] = project_root
        captured["runner_kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(
        "ipo_risk.uploads.ProspectusUploadService",
        FakeUploadService,
    )
    upload = BatchUploadSpec(
        file_name="prospectus.pdf",
        content=b"%PDF-test",
        company_id="00001",
        company_name="上传公司",
        document_date=date(2025, 1, 2),
        source_hash="a" * 64,
        page_count=10,
    )
    job = BatchAnalysisJob.from_uploads(
        project_root=tmp_path,
        experiment_name="upload-test",
        uploads=(upload,),
        runner=runner,
    )

    job._run()

    assert job.companies == (
        BatchCompany(
            company_id="00001",
            company_folder="00001_上传公司",
            company_name="上传公司",
            prospectus_date="2025-01-02",
            split="upload",
        ),
    )
    runner_kwargs = captured["runner_kwargs"]
    assert runner_kwargs["company_folders"] == ("00001_上传公司",)
    assert runner_kwargs["company_rows"][0]["source_hash"] == "a" * 64


def test_uploaded_batch_exposes_registration_progress_immediately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration_started = Event()
    release_registration = Event()

    class SlowUploadService:
        def __init__(self, *, project_root: Path) -> None:
            assert project_root == tmp_path

        def register(self, **_kwargs):
            registration_started.set()
            assert release_registration.wait(timeout=2)
            return SimpleNamespace(
                company_folder="00001_上传公司",
                source_hash="a" * 64,
            )

    monkeypatch.setattr(
        "ipo_risk.uploads.ProspectusUploadService",
        SlowUploadService,
    )
    upload = BatchUploadSpec(
        file_name="prospectus.pdf",
        content=b"%PDF-test",
        company_id="00001",
        company_name="上传公司",
        document_date=date(2025, 1, 2),
        source_hash="a" * 64,
        page_count=10,
    )
    job = BatchAnalysisJob.from_uploads(
        project_root=tmp_path,
        experiment_name="upload-progress-test",
        uploads=(upload,),
        runner=lambda *_args, **_kwargs: SimpleNamespace(),
    )

    job.start()
    assert registration_started.wait(timeout=1)
    snapshot = job.snapshot()

    assert snapshot.phase == "preparing"
    assert snapshot.companies[0].state == "preparing"
    assert snapshot.companies[0].preparation_state == "registering"
    assert snapshot.phase_detail == "正在登记第1/1份：prospectus.pdf"

    release_registration.set()
    assert job._thread is not None
    job._thread.join(timeout=2)
