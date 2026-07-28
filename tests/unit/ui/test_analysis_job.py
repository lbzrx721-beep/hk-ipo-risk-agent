from __future__ import annotations

from datetime import date
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from ipo_risk.agents.intervention import (
    RiskScopeCandidate,
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
    RiskScopeSelection,
)
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.ui.analysis_job import AnalysisJob


class _UploadService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def register(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            reused=False,
            page_count=12,
            document_id="DOC-BACKGROUND",
            company_folder="03696_测试公司",
        )


def test_analysis_job_continues_independently_of_page_rerenders(
    tmp_path: Path,
) -> None:
    entered_analyzer = Event()
    release_analyzer = Event()
    artifact = SimpleNamespace(report=SimpleNamespace(run_id="RUN-BACKGROUND"))

    def analyze(_company_folder: str, **kwargs: object) -> SimpleNamespace:
        progress = kwargs["progress_callback"]
        assert callable(progress)
        progress("agent:finance:started", "财务Agent开始分析。")
        entered_analyzer.set()
        assert release_analyzer.wait(timeout=2)
        progress("agent:finance:completed", "财务Agent完成分析。")
        return artifact

    job = AnalysisJob(
        upload_service=_UploadService(tmp_path),
        analyze=analyze,
        file_name="prospectus.pdf",
        content=b"pdf",
        company_id="03696",
        company_name="测试公司",
        document_date=date(2026, 7, 22),
    )

    job.start()
    assert entered_analyzer.wait(timeout=2)
    running = job.snapshot()

    assert running.status == "running"
    assert any(event.stage == "agent:finance:started" for event in running.events)

    release_analyzer.set()
    assert job.wait(timeout=2)
    completed = job.snapshot()

    assert completed.status == "completed"
    assert completed.artifacts is artifact
    assert completed.error is None


def test_analysis_job_exposes_scope_review_and_resumes_after_submission(
    tmp_path: Path,
) -> None:
    artifact = SimpleNamespace(report=SimpleNamespace(run_id="RUN-SCOPE-REVIEW"))

    def analyze(_company_folder: str, **kwargs: object) -> SimpleNamespace:
        controller = kwargs["risk_scope_review_controller"]
        assert isinstance(controller, RiskScopeReviewController)
        request = RiskScopeReviewRequest(
            run_id="RUN-SCOPE-REVIEW",
            candidates=(
                RiskScopeCandidate(
                    risk_type="cash_burn",
                    title="现金消耗压力",
                    owner=AgentRole.FINANCE,
                    severity=RiskSeverity.HIGH,
                    ai_recommended=True,
                    rationale="研发投入较高。",
                    review_question="核验现金消耗与融资依赖。",
                    source="document_discovery",
                ),
            ),
        )
        controller.request_review(request)
        return artifact

    job = AnalysisJob(
        upload_service=_UploadService(tmp_path),
        analyze=analyze,
        file_name="prospectus.pdf",
        content=b"pdf",
        company_id="03696",
        company_name="测试公司",
        document_date=date(2026, 7, 22),
    )

    job.start()
    assert job.wait_for_risk_scope_review(timeout=2)
    waiting = job.snapshot()

    assert waiting.status == "awaiting_review"
    assert waiting.risk_scope_review is not None
    assert waiting.risk_scope_review.run_id == "RUN-SCOPE-REVIEW"

    job.submit_risk_scope_review(
        RiskScopeReviewDecision(
            run_id="RUN-SCOPE-REVIEW",
            reviewer="人工复核员",
            selections=(
                RiskScopeSelection(
                    risk_type="cash_burn",
                    included=True,
                    owner=AgentRole.MARKET,
                    severity=RiskSeverity.MEDIUM,
                    analysis_requirement="关注融资窗口与市场环境。",
                ),
            ),
        )
    )

    assert job.wait(timeout=2)
    completed = job.snapshot()
    assert completed.status == "completed"
    assert completed.risk_scope_review is None
    assert completed.artifacts is artifact


def test_analysis_job_marks_budget_limit_as_safe_pause(tmp_path: Path) -> None:
    def analyze(_company_folder: str, **_kwargs: object) -> SimpleNamespace:
        raise ModelBudgetExceededError("模型调用预算已达到 40 次")

    job = AnalysisJob(
        upload_service=_UploadService(tmp_path),
        analyze=analyze,
        file_name="prospectus.pdf",
        content=b"pdf",
        company_id="03696",
        company_name="测试公司",
        document_date=date(2026, 7, 22),
    )

    job.start()
    assert job.wait(timeout=2)
    snapshot = job.snapshot()

    assert snapshot.status == "paused"
    assert "预算" in str(snapshot.error)
    assert snapshot.events[-1].stage == "model_budget_paused"


def test_analysis_job_coalesces_high_frequency_deltas_per_model_stream(
    tmp_path: Path,
) -> None:
    job = AnalysisJob(
        upload_service=_UploadService(tmp_path),
        analyze=lambda *_args, **_kwargs: SimpleNamespace(),
        file_name="prospectus.pdf",
        content=b"pdf",
        company_id="03696",
        company_name="测试公司",
        document_date=date(2026, 7, 22),
    )

    for token in ("甲", "乙", "丙"):
        job._emit("agent:finance:delta", token)
    for token in ("一", "二"):
        job._emit("agent:discovery:delta", f"[[batch:1/2]]{token}")
    job._emit("agent:discovery:delta", "[[batch:2/2]]另")

    events = job.snapshot().events
    assert len(events) == 3
    assert events[0].detail == "甲乙丙"
    assert events[1].detail == "[[batch:1/2]]一二"
    assert events[2].detail == "[[batch:2/2]]另"
