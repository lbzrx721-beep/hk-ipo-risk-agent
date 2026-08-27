"""对已保存风险执行低成本、可审计的专业核验重试。"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from ipo_risk.agents.analysis import EvidenceAnalysis, OpenAICompatibleJsonChatProvider
from ipo_risk.agents.decision import DeepSeekDecisionAnalyzer
from ipo_risk.agents.grounded import AnalyzedRisk, RetrievalCandidate
from ipo_risk.agents.orchestrator import AgentProposal
from ipo_risk.agents.specialist import (
    DeepSeekSpecialistBatchAnalyzer,
    SpecialistAssessment,
    SpecialistStatus,
    SpecialistTask,
)
from ipo_risk.agents.usage import ModelPricing, ModelUsageLedger
from ipo_risk.config import load_config
from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    Evidence,
    EvidenceType,
    RiskFinding,
    ToolCall,
    ToolCallStatus,
    TraceStatus,
    VerificationStatus,
)
from ipo_risk.reporting import RiskAnalysisReport, write_risk_report
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import (
    RetryExhaustedError,
    RetryPolicy,
    retry_model_node_error,
    retry_transient_api_error,
)
from ipo_risk.storage.files import read_json
from ipo_risk.storage.repositories import CatalogRepository


RetryState = Literal["queued", "running", "succeeded", "failed"]
_PROFESSIONAL_ROLES = {
    AgentRole.FINANCE,
    AgentRole.LEGAL,
    AgentRole.BUSINESS,
    AgentRole.MARKET,
}
_LEGACY_FAILURE_PREFIX = "专业Agent批量核验失败"


def finding_has_retryable_failure(finding: RiskFinding) -> bool:
    """兼容新结构字段与旧报告中的错误文本。"""
    if finding.specialist_status == SpecialistStatus.TECHNICAL_FAILURE.value:
        return True
    uncertainty = finding.uncertainty or ""
    if _LEGACY_FAILURE_PREFIX not in uncertainty and "Agent技术失败" not in uncertainty:
        return False
    return retry_transient_api_error(RuntimeError(uncertainty))


@dataclass
class RetryItemSnapshot:
    finding_id: str
    state: RetryState = "queued"
    attempt: int = 0
    stage: str = "等待重试"
    error: str | None = None


@dataclass
class RetryJobSnapshot:
    job_id: str
    run_id: str
    scope: str
    state: RetryState
    max_attempts: int
    created_at: str
    updated_at: str
    detail: str
    items: dict[str, RetryItemSnapshot] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = {key: asdict(value) for key, value in self.items.items()}
        return payload


class FindingRetryService:
    """后台重跑专业与决策 Agent；复用已保存证据，不重做PDF和向量化。"""

    def __init__(
        self,
        *,
        repository: CatalogRepository,
        project_root: Path,
        max_workers: int = 2,
    ) -> None:
        self.repository = repository
        self.project_root = project_root.resolve()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finding-retry")
        self._lock = Lock()
        self._run_locks: dict[str, Lock] = {}
        self._jobs: dict[str, RetryJobSnapshot] = {}
        self._active_findings: set[tuple[str, str]] = set()

    def list_jobs(self, run_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.run_id == run_id]
            jobs.sort(key=lambda item: item.created_at, reverse=True)
            return tuple(job.as_dict() for job in jobs)

    def start(
        self,
        *,
        run_id: str,
        finding_ids: tuple[str, ...] | None = None,
        agent: AgentRole | None = None,
        scope: str,
    ) -> dict[str, Any]:
        findings = self.repository.list_findings(run_id)
        if not findings:
            raise KeyError(f"analysis run does not exist or has no findings: {run_id}")
        requested = set(finding_ids or ())
        selected = [
            finding
            for finding in findings
            if (not requested or finding.finding_id in requested)
            and (agent is None or finding.created_by is agent)
            and finding.status
            not in {VerificationStatus.VERIFIED, VerificationStatus.REJECTED}
            and finding_has_retryable_failure(finding)
        ]
        if requested - {finding.finding_id for finding in findings}:
            missing = sorted(requested - {finding.finding_id for finding in findings})
            raise KeyError(f"finding does not exist: {', '.join(missing)}")
        if not selected:
            raise ValueError("没有可重试的技术失败风险；证据不足项仍应由人工判断。")

        config = load_config(
            config_path=self.project_root / "configs" / "base.yaml",
            providers_path=self.project_root / "configs" / "providers.local.yaml",
        )
        now = _now().isoformat()
        job = RetryJobSnapshot(
            job_id=f"RETRY-{uuid4().hex[:12]}",
            run_id=run_id,
            scope=scope,
            state="queued",
            max_attempts=config.agents.max_retries + 1,
            created_at=now,
            updated_at=now,
            detail=f"已排队，准备复用现有证据重试{len(selected)}条风险。",
            items={finding.finding_id: RetryItemSnapshot(finding.finding_id) for finding in selected},
        )
        with self._lock:
            conflicts = [
                finding.finding_id
                for finding in selected
                if (run_id, finding.finding_id) in self._active_findings
            ]
            if conflicts:
                raise RuntimeError(f"风险正在重试，请勿重复提交：{', '.join(conflicts)}")
            self._jobs[job.job_id] = job
            self._active_findings.update((run_id, finding.finding_id) for finding in selected)
        self._executor.submit(self._run_job, job.job_id, tuple(selected))
        return job.as_dict()

    def _run_job(self, job_id: str, findings: tuple[RiskFinding, ...]) -> None:
        try:
            self._set_job(job_id, state="running", detail="正在加载已保存证据。")
            report = self._load_report(findings[0], self._jobs[job_id].run_id)
            evidence_by_id = {item.evidence_id: item for item in report.evidences}
            config, specialist, decision = self._build_models(job_id)
            analyses: dict[str, AnalyzedRisk] = {}
            assessments: dict[str, SpecialistAssessment] = {}
            traces: list[AgentTrace] = []

            grouped: dict[AgentRole, list[RiskFinding]] = defaultdict(list)
            for finding in findings:
                if finding.created_by not in _PROFESSIONAL_ROLES:
                    raise ValueError(f"风险{finding.finding_id}没有可重试的专业Agent归属。")
                grouped[finding.created_by].append(finding)

            for role, role_findings in grouped.items():
                started = _now()
                self._set_items(job_id, role_findings, state="running", stage=f"{_role_label(role)} Agent正在重试")
                tasks = tuple(
                    SpecialistTask(
                        rule=_finding_rule(finding),
                        candidates=tuple(
                            _evidence_chunk(evidence_by_id[evidence_id])
                            for evidence_id in finding.evidence_ids
                            if evidence_id in evidence_by_id
                        ),
                    )
                    for finding in role_findings
                )
                result = specialist.analyze_batch(tasks)
                ended = _now()
                by_risk = result.by_risk()
                errors: list[str] = []
                for finding, task in zip(role_findings, tasks, strict=True):
                    assessment = by_risk[task.rule.id]
                    assessments[finding.finding_id] = assessment
                    analyses[finding.finding_id] = _analyzed_risk(
                        task,
                        assessment,
                        attempts=result.attempts_by_risk.get(task.rule.id, 1),
                        errors=result.errors_by_risk.get(task.rule.id, ()),
                    )
                    attempts = result.attempts_by_risk.get(task.rule.id, 1)
                    if assessment.status is SpecialistStatus.TECHNICAL_FAILURE:
                        errors.append(assessment.uncertainty)
                        self._set_item(
                            job_id,
                            finding.finding_id,
                            state="failed",
                            attempt=attempts,
                            stage="专业Agent重试失败",
                            error=assessment.uncertainty,
                        )
                    else:
                        self._set_item(
                            job_id,
                            finding.finding_id,
                            attempt=attempts,
                            stage="专业核验完成，等待决策整合",
                        )
                traces.append(
                    _retry_trace(
                        run_id=report.run_id,
                        role=role,
                        finding_ids=tuple(item.finding_id for item in role_findings),
                        started_at=started,
                        ended_at=ended,
                        attempts=max(result.attempts_by_risk.values(), default=1),
                        errors=tuple(errors),
                        model_name=specialist.model_name,
                        failed=bool(errors),
                    )
                )

            proposals = tuple(
                AgentProposal(
                    proposal_id=f"retry_{uuid4().hex[:12]}",
                    agent=finding.created_by,
                    analyzed_risk=analyses[finding.finding_id],
                )
                for finding in findings
                if assessments[finding.finding_id].status is SpecialistStatus.SUPPORTED
            )
            verdicts: dict[str, Any] = {}
            if proposals:
                started = _now()
                proposal_findings = [
                    finding
                    for finding in findings
                    if assessments[finding.finding_id].status is SpecialistStatus.SUPPORTED
                ]
                self._set_items(job_id, proposal_findings, stage="决策Agent正在重新整合")
                policy = RetryPolicy(
                    max_retries=config.agents.max_retries,
                    initial_seconds=config.agents.retry_initial_seconds,
                    backoff_factor=config.agents.retry_backoff_factor,
                    max_seconds=config.agents.retry_max_seconds,
                    retry_if=retry_model_node_error,
                )
                try:
                    outcome = policy.run_with_feedback(
                        lambda feedback: decision.adjudicate_with_feedback(
                            proposals,
                            correction_feedback=feedback,
                        ),
                        on_retry=lambda error, attempt, wait: self._report_decision_retry(
                            job_id, proposal_findings, error, attempt, wait
                        ),
                    )
                    verdicts = {item.risk_type: item for item in outcome.value.decisions}
                    decision_errors = outcome.errors
                    decision_attempts = outcome.attempts
                    decision_failed = False
                except RetryExhaustedError as error:
                    decision_attempts = error.attempts
                    decision_errors = (str(error.last_error),)
                    decision_failed = True
                    for finding in proposal_findings:
                        self._set_item(
                            job_id,
                            finding.finding_id,
                            state="failed",
                            attempt=error.attempts,
                            stage="决策Agent重试失败",
                            error=str(error.last_error),
                        )
                ended = _now()
                traces.append(
                    _retry_trace(
                        run_id=report.run_id,
                        role=AgentRole.DECISION,
                        finding_ids=tuple(item.finding_id for item in proposal_findings),
                        started_at=started,
                        ended_at=ended,
                        attempts=decision_attempts,
                        errors=decision_errors,
                        model_name=decision.model_name,
                        failed=decision_failed,
                    )
                )

            replacements: dict[str, RiskFinding] = {}
            for finding in findings:
                assessment = assessments[finding.finding_id]
                verdict = verdicts.get(finding.risk_type)
                replacements[finding.finding_id] = _updated_finding(finding, assessment, verdict)
                if assessment.status is SpecialistStatus.TECHNICAL_FAILURE:
                    continue
                if assessment.status is SpecialistStatus.SUPPORTED and verdict is None:
                    continue
                self._set_item(
                    job_id,
                    finding.finding_id,
                    state="succeeded",
                    stage=("已重新分析，进入质量复核" if verdict is not None else "证据不足，机器未纳入正式结论"),
                    error=None,
                )

            self._save_report(report, replacements, tuple(traces))
            item_states = [item.state for item in self._jobs[job_id].items.values()]
            failed = item_states.count("failed")
            self._set_job(
                job_id,
                state="failed" if failed == len(item_states) else "succeeded",
                detail=(
                    f"重试完成：{len(item_states) - failed}条已更新，{failed}条仍为技术失败。"
                ),
            )
        except Exception as error:
            self._fail_job(job_id, error)
        finally:
            with self._lock:
                job = self._jobs[job_id]
                for finding_id in job.items:
                    self._active_findings.discard((job.run_id, finding_id))

    def _build_models(
        self,
        job_id: str,
    ) -> tuple[Any, DeepSeekSpecialistBatchAnalyzer, DeepSeekDecisionAnalyzer]:
        config = load_config(
            config_path=self.project_root / "configs" / "base.yaml",
            providers_path=self.project_root / "configs" / "providers.local.yaml",
        )
        if not config.llm.is_configured:
            raise RuntimeError("风险模型尚未配置，无法重新分析。")
        specialist_config = (
            config.specialist_llm
            if config.specialist_llm is not None and config.specialist_llm.is_configured
            else config.llm
        )
        ledger = ModelUsageLedger()

        def provider(settings: Any) -> OpenAICompatibleJsonChatProvider:
            return OpenAICompatibleJsonChatProvider(
                api_key=settings.require_api_key(),
                base_url=settings.base_url,
                model_name=settings.model_name,
                provider_name=settings.provider,
                request_extra_body=None if settings.disable_thinking else {},
                supports_json_object=settings.supports_json_object,
                reasoning_effort=settings.reasoning_effort,
                usage_ledger=ledger,
                pricing=ModelPricing(
                    input_per_million_rmb=settings.input_price_per_million_rmb,
                    output_per_million_rmb=settings.output_price_per_million_rmb,
                    cached_input_per_million_rmb=settings.cached_input_price_per_million_rmb,
                ),
            )

        role_providers = {role: provider(specialist_config) for role in _PROFESSIONAL_ROLES}
        role_prompts = {
            role: (self.project_root / "prompts" / "roles" / f"{role.value}_v2.md").read_text(encoding="utf-8")
            for role in _PROFESSIONAL_ROLES
        }
        specialist = DeepSeekSpecialistBatchAnalyzer(
            role_providers[AgentRole.FINANCE],
            temperature=specialist_config.temperature,
            max_output_tokens=specialist_config.max_output_tokens,
            batch_size=config.agents.specialist_batch_size,
            system_prompt=(self.project_root / "prompts" / "decision" / "evidence_analysis_v5.md").read_text(encoding="utf-8"),
            role_prompts=role_prompts,
            providers_by_role=role_providers,
            progress_callback=lambda stage, detail: self._report_progress(job_id, stage, detail),
            retry_policy=RetryPolicy(
                max_retries=config.agents.max_retries,
                initial_seconds=config.agents.retry_initial_seconds,
                backoff_factor=config.agents.retry_backoff_factor,
                max_seconds=config.agents.retry_max_seconds,
                retry_if=retry_model_node_error,
            ),
        )
        decision = DeepSeekDecisionAnalyzer(
            provider(config.llm),
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_output_tokens,
            system_prompt=(self.project_root / "prompts" / "decision" / "decision_adjudication_v4.md").read_text(encoding="utf-8"),
            progress_callback=lambda stage, detail: self._report_progress(job_id, stage, detail),
        )
        return config, specialist, decision

    def _load_report(self, finding: RiskFinding, run_id: str) -> RiskAnalysisReport:
        reference = self.repository.get_analysis_report_path(run_id)
        if not reference:
            raise FileNotFoundError("分析报告路径不存在，无法复用已保存证据。")
        path = Path(reference)
        if not path.is_absolute():
            path = self.project_root / path
        report = RiskAnalysisReport.model_validate(read_json(path.resolve()))
        if report.company_id != finding.company_id:
            raise ValueError("风险与分析报告公司不一致。")
        return report

    def _save_report(
        self,
        report: RiskAnalysisReport,
        replacements: dict[str, RiskFinding],
        traces: tuple[AgentTrace, ...],
    ) -> None:
        with self._lock:
            run_lock = self._run_locks.setdefault(report.run_id, Lock())
        with run_lock:
            reference = self.repository.get_analysis_report_path(report.run_id)
            if not reference:
                raise FileNotFoundError("分析报告路径不存在。")
            path = Path(reference)
            if not path.is_absolute():
                path = self.project_root / path
            latest = RiskAnalysisReport.model_validate(read_json(path.resolve()))
            current = {item.finding_id: item for item in self.repository.list_findings(report.run_id)}
            findings = tuple(
                (
                    replacements.get(item.finding_id, item)
                    if current.get(item.finding_id, item).status
                    not in {VerificationStatus.VERIFIED, VerificationStatus.REJECTED}
                    else current[item.finding_id]
                )
                for item in latest.risk_findings
            )
            updated = latest.model_copy(
                update={"risk_findings": findings, "agent_traces": (*latest.agent_traces, *traces)}
            )
            write_risk_report(updated, path.resolve())
            self.repository.save_analysis_report(updated, reference)

    def _report_progress(self, job_id: str, stage: str, detail: str) -> None:
        if stage.endswith(":delta") or stage.endswith(":structured"):
            return
        attempt_match = re.search(r"第(\d+)次尝试", detail)
        attempt = int(attempt_match.group(1)) if attempt_match else None
        with self._lock:
            job = self._jobs[job_id]
            job.detail = detail
            job.updated_at = _now().isoformat()
            if attempt is not None:
                for item in job.items.values():
                    if item.state == "running":
                        item.attempt = attempt
                        item.stage = detail

    def _report_decision_retry(
        self,
        job_id: str,
        findings: list[RiskFinding],
        error: Exception,
        attempt: int,
        wait: float,
    ) -> None:
        detail = f"决策Agent连接中断，{wait:g}秒后进行第{attempt}次尝试。"
        self._set_job(job_id, detail=detail)
        for finding in findings:
            self._set_item(job_id, finding.finding_id, attempt=attempt, stage=detail, error=str(error))

    def _set_job(self, job_id: str, *, state: RetryState | None = None, detail: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if state is not None:
                job.state = state
            if detail is not None:
                job.detail = detail
            job.updated_at = _now().isoformat()

    def _set_items(self, job_id: str, findings: list[RiskFinding], **updates: Any) -> None:
        for finding in findings:
            self._set_item(job_id, finding.finding_id, **updates)

    def _set_item(self, job_id: str, finding_id: str, **updates: Any) -> None:
        with self._lock:
            item = self._jobs[job_id].items[finding_id]
            for key, value in updates.items():
                setattr(item, key, value)
            self._jobs[job_id].updated_at = _now().isoformat()

    def _fail_job(self, job_id: str, error: Exception) -> None:
        message = str(error).strip() or type(error).__name__
        with self._lock:
            job = self._jobs[job_id]
            job.state = "failed"
            job.detail = message
            job.updated_at = _now().isoformat()
            for item in job.items.values():
                if item.state in {"queued", "running"}:
                    item.state = "failed"
                    item.stage = "重试任务异常结束"
                    item.error = message


def _finding_rule(finding: RiskFinding) -> RiskRule:
    return RiskRule(
        id=finding.risk_type,
        name=finding.title,
        owner=finding.created_by,
        severity=finding.severity,
        query=finding.title,
        required_keywords=(finding.title,),
        claim=finding.claim,
        max_evidence=max(1, len(finding.evidence_ids)),
    )


def _evidence_chunk(evidence: Evidence) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=evidence.evidence_id,
        company_id=evidence.company_id,
        document_id=evidence.document_id,
        source_file=evidence.source_file,
        source_hash=evidence.source_hash,
        page=evidence.page,
        printed_page_label=evidence.printed_page_label,
        bbox=evidence.bbox,
        text=evidence.quote,
        is_ocr=evidence.evidence_type is EvidenceType.OCR_TEXT,
        font_size=0.0,
        available_at=evidence.available_at,
        chunk_kind="table" if evidence.evidence_type is EvidenceType.TABLE else "text",
        table_id=evidence.table_id,
        table_cell_refs=evidence.table_cell_refs,
        table_cell_values=evidence.table_cell_values,
    )


def _analyzed_risk(
    task: SpecialistTask,
    assessment: SpecialistAssessment,
    *,
    attempts: int,
    errors: tuple[str, ...],
) -> AnalyzedRisk:
    chunks = {chunk.chunk_id: chunk for chunk in task.candidates}
    supported = assessment.status is SpecialistStatus.SUPPORTED
    selected = tuple(chunks[item] for item in assessment.evidence_chunk_ids if item in chunks)
    analysis = EvidenceAnalysis(
        supported=supported,
        claim=assessment.claim if supported else "",
        severity=assessment.severity,
        confidence=assessment.confidence,
        evidence_chunk_ids=assessment.evidence_chunk_ids if supported else (),
        uncertainty=assessment.uncertainty,
    )
    return AnalyzedRisk(
        rule=task.rule,
        analysis=analysis,
        candidates=tuple(
            RetrievalCandidate(chunk=chunk, score=1.0, rank=index)
            for index, chunk in enumerate(task.candidates, start=1)
        ),
        selected_chunks=selected,
        retrieval_mode="stored_evidence_reuse",
        analysis_mode="manual_specialist_retry",
        analysis_attempts=attempts,
        search_queries=(task.rule.query,),
        analysis_error=(assessment.uncertainty if assessment.status is SpecialistStatus.TECHNICAL_FAILURE else None),
        analysis_retry_errors=errors,
        specialist_status=assessment.status,
    )


def _updated_finding(finding: RiskFinding, assessment: SpecialistAssessment, verdict: Any) -> RiskFinding:
    payload = finding.model_dump(mode="python")
    payload["specialist_status"] = assessment.status.value
    payload["confidence"] = assessment.confidence
    if assessment.status is not SpecialistStatus.SUPPORTED:
        payload["uncertainty"] = f"专业Agent状态：{assessment.status.value}。{assessment.uncertainty}"
        return RiskFinding.model_validate(payload)
    if verdict is None:
        payload["specialist_status"] = SpecialistStatus.TECHNICAL_FAILURE.value
        payload["uncertainty"] = "决策Agent技术失败；专业核验结果已保留，可再次重试。"
        return RiskFinding.model_validate(payload)
    if verdict.accepted:
        payload.update(
            claim=verdict.claim,
            severity=verdict.severity,
            confidence=verdict.confidence,
            evidence_ids=verdict.evidence_chunk_ids,
            uncertainty=verdict.uncertainty,
            specialist_status=SpecialistStatus.SUPPORTED.value,
        )
    else:
        payload["specialist_status"] = SpecialistStatus.CONFLICT.value
        payload["uncertainty"] = f"决策Agent最终未纳入；证据边界：{verdict.rationale}"
    return RiskFinding.model_validate(payload)


def _retry_trace(
    *,
    run_id: str,
    role: AgentRole,
    finding_ids: tuple[str, ...],
    started_at: datetime,
    ended_at: datetime,
    attempts: int,
    errors: tuple[str, ...],
    model_name: str,
    failed: bool,
) -> AgentTrace:
    suffix = uuid4().hex[:12]
    call = ToolCall(
        tool_call_id=f"TC-{run_id}-retry-{suffix}",
        tool_name="retry_specialist_and_decision",
        started_at=started_at,
        ended_at=ended_at,
        inputs={"finding_ids": list(finding_ids), "reuse_stored_evidence": True},
        output={"model": model_name, "attempts": attempts, "errors": list(errors)},
        status=ToolCallStatus.FAILED if failed else ToolCallStatus.SUCCEEDED,
        error="；".join(errors)[:4000] if failed else None,
    )
    return AgentTrace(
        trace_id=f"T-{run_id}-retry-{suffix}",
        run_id=run_id,
        agent=role,
        started_at=started_at,
        ended_at=ended_at,
        input_refs=finding_ids,
        tool_calls=(call,),
        output_refs=finding_ids,
        status=TraceStatus.FAILED if failed else TraceStatus.SUCCEEDED,
        decision=("技术重试仍失败，保留原风险。" if failed else "复用原证据完成专业核验与裁决重试。"),
        error="；".join(errors)[:4000] if failed else None,
    )


def _role_label(role: AgentRole) -> str:
    return {
        AgentRole.FINANCE: "财务",
        AgentRole.LEGAL: "法律",
        AgentRole.BUSINESS: "业务",
        AgentRole.MARKET: "市场",
        AgentRole.DECISION: "决策",
    }.get(role, role.value)


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["FindingRetryService", "finding_has_retryable_failure"]
