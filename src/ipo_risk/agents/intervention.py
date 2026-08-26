"""分析中途的风险范围人工确认数据契约与线程安全控制器。"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from threading import Condition
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.domain import AgentRole, RiskSeverity


def _now() -> datetime:
    return datetime.now(timezone.utc)


LlmRecoveryAction = Literal["retry_llm", "fallback_bm25"]


def sanitize_llm_error(error: BaseException | str) -> tuple[str, str]:
    """把模型异常转换为可展示原因，同时移除可能出现的密钥。"""
    raw = str(error).strip() or type(error).__name__
    detail = re.sub(r"(?i)bearer\s+[a-z0-9._-]+", "Bearer ***", raw)
    detail = re.sub(r"(?i)(?:sk|ark)-[a-z0-9_-]{8,}", "***", detail)
    detail = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+",
        r"\1***",
        detail,
    )
    detail = detail[:1_000]
    lowered = detail.lower()
    if any(
        token in lowered
        for token in (
            "401",
            "403",
            "unauthorized",
            "api key",
            "apikey",
            "密钥",
            "未配置",
        )
    ):
        reason = "模型认证失败：API Key 缺失、无效或没有该模型权限。"
    elif any(token in lowered for token in ("429", "too many requests", "overload", "rate limit")):
        reason = "模型服务繁忙或触发限流（429），当前请求未完成。"
    elif any(token in lowered for token in ("timeout", "timed out")):
        reason = "模型请求超时，当前请求未完成。"
    elif any(token in lowered for token in ("502", "503", "connection", "peer closed", "reset")):
        reason = "模型服务连接异常，当前请求未完整返回。"
    elif any(token in lowered for token in ("validation error", "invalid json", "json")):
        reason = "模型返回格式不符合结构化结果要求。"
    else:
        reason = "模型调用失败，当前公司已暂停。"
    return reason, detail


class LlmFailureRecoveryRequest(BaseModel):
    """模型失败后交给使用者选择重试或BM25降级的载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    run_id: str = Field(min_length=1)
    stage: str = "risk_discovery"
    agent: str = "discovery"
    provider: str | None = None
    model: str | None = None
    reason: str = Field(min_length=1)
    technical_detail: str = Field(min_length=1, max_length=1_000)
    attempt: int = Field(default=1, ge=1)
    requested_at: datetime = Field(default_factory=_now)


class LlmFailureRecoveryDecision(BaseModel):
    """使用者对一次模型失败的恢复选择。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action: LlmRecoveryAction
    submitted_at: datetime = Field(default_factory=_now)


class LlmFailureRecoveryController:
    """让后台线程在模型故障处等待使用者选择，且页面刷新不丢状态。"""

    def __init__(self) -> None:
        self._condition = Condition()
        self._request: LlmFailureRecoveryRequest | None = None
        self._decision: LlmFailureRecoveryDecision | None = None
        self._wait_seconds = 0.0

    def request_resolution(
        self,
        request: LlmFailureRecoveryRequest,
    ) -> LlmFailureRecoveryDecision:
        with self._condition:
            if self._request is not None:
                raise RuntimeError("another LLM failure recovery is already pending")
            self._request = request
            self._decision = None
            self._condition.notify_all()
            wait_started = perf_counter()
            while self._decision is None:
                self._condition.wait()
            self._wait_seconds += max(0.0, perf_counter() - wait_started)
            decision = self._decision
            self._request = None
            self._decision = None
            return decision

    def submit(self, decision: LlmFailureRecoveryDecision) -> None:
        with self._condition:
            request = self._request
            if request is None:
                raise RuntimeError("no LLM failure recovery is pending")
            if decision.request_id != request.request_id or decision.run_id != request.run_id:
                raise ValueError("LLM recovery decision belongs to another request")
            self._decision = decision
            self._request = None
            self._condition.notify_all()

    def pending_request(self) -> LlmFailureRecoveryRequest | None:
        with self._condition:
            return self._request

    def wait_seconds(self) -> float:
        with self._condition:
            return self._wait_seconds

    def wait_for_request(self, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self._request is not None, timeout=timeout)


class RiskScopeCandidate(BaseModel):
    """等待人工确认的一条AI候选风险。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    owner: AgentRole
    severity: RiskSeverity
    ai_recommended: bool
    rationale: str = Field(min_length=1)
    review_question: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_candidate_count: int = Field(default=1, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RiskScopeFunnel(BaseModel):
    """交给使用者的风险候选漏斗口径。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_batches: int = Field(default=0, ge=0)
    raw_candidates: int = Field(default=0, ge=0)
    consolidated_candidates: int = Field(default=0, ge=0)
    catalog_added: int = Field(default=0, ge=0)
    catalog_merged: int = Field(default=0, ge=0)
    scoped_in: int = Field(default=0, ge=0)
    scoped_out: int = Field(default=0, ge=0)
    consolidation_mode: str = "not_recorded"
    consolidation_error: str | None = None


class RiskScopeReviewRequest(BaseModel):
    """AI完成全文发现和适用性判断后交给人的确认载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    industry: str = "待确认"
    business_model: str = "待确认"
    revenue_model: str = "待确认"
    candidates: tuple[RiskScopeCandidate, ...]
    funnel: RiskScopeFunnel = Field(default_factory=RiskScopeFunnel)
    requested_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def candidates_must_be_unique(self) -> RiskScopeReviewRequest:
        risk_types = [candidate.risk_type for candidate in self.candidates]
        if not risk_types:
            raise ValueError("risk scope review requires candidates")
        if len(risk_types) != len(set(risk_types)):
            raise ValueError("risk scope review contains duplicate risk types")
        return self


class RiskScopeSelection(BaseModel):
    """人工对一条AI候选风险的取舍和调整。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    included: bool
    owner: AgentRole
    severity: RiskSeverity
    analysis_requirement: str = ""


class HumanAddedRisk(BaseModel):
    """人工在AI候选之外补充的一条开放风险。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=2, max_length=120)
    owner: AgentRole
    severity: RiskSeverity = RiskSeverity.MEDIUM
    analysis_requirement: str = Field(min_length=4, max_length=1_000)


class RiskScopeReviewDecision(BaseModel):
    """人工确认后用于恢复专业Agent分析的结构化决定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1, max_length=80)
    selections: tuple[RiskScopeSelection, ...]
    added_risks: tuple[HumanAddedRisk, ...] = ()
    global_requirement: str = Field(default="", max_length=2_000)
    submitted_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def decision_must_be_actionable(self) -> RiskScopeReviewDecision:
        risk_types = [selection.risk_type for selection in self.selections]
        if len(risk_types) != len(set(risk_types)):
            raise ValueError("risk scope decision contains duplicate selections")
        if not any(selection.included for selection in self.selections) and not self.added_risks:
            raise ValueError("risk scope decision must keep or add at least one risk")
        return self


class RiskScopeReviewController:
    """让后台分析线程在风险范围节点等待页面提交，页面重跑不会丢失状态。"""

    def __init__(self) -> None:
        self._condition = Condition()
        self._request: RiskScopeReviewRequest | None = None
        self._decision: RiskScopeReviewDecision | None = None
        self._wait_seconds = 0.0

    def request_review(
        self,
        request: RiskScopeReviewRequest,
    ) -> RiskScopeReviewDecision:
        """发布确认请求并阻塞当前图节点，直到页面提交决定。"""
        with self._condition:
            if self._request is not None:
                raise RuntimeError("another risk scope review is already pending")
            self._request = request
            self._decision = None
            self._condition.notify_all()
            wait_started = perf_counter()
            while self._decision is None:
                self._condition.wait()
            self._wait_seconds += max(0.0, perf_counter() - wait_started)
            decision = self._decision
            self._request = None
            self._decision = None
            return decision

    def submit(self, decision: RiskScopeReviewDecision) -> None:
        """校验并提交页面决定，唤醒等待中的LangGraph节点。"""
        with self._condition:
            request = self._request
            if request is None:
                raise RuntimeError("no risk scope review is pending")
            if decision.run_id != request.run_id:
                raise ValueError("risk scope decision belongs to another run")
            expected = {candidate.risk_type for candidate in request.candidates}
            returned = {selection.risk_type for selection in decision.selections}
            if returned != expected:
                raise ValueError(
                    "risk scope decision must cover every candidate: "
                    f"expected={sorted(expected)}, returned={sorted(returned)}"
                )
            self._decision = decision
            self._request = None
            self._condition.notify_all()

    def pending_request(self) -> RiskScopeReviewRequest | None:
        with self._condition:
            return self._request

    def wait_seconds(self) -> float:
        """返回该控制器累计等待人工提交的秒数。"""
        with self._condition:
            return self._wait_seconds

    def wait_for_request(self, timeout: float | None = None) -> bool:
        """仅供测试和受控协调使用，页面通过快照轮询。"""
        with self._condition:
            return self._condition.wait_for(
                lambda: self._request is not None,
                timeout=timeout,
            )


__all__ = [
    "HumanAddedRisk",
    "LlmFailureRecoveryController",
    "LlmFailureRecoveryDecision",
    "LlmFailureRecoveryRequest",
    "LlmRecoveryAction",
    "RiskScopeCandidate",
    "RiskScopeFunnel",
    "RiskScopeReviewController",
    "RiskScopeReviewDecision",
    "RiskScopeReviewRequest",
    "RiskScopeSelection",
    "sanitize_llm_error",
]
