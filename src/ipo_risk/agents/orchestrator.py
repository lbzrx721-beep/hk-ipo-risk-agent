"""固定状态图：按角色分发证据分析，并形成可追踪的裁决结果。"""

from __future__ import annotations

import operator
import json
from inspect import signature
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from functools import partial
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from ipo_risk.observability.trace import trace_step
from langgraph.types import RetryPolicy as LangGraphRetryPolicy
from pydantic import JsonValue

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import DecisionAnalyzer, DecisionBatch, DecisionVerdict
from ipo_risk.agents.discovery import RiskDiscoverer
from ipo_risk.agents.grounded import AnalyzedRisk, GroundedAnalysisEngine
from ipo_risk.agents.intervention import (
    HumanAddedRisk,
    LlmFailureRecoveryController,
    LlmFailureRecoveryRequest,
    RiskScopeCandidate,
    RiskScopeFunnel,
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
    sanitize_llm_error,
)
from ipo_risk.agents.risk_scoping import RiskScopePlanner
from ipo_risk.agents.review_planning import PROFILE_QUERIES, CompanyReviewPlanner
from ipo_risk.agents.search_planning import RoleSearchPlanner
from ipo_risk.agents.specialist import SpecialistStatus
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
    RiskSeverity,
    ToolCall,
    ToolCallStatus,
    TraceStatus,
)
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryExhaustedError, RetryPolicy, retry_model_node_error


ProgressCallback = Callable[[str, str], None]
_ROLE_NAMES = {
    AgentRole.FINANCE: "财务",
    AgentRole.LEGAL: "法律",
    AgentRole.BUSINESS: "业务",
    AgentRole.MARKET: "市场",
}


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, detail)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AgentProposal:
    """一个角色 Agent 对一条风险规则作出的证据约束提案。"""

    proposal_id: str
    agent: AgentRole
    analyzed_risk: AnalyzedRisk


@dataclass(frozen=True)
class AdjudicatedRisk:
    """决策节点接受的一条风险，以及支撑它的角色提案。"""

    risk_type: str
    selected: AnalyzedRisk
    analysis: EvidenceAnalysis
    source_proposal_ids: tuple[str, ...]
    rationale: str
    decision_mode: str
    decision_attempts: int
    decision_error: str | None = None
    conflict: bool = False
    specialist_status: SpecialistStatus = SpecialistStatus.SUPPORTED


@dataclass(frozen=True)
class AgentConflict:
    """专业角色与裁决节点之间一项显式、可展示的分歧。"""

    risk_type: str
    kind: str
    proposal_ids: tuple[str, ...]
    detail: str
    resolution: str


@dataclass(frozen=True)
class MultiAgentResult:
    """一次多角色分析的提案、裁决和完整父子轨迹。"""

    proposals: tuple[AgentProposal, ...]
    decisions: tuple[AdjudicatedRisk, ...]
    conflicts: tuple[AgentConflict, ...]
    traces: tuple[AgentTrace, ...]
    root_trace: AgentTrace


@dataclass(frozen=True)
class _RoleExecution:
    role: AgentRole
    proposals: tuple[AgentProposal, ...]
    trace: AgentTrace | None


@dataclass(frozen=True)
class _DecisionExecution:
    proposals: tuple[AgentProposal, ...]
    decisions: tuple[AdjudicatedRisk, ...]
    conflicts: tuple[AgentConflict, ...]
    trace: AgentTrace


class _ManualBm25Fallback(RuntimeError):
    """使用者明确选择以BM25目录检索替代本轮风险发现模型。"""

    def __init__(self, reason: str, technical_detail: str, attempt: int) -> None:
        super().__init__(technical_detail)
        self.reason = reason
        self.technical_detail = technical_detail
        self.attempt = attempt


class _ProfessionalGraphState(TypedDict):
    """LangGraph 专业 Agent 扇出/汇合阶段的共享状态。"""

    run_id: str
    root_trace_id: str
    active_rules: tuple[RiskRule, ...]
    candidate_rules: tuple[RiskRule, ...]
    scope_review_request: RiskScopeReviewRequest
    discovery_seed_ids_by_risk: dict[str, tuple[str, ...]]
    progress_callback: ProgressCallback | None
    role_results: Annotated[list[_RoleExecution], operator.add]
    scope_review_decision: NotRequired[RiskScopeReviewDecision]
    decision_execution: NotRequired[_DecisionExecution]


def _tool_call(
    *,
    run_id: str,
    proposal: AgentProposal,
    started_at: datetime,
    ended_at: datetime,
) -> ToolCall:
    analyzed = proposal.analyzed_risk
    return ToolCall(
        tool_call_id=f"TC-{run_id}-{proposal.proposal_id}",
        tool_name="grounded_risk_analysis",
        started_at=started_at,
        ended_at=ended_at,
        inputs={
            "risk_type": analyzed.rule.id,
            "query": analyzed.rule.query,
            "search_queries": list(analyzed.search_queries),
            "required_keywords": list(analyzed.rule.required_keywords),
        },
        output={
            "retrieval_mode": analyzed.retrieval_mode,
            "analysis_mode": analyzed.analysis_mode,
            "specialist_status": analyzed.specialist_status.value,
            "followup_queries": list(analyzed.followup_queries),
            "supported": analyzed.analysis.supported,
            "severity": analyzed.analysis.severity.value,
            "confidence": analyzed.analysis.confidence,
            "candidate_chunk_ids": [candidate.chunk.chunk_id for candidate in analyzed.candidates],
            "selected_chunk_ids": list(analyzed.analysis.evidence_chunk_ids),
            "attempts": analyzed.analysis_attempts,
            "retries": max(0, analyzed.analysis_attempts - 1),
            "repair_errors": list(analyzed.analysis_retry_errors),
            "prompt_version": analyzed.prompt_version,
            "prompt_sha256": analyzed.prompt_sha256,
            "prompt_snapshot": analyzed.prompt_snapshot,
            "retrieval_error": analyzed.retrieval_error,
            "analysis_error": analyzed.analysis_error,
        },
        status=(
            ToolCallStatus.FAILED
            if analyzed.analysis_error is not None
            else ToolCallStatus.SUCCEEDED
        ),
        error=analyzed.analysis_error,
    )


class MultiAgentOrchestrator:
    """将规则分配给角色 Agent，再对有证据的提案进行统一裁决。"""

    _ROLES = (
        AgentRole.FINANCE,
        AgentRole.LEGAL,
        AgentRole.BUSINESS,
        AgentRole.MARKET,
    )

    def __init__(
        self,
        engine: GroundedAnalysisEngine,
        rules: Sequence[RiskRule],
        *,
        decision_analyzer: DecisionAnalyzer | None = None,
        decision_retry_policy: RetryPolicy | None = None,
        search_planner: RoleSearchPlanner | None = None,
        search_retry_policy: RetryPolicy | None = None,
        risk_discoverer: RiskDiscoverer | None = None,
        review_planner: CompanyReviewPlanner | None = None,
        risk_scope_planner: RiskScopePlanner | None = None,
        risk_scope_retry_policy: RetryPolicy | None = None,
        risk_scope_review_controller: RiskScopeReviewController | None = None,
        llm_failure_recovery_controller: LlmFailureRecoveryController | None = None,
        decision_conflict_batch_size: int = 12,
        material_debate_max_items: int = 0,
    ) -> None:
        if decision_conflict_batch_size < 2:
            raise ValueError("decision_conflict_batch_size must be at least 2")
        if material_debate_max_items < 0:
            raise ValueError("material_debate_max_items cannot be negative")
        self._engine = engine
        self._rules = tuple(rules)
        self._decision_analyzer = decision_analyzer
        self._search_planner = search_planner
        self._risk_discoverer = risk_discoverer
        self._review_planner = review_planner
        self._risk_scope_planner = risk_scope_planner
        self._risk_scope_review_controller = risk_scope_review_controller
        self._llm_failure_recovery_controller = llm_failure_recovery_controller
        self._decision_conflict_batch_size = decision_conflict_batch_size
        self._material_debate_max_items = material_debate_max_items
        self._risk_scope_retry_policy = risk_scope_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )
        self._search_retry_policy = search_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )
        self._decision_retry_policy = decision_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )
        self._professional_graph = self._build_professional_graph()

    def _discover_with_recovery(
        self,
        invocation: Callable[[], Any],
        *,
        run_id: str,
        discoverer: RiskDiscoverer,
        progress_callback: ProgressCallback | None,
    ) -> Any:
        """模型彻底失败时等待用户选择；重试只重新执行风险发现调用。"""
        controller = self._llm_failure_recovery_controller
        if controller is None:
            return invocation()
        attempt = 0
        while True:
            attempt += 1
            try:
                result = invocation()
                failed_batches = len(result.batch_errors)
                all_batches_failed = result.batch_count > 0 and failed_batches >= result.batch_count
                if not all_batches_failed:
                    return result
                batch_label = (
                    "风险原文包"
                    if result.discovery_mode == "vector_guided_original_text"
                    else "全文扫描批次"
                )
                last_error = (
                    str(result.batch_errors[-1]) if result.batch_errors else "未返回错误明细"
                )
                failure: BaseException | str = (
                    f"全部{result.batch_count}个{batch_label}失败；最后错误：{last_error}"
                )
            except ModelBudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001 - 转为页面可恢复状态
                failure = exc

            reason, technical_detail = sanitize_llm_error(failure)
            request = LlmFailureRecoveryRequest(
                run_id=run_id,
                provider=getattr(discoverer, "provider_name", None),
                model=getattr(discoverer, "model_name", None),
                reason=reason,
                technical_detail=technical_detail,
                attempt=attempt,
            )
            _emit_progress(
                progress_callback,
                "llm_recovery_requested",
                f"风险发现Agent调用失败，分析已暂停：{reason} 技术详情：{technical_detail}",
            )
            decision = controller.request_resolution(request)
            if decision.action == "retry_llm":
                _emit_progress(
                    progress_callback,
                    "llm_recovery_retry_selected",
                    "已选择重试风险发现LLM；复用现有PDF解析与向量索引。",
                )
                continue
            _emit_progress(
                progress_callback,
                "llm_recovery_bm25_selected",
                "已选择降级使用BM25目录检索；不会再次调用风险发现LLM。",
            )
            raise _ManualBm25Fallback(reason, technical_detail, attempt)

    def _build_professional_graph(self) -> Any:
        """构建四路并行、统一汇合到决策 Agent 的 LangGraph。"""
        builder = StateGraph(_ProfessionalGraphState)
        builder.add_node("human_scope_review", self._scope_review_graph_node)
        builder.add_edge(START, "human_scope_review")
        role_nodes: list[str] = []
        role_retry_policy = LangGraphRetryPolicy(
            initial_interval=self._search_retry_policy.initial_seconds,
            backoff_factor=self._search_retry_policy.backoff_factor,
            max_interval=self._search_retry_policy.max_seconds,
            max_attempts=max(1, self._search_retry_policy.max_retries + 1),
            jitter=False,
            retry_on=retry_model_node_error,
        )
        for role in self._ROLES:
            node_name = f"{role.value}_agent"
            role_nodes.append(node_name)

            def role_node(
                state: _ProfessionalGraphState,
                selected_role: AgentRole = role,
            ) -> dict[str, list[_RoleExecution]]:
                return {
                    "role_results": [
                        self._execute_role(
                            selected_role,
                            active_rules=state["active_rules"],
                            discovery_seed_ids_by_risk=state["discovery_seed_ids_by_risk"],
                            run_id=state["run_id"],
                            root_trace_id=state["root_trace_id"],
                            progress_callback=state["progress_callback"],
                        )
                    ]
                }

            builder.add_node(
                node_name,
                role_node,
                retry_policy=role_retry_policy,
            )
            builder.add_edge("human_scope_review", node_name)

        builder.add_node(
            "decision_agent",
            self._decision_graph_node,
            retry_policy=LangGraphRetryPolicy(
                initial_interval=self._decision_retry_policy.initial_seconds,
                backoff_factor=self._decision_retry_policy.backoff_factor,
                max_interval=self._decision_retry_policy.max_seconds,
                max_attempts=max(
                    1,
                    self._decision_retry_policy.max_retries + 1,
                ),
                jitter=False,
                retry_on=retry_model_node_error,
            ),
        )
        builder.add_edge(role_nodes, "decision_agent")
        builder.add_edge("decision_agent", END)
        return builder.compile()

    def _scope_review_graph_node(
        self,
        state: _ProfessionalGraphState,
    ) -> dict[str, object]:
        """在专业Agent扇出前等待人工确认AI生成的风险范围。"""
        controller = self._risk_scope_review_controller
        if controller is None:
            return {}
        request = state["scope_review_request"]
        progress_callback = state["progress_callback"]
        _emit_progress(
            progress_callback,
            "risk_scope_review_requested",
            (f"已生成{len(request.candidates)}项待查事项，等待人工确认分析范围。"),
        )
        decision = controller.request_review(request)
        active_rules = self._apply_scope_review_decision(
            state["candidate_rules"],
            decision,
        )
        mandatory_ids = (
            {rule.id for rule in self._rules}
            if self._review_planner is not None else {"disclosure_embellishment"}
        )
        active_ids = {rule.id for rule in active_rules}
        active_rules = (
            *active_rules,
            *(
                rule
                for rule in state["candidate_rules"]
                if rule.id in mandatory_ids and rule.id not in active_ids
            ),
        )
        _emit_progress(
            progress_callback,
            "risk_scope_review_completed",
            (
                f"{decision.reviewer}已确认风险范围："
                f"保留及新增{len(active_rules)}条，专业Agent继续运行。"
            ),
        )
        return {
            "active_rules": tuple(active_rules),
            "scope_review_decision": decision,
        }

    @staticmethod
    def _apply_scope_review_decision(
        candidate_rules: Sequence[RiskRule],
        decision: RiskScopeReviewDecision,
    ) -> tuple[RiskRule, ...]:
        rule_by_id = {rule.id: rule for rule in candidate_rules}
        selected_rules: list[RiskRule] = []
        for selection in decision.selections:
            if not selection.included:
                continue
            rule = rule_by_id[selection.risk_type]
            requirements = tuple(
                item
                for item in (
                    selection.analysis_requirement,
                    decision.global_requirement,
                )
                if item
            )
            claim_suffix = "；".join(requirements)
            selected_rules.append(
                rule.model_copy(
                    update={
                        "owner": selection.owner,
                        "severity": selection.severity,
                        "query_variants": tuple(
                            dict.fromkeys((*rule.query_variants, *requirements))
                        ),
                        "claim": (
                            f"{rule.claim} 人工补充分析要求：{claim_suffix}"
                            if claim_suffix
                            else rule.claim
                        ),
                    }
                )
            )
        for index, added in enumerate(decision.added_risks, start=1):
            selected_rules.append(
                MultiAgentOrchestrator._human_added_rule(
                    added,
                    run_id=decision.run_id,
                    index=index,
                    global_requirement=decision.global_requirement,
                )
            )
        return tuple(selected_rules)

    @staticmethod
    def _human_added_rule(
        added: HumanAddedRisk,
        *,
        run_id: str,
        index: int,
        global_requirement: str,
    ) -> RiskRule:
        digest = sha256(
            (f"{run_id}\0{index}\0{added.title}\0{added.analysis_requirement}").encode("utf-8")
        ).hexdigest()[:12]
        requirements = tuple(
            item for item in (added.analysis_requirement, global_requirement) if item
        )
        return RiskRule(
            id=f"open_human_{digest}",
            name=added.title,
            owner=added.owner,
            evidence_fields=("人工补充风险",),
            severity=added.severity,
            query=added.analysis_requirement,
            query_variants=tuple(dict.fromkeys((added.title, *requirements[1:]))),
            required_keywords=(added.title,),
            claim=(f"请根据招股书原文核验“{added.title}”。人工分析要求：{'；'.join(requirements)}"),
            max_evidence=3,
        )

    def _decision_graph_node(
        self,
        state: _ProfessionalGraphState,
    ) -> dict[str, _DecisionExecution]:
        role_order = {role: index for index, role in enumerate(self._ROLES)}
        role_results = sorted(
            state["role_results"],
            key=lambda result: role_order[result.role],
        )
        proposals = tuple(proposal for result in role_results for proposal in result.proposals)
        return {
            "decision_execution": self._execute_decision(
                proposals,
                run_id=state["run_id"],
                root_trace_id=state["root_trace_id"],
                progress_callback=state["progress_callback"],
            )
        }

    @staticmethod
    def _deterministic_decisions(
        proposals: Sequence[AgentProposal],
    ) -> DecisionBatch:
        grouped: dict[str, list[AgentProposal]] = defaultdict(list)
        for proposal in proposals:
            grouped[proposal.analyzed_risk.rule.id].append(proposal)
        verdicts: list[DecisionVerdict] = []
        for risk_type, risk_proposals in grouped.items():
            supported = [
                proposal
                for proposal in risk_proposals
                if proposal.analyzed_risk.analysis.supported
                and proposal.analyzed_risk.analysis.evidence_chunk_ids
            ]
            if not supported:
                continue
            selected = max(
                supported,
                key=lambda proposal: proposal.analyzed_risk.analysis.confidence,
            )
            analysis = selected.analyzed_risk.analysis
            verdicts.append(
                DecisionVerdict(
                    risk_type=risk_type,
                    accepted=True,
                    claim=analysis.claim,
                    severity=analysis.severity,
                    confidence=analysis.confidence,
                    evidence_chunk_ids=analysis.evidence_chunk_ids,
                    uncertainty=analysis.uncertainty,
                    rationale="接受有原文证据的最高置信度角色提案，进入质量复核。",
                    source_proposal_ids=tuple(proposal.proposal_id for proposal in supported),
                )
            )
        return DecisionBatch(decisions=tuple(verdicts))

    def _add_material_risk_debates(
        self,
        decision_batch: DecisionBatch,
        proposals: Sequence[AgentProposal],
    ) -> tuple[DecisionBatch, set[str]]:
        """为少量重大风险增加反证检查，不制造无业务意义的全量辩论。"""
        requested = {
            verdict.risk_type for verdict in decision_batch.decisions if verdict.needs_debate
        }
        remaining = max(0, self._material_debate_max_items - len(requested))
        if remaining == 0:
            return decision_batch, set()

        proposals_by_risk: dict[str, list[AgentProposal]] = defaultdict(list)
        for proposal in proposals:
            proposals_by_risk[proposal.analyzed_risk.rule.id].append(proposal)
        severity_rank = {RiskSeverity.CRITICAL: 0, RiskSeverity.HIGH: 1}
        candidates = sorted(
            (
                verdict
                for verdict in decision_batch.decisions
                if verdict.risk_type not in requested
                and verdict.accepted
                and verdict.severity in severity_rank
                and verdict.risk_type in proposals_by_risk
            ),
            key=lambda verdict: (severity_rank[verdict.severity], verdict.confidence),
        )[:remaining]
        if not candidates:
            return decision_batch, set()

        injected: dict[str, DecisionVerdict] = {}
        for verdict in candidates:
            original = proposals_by_risk[verdict.risk_type][0]
            risk_name = original.analyzed_risk.rule.name
            injected[verdict.risk_type] = verdict.model_copy(
                update={
                    "needs_debate": True,
                    "challenge": (
                        "该重大风险是否存在招股书已披露的终止、缓释、期后改善或"
                        "统计口径限制，足以改变当前风险结论或严重程度？"
                    ),
                    "challenge_role": original.agent,
                    "challenge_queries": (
                        f"{risk_name} 终止 缓释 期后变化",
                        f"{risk_name} 最新状态 统计口径 风险因素",
                    ),
                }
            )
        return (
            DecisionBatch(
                decisions=tuple(
                    injected.get(verdict.risk_type, verdict)
                    for verdict in decision_batch.decisions
                )
            ),
            set(injected),
        )

    @staticmethod
    def _requires_conflict_escalation(
        proposals: Sequence[AgentProposal],
    ) -> bool:
        """只把真实分歧或复杂重大项交给冲突模型。"""
        supported = [
            proposal
            for proposal in proposals
            if proposal.analyzed_risk.analysis.supported
            and proposal.analyzed_risk.analysis.evidence_chunk_ids
        ]
        if not supported:
            return False
        if any(
            proposal.analyzed_risk.specialist_status.value == "conflict" for proposal in proposals
        ):
            return True
        if len(supported) > 1:
            severities = {proposal.analyzed_risk.analysis.severity for proposal in supported}
            claims = {
                "".join(proposal.analyzed_risk.analysis.claim.lower().split())
                for proposal in supported
            }
            evidence_sets = {
                tuple(sorted(proposal.analyzed_risk.analysis.evidence_chunk_ids))
                for proposal in supported
            }
            if len(severities) > 1 or len(claims) > 1 or len(evidence_sets) > 1:
                return True
        return any(
            proposal.analyzed_risk.analysis.severity.value == "critical" for proposal in supported
        )

    def _split_integration_work(
        self,
        proposals: Sequence[AgentProposal],
    ) -> tuple[tuple[AgentProposal, ...], tuple[tuple[AgentProposal, ...], ...]]:
        """返回可直接整合的提案，以及按风险完整分组的冲突升级批次。"""
        grouped: dict[str, list[AgentProposal]] = defaultdict(list)
        for proposal in proposals:
            grouped[proposal.analyzed_risk.rule.id].append(proposal)
        direct: list[AgentProposal] = []
        escalation_groups: list[list[AgentProposal]] = []
        for risk_type in sorted(grouped):
            group = grouped[risk_type]
            if self._requires_conflict_escalation(group):
                escalation_groups.append(group)
            else:
                direct.extend(group)
        batches: list[tuple[AgentProposal, ...]] = []
        current: list[AgentProposal] = []
        current_risk_count = 0
        for group in escalation_groups:
            if current and current_risk_count >= self._decision_conflict_batch_size:
                batches.append(tuple(current))
                current = []
                current_risk_count = 0
            current.extend(group)
            current_risk_count += 1
        if current:
            batches.append(tuple(current))
        return tuple(direct), tuple(batches)

    @staticmethod
    def _validate_verdict(
        verdict: DecisionVerdict,
        proposal_by_id: dict[str, AgentProposal],
    ) -> None:
        unknown_proposals = set(verdict.source_proposal_ids) - set(proposal_by_id)
        if unknown_proposals:
            raise ValueError(f"decision cited unknown proposals: {sorted(unknown_proposals)}")
        cited = [proposal_by_id[item] for item in verdict.source_proposal_ids]
        if any(proposal.analyzed_risk.rule.id != verdict.risk_type for proposal in cited):
            raise ValueError("decision cited a proposal for another risk_type")
        allowed_evidence = {
            chunk_id
            for proposal in cited
            for chunk_id in proposal.analyzed_risk.analysis.evidence_chunk_ids
        }
        unknown_evidence = set(verdict.evidence_chunk_ids) - allowed_evidence
        if unknown_evidence:
            raise ValueError(f"decision cited unknown evidence chunks: {sorted(unknown_evidence)}")

    def _scope_active_rules(
        self,
        rules: Sequence[RiskRule],
        *,
        run_id: str,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[RiskRule], ToolCall | None]:
        """在动态发现之后核验候选风险与当前公司业务是否匹配。"""
        selected_rules = tuple(rules)
        risk_scope_planner = self._risk_scope_planner
        if risk_scope_planner is None or not selected_rules:
            return list(selected_rules), None

        scope_started = _now()
        scope_error: str | None = None
        scope_attempts = 1
        _emit_progress(
            progress_callback,
            "risk_scoping_started",
            "正在识别公司画像并核验动态风险与当前业务是否匹配。",
        )
        try:
            plan_with_feedback = getattr(
                risk_scope_planner,
                "plan_with_feedback",
                None,
            )
            if callable(plan_with_feedback):
                scope_outcome = self._risk_scope_retry_policy.run_with_feedback(
                    lambda correction_feedback: plan_with_feedback(
                        selected_rules,
                        self._engine.chunks,
                        correction_feedback=correction_feedback,
                    )
                )
            else:
                scope_outcome = self._risk_scope_retry_policy.run(
                    lambda: risk_scope_planner.plan(
                        selected_rules,
                        self._engine.chunks,
                    )
                )
            scope = scope_outcome.value
            scope_attempts = scope_outcome.attempts
            expected = {rule.id for rule in selected_rules}
            returned = {risk.risk_type for risk in scope.risks}
            if returned != expected:
                raise ValueError(
                    "risk scope must cover every candidate risk type: "
                    f"expected={sorted(expected)}, returned={sorted(returned)}"
                )
            # 画像节点只提供范围建议。为优先保证赛题召回率，在人工确认前不
            # 自动删除任何动态发现或目录兜底候选。
            active_rules = list(selected_rules)
            prompt_metadata: dict[str, str] = {}
            prompt_audit_getter = getattr(
                risk_scope_planner,
                "prompt_audit",
                None,
            )
            if callable(prompt_audit_getter):
                prompt_metadata = prompt_audit_getter()
            scope_output: JsonValue = {
                "mode": risk_scope_planner.provider_name,
                "model": risk_scope_planner.model_name,
                "prompt_version": risk_scope_planner.prompt_version,
                "prompt_sha256": prompt_metadata.get("prompt_sha256"),
                "prompt_snapshot": prompt_metadata.get("prompt_snapshot"),
                "attempts": scope_attempts,
                "retries": max(0, scope_attempts - 1),
                "repair_errors": list(scope_outcome.errors),
                "industry": scope.industry,
                "business_model": scope.business_model,
                "revenue_model": scope.revenue_model,
                "applicable_risk_types": list(scope.applicable_risk_types),
                "excluded_risk_types": list(scope.excluded_risk_types),
                "auto_removed_risk_types": [],
                "decisions": [
                    {
                        "risk_type": risk.risk_type,
                        "applicable": risk.applicable,
                        "rationale": risk.rationale,
                    }
                    for risk in scope.risks
                ],
            }
            scope_status = ToolCallStatus.SUCCEEDED
            _emit_progress(
                progress_callback,
                "risk_scoping_completed",
                (
                    f"公司画像：{scope.industry}；AI建议纳入"
                    f"{len(scope.applicable_risk_types)}条、存疑或建议排除"
                    f"{len(scope.excluded_risk_types)}条；人工确认前保留全部"
                    f"{len(active_rules)}条候选。"
                ),
            )
        except RetryExhaustedError as error:
            if isinstance(error.last_error, ModelBudgetExceededError):
                raise error.last_error
            scope_attempts = error.attempts
            scope_error = str(error.last_error).strip() or type(error.last_error).__name__
            active_rules = list(selected_rules)
            scope_output = {
                "mode": "candidate_rules_fallback",
                "model": risk_scope_planner.model_name,
                "prompt_version": risk_scope_planner.prompt_version,
                "attempts": scope_attempts,
                "retries": max(0, scope_attempts - 1),
                "repair_errors": [scope_error],
                "applicable_risk_types": [rule.id for rule in selected_rules],
                "excluded_risk_types": [],
                "auto_removed_risk_types": [],
            }
            scope_status = ToolCallStatus.FAILED
            _emit_progress(
                progress_callback,
                "risk_scoping_fallback",
                f"业务适用性核验失败，保留全部动态候选：{scope_error}",
            )
        scope_ended = _now()
        call = ToolCall(
            tool_call_id=f"TC-{run_id}-company-risk-scoping",
            tool_name="company_risk_scoping",
            started_at=scope_started,
            ended_at=scope_ended,
            inputs={
                "risk_types": [rule.id for rule in selected_rules],
                "document_chunk_count": len(self._engine.chunks),
                "execution_order": "after_document_first_discovery",
            },
            output=scope_output,
            status=scope_status,
            error=scope_error,
        )
        return active_rules, call

    @staticmethod
    def _build_scope_review_request(
        *,
        run_id: str,
        candidate_rules: Sequence[RiskRule],
        active_rules: Sequence[RiskRule],
        scope_call: ToolCall | None,
        funnel: RiskScopeFunnel,
        evidence_counts_by_risk: dict[str, int],
        source_candidate_counts_by_risk: dict[str, int],
        confidences_by_risk: dict[str, float],
    ) -> RiskScopeReviewRequest:
        output = (
            scope_call.output
            if scope_call is not None and isinstance(scope_call.output, dict)
            else {}
        )
        rationale_by_risk: dict[str, str] = {}
        recommendation_by_risk: dict[str, bool] = {}
        raw_decisions = output.get("decisions")
        if isinstance(raw_decisions, list):
            for item in raw_decisions:
                if not isinstance(item, dict):
                    continue
                risk_type = item.get("risk_type")
                rationale = item.get("rationale")
                if isinstance(risk_type, str) and isinstance(rationale, str):
                    rationale_by_risk[risk_type] = rationale
                applicable = item.get("applicable")
                if isinstance(risk_type, str) and isinstance(applicable, bool):
                    recommendation_by_risk[risk_type] = applicable
        active_ids = {rule.id for rule in active_rules}
        candidates: list[RiskScopeCandidate] = []
        is_review_plan = funnel.consolidation_mode == "company_profile_review_plan"
        for rule in candidate_rules:
            is_document_discovery = rule.id.startswith("open_")
            if is_review_plan:
                ai_recommended = True
                rationale = (
                    "根据公司画像提出的专项问题，尚不代表风险成立；专业Agent将独立检索全文。"
                    if rule.id.startswith("check_")
                    else "基础必查项，始终保留；由专业Agent检索支持证据与反证后判断。"
                )
            elif is_document_discovery:
                ai_recommended = recommendation_by_risk.get(
                    rule.id,
                    rule.id in active_ids,
                )
                rationale = rationale_by_risk.get(
                    rule.id,
                    (
                        "自动适用性判断未启用或已降级，默认保留候选。"
                        if rule.id in active_ids
                        else "AI认为该风险与当前公司业务不匹配。"
                    ),
                )
            else:
                # Catalog rules are fallback directions, not document-grounded
                # discoveries. They must not look like AI findings before a
                # specialist Agent has retrieved and verified source evidence.
                ai_recommended = False
                rationale = (
                    "开放式风险发现未形成可验证候选；该项仅为目录兜底方向，"
                    "需人工选择后由专业 Agent 检索原文核验。"
                )
            candidates.append(
                RiskScopeCandidate(
                    risk_type=rule.id,
                    title=rule.name,
                    owner=rule.owner,
                    severity=rule.severity,
                    ai_recommended=ai_recommended,
                    rationale=rationale,
                    review_question=rule.claim,
                    source=(
                        ("profile_review_question" if rule.id.startswith("check_")
                         else "mandatory_review") if is_review_plan
                        else ("document_discovery" if is_document_discovery else "catalog_fallback")
                    ),
                    source_candidate_count=source_candidate_counts_by_risk.get(
                        rule.id,
                        1 if is_document_discovery else 0,
                    ),
                    evidence_count=evidence_counts_by_risk.get(rule.id, 0),
                    confidence=(
                        confidences_by_risk.get(rule.id) if is_document_discovery else None
                    ),
                )
            )
        recommended_count = sum(candidate.ai_recommended for candidate in candidates)
        normalized_funnel = funnel.model_copy(
            update={
                "scoped_in": recommended_count,
                "scoped_out": len(candidates) - recommended_count,
            }
        )
        return RiskScopeReviewRequest(
            run_id=run_id,
            industry=str(output.get("industry") or "待确认"),
            business_model=str(output.get("business_model") or "待确认"),
            revenue_model=str(output.get("revenue_model") or "待确认"),
            candidates=tuple(candidates),
            funnel=normalized_funnel,
        )

    def _execute_role(
        self,
        role: AgentRole,
        *,
        active_rules: Sequence[RiskRule],
        discovery_seed_ids_by_risk: dict[str, tuple[str, ...]],
        run_id: str,
        root_trace_id: str,
        progress_callback: ProgressCallback | None,
    ) -> _RoleExecution:
        """执行一条专业 Agent 分支；四个角色由 LangGraph 并行调用。"""
        role_rules = [rule for rule in active_rules if rule.owner is role]
        if not role_rules:
            _emit_progress(
                progress_callback,
                f"agent:{role.value}:skipped",
                f"{_ROLE_NAMES[role]} Agent本次未分配风险，无需调用模型。",
            )
            return _RoleExecution(role=role, proposals=(), trace=None)

        role_started = _now()
        _emit_progress(
            progress_callback,
            f"agent:{role.value}:started",
            f"{_ROLE_NAMES[role]} Agent开始分析{len(role_rules)}类风险。",
        )
        _emit_progress(
            progress_callback,
            f"agent:{role.value}:risk_catalog",
            json.dumps(
                {rule.id: rule.name for rule in role_rules},
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        role_tool_calls: list[ToolCall] = []
        # 动态发现阶段已经为 open_* 风险生成支持/反证/期间检索问题，专业Agent直接
        # 使用这些问题，避免四个角色再各调用一次模型重复规划。
        planned_queries: dict[str, tuple[str, ...]] = {
            rule.id: rule.search_queries for rule in role_rules if rule.id.startswith("open_")
        }
        rules_requiring_planning = [rule for rule in role_rules if rule.id not in planned_queries]
        search_planner = self._search_planner
        if search_planner is not None and rules_requiring_planning:
            planning_started = _now()
            search_attempts = 1
            search_error: str | None = None
            search_repair_errors: list[str] = []
            search_mode = "fixed_query_fallback"
            try:
                plan_with_feedback = getattr(
                    search_planner,
                    "plan_with_feedback",
                    None,
                )
                if callable(plan_with_feedback):
                    search_outcome = self._search_retry_policy.run_with_feedback(
                        lambda correction_feedback: plan_with_feedback(
                            rules_requiring_planning,
                            correction_feedback=correction_feedback,
                        )
                    )
                else:
                    search_outcome = self._search_retry_policy.run(
                        lambda: search_planner.plan(rules_requiring_planning)
                    )
                search_plan = search_outcome.value
                search_attempts = search_outcome.attempts
                search_repair_errors = list(search_outcome.errors)
                expected_risk_types = {rule.id for rule in rules_requiring_planning}
                if search_plan.role is not role:
                    raise ValueError("search plan role does not match assigned Agent")
                if set(search_plan.queries_by_risk()) != expected_risk_types:
                    raise ValueError("search plan must cover every risk assigned to the Agent")
                planned_queries.update(search_plan.queries_by_risk())
                search_mode = search_planner.provider_name
                _emit_progress(
                    progress_callback,
                    f"agent:{role.value}:search_planned",
                    (
                        f"{_ROLE_NAMES[role]} Agent已生成"
                        f"{sum(len(items) for items in planned_queries.values())}个"
                        "语义搜索问题。"
                    ),
                )
            except RetryExhaustedError as error:
                if isinstance(error.last_error, ModelBudgetExceededError):
                    raise error.last_error
                search_attempts = error.attempts
                search_error = str(error.last_error)
                search_repair_errors = [search_error] if search_error else []
                _emit_progress(
                    progress_callback,
                    f"agent:{role.value}:search_fallback",
                    f"搜索规划失败，启用固定查询兜底：{search_error}",
                )
            planning_ended = _now()
            prompt_metadata: dict[str, str] = {}
            prompt_audit_getter = getattr(search_planner, "prompt_audit", None)
            if callable(prompt_audit_getter):
                prompt_metadata = prompt_audit_getter(role)
            model_name_for = getattr(search_planner, "model_name_for", None)
            planned_model_name = (
                model_name_for(role)
                if planned_queries and callable(model_name_for)
                else search_planner.model_name if planned_queries else None
            )
            role_tool_calls.append(
                ToolCall(
                    tool_call_id=f"TC-{run_id}-{role.value}-search-planning",
                    tool_name="agent_search_planning",
                    started_at=planning_started,
                    ended_at=planning_ended,
                    inputs={
                        "role": role.value,
                        "risk_types": [rule.id for rule in rules_requiring_planning],
                    },
                    output={
                        "mode": search_mode,
                        "model": planned_model_name,
                        "prompt_version": (
                            search_planner.prompt_version if planned_queries else None
                        ),
                        "prompt_sha256": (
                            prompt_metadata.get("prompt_sha256") if planned_queries else None
                        ),
                        "prompt_snapshot": (
                            prompt_metadata.get("prompt_snapshot") if planned_queries else None
                        ),
                        "queries_by_risk": {
                            risk_type: list(queries)
                            for risk_type, queries in planned_queries.items()
                        },
                        "attempts": search_attempts,
                        "retries": max(0, search_attempts - 1),
                        "repair_errors": cast(JsonValue, search_repair_errors),
                        "fallback_error": search_error,
                    },
                    status=ToolCallStatus.SUCCEEDED,
                )
            )

        _emit_progress(
            progress_callback,
            f"agent:{role.value}:prompt",
            (
                f"{_ROLE_NAMES[role]} Agent已接收全部{len(role_rules)}条风险，"
                "开始角色内批量核验：" + "、".join(rule.id for rule in role_rules)
            ),
        )
        if any(rule.id == "disclosure_embellishment" for rule in role_rules):
            _emit_progress(
                progress_callback,
                "text_embellishment_started",
                "披露语言风险Agent正在对照业务概览、风险因素与财务披露，检查宣传性表述、量化支撑和期间一致性。",
            )
        analyzed_risks = self._engine.analyze_rules_batched(
            role_rules,
            search_queries_by_risk=planned_queries,
            seed_chunk_ids_by_risk=discovery_seed_ids_by_risk,
        )
        for index, analyzed in enumerate(analyzed_risks, start=1):
            rule = analyzed.rule
            evidence_count = len(analyzed.analysis.evidence_chunk_ids)
            conclusion = (
                f"形成有证据结论，引用{evidence_count}个原文片段"
                if analyzed.analysis.supported and evidence_count
                else f"标记为{analyzed.specialist_status.value}"
            )
            _emit_progress(
                progress_callback,
                f"agent:{role.value}:progress",
                (
                    f"[{index}/{len(role_rules)}] {rule.name}（{rule.id}）："
                    f"召回{len(analyzed.candidates)}个候选片段，{conclusion}。"
                ),
            )

        embellishment_result = next(
            (
                analyzed
                for analyzed in analyzed_risks
                if analyzed.rule.id == "disclosure_embellishment"
            ),
            None,
        )
        if embellishment_result is not None:
            level = (
                embellishment_result.analysis.severity.value
                if embellishment_result.analysis.supported
                else "low"
            )
            _emit_progress(
                progress_callback,
                "text_embellishment_completed",
                (
                    f"文本粉饰分析完成：等级{level}；"
                    f"引用{len(embellishment_result.analysis.evidence_chunk_ids)}个原文片段。"
                ),
            )

        role_ended = _now()
        role_proposals = tuple(
            AgentProposal(
                proposal_id=f"P-{role.value}-{analyzed.rule.id}",
                agent=role,
                analyzed_risk=analyzed,
            )
            for analyzed in analyzed_risks
        )
        supported_count = sum(
            proposal.analyzed_risk.analysis.supported
            and bool(proposal.analyzed_risk.analysis.evidence_chunk_ids)
            for proposal in role_proposals
        )
        _emit_progress(
            progress_callback,
            f"agent:{role.value}:completed",
            (
                f"{_ROLE_NAMES[role]} Agent完成{len(role_proposals)}类风险分析，"
                f"形成{supported_count}条有证据提案。"
            ),
        )
        trace = AgentTrace(
            trace_id=f"T-{run_id}-{role.value}",
            run_id=run_id,
            parent_trace_id=root_trace_id,
            agent=role,
            started_at=role_started,
            ended_at=role_ended,
            input_refs=tuple(rule.id for rule in role_rules),
            tool_calls=tuple(role_tool_calls)
            + tuple(
                _tool_call(
                    run_id=run_id,
                    proposal=proposal,
                    started_at=role_started,
                    ended_at=role_ended,
                )
                for proposal in role_proposals
            ),
            output_refs=tuple(proposal.proposal_id for proposal in role_proposals),
            status=TraceStatus.SUCCEEDED,
            decision=(
                f"{role.value} Agent完成{len(role_proposals)}条规则分析，"
                "所有结论均保留候选与选中证据ID。"
            ),
        )
        return _RoleExecution(
            role=role,
            proposals=role_proposals,
            trace=trace,
        )

    def _execute_decision(
        self,
        proposals: Sequence[AgentProposal],
        *,
        run_id: str,
        root_trace_id: str,
        progress_callback: ProgressCallback | None,
    ) -> _DecisionExecution:
        """汇合四路专业结果，并由决策模型逐项复核后交给人工。"""
        ordered_proposals = tuple(proposals)
        decision_started = _now()
        direct_proposals, escalation_batches = self._split_integration_work(ordered_proposals)
        eligible_proposals = tuple(
            proposal
            for proposal in ordered_proposals
            if proposal.analyzed_risk.analysis.supported
            and proposal.analyzed_risk.analysis.evidence_chunk_ids
        )
        unresolved_proposals = tuple(
            proposal for proposal in ordered_proposals if proposal not in eligible_proposals
        )
        _emit_progress(
            progress_callback,
            "agent:decision:started",
            (
                f"决策Agent接收{len(ordered_proposals)}条专业提案："
                f"{len(eligible_proposals)}条进入模型裁决，"
                f"{len(unresolved_proposals)}条证据不足项由机器直接不纳入。"
            ),
        )
        _emit_progress(
            progress_callback,
            "agent:decision:risk_catalog",
            json.dumps(
                {
                    proposal.analyzed_risk.rule.id: proposal.analyzed_risk.rule.name
                    for proposal in ordered_proposals
                },
                ensure_ascii=False,
            ),
        )
        decision_error: str | None = None
        decision_attempts = 0
        decision_mode = "deterministic_integration"
        analyzer = self._decision_analyzer
        repair_errors: list[str] = []
        verdicts: list[DecisionVerdict] = []
        proposals_by_risk: dict[str, list[AgentProposal]] = defaultdict(list)
        for proposal in eligible_proposals:
            proposals_by_risk[proposal.analyzed_risk.rule.id].append(proposal)
        packed_batches: list[tuple[AgentProposal, ...]] = []
        current_batch: list[AgentProposal] = []
        for risk_group in proposals_by_risk.values():
            if (
                current_batch
                and len(current_batch) + len(risk_group) > self._decision_conflict_batch_size
            ):
                packed_batches.append(tuple(current_batch))
                current_batch = []
            current_batch.extend(risk_group)
        if current_batch:
            packed_batches.append(tuple(current_batch))
        model_batches = tuple(packed_batches)
        output_batch_count = len(model_batches) + (1 if unresolved_proposals else 0)
        model_failed = False
        if analyzer is None:
            verdicts.extend(self._deterministic_decisions(eligible_proposals).decisions)
        for batch_index, decision_batch_proposals in enumerate(model_batches, start=1):
            if analyzer is None:
                break
            try:
                adjudicate_with_feedback = getattr(
                    analyzer,
                    "adjudicate_with_feedback",
                    None,
                )
                if callable(adjudicate_with_feedback):
                    feedback_adjudicator = cast(Callable[..., Any], adjudicate_with_feedback)
                    stream_marker = f"[[batch:{batch_index}/{max(1, output_batch_count)}]]"
                    try:
                        supports_stream_marker = (
                            "stream_marker" in signature(adjudicate_with_feedback).parameters
                        )
                    except (TypeError, ValueError):
                        supports_stream_marker = False
                    outcome = self._decision_retry_policy.run_with_feedback(
                        lambda correction_feedback: feedback_adjudicator(
                            decision_batch_proposals,
                            correction_feedback=correction_feedback,
                            **({"stream_marker": stream_marker} if supports_stream_marker else {}),
                        )
                    )
                else:
                    outcome = self._decision_retry_policy.run(
                        lambda: analyzer.adjudicate(decision_batch_proposals)
                    )
                verdicts.extend(outcome.value.decisions)
                decision_attempts += outcome.attempts
                repair_errors.extend(outcome.errors)
                decision_mode = f"{analyzer.provider_name}_full_adjudication"
                _emit_progress(
                    progress_callback,
                    "agent:decision:structured",
                    (
                        f"[[batch:{batch_index}/{max(1, output_batch_count)}]]"
                        + json.dumps(
                            outcome.value.model_dump(mode="json"),
                            ensure_ascii=False,
                        )
                    ),
                )
                _emit_progress(
                    progress_callback,
                    "agent:decision:progress",
                    (
                        f"决策Agent已完成第{batch_index}/{len(model_batches)}批，"
                        f"累计复核{len(verdicts)}条风险。"
                    ),
                )
            except RetryExhaustedError as error:
                if isinstance(error.last_error, ModelBudgetExceededError):
                    raise error.last_error
                decision_attempts += error.attempts
                decision_error = str(error.last_error)
                repair_errors.append(decision_error)
                # 模型失败时不删除风险；由确定性规则选择最高置信度提案完成机器裁决。
                verdicts.extend(self._deterministic_decisions(decision_batch_proposals).decisions)
                model_failed = True
        decision_batch = DecisionBatch(decisions=tuple(verdicts))
        if model_failed:
            decision_mode = "model_adjudication_with_deterministic_fallback"
        material_review_risk_types: set[str] = set()
        if analyzer is not None and not model_failed:
            decision_batch, material_review_risk_types = self._add_material_risk_debates(
                decision_batch,
                eligible_proposals,
            )

        # 第一轮只对Decision明确提出的实质争议进行定向补证。补证从同一向量索引
        # 重新检索，并由Decision指定的专业角色回应；同时对少量重大风险执行反证检查。
        debate_tool_calls: list[ToolCall] = []
        debate_records: list[dict[str, JsonValue]] = []
        debate_proposals: list[AgentProposal] = []
        if analyzer is not None and not model_failed:
            revised_verdicts: dict[str, DecisionVerdict] = {}
            first_round_by_risk = {
                verdict.risk_type: verdict for verdict in decision_batch.decisions
            }
            for debate_index, first_verdict in enumerate(
                (verdict for verdict in decision_batch.decisions if verdict.needs_debate),
                start=1,
            ):
                original = tuple(
                    proposal
                    for proposal in eligible_proposals
                    if proposal.analyzed_risk.rule.id == first_verdict.risk_type
                )
                if not original:
                    continue
                debate_started = _now()
                challenge_role = first_verdict.challenge_role or original[0].agent
                challenge_queries = tuple(first_verdict.challenge_queries)
                debate_trigger = (
                    "material_risk_review"
                    if first_verdict.risk_type in material_review_risk_types
                    else "decision_challenge"
                )
                _emit_progress(
                    progress_callback,
                    "agent:decision:debate_requested",
                    (
                        (
                            "决策Agent对重大风险发起对抗性复核："
                            if debate_trigger == "material_risk_review"
                            else "决策Agent提出实质质疑："
                        )
                        + f"《{original[0].analyzed_risk.rule.name}》"
                        f"{first_verdict.challenge}；已交由{_ROLE_NAMES[challenge_role]}Agent查证。"
                    ),
                )
                _emit_progress(
                    progress_callback,
                    f"agent:{challenge_role.value}:debate_retrieval",
                    (
                        f"{_ROLE_NAMES[challenge_role]}Agent正在使用{len(challenge_queries)}个"
                        "质疑问题重新检索招股书向量库。"
                    ),
                )
                debate_error: str | None = None
                response_proposal: AgentProposal | None = None
                final_verdict = first_verdict
                try:
                    challenge_rule = original[0].analyzed_risk.rule.model_copy(
                        update={"owner": challenge_role}
                    )
                    response = self._engine.analyze_rules_batched(
                        (challenge_rule,),
                        search_queries_by_risk={
                            challenge_rule.id: challenge_queries,
                        },
                    )[0]
                    response_proposal = AgentProposal(
                        proposal_id=(
                            f"P-debate-{debate_index}-{challenge_role.value}-"
                            f"{challenge_rule.id}"
                        ),
                        agent=challenge_role,
                        analyzed_risk=response,
                    )
                    debate_proposals.append(response_proposal)
                    second_round_inputs = (*original, response_proposal)
                    adjudicate_with_feedback = getattr(
                        analyzer,
                        "adjudicate_with_feedback",
                        None,
                    )
                    final_round_instruction = (
                        "补证辩论后的最终轮：不得再次发起辩论。"
                        f"第一轮质疑：{first_verdict.challenge}。"
                        f"回应角色：{challenge_role.value}；"
                        f"回应状态：{response.specialist_status.value}。"
                    )
                    if callable(adjudicate_with_feedback):
                        try:
                            supports_stream_marker = (
                                "stream_marker"
                                in signature(adjudicate_with_feedback).parameters
                            )
                        except (TypeError, ValueError):
                            supports_stream_marker = False
                        debate_outcome = self._decision_retry_policy.run_with_feedback(
                            lambda correction: adjudicate_with_feedback(
                                second_round_inputs,
                                correction_feedback=(
                                    final_round_instruction
                                    + (f" 输出校验反馈：{correction}" if correction else "")
                                ),
                                **(
                                    {"stream_marker": f"[[debate:{debate_index}]]"}
                                    if supports_stream_marker
                                    else {}
                                ),
                            )
                        )
                    else:
                        debate_outcome = self._decision_retry_policy.run(
                            lambda: analyzer.adjudicate(second_round_inputs)
                        )
                    if len(debate_outcome.value.decisions) != 1:
                        raise ValueError("debate final round must return exactly one decision")
                    raw_final = debate_outcome.value.decisions[0]
                    final_verdict = DecisionVerdict.model_validate(
                        {
                            **raw_final.model_dump(mode="python"),
                            "needs_debate": False,
                            "challenge": "",
                            "challenge_role": None,
                            "challenge_queries": (),
                        }
                    )
                    proposal_lookup = {
                        proposal.proposal_id: proposal for proposal in second_round_inputs
                    }
                    self._validate_verdict(final_verdict, proposal_lookup)
                    revised_verdicts[first_verdict.risk_type] = final_verdict
                    decision_attempts += debate_outcome.attempts
                    repair_errors.extend(debate_outcome.errors)
                    _emit_progress(
                        progress_callback,
                        f"agent:{challenge_role.value}:debate_response",
                        (
                            f"{_ROLE_NAMES[challenge_role]}Agent已提交补证回应："
                            f"召回{len(response.candidates)}个候选片段，"
                            f"结论状态为{response.specialist_status.value}。"
                        ),
                    )
                    _emit_progress(
                        progress_callback,
                        "agent:decision:debate_completed",
                        (
                            f"决策Agent已完成《{challenge_rule.name}》第二轮裁决："
                            f"{'纳入正式风险' if final_verdict.accepted else '不纳入正式风险'}。"
                        ),
                    )
                except ModelBudgetExceededError:
                    raise
                except Exception as error:  # noqa: BLE001 - 单项辩论不得拖断整家公司
                    debate_error = str(error).strip() or type(error).__name__
                    _emit_progress(
                        progress_callback,
                        f"agent:{challenge_role.value}:debate_failed",
                        (
                            f"{_ROLE_NAMES[challenge_role]}Agent本轮补证未完成，"
                            "已结束补证并保留首轮专业结论。"
                        ),
                    )
                    _emit_progress(
                        progress_callback,
                        "agent:decision:debate_failed",
                        f"补证辩论未完成，保留第一轮裁决：{debate_error}",
                    )
                debate_ended = _now()
                record: dict[str, JsonValue] = {
                    "debate_id": f"{debate_index}-{first_verdict.risk_type}",
                    "risk_type": first_verdict.risk_type,
                    "risk_name": original[0].analyzed_risk.rule.name,
                    "trigger": debate_trigger,
                    "challenge": first_verdict.challenge,
                    "challenge_role": challenge_role.value,
                    "challenge_queries": list(challenge_queries),
                    "response_proposal_id": (
                        response_proposal.proposal_id if response_proposal is not None else None
                    ),
                    "first_round_accepted": first_round_by_risk[
                        first_verdict.risk_type
                    ].accepted,
                    "final_accepted": final_verdict.accepted,
                    "response_status": (
                        response_proposal.analyzed_risk.specialist_status.value
                        if response_proposal is not None
                        else "failed"
                    ),
                    "recalled_evidence_count": (
                        len(response_proposal.analyzed_risk.candidates)
                        if response_proposal is not None
                        else 0
                    ),
                    "new_evidence_count": (
                        len(
                            {
                                candidate.chunk.chunk_id
                                for candidate in response_proposal.analyzed_risk.candidates
                            }
                            - {
                                candidate.chunk.chunk_id
                                for proposal in original
                                for candidate in proposal.analyzed_risk.candidates
                            }
                        )
                        if response_proposal is not None
                        else 0
                    ),
                    "final_rationale": final_verdict.rationale,
                    "status": "failed" if debate_error else "completed",
                    "error": debate_error,
                }
                debate_records.append(record)
                _emit_progress(
                    progress_callback,
                    "debate_record",
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                )
                debate_tool_calls.append(
                    ToolCall(
                        tool_call_id=(
                            f"TC-{run_id}-risk-debate-{debate_index}-"
                            f"{first_verdict.risk_type}"
                        ),
                        tool_name="risk_debate_verification",
                        started_at=debate_started,
                        ended_at=debate_ended,
                        inputs={
                            "risk_type": first_verdict.risk_type,
                            "challenge": first_verdict.challenge,
                            "challenge_role": challenge_role.value,
                            "challenge_queries": list(challenge_queries),
                        },
                        output=record,
                        status=(
                            ToolCallStatus.FAILED
                            if debate_error
                            else ToolCallStatus.SUCCEEDED
                        ),
                        error=debate_error,
                    )
                )
            if revised_verdicts:
                decision_batch = DecisionBatch(
                    decisions=tuple(
                        revised_verdicts.get(verdict.risk_type, verdict)
                        for verdict in decision_batch.decisions
                    )
                )
                decision_mode = f"{decision_mode}_with_targeted_debate"

        ordered_proposals = (*ordered_proposals, *debate_proposals)
        decision_repair_errors = tuple(repair_errors)

        proposal_by_id = {proposal.proposal_id: proposal for proposal in ordered_proposals}
        decisions: list[AdjudicatedRisk] = []
        conflicts: list[AgentConflict] = []
        for verdict in decision_batch.decisions:
            self._validate_verdict(verdict, proposal_by_id)
            sources = [proposal_by_id[item] for item in verdict.source_proposal_ids]
            selected = max(
                sources,
                key=lambda proposal: proposal.analyzed_risk.analysis.confidence,
            )
            if not verdict.accepted:
                if any(proposal.analyzed_risk.analysis.supported for proposal in sources):
                    conflicts.append(
                        AgentConflict(
                            risk_type=verdict.risk_type,
                            kind="unresolved_support_conflict",
                            proposal_ids=verdict.source_proposal_ids,
                            detail="专业Agent认为风险成立，决策Agent最终裁定不纳入正式风险。",
                            resolution=f"机器最终不纳入：{verdict.rationale}",
                        )
                    )
                decisions.append(
                    AdjudicatedRisk(
                        risk_type=verdict.risk_type,
                        selected=selected.analyzed_risk,
                        analysis=EvidenceAnalysis(
                            supported=False,
                            claim="",
                            severity=verdict.severity,
                            confidence=verdict.confidence,
                            evidence_chunk_ids=(),
                            uncertainty=verdict.uncertainty,
                        ),
                        source_proposal_ids=verdict.source_proposal_ids,
                        rationale=verdict.rationale,
                        decision_mode=decision_mode,
                        decision_attempts=decision_attempts,
                        decision_error=decision_error,
                        conflict=True,
                        specialist_status=selected.analyzed_risk.specialist_status,
                    )
                )
                continue
            severity_disagreement = any(
                proposal.analyzed_risk.analysis.severity is not verdict.severity
                for proposal in sources
            )
            if severity_disagreement:
                source_levels = ", ".join(
                    sorted({proposal.analyzed_risk.analysis.severity.value for proposal in sources})
                )
                conflicts.append(
                    AgentConflict(
                        risk_type=verdict.risk_type,
                        kind="severity_disagreement",
                        proposal_ids=verdict.source_proposal_ids,
                        detail=(
                            f"专业Agent严重程度为{source_levels}，"
                            f"决策Agent裁定为{verdict.severity.value}。"
                        ),
                        resolution=verdict.rationale,
                    )
                )
            final_analysis = EvidenceAnalysis(
                supported=True,
                claim=verdict.claim,
                severity=verdict.severity,
                confidence=verdict.confidence,
                evidence_chunk_ids=verdict.evidence_chunk_ids,
                uncertainty=verdict.uncertainty,
            )
            decisions.append(
                AdjudicatedRisk(
                    risk_type=verdict.risk_type,
                    selected=selected.analyzed_risk,
                    analysis=final_analysis,
                    source_proposal_ids=verdict.source_proposal_ids,
                    rationale=verdict.rationale,
                    decision_mode=decision_mode,
                    decision_attempts=decision_attempts,
                    decision_error=decision_error,
                    conflict=severity_disagreement,
                    specialist_status=selected.analyzed_risk.specialist_status,
                )
            )

        decided_risk_types = {decision.risk_type for decision in decisions}
        unresolved_cards: list[dict[str, Any]] = []
        for proposal in unresolved_proposals:
            analyzed = proposal.analyzed_risk
            risk_type = analyzed.rule.id
            if risk_type in decided_risk_types:
                continue
            status = analyzed.specialist_status
            rationale = (
                "专业Agent未获得足以支持正式结论的证据；"
                "机器最终不纳入正式风险清单，原始提案保留在审计记录中。"
            )
            decisions.append(
                AdjudicatedRisk(
                    risk_type=risk_type,
                    selected=analyzed,
                    analysis=EvidenceAnalysis(
                        supported=False,
                        claim="",
                        severity=analyzed.analysis.severity,
                        confidence=analyzed.analysis.confidence,
                        evidence_chunk_ids=(),
                        uncertainty=(
                            analyzed.analysis.uncertainty
                            or "现有检索证据不足以形成可验证风险结论。"
                        ),
                    ),
                    source_proposal_ids=(proposal.proposal_id,),
                    rationale=rationale,
                    decision_mode="excluded_insufficient_evidence",
                    decision_attempts=0,
                    specialist_status=status,
                )
            )
            decided_risk_types.add(risk_type)
            unresolved_cards.append(
                {
                    "risk_type": risk_type,
                    "title": analyzed.rule.name,
                    "accepted": False,
                    "status": status.value,
                    "claim": analyzed.rule.claim,
                    "severity": analyzed.analysis.severity.value,
                    "confidence": analyzed.analysis.confidence,
                    "evidence_chunk_ids": [],
                    "source_proposal_ids": [proposal.proposal_id],
                    "rationale": rationale,
                    "uncertainty": analyzed.analysis.uncertainty,
                }
            )
        if unresolved_cards:
            _emit_progress(
                progress_callback,
                "agent:decision:structured",
                (
                    f"[[batch:{len(model_batches) + 1}/{output_batch_count}]]"
                    + json.dumps(
                        {"decisions": unresolved_cards},
                        ensure_ascii=False,
                    )
                ),
            )

        decision_ended = _now()
        prompt_metadata: dict[str, str] = {}
        if analyzer is not None:
            prompt_audit_getter = getattr(analyzer, "prompt_audit", None)
            if callable(prompt_audit_getter):
                prompt_metadata = prompt_audit_getter()
        used_decision_model = analyzer is not None and decision_attempts > 0
        decision_call = ToolCall(
            tool_call_id=f"TC-{run_id}-decision-adjudication",
            tool_name="risk_adjudication",
            started_at=decision_started,
            ended_at=decision_ended,
            inputs={
                "proposal_ids": [proposal.proposal_id for proposal in ordered_proposals],
            },
            output={
                "mode": decision_mode,
                "model": (
                    analyzer.model_name if analyzer is not None and used_decision_model else None
                ),
                "prompt_version": (
                    analyzer.prompt_version
                    if analyzer is not None and used_decision_model
                    else None
                ),
                "prompt_sha256": (
                    prompt_metadata.get("prompt_sha256") if used_decision_model else None
                ),
                "prompt_snapshot": (
                    prompt_metadata.get("prompt_snapshot") if used_decision_model else None
                ),
                "integrated_risk_types": [decision.risk_type for decision in decisions],
                "proposal_count": len(ordered_proposals),
                "model_adjudicated_count": len(eligible_proposals),
                "excluded_insufficient_evidence_count": len(unresolved_proposals),
                "direct_proposal_count": len(direct_proposals),
                "escalation_batch_count": len(escalation_batches),
                "decision_batch_count": len(model_batches),
                "proposal_ledger": [
                    {
                        "proposal_id": proposal.proposal_id,
                        "risk_type": proposal.analyzed_risk.rule.id,
                        "specialist_status": (proposal.analyzed_risk.specialist_status.value),
                        "evidence_chunk_ids": list(
                            proposal.analyzed_risk.analysis.evidence_chunk_ids
                        ),
                    }
                    for proposal in ordered_proposals
                ],
                "conflict_count": len(conflicts),
                "debate_count": len(debate_records),
                "debates": cast(JsonValue, debate_records),
                "conflicts": [
                    {
                        "risk_type": conflict.risk_type,
                        "kind": conflict.kind,
                        "proposal_ids": list(conflict.proposal_ids),
                        "detail": conflict.detail,
                        "resolution": conflict.resolution,
                    }
                    for conflict in conflicts
                ],
                "attempts": decision_attempts,
                "retries": max(0, decision_attempts - 1),
                "repair_errors": list(decision_repair_errors),
                "fallback_error": decision_error,
            },
            status=ToolCallStatus.SUCCEEDED,
        )
        decision_trace = AgentTrace(
            trace_id=f"T-{run_id}-decision",
            run_id=run_id,
            parent_trace_id=root_trace_id,
            agent=AgentRole.DECISION,
            started_at=decision_started,
            ended_at=decision_ended,
            input_refs=tuple(proposal.proposal_id for proposal in ordered_proposals),
            tool_calls=(decision_call, *debate_tool_calls),
            output_refs=tuple(decision.risk_type for decision in decisions),
            status=TraceStatus.SUCCEEDED,
            decision=(
                f"决策Agent整合{len(decisions)}条风险提案；"
                f"其中{len(eligible_proposals)}条由模型裁决，"
                f"{len(unresolved_proposals)}条证据不足项未纳入正式风险；"
                f"{len(debate_records)}条争议风险完成定向质疑、补证与再裁决；"
                "原始提案与裁决理由完整保留在审计记录中。"
            ),
        )
        _emit_progress(
            progress_callback,
            "agent:decision:completed",
            (
                f"决策Agent裁决完成：机器采纳"
                f"{sum(item.analysis.supported for item in decisions)}条风险，"
                f"不纳入{sum(not item.analysis.supported for item in decisions)}条；"
                f"完成{len(debate_records)}项补证辩论，记录{len(conflicts)}项裁决分歧。"
            ),
        )
        return _DecisionExecution(
            proposals=ordered_proposals,
            decisions=tuple(decisions),
            conflicts=tuple(conflicts),
            trace=decision_trace,
        )

    @trace_step("ipo.multi_agent", inputs=("run_id", "source_ref"))
    def run(
        self,
        *,
        run_id: str,
        source_ref: str,
        progress_callback: ProgressCallback | None = None,
    ) -> MultiAgentResult:
        """运行角色节点和确定性首版裁决，所有节点共享同一检索索引。"""
        root_started = _now()
        root_trace_id = f"T-{run_id}-orchestrator"
        proposals: list[AgentProposal] = []
        child_traces: list[AgentTrace] = []
        root_tool_calls: list[ToolCall] = []
        active_rules = list(self._rules)
        discovery_seed_ids_by_risk: dict[str, tuple[str, ...]] = {}
        discovery_source_counts_by_risk: dict[str, int] = {}
        discovery_confidences_by_risk: dict[str, float] = {}
        discovery_funnel = RiskScopeFunnel(catalog_added=len(active_rules))

        discoverer = self._risk_discoverer
        if self._review_planner is not None:
            planner = self._review_planner
            active_rules = [r.model_copy(update={"requires_model_verification": True})
                            for r in self._rules]
            discovery_started = _now()
            _emit_progress(progress_callback, "agent:discovery:started",
                           "正在检索公司概况，形成有原文引用的画像与审查问题。")
            plan_error = None
            try:
                review_plan = planner.plan(self._engine, self._rules)
                active_rules = list(review_plan.rules)
                plan_output = review_plan.model_dump(mode="json", exclude={"rules"})
                plan_output["base_checks"] = [
                    {"id": r.id, "title": r.name, "owner": r.owner.value,
                     "question": r.claim} for r in self._rules
                ]
                plan_output["status"] = "degraded" if review_plan.warnings else "completed"
                summary = (f"已形成{len(review_plan.profile.facts)}条画像事实、"
                           f"{len(review_plan.checks)}项专项问题；保留{len(self._rules)}项基础检查。"
                           "这些是待查事项，专业Agent将独立检索全文核验。")
                if review_plan.warnings:
                    summary += "部分画像信息未获取，基础检查继续。"
            except ModelBudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep mandatory coverage
                plan_error = type(exc).__name__
                plan_output = {"status": "degraded", "profile": {"facts": []},
                               "checks": [], "unknown_fields": list(PROFILE_QUERIES),
                               "base_checks": [{"id": r.id, "title": r.name,
                                                "owner": r.owner.value, "question": r.claim}
                                               for r in self._rules],
                               "warnings": ["画像规划失败，保留基础检查，不生成公司事实。"]}
                summary = "画像规划未完成；保留全部基础检查，专业Agent继续查证。"
            plan_output.update({"model": planner.model_name, "mode": planner.provider_name,
                                "prompt_version": planner.prompt_version})
            plan_output["review_tasks"] = [
                {"id": r.id, "owner": r.owner.value, "question": r.claim,
                 "queries": list(r.search_queries)} for r in active_rules
            ]
            _emit_progress(progress_callback, "company_review_plan",
                           json.dumps(plan_output, ensure_ascii=False))
            _emit_progress(progress_callback,
                           "agent:discovery:degraded" if plan_output["status"] == "degraded"
                           else "agent:discovery:completed",
                           summary)
            discovery_ended = _now()
            child_traces.append(AgentTrace(
                trace_id=f"T-{run_id}-discovery", run_id=run_id,
                parent_trace_id=root_trace_id, agent=AgentRole.DISCOVERY,
                started_at=discovery_started, ended_at=discovery_ended, input_refs=(source_ref,),
                tool_calls=(ToolCall(
                    tool_call_id=f"TC-{run_id}-company-review-plan",
                    tool_name="company_profile_review_planning",
                    started_at=discovery_started, ended_at=discovery_ended,
                    inputs={"chunk_count": len(self._engine.chunks),
                            "workflow": "profile_retrieval_then_review_questions"},
                    output={**plan_output, **planner.prompt_metadata()},
                    status=ToolCallStatus.FAILED if plan_error else ToolCallStatus.SUCCEEDED,
                    error=plan_error,
                ),),
                output_refs=tuple(r.id for r in active_rules),
                status=TraceStatus.FAILED if plan_error else TraceStatus.SUCCEEDED,
                decision=summary, error=plan_error,
            ))
            discovery_funnel = RiskScopeFunnel(
                catalog_added=len(self._rules), scoped_in=len(active_rules),
                consolidation_mode="company_profile_review_plan",
            )
        elif discoverer is not None:
            discovery_started = _now()
            guided_discover_value = getattr(
                discoverer,
                "discover_with_retrieval",
                None,
            )
            guided_discover = (
                cast(Callable[..., Any], guided_discover_value)
                if callable(guided_discover_value)
                else None
            )
            vector_guided = guided_discover is not None
            _emit_progress(
                progress_callback,
                "agent:discovery:started",
                (
                    (
                        "风险发现Agent将先用语义索引定位原文，再阅读命中段落、"
                        "相邻上下文与章节覆盖样本。"
                    )
                    if vector_guided
                    else (f"风险发现Agent开始长上下文阅读{len(self._engine.chunks)}个全文切片。")
                ),
            )
            discovery_error: str | None = None
            tool_output: JsonValue | None
            try:
                if guided_discover is not None:
                    invocation = partial(
                        guided_discover,
                        self._engine.chunks,
                        self._rules,
                        corpus_selector=self._engine.select_discovery_corpus,
                    )
                else:
                    invocation = partial(
                        discoverer.discover,
                        self._engine.chunks,
                        self._rules,
                    )
                discovery_result = self._discover_with_recovery(
                    invocation,
                    run_id=run_id,
                    discoverer=discoverer,
                    progress_callback=progress_callback,
                )
                discovered = tuple(discovery_result.risks)
                discovery_seed_ids_by_risk = {
                    risk.risk_type: risk.evidence_chunk_ids for risk in discovered
                }
                discovery_source_counts_by_risk = {
                    risk.risk_type: max(1, len(risk.source_candidate_ids)) for risk in discovered
                }
                discovery_confidences_by_risk = {
                    risk.risk_type: risk.confidence for risk in discovered
                }
                output_refs = tuple(risk.risk_type for risk in discovered)
                failed_batches = len(discovery_result.batch_errors)
                all_batches_failed = (
                    discovery_result.batch_count > 0
                    and failed_batches >= discovery_result.batch_count
                )
                if all_batches_failed:
                    active_rules = list(self._rules)
                    fallback_catalog_used = True
                    analysis_strategy = "fixed_catalog_fallback"
                    catalog_added_count = len(active_rules)
                    catalog_merged_count = 0
                    batch_label = (
                        "风险原文包"
                        if discovery_result.discovery_mode == "vector_guided_original_text"
                        else "全文扫描批次"
                    )
                    discovery_error = f"全部{discovery_result.batch_count}个{batch_label}失败"
                    discovery_status = TraceStatus.FAILED
                    tool_status = ToolCallStatus.FAILED
                    discovery_decision = "开放式风险发现全部批次失败，固定风险分析继续运行。"
                    _emit_progress(
                        progress_callback,
                        "agent:discovery:failed",
                        f"风险发现Agent已降级：{discovery_error}。",
                    )
                else:
                    dynamic_rules = [risk.to_rule() for risk in discovered]
                    matched_catalog_types = {
                        risk.related_catalog_risk_type
                        for risk in discovered
                        if risk.related_catalog_risk_type is not None
                    }
                    catalog_merged_count = len(matched_catalog_types)
                    fallback_catalog_used = bool(failed_batches or not dynamic_rules)
                    if dynamic_rules and not failed_batches:
                        active_rules = dynamic_rules
                        analysis_strategy = (
                            "vector_guided_dynamic"
                            if discovery_result.discovery_mode == "vector_guided_original_text"
                            else "document_first_dynamic"
                        )
                        catalog_added_count = 0
                    elif dynamic_rules:
                        catalog_additions = [
                            rule for rule in self._rules if rule.id not in matched_catalog_types
                        ]
                        active_rules = [*dynamic_rules, *catalog_additions]
                        analysis_strategy = "dynamic_with_catalog_fallback"
                        catalog_added_count = len(catalog_additions)
                    else:
                        active_rules = list(self._rules)
                        analysis_strategy = "fixed_catalog_fallback"
                        catalog_added_count = len(active_rules)
                    discovery_status = TraceStatus.SUCCEEDED
                    tool_status = ToolCallStatus.SUCCEEDED
                    partial_note = f"；另有{failed_batches}个批次重试耗尽" if failed_batches else ""
                    source_summary = (
                        (
                            f"从{discovery_result.source_chunk_count}个全文切片中检索"
                            f"{discovery_result.selected_chunk_count}个原文切片"
                        )
                        if discovery_result.discovery_mode == "vector_guided_original_text"
                        else "全文扫描"
                    )
                    discovery_decision = (
                        f"{source_summary}形成"
                        f"{discovery_result.raw_candidate_count}条原始候选，"
                        f"全局合并为{len(discovered)}条动态风险，"
                        f"已分派给专业Agent重新检索和核验{partial_note}。"
                    )
                    _emit_progress(
                        progress_callback,
                        (
                            "agent:discovery:degraded"
                            if failed_batches
                            else "agent:discovery:completed"
                        ),
                        discovery_decision,
                    )
                raw_candidate_count = discovery_result.raw_candidate_count or len(discovered)
                consolidated_candidate_count = discovery_result.consolidated_candidate_count or len(
                    discovered
                )
                discovery_funnel = RiskScopeFunnel(
                    document_batches=discovery_result.batch_count,
                    raw_candidates=raw_candidate_count,
                    consolidated_candidates=consolidated_candidate_count,
                    catalog_added=catalog_added_count,
                    catalog_merged=catalog_merged_count,
                    scoped_in=len(active_rules),
                    consolidation_mode=discovery_result.consolidation_mode,
                    consolidation_error=discovery_result.consolidation_error,
                )
                discovery_prompt_metadata: dict[str, str] = {}
                discovery_audit_getter = getattr(
                    discoverer,
                    "discovery_prompt_audit",
                    None,
                )
                if callable(discovery_audit_getter):
                    discovery_prompt_metadata = discovery_audit_getter()
                consolidation_prompt_metadata: dict[str, str] = {}
                consolidation_audit_getter = getattr(
                    discoverer,
                    "consolidation_prompt_audit",
                    None,
                )
                if callable(consolidation_audit_getter):
                    consolidation_prompt_metadata = consolidation_audit_getter()
                retrieval_prompt_metadata: dict[str, str] = {}
                retrieval_audit_getter = getattr(
                    discoverer,
                    "retrieval_prompt_audit",
                    None,
                )
                if callable(retrieval_audit_getter):
                    retrieval_prompt_metadata = retrieval_audit_getter()
                tool_output = {
                    "mode": discoverer.provider_name,
                    "model": discoverer.model_name,
                    "prompt_version": discoverer.prompt_version,
                    "prompt_sha256": discovery_prompt_metadata.get("prompt_sha256"),
                    "prompt_snapshot": discovery_prompt_metadata.get("prompt_snapshot"),
                    "consolidation_prompt_version": (
                        consolidation_prompt_metadata.get("prompt_version")
                    ),
                    "consolidation_prompt_sha256": (
                        consolidation_prompt_metadata.get("prompt_sha256")
                    ),
                    "consolidation_prompt_snapshot": (
                        consolidation_prompt_metadata.get("prompt_snapshot")
                    ),
                    "retrieval_prompt_version": (retrieval_prompt_metadata.get("prompt_version")),
                    "retrieval_prompt_sha256": (retrieval_prompt_metadata.get("prompt_sha256")),
                    "retrieval_prompt_snapshot": (retrieval_prompt_metadata.get("prompt_snapshot")),
                    "discovery_mode": discovery_result.discovery_mode,
                    "source_chunk_count": discovery_result.source_chunk_count,
                    "selected_chunk_count": discovery_result.selected_chunk_count,
                    "retrieval_query_count": discovery_result.retrieval_query_count,
                    "analysis_strategy": analysis_strategy,
                    "fallback_catalog_used": fallback_catalog_used,
                    "batch_count": discovery_result.batch_count,
                    "model_attempts": discovery_result.model_attempts,
                    "batch_errors": list(discovery_result.batch_errors),
                    "batch_warnings": list(discovery_result.batch_warnings),
                    "raw_candidate_count": raw_candidate_count,
                    "consolidated_candidate_count": (consolidated_candidate_count),
                    "consolidation_mode": (discovery_result.consolidation_mode),
                    "consolidation_attempts": (discovery_result.consolidation_attempts),
                    "consolidation_error": (discovery_result.consolidation_error),
                    "catalog_merged_count": catalog_merged_count,
                    "catalog_added_count": catalog_added_count,
                    "raw_candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in discovery_result.raw_candidates
                    ],
                    "consolidation_mappings": [
                        mapping.model_dump(mode="json")
                        for mapping in discovery_result.consolidation_mappings
                    ],
                    "discovered_risks": [
                        {
                            "risk_type": risk.risk_type,
                            "title": risk.title,
                            "owner": risk.owner.value,
                            "confidence": risk.confidence,
                            "seed_evidence_chunk_ids": list(risk.evidence_chunk_ids),
                            "source_candidate_ids": list(risk.source_candidate_ids),
                            "related_catalog_risk_type": (risk.related_catalog_risk_type),
                        }
                        for risk in discovered
                    ],
                }
            except ModelBudgetExceededError:
                raise
            except _ManualBm25Fallback as exc:
                discovery_error = exc.technical_detail
                discovered = ()
                active_rules = list(self._rules)
                self._engine.enable_bm25_fallback(
                    f"使用者在风险发现模型失败后选择BM25：{exc.reason}"
                )
                discovery_funnel = RiskScopeFunnel(
                    catalog_added=len(active_rules),
                    scoped_in=len(active_rules),
                    consolidation_mode="manual_bm25_fallback",
                    consolidation_error=discovery_error,
                )
                discovery_status = TraceStatus.FAILED
                tool_status = ToolCallStatus.FAILED
                tool_output = {
                    "mode": "manual_bm25_fallback",
                    "analysis_strategy": "fixed_catalog_bm25_fallback",
                    "fallback_catalog_used": True,
                    "recovery_action": "fallback_bm25",
                    "recovery_attempt": exc.attempt,
                    "failure_reason": exc.reason,
                }
                output_refs = ()
                discovery_decision = "风险发现LLM失败；使用者明确选择BM25目录检索继续分析。"
            except Exception as exc:  # noqa: BLE001
                discovery_error = str(exc).strip() or type(exc).__name__
                discovered = ()
                active_rules = list(self._rules)
                discovery_funnel = RiskScopeFunnel(
                    catalog_added=len(active_rules),
                    scoped_in=len(active_rules),
                    consolidation_mode="failed_before_consolidation",
                    consolidation_error=discovery_error,
                )
                discovery_status = TraceStatus.FAILED
                tool_status = ToolCallStatus.FAILED
                tool_output = {
                    "mode": "fixed_catalog_fallback",
                    "analysis_strategy": "fixed_catalog_fallback",
                    "fallback_catalog_used": True,
                }
                output_refs = ()
                discovery_decision = "开放式风险发现失败，固定风险分析继续运行。"
                _emit_progress(
                    progress_callback,
                    "agent:discovery:failed",
                    f"风险发现Agent失败，继续固定风险分析：{discovery_error}",
                )
            discovery_ended = _now()
            discovery_call = ToolCall(
                tool_call_id=f"TC-{run_id}-open-risk-discovery",
                tool_name="open_risk_discovery",
                started_at=discovery_started,
                ended_at=discovery_ended,
                inputs={
                    "chunk_count": len(self._engine.chunks),
                    "known_risk_types": [],
                    "catalog_matching_types": [rule.id for rule in self._rules],
                    "topic_source": (
                        "vector_retrieved_original_text_with_section_coverage"
                        if vector_guided
                        else "prospectus_full_text"
                    ),
                },
                output=tool_output,
                status=tool_status,
                error=discovery_error,
            )
            child_traces.append(
                AgentTrace(
                    trace_id=f"T-{run_id}-discovery",
                    run_id=run_id,
                    parent_trace_id=root_trace_id,
                    agent=AgentRole.DISCOVERY,
                    started_at=discovery_started,
                    ended_at=discovery_ended,
                    input_refs=(source_ref,),
                    tool_calls=(discovery_call,),
                    output_refs=output_refs,
                    status=discovery_status,
                    decision=discovery_decision,
                    error=discovery_error,
                )
            )

        # 文本粉饰是任务书要求的显式结果项，即使开放式发现没有单独命名，也必须
        # 进入专业核验；它只分析当前招股书，不使用上市后信息或历史标签。
        mandatory_risk_ids = {"disclosure_embellishment"}
        active_ids = {rule.id for rule in active_rules}
        mandatory_additions = [
            rule
            for rule in self._rules
            if rule.id in mandatory_risk_ids and rule.id not in active_ids
        ]
        if mandatory_additions:
            active_rules.extend(mandatory_additions)
            _emit_progress(
                progress_callback,
                "text_embellishment_registered",
                "已加入独立的招股书文本粉饰与选择性披露分析模块。",
            )

        candidate_rules = tuple(active_rules)
        # New planning already extracts a grounded profile. Do not run the legacy
        # broad-document profile stage again or turn questions into risk candidates.
        scope_call: ToolCall | None
        if self._review_planner is not None:
            profile_values: dict[str, list[str]] = defaultdict(list)
            for fact in plan_output.get("profile", {}).get("facts", []):
                profile_values[fact["field"]].append(fact["value"])
            scope_call = ToolCall(
                tool_call_id=f"TC-{run_id}-review-plan-scope", tool_name="review_plan_scope",
                started_at=discovery_ended, ended_at=discovery_ended, inputs={},
                output={"industry": "；".join(profile_values["industry"]) or "未知",
                        "business_model": "；".join(profile_values["business"]) or "未知",
                        "revenue_model": "；".join(profile_values["revenue_model"]) or "未知"},
                status=ToolCallStatus.SUCCEEDED,
            )
        else:
            active_rules, scope_call = self._scope_active_rules(
                active_rules,
                run_id=run_id,
                progress_callback=progress_callback,
            )
        discovery_funnel = discovery_funnel.model_copy(
            update={
                "scoped_in": len(active_rules),
                "scoped_out": max(0, len(candidate_rules) - len(active_rules)),
            }
        )
        if scope_call is not None:
            root_tool_calls.append(scope_call)
        scope_review_request = self._build_scope_review_request(
            run_id=run_id,
            candidate_rules=candidate_rules,
            active_rules=active_rules,
            scope_call=scope_call,
            funnel=discovery_funnel,
            evidence_counts_by_risk={
                risk_type: len(chunk_ids)
                for risk_type, chunk_ids in discovery_seed_ids_by_risk.items()
            },
            source_candidate_counts_by_risk=(discovery_source_counts_by_risk),
            confidences_by_risk=discovery_confidences_by_risk,
        )

        graph_started = _now()
        _emit_progress(
            progress_callback,
            "agents_parallel_started",
            (
                "四个专业分支已同时调度："
                + "、".join(
                    f"{_ROLE_NAMES[role]}{sum(rule.owner is role for rule in active_rules)}条"
                    for role in self._ROLES
                )
                + "；无任务角色将直接标记为跳过。"
            ),
        )
        graph_state = self._professional_graph.invoke(
            {
                "run_id": run_id,
                "root_trace_id": root_trace_id,
                "active_rules": tuple(active_rules),
                "candidate_rules": candidate_rules,
                "scope_review_request": scope_review_request,
                "discovery_seed_ids_by_risk": discovery_seed_ids_by_risk,
                "progress_callback": progress_callback,
                "role_results": [],
            },
            config={
                "configurable": {"thread_id": run_id},
                "max_concurrency": len(self._ROLES),
            },
        )
        graph_ended = _now()
        active_rules = list(graph_state["active_rules"])
        decision_execution = graph_state["decision_execution"]
        proposals = list(decision_execution.proposals)
        decisions = list(decision_execution.decisions)
        conflicts = list(decision_execution.conflicts)
        decision_trace = decision_execution.trace
        role_order = {role: index for index, role in enumerate(self._ROLES)}
        role_results = sorted(
            graph_state["role_results"],
            key=lambda result: role_order[result.role],
        )
        child_traces.extend(result.trace for result in role_results if result.trace is not None)
        scope_review_decision = graph_state.get("scope_review_decision")
        if isinstance(scope_review_decision, RiskScopeReviewDecision):
            root_tool_calls.append(
                ToolCall(
                    tool_call_id=f"TC-{run_id}-human-risk-scope-review",
                    tool_name="human_risk_scope_review",
                    started_at=scope_review_request.requested_at,
                    ended_at=scope_review_decision.submitted_at,
                    inputs={
                        "candidate_risks": [
                            {
                                "risk_type": candidate.risk_type,
                                "ai_recommended": candidate.ai_recommended,
                                "owner": candidate.owner.value,
                                "severity": candidate.severity.value,
                            }
                            for candidate in scope_review_request.candidates
                        ],
                        "industry": scope_review_request.industry,
                        "business_model": scope_review_request.business_model,
                        "revenue_model": scope_review_request.revenue_model,
                        "risk_funnel": scope_review_request.funnel.model_dump(mode="json"),
                    },
                    output={
                        "reviewer": scope_review_decision.reviewer,
                        "selections": [
                            selection.model_dump(mode="json")
                            for selection in scope_review_decision.selections
                        ],
                        "added_risks": [
                            risk.model_dump(mode="json")
                            for risk in scope_review_decision.added_risks
                        ],
                        "global_requirement": (scope_review_decision.global_requirement),
                        "final_risk_types": [rule.id for rule in active_rules],
                    },
                    status=ToolCallStatus.SUCCEEDED,
                )
            )
        root_tool_calls.append(
            ToolCall(
                tool_call_id=f"TC-{run_id}-langgraph-orchestration",
                tool_name="langgraph_state_graph",
                started_at=graph_started,
                ended_at=graph_ended,
                inputs={
                    "parallel_nodes": [f"{role.value}_agent" for role in self._ROLES],
                    "join_node": "decision_agent",
                    "active_risk_types": [rule.id for rule in active_rules],
                },
                output={
                    "execution_model": "parallel_fan_out_fan_in",
                    "assigned_risks_by_role": {
                        role.value: sum(rule.owner is role for rule in active_rules)
                        for role in self._ROLES
                    },
                    "completed_roles": [
                        result.role.value for result in role_results if result.trace is not None
                    ],
                    "proposal_count": len(proposals),
                    "accepted_count": len(decisions),
                },
                status=ToolCallStatus.SUCCEEDED,
            )
        )

        root_ended = _now()
        root_trace = AgentTrace(
            trace_id=root_trace_id,
            run_id=run_id,
            agent=AgentRole.ORCHESTRATOR,
            started_at=root_started,
            ended_at=root_ended,
            input_refs=(source_ref,),
            tool_calls=tuple(root_tool_calls),
            output_refs=tuple(decision.risk_type for decision in decisions),
            status=TraceStatus.SUCCEEDED,
            decision=(
                f"分发{len(proposals)}条角色提案并完成决策节点裁决，接受{len(decisions)}条。"
            ),
        )
        traces = (root_trace, *child_traces, decision_trace)
        return MultiAgentResult(
            proposals=tuple(proposals),
            decisions=tuple(decisions),
            conflicts=tuple(conflicts),
            traces=traces,
            root_trace=root_trace,
        )


__all__ = [
    "AdjudicatedRisk",
    "AgentConflict",
    "AgentProposal",
    "MultiAgentOrchestrator",
    "MultiAgentResult",
]
