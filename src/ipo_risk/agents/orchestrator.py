"""固定状态图：按角色分发证据分析，并形成可追踪的裁决结果。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import DecisionAnalyzer, DecisionBatch, DecisionVerdict
from ipo_risk.agents.grounded import AnalyzedRisk, GroundedAnalysisEngine
from ipo_risk.agents.search_planning import RoleSearchPlanner
from ipo_risk.domain import (
    AgentRole,
    AgentTrace,
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
            "supported": analyzed.analysis.supported,
            "severity": analyzed.analysis.severity.value,
            "confidence": analyzed.analysis.confidence,
            "candidate_chunk_ids": [
                candidate.chunk.chunk_id for candidate in analyzed.candidates
            ],
            "selected_chunk_ids": list(analyzed.analysis.evidence_chunk_ids),
            "attempts": analyzed.analysis_attempts,
            "retrieval_error": analyzed.retrieval_error,
            "analysis_error": analyzed.analysis_error,
        },
        status=ToolCallStatus.SUCCEEDED,
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
    ) -> None:
        self._engine = engine
        self._rules = tuple(rules)
        self._decision_analyzer = decision_analyzer
        self._search_planner = search_planner
        self._search_retry_policy = search_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )
        self._decision_retry_policy = decision_retry_policy or RetryPolicy(
            retry_if=retry_model_node_error
        )

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
                    rationale="接受有原文证据的最高置信度角色提案，等待人工复核。",
                    source_proposal_ids=tuple(
                        proposal.proposal_id for proposal in supported
                    ),
                )
            )
        return DecisionBatch(decisions=tuple(verdicts))

    @staticmethod
    def _validate_verdict(
        verdict: DecisionVerdict,
        proposal_by_id: dict[str, AgentProposal],
    ) -> None:
        unknown_proposals = set(verdict.source_proposal_ids) - set(proposal_by_id)
        if unknown_proposals:
            raise ValueError(
                f"decision cited unknown proposals: {sorted(unknown_proposals)}"
            )
        cited = [proposal_by_id[item] for item in verdict.source_proposal_ids]
        if any(
            proposal.analyzed_risk.rule.id != verdict.risk_type
            for proposal in cited
        ):
            raise ValueError("decision cited a proposal for another risk_type")
        allowed_evidence = {
            chunk_id
            for proposal in cited
            for chunk_id in proposal.analyzed_risk.analysis.evidence_chunk_ids
        }
        unknown_evidence = set(verdict.evidence_chunk_ids) - allowed_evidence
        if unknown_evidence:
            raise ValueError(
                f"decision cited unknown evidence chunks: {sorted(unknown_evidence)}"
            )

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

        for role in self._ROLES:
            role_rules = [rule for rule in self._rules if rule.owner is role]
            if not role_rules:
                continue
            role_started = _now()
            _emit_progress(
                progress_callback,
                f"agent:{role.value}:started",
                f"{_ROLE_NAMES[role]} Agent开始分析{len(role_rules)}类风险。",
            )
            role_tool_calls: list[ToolCall] = []
            planned_queries: dict[str, tuple[str, ...]] = {}
            search_planner = self._search_planner
            if search_planner is not None:
                planning_started = _now()
                search_attempts = 1
                search_error: str | None = None
                search_mode = "fixed_query_fallback"
                try:
                    search_outcome = self._search_retry_policy.run(
                        lambda: search_planner.plan(role_rules)
                    )
                    search_plan = search_outcome.value
                    search_attempts = search_outcome.attempts
                    expected_risk_types = {rule.id for rule in role_rules}
                    if search_plan.role is not role:
                        raise ValueError("search plan role does not match assigned Agent")
                    if set(search_plan.queries_by_risk()) != expected_risk_types:
                        raise ValueError(
                            "search plan must cover every risk assigned to the Agent"
                        )
                    planned_queries = search_plan.queries_by_risk()
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
                    search_attempts = error.attempts
                    search_error = str(error.last_error)
                    _emit_progress(
                        progress_callback,
                        f"agent:{role.value}:search_fallback",
                        f"搜索规划失败，启用固定查询兜底：{search_error}",
                    )
                planning_ended = _now()
                role_tool_calls.append(
                    ToolCall(
                        tool_call_id=f"TC-{run_id}-{role.value}-search-planning",
                        tool_name="agent_search_planning",
                        started_at=planning_started,
                        ended_at=planning_ended,
                        inputs={
                            "role": role.value,
                            "risk_types": [rule.id for rule in role_rules],
                        },
                        output={
                            "mode": search_mode,
                            "model": (
                                search_planner.model_name
                                if planned_queries
                                else None
                            ),
                            "prompt_version": (
                                search_planner.prompt_version
                                if planned_queries
                                else None
                            ),
                            "queries_by_risk": {
                                risk_type: list(queries)
                                for risk_type, queries in planned_queries.items()
                            },
                            "attempts": search_attempts,
                            "retries": max(0, search_attempts - 1),
                            "fallback_error": search_error,
                        },
                        status=ToolCallStatus.SUCCEEDED,
                    )
                )
            analyzed_risks: list[AnalyzedRisk] = []
            for index, rule in enumerate(role_rules, start=1):
                _emit_progress(
                    progress_callback,
                    f"agent:{role.value}:prompt",
                    f"[{index}/{len(role_rules)}] 正在分析：{rule.name}（{rule.id}）。",
                )
                analyzed = self._engine.analyze_rules(
                    (rule,),
                    search_queries_by_risk=planned_queries,
                )[0]
                analyzed_risks.append(analyzed)
                evidence_count = len(analyzed.analysis.evidence_chunk_ids)
                conclusion = (
                    f"形成有证据结论，引用{evidence_count}个原文片段"
                    if analyzed.analysis.supported and evidence_count
                    else "暂未形成有证据结论"
                )
                _emit_progress(
                    progress_callback,
                    f"agent:{role.value}:progress",
                    (
                        f"[{index}/{len(role_rules)}] {rule.name}（{rule.id}）："
                        f"召回{len(analyzed.candidates)}个候选片段，{conclusion}。"
                    ),
                )
            role_ended = _now()
            role_proposals = [
                AgentProposal(
                    proposal_id=f"P-{role.value}-{analyzed.rule.id}",
                    agent=role,
                    analyzed_risk=analyzed,
                )
                for analyzed in analyzed_risks
            ]
            proposals.extend(role_proposals)
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
            child_traces.append(
                AgentTrace(
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
                    output_refs=tuple(
                        proposal.proposal_id for proposal in role_proposals
                    ),
                    status=TraceStatus.SUCCEEDED,
                    decision=(
                        f"{role.value} Agent完成{len(role_proposals)}条规则分析，"
                        "所有结论均保留候选与选中证据ID。"
                    ),
                )
            )

        decision_started = _now()
        _emit_progress(
            progress_callback,
            "agent:decision:started",
            f"决策Agent开始交叉核验{len(proposals)}条专业Agent提案。",
        )
        decision_error: str | None = None
        decision_attempts = 1
        decision_mode = "deterministic"
        analyzer = self._decision_analyzer
        if analyzer is not None and proposals:
            try:
                outcome = self._decision_retry_policy.run(
                    lambda: analyzer.adjudicate(proposals)
                )
                decision_batch = outcome.value
                decision_attempts = outcome.attempts
                decision_mode = analyzer.provider_name
            except RetryExhaustedError as error:
                decision_attempts = error.attempts
                decision_error = str(error.last_error)
                decision_batch = self._deterministic_decisions(proposals)
                decision_mode = "deterministic_fallback"
        else:
            decision_batch = self._deterministic_decisions(proposals)

        proposal_by_id = {proposal.proposal_id: proposal for proposal in proposals}
        decisions: list[AdjudicatedRisk] = []
        conflicts: list[AgentConflict] = []
        for verdict in decision_batch.decisions:
            self._validate_verdict(verdict, proposal_by_id)
            sources = [proposal_by_id[item] for item in verdict.source_proposal_ids]
            if not verdict.accepted:
                if any(
                    proposal.analyzed_risk.analysis.supported
                    for proposal in sources
                ):
                    conflicts.append(
                        AgentConflict(
                            risk_type=verdict.risk_type,
                            kind="support_disagreement",
                            proposal_ids=verdict.source_proposal_ids,
                            detail="专业Agent认为风险成立，决策Agent认为证据不足。",
                            resolution=f"决策Agent驳回：{verdict.rationale}",
                        )
                    )
                continue
            selected = max(
                sources,
                key=lambda proposal: proposal.analyzed_risk.analysis.confidence,
            )
            severity_disagreement = any(
                proposal.analyzed_risk.analysis.severity is not verdict.severity
                for proposal in sources
            )
            if severity_disagreement:
                source_levels = ", ".join(
                    sorted(
                        {
                            proposal.analyzed_risk.analysis.severity.value
                            for proposal in sources
                        }
                    )
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
                )
            )
        decision_ended = _now()
        decision_call = ToolCall(
            tool_call_id=f"TC-{run_id}-decision-adjudication",
            tool_name="risk_adjudication",
            started_at=decision_started,
            ended_at=decision_ended,
            inputs={
                "proposal_ids": [proposal.proposal_id for proposal in proposals],
            },
            output={
                "mode": decision_mode,
                "model": (
                    self._decision_analyzer.model_name
                    if self._decision_analyzer is not None
                    and decision_mode == self._decision_analyzer.provider_name
                    else None
                ),
                "prompt_version": (
                    self._decision_analyzer.prompt_version
                    if self._decision_analyzer is not None
                    and decision_mode == self._decision_analyzer.provider_name
                    else None
                ),
                "accepted_risk_types": [decision.risk_type for decision in decisions],
                "conflict_count": len(conflicts),
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
            input_refs=tuple(proposal.proposal_id for proposal in proposals),
            tool_calls=(decision_call,),
            output_refs=tuple(decision.risk_type for decision in decisions),
            status=TraceStatus.SUCCEEDED,
            decision=f"裁决接受{len(decisions)}条具备原文证据的风险提案。",
        )
        _emit_progress(
            progress_callback,
            "agent:decision:completed",
            (
                f"决策Agent接受{len(decisions)}条有证据风险，"
                f"记录{len(conflicts)}项角色分歧。"
            ),
        )

        root_ended = _now()
        root_trace = AgentTrace(
            trace_id=root_trace_id,
            run_id=run_id,
            agent=AgentRole.ORCHESTRATOR,
            started_at=root_started,
            ended_at=root_ended,
            input_refs=(source_ref,),
            output_refs=tuple(decision.risk_type for decision in decisions),
            status=TraceStatus.SUCCEEDED,
            decision=(
                f"分发{len(proposals)}条角色提案并完成决策节点裁决，"
                f"接受{len(decisions)}条。"
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
