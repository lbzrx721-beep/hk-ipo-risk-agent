from __future__ import annotations

from datetime import date

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import (
    DecisionBatch,
    DecisionVerdict,
    DeepSeekDecisionAnalyzer,
)
from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator
from ipo_risk.agents.search_planning import PlannedRiskSearch, RoleSearchPlan
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryPolicy, retry_model_node_error


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


def _rule(risk_id: str, owner: AgentRole, query: str) -> RiskRule:
    return RiskRule(
        id=risk_id,
        name=risk_id,
        owner=owner,
        severity=RiskSeverity.HIGH,
        query=query,
        required_keywords=(query,),
        claim=f"{risk_id} is material.",
    )


class OwnerAwareAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "evidence-analysis-v1"

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        return EvidenceAnalysis(
            supported=True,
            claim=f"{rule.owner.value} agent identified {rule.id}.",
            severity=rule.severity,
            confidence=0.88,
            evidence_chunk_ids=(candidates[0].chunk_id,),
            uncertainty="Human verification remains required.",
        )


class AgentDirectedEmbeddingProvider:
    model_name = "agent-directed-semantic"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if "Agent专属问题" in text or "目标证据" in text
            else [0.0, 1.0]
            for text in texts
        ]


class FinanceSearchPlanner:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "role-search-planning-v1"

    def plan(self, rules: list[RiskRule], *, feedback: object = None) -> RoleSearchPlan:
        assert [rule.owner for rule in rules] == [AgentRole.FINANCE]
        return RoleSearchPlan(
            role=AgentRole.FINANCE,
            plans=(
                PlannedRiskSearch(
                    risk_type="cash_burn",
                    queries=("Agent专属问题：研发现金跑道与后续融资",),
                    coverage_note="覆盖现金跑道和融资依赖表述。",
                ),
            ),
        )


class FlakyFinanceSearchPlanner(FinanceSearchPlanner):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def plan(self, rules: list[RiskRule], *, feedback: object = None) -> RoleSearchPlan:
        self.calls += 1
        if self.calls <= self.failures:
            raise ValueError("temporary malformed search plan")
        return super().plan(rules, feedback=feedback)


def test_role_agent_plans_queries_that_drive_semantic_evidence_search() -> None:
    target = _chunk("C-TARGET", "目标证据：公司现金跑道较短并需要后续融资。")
    distractor = _chunk("C-DISTRACTOR", "普通公司治理和董事会会议安排。")
    rule = _rule("cash_burn", AgentRole.FINANCE, "固定查询不会命中")
    engine = GroundedAnalysisEngine(
        [target, distractor],
        [rule],
        embedding_provider=AgentDirectedEmbeddingProvider(),
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=2,
        semantic_top_k=2,
        rerank_top_k=1,
    )

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        search_planner=FinanceSearchPlanner(),
    ).run(run_id="RUN-03696-AGENT-SEARCH", source_ref="prospectus.pdf")

    proposal = result.proposals[0].analyzed_risk
    assert proposal.search_queries == ("Agent专属问题：研发现金跑道与后续融资",)
    assert proposal.retrieval_mode == "agentic_semantic"
    assert proposal.selected_chunks == (target,)
    role_trace = next(
        trace for trace in result.traces if trace.agent is AgentRole.FINANCE
    )
    search_call = role_trace.tool_calls[0]
    assert search_call.tool_name == "agent_search_planning"
    assert search_call.output["mode"] == "deepseek"
    assert search_call.output["queries_by_risk"] == {
        "cash_burn": ["Agent专属问题：研发现金跑道与后续融资"]
    }


def test_search_planning_retries_before_using_agent_queries() -> None:
    target = _chunk("C-TARGET", "目标证据：公司现金跑道较短并需要后续融资。")
    rule = _rule("cash_burn", AgentRole.FINANCE, "固定查询不会命中")
    planner = FlakyFinanceSearchPlanner(failures=1)
    engine = GroundedAnalysisEngine(
        [target],
        [rule],
        embedding_provider=AgentDirectedEmbeddingProvider(),
        evidence_analyzer=OwnerAwareAnalyzer(),
    )

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        search_planner=planner,
        search_retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    ).run(run_id="RUN-RETRY", source_ref="prospectus.pdf")

    assert planner.calls == 2
    search_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.FINANCE
    ).tool_calls[0]
    assert search_call.output["attempts"] == 2
    assert search_call.output["retries"] == 1
    assert search_call.output["fallback_error"] is None


def test_search_planning_falls_back_to_fixed_retrieval_and_audits_reason() -> None:
    chunk = _chunk("C-FIXED", "cash burn remains material")
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    planner = FlakyFinanceSearchPlanner(failures=99)
    engine = GroundedAnalysisEngine(
        [chunk],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )
    progress_events: list[tuple[str, str]] = []

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        search_planner=planner,
        search_retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    ).run(
        run_id="RUN-FALLBACK",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append((stage, detail)),
    )

    proposal = result.proposals[0].analyzed_risk
    assert proposal.search_queries == ("cash burn",)
    assert proposal.retrieval_mode == "bm25"
    assert ("agent:finance:search_fallback", "搜索规划失败，启用固定查询兜底：temporary malformed search plan") in progress_events
    search_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.FINANCE
    ).tool_calls[0]
    assert search_call.output["mode"] == "fixed_query_fallback"
    assert search_call.output["attempts"] == 2
    assert search_call.output["fallback_error"] == "temporary malformed search plan"


def test_orchestrator_dispatches_rules_to_role_agents_with_parent_child_traces() -> None:
    chunks = [
        _chunk("C-FIN", "cash burn remains material"),
        _chunk("C-LEGAL", "license compliance remains material"),
    ]
    rules = [
        _rule("cash_burn", AgentRole.FINANCE, "cash burn"),
        _rule("license_dependency", AgentRole.LEGAL, "license compliance"),
    ]
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
        rerank_top_k=2,
    )

    progress_events: list[tuple[str, str]] = []
    result = MultiAgentOrchestrator(engine, rules).run(
        run_id="RUN-03696-TEST",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append(
            (stage, detail)
        ),
    )

    assert {proposal.agent for proposal in result.proposals} == {
        AgentRole.FINANCE,
        AgentRole.LEGAL,
    }
    role_traces = {
        trace.agent: trace
        for trace in result.traces
        if trace.agent in {AgentRole.FINANCE, AgentRole.LEGAL}
    }
    assert set(role_traces) == {AgentRole.FINANCE, AgentRole.LEGAL}
    assert all(trace.parent_trace_id == result.root_trace.trace_id for trace in role_traces.values())
    assert all(trace.tool_calls for trace in role_traces.values())
    assert {decision.risk_type for decision in result.decisions} == {
        "cash_burn",
        "license_dependency",
    }
    assert [stage for stage, _detail in progress_events] == [
        "agent:finance:started",
        "agent:finance:prompt",
        "agent:finance:progress",
        "agent:finance:completed",
        "agent:legal:started",
        "agent:legal:prompt",
        "agent:legal:progress",
        "agent:legal:completed",
        "agent:decision:started",
        "agent:decision:completed",
    ]
    assert "cash_burn" in progress_events[1][1]
    assert "license_dependency" in progress_events[5][1]


class SeverityReviewingDecisionAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "decision-adjudication-v1"

    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        proposal = proposals[0]
        analyzed = proposal.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=True,
                    claim="Evidence supports cash burn, but current liquidity reduces immediacy.",
                    severity=RiskSeverity.MEDIUM,
                    confidence=0.81,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty="Liquidity runway still requires human verification.",
                    rationale="The evidence supports the direction but not high immediacy.",
                    source_proposal_ids=(proposal.proposal_id,),
                ),
            )
        )


def test_decision_agent_adjustment_is_applied_and_recorded_as_a_conflict() -> None:
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    engine = GroundedAnalysisEngine(
        [_chunk("C-FIN", "cash burn remains material")],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=SeverityReviewingDecisionAnalyzer(),
    ).run(run_id="RUN-03696-DECISION", source_ref="prospectus.pdf")

    assert result.decisions[0].analysis.severity is RiskSeverity.MEDIUM
    assert result.decisions[0].analysis.claim.startswith("Evidence supports cash burn")
    assert result.decisions[0].decision_mode == "deepseek"
    assert result.conflicts[0].kind == "severity_disagreement"
    assert "high" in result.conflicts[0].detail
    assert "medium" in result.conflicts[0].detail
    decision_trace = next(
        trace for trace in result.traces if trace.agent is AgentRole.DECISION
    )
    assert decision_trace.tool_calls[0].output["conflict_count"] == 1


class GroundedDecisionProvider:
    model_name = "deepseek-v4-flash"

    def __init__(self) -> None:
        self.user_prompt = ""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, object]:
        self.user_prompt = user_prompt
        return {
            "decisions": [
                {
                    "risk_type": "cash_burn",
                    "accepted": True,
                    "claim": "Cash burn is supported by the cited prospectus text.",
                    "severity": "high",
                    "confidence": 0.86,
                    "evidence_chunk_ids": ["C-FIN"],
                    "uncertainty": "The exact runway still needs verification.",
                    "rationale": "The specialist conclusion is directly grounded.",
                    "source_proposal_ids": ["P-finance-cash_burn"],
                }
            ]
        }


class StreamingGroundedDecisionProvider(GroundedDecisionProvider):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: object | None = None,
    ) -> dict[str, object]:
        if callable(on_text_delta):
            on_text_delta('{"decisions":[{"risk_type":"cash_burn"}]}')
        return super().complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def test_deepseek_decision_agent_streams_content_to_decision_channel() -> None:
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    progress_events: list[tuple[str, str]] = []
    decision_analyzer = DeepSeekDecisionAnalyzer(
        StreamingGroundedDecisionProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        progress_callback=lambda stage, detail: progress_events.append(
            (stage, detail)
        ),
    )
    engine = GroundedAnalysisEngine(
        [_chunk("C-FIN", "cash burn remains material")],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )

    MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=decision_analyzer,
    ).run(run_id="RUN-03696-STREAM", source_ref="prospectus.pdf")

    assert progress_events == [
        (
            "agent:decision:delta",
            '{"decisions":[{"risk_type":"cash_burn"}]}',
        )
    ]


def test_deepseek_decision_agent_receives_only_grounded_role_proposals() -> None:
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    provider = GroundedDecisionProvider()
    decision_analyzer = DeepSeekDecisionAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=1024,
    )
    engine = GroundedAnalysisEngine(
        [_chunk("C-FIN", "cash burn remains material")],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=decision_analyzer,
    ).run(run_id="RUN-03696-DEEPSEEK", source_ref="prospectus.pdf")

    assert result.decisions[0].analysis.evidence_chunk_ids == ("C-FIN",)
    assert "cash burn remains material" in provider.user_prompt
    assert '"allowed_evidence_chunk_ids": ["C-FIN"]' in provider.user_prompt
    assert result.decisions[0].decision_mode == "deepseek"


class RejectingDecisionAnalyzer(SeverityReviewingDecisionAnalyzer):
    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        proposal = proposals[0]
        analyzed = proposal.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=False,
                    severity=RiskSeverity.LOW,
                    confidence=0.35,
                    rationale="The passage describes a possibility, not a supported current risk.",
                    source_proposal_ids=(proposal.proposal_id,),
                ),
            )
        )


def test_decision_rejection_of_supported_proposal_is_a_support_conflict() -> None:
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    engine = GroundedAnalysisEngine(
        [_chunk("C-FIN", "cash burn remains material")],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=RejectingDecisionAnalyzer(),
    ).run(run_id="RUN-03696-REJECT", source_ref="prospectus.pdf")

    assert result.decisions == ()
    assert result.conflicts[0].kind == "support_disagreement"
    assert "驳回" in result.conflicts[0].resolution
    decision_trace = next(
        trace for trace in result.traces if trace.agent is AgentRole.DECISION
    )
    assert decision_trace.tool_calls[0].output["conflict_count"] == 1
