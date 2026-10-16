from __future__ import annotations

from datetime import date
from threading import Barrier, Lock, Thread

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import (
    DecisionBatch,
    DecisionVerdict,
    DeepSeekDecisionAnalyzer,
)
from ipo_risk.agents.discovery import DiscoveredRisk, RiskDiscoveryResult
from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.agents.intervention import (
    RiskScopeFunnel,
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeSelection,
)
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator, MultiAgentResult
from ipo_risk.agents.risk_scoping import CompanyRiskScope, RiskApplicability
from ipo_risk.agents.search_planning import PlannedRiskSearch, RoleSearchPlan
from ipo_risk.agents.specialist import SpecialistStatus
from ipo_risk.domain import AgentRole, RiskSeverity, TraceStatus
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


def test_catalog_fallback_is_not_presented_as_an_ai_discovered_risk() -> None:
    catalog_rule = _rule("cash_burn", AgentRole.FINANCE, "现金")

    request = MultiAgentOrchestrator._build_scope_review_request(
        run_id="RUN-FALLBACK",
        candidate_rules=(catalog_rule,),
        active_rules=(catalog_rule,),
        scope_call=None,
        funnel=RiskScopeFunnel(catalog_added=1, scoped_in=1),
        evidence_counts_by_risk={},
        source_candidate_counts_by_risk={},
        confidences_by_risk={},
    )

    candidate = request.candidates[0]
    assert candidate.source == "catalog_fallback"
    assert candidate.ai_recommended is False
    assert candidate.confidence is None
    assert candidate.evidence_count == 0
    assert "人工" in candidate.rationale


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


class PartiallyUnsupportedAnalyzer(OwnerAwareAnalyzer):
    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        if rule.id == "share_award_dilution":
            return EvidenceAnalysis(
                supported=False,
                severity=RiskSeverity.MEDIUM,
                confidence=0.45,
                uncertainty=(
                    "现有证据仅说明附属公司中间股东安排，不能证明发行人股份被稀释。"
                ),
            )
        return super().analyze(rule, candidates)


def test_unsupported_specialist_candidate_is_preserved_for_human_review() -> None:
    rules = [
        _rule("cash_burn", AgentRole.FINANCE, "现金消耗"),
        _rule("share_award_dilution", AgentRole.LEGAL, "股份奖励"),
    ]
    chunks = [
        _chunk("C-CASH", "现金消耗持续增加，现金消耗存在压力。"),
        _chunk(
            "C-AWARD",
            "附属公司通过中间股东向雇员授予权益，属于股份奖励安排。",
        ),
    ]
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        evidence_analyzer=PartiallyUnsupportedAnalyzer(),
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(engine, rules).run(
        run_id="RUN-PRESERVE-UNSUPPORTED",
        source_ref="prospectus.pdf",
    )

    assert {decision.risk_type for decision in result.decisions} == {
        "cash_burn",
        "share_award_dilution",
    }
    unresolved = next(
        decision
        for decision in result.decisions
        if decision.risk_type == "share_award_dilution"
    )
    assert unresolved.analysis.supported is False
    assert unresolved.specialist_status is SpecialistStatus.EVIDENCE_INSUFFICIENT
    assert unresolved.selected.candidates


class AgentDirectedEmbeddingProvider:
    model_name = "agent-directed-semantic"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "Agent专属问题" in text or "目标证据" in text else [0.0, 1.0]
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
    role_trace = next(trace for trace in result.traces if trace.agent is AgentRole.FINANCE)
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
    assert (
        "agent:finance:search_fallback",
        "搜索规划失败，启用固定查询兜底：temporary malformed search plan",
    ) in progress_events
    search_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.FINANCE
    ).tool_calls[0]
    assert search_call.output["mode"] == "fixed_query_fallback"
    assert search_call.output["attempts"] == 2
    assert search_call.output["fallback_error"] == "temporary malformed search plan"


def test_search_planning_fallback_keeps_existing_open_risk_queries() -> None:
    chunks = [
        _chunk("C-OPEN", "dynamic business risk remains material"),
        _chunk("C-KNOWN", "cash burn remains material"),
    ]
    rules = [
        _rule("open_dynamic_business_risk", AgentRole.FINANCE, "dynamic business risk"),
        _rule("cash_burn", AgentRole.FINANCE, "cash burn"),
    ]
    planner = FlakyFinanceSearchPlanner(failures=99)
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        evidence_analyzer=OwnerAwareAnalyzer(),
    )

    result = MultiAgentOrchestrator(
        engine,
        rules,
        search_planner=planner,
        search_retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    ).run(run_id="RUN-OPEN-FALLBACK", source_ref="prospectus.pdf")

    search_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.FINANCE
    ).tool_calls[0]
    assert search_call.output["mode"] == "fixed_query_fallback"
    assert search_call.output["queries_by_risk"] == {
        "open_dynamic_business_risk": ["dynamic business risk"]
    }
    assert search_call.output["repair_errors"] == ["temporary malformed search plan"]
    assert {proposal.analyzed_risk.rule.id for proposal in result.proposals} == {
        "open_dynamic_business_risk",
        "cash_burn",
    }


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
        progress_callback=lambda stage, detail: progress_events.append((stage, detail)),
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
    assert all(
        trace.parent_trace_id == result.root_trace.trace_id for trace in role_traces.values()
    )
    assert all(trace.tool_calls for trace in role_traces.values())
    assert {decision.risk_type for decision in result.decisions} == {
        "cash_burn",
        "license_dependency",
    }
    stages = [stage for stage, _detail in progress_events]
    for role in ("finance", "legal"):
        role_stages = [
            f"agent:{role}:started",
            f"agent:{role}:prompt",
            f"agent:{role}:progress",
            f"agent:{role}:completed",
        ]
        assert [stages.index(stage) for stage in role_stages] == sorted(
            stages.index(stage) for stage in role_stages
        )
    decision_started = stages.index("agent:decision:started")
    assert decision_started > stages.index("agent:finance:completed")
    assert decision_started > stages.index("agent:legal:completed")
    assert stages[-1] == "agent:decision:completed"
    assert "cash_burn" in next(
        detail for stage, detail in progress_events if stage == "agent:finance:prompt"
    )
    assert "license_dependency" in next(
        detail for stage, detail in progress_events if stage == "agent:legal:prompt"
    )


class ParallelBarrierAnalyzer(OwnerAwareAnalyzer):
    def __init__(self, parties: int) -> None:
        self._barrier = Barrier(parties, timeout=3)
        self._lock = Lock()
        self.completed_roles: set[AgentRole] = set()

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        self._barrier.wait()
        with self._lock:
            self.completed_roles.add(rule.owner)
        return super().analyze(rule, candidates)


def test_professional_agents_run_in_parallel_before_decision_agent() -> None:
    role_queries = {
        AgentRole.FINANCE: "现金流消耗压力",
        AgentRole.LEGAL: "监管许可合规风险",
        AgentRole.BUSINESS: "客户集中度风险",
        AgentRole.MARKET: "估值与市场波动",
    }
    rules = [_rule(f"{role.value}_risk", role, query) for role, query in role_queries.items()]
    chunks = [
        _chunk(
            f"C-{role.value.upper()}",
            f"招股书披露公司存在{query}，相关风险可能在上市后继续产生重大影响。",
        )
        for role, query in role_queries.items()
    ]
    analyzer = ParallelBarrierAnalyzer(parties=len(role_queries))
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        evidence_analyzer=analyzer,
        lexical_top_k=8,
        rerank_top_k=4,
    )
    events: list[str] = []

    result = MultiAgentOrchestrator(engine, rules).run(
        run_id="RUN-LANGGRAPH-PARALLEL",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, _detail: events.append(stage),
    )

    assert analyzer.completed_roles == set(role_queries)
    decision_started = events.index("agent:decision:started")
    assert all(
        events.index(f"agent:{role.value}:completed") < decision_started for role in role_queries
    )
    assert {proposal.agent for proposal in result.proposals} == set(role_queries)
    graph_call = next(
        call for call in result.root_trace.tool_calls if call.tool_name == "langgraph_state_graph"
    )
    assert graph_call.output["execution_model"] == "parallel_fan_out_fan_in"
    assert set(graph_call.output["completed_roles"]) == {role.value for role in role_queries}


def test_human_scope_review_gates_agents_and_applies_edits() -> None:
    chunk = _chunk(
        "C-CASH",
        "招股书披露公司存在现金流消耗压力，后续融资窗口可能受到市场环境影响。",
    )
    rule = _rule("cash_burn", AgentRole.FINANCE, "现金流消耗压力")
    engine = GroundedAnalysisEngine(
        [chunk],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )
    controller = RiskScopeReviewController()
    events: list[str] = []
    results: list[MultiAgentResult] = []
    orchestrator = MultiAgentOrchestrator(
        engine,
        [rule],
        risk_scope_review_controller=controller,
    )
    thread = Thread(
        target=lambda: results.append(
            orchestrator.run(
                run_id="RUN-HUMAN-GATE",
                source_ref="prospectus.pdf",
                progress_callback=lambda stage, _detail: events.append(stage),
            )
        )
    )

    thread.start()
    assert controller.wait_for_request(timeout=2)
    request = controller.pending_request()
    assert request is not None
    assert request.candidates[0].risk_type == "cash_burn"
    assert "risk_scope_review_requested" in events
    assert not any(
        stage.startswith("agent:finance:") or stage.startswith("agent:market:") for stage in events
    )

    controller.submit(
        RiskScopeReviewDecision(
            run_id=request.run_id,
            reviewer="投资经理A",
            selections=(
                RiskScopeSelection(
                    risk_type="cash_burn",
                    included=True,
                    owner=AgentRole.MARKET,
                    severity=RiskSeverity.MEDIUM,
                    analysis_requirement="重点核验后续融资窗口与市场环境。",
                ),
            ),
            global_requirement="优先引用最近报告期数据。",
        )
    )
    thread.join(timeout=3)

    assert not thread.is_alive()
    result = results[0]
    assert result.proposals[0].agent is AgentRole.MARKET
    edited_rule = result.proposals[0].analyzed_risk.rule
    assert edited_rule.severity is RiskSeverity.MEDIUM
    assert "重点核验后续融资窗口与市场环境。" in edited_rule.search_queries
    assert events.index("risk_scope_review_completed") < events.index("agent:market:started")
    review_call = next(
        call for call in result.root_trace.tool_calls if call.tool_name == "human_risk_scope_review"
    )
    assert review_call.output["reviewer"] == "投资经理A"
    assert review_call.output["selections"][0]["owner"] == "market"


class OpenRiskDiscoverer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "open-risk-discovery-v1"

    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        assert chunks[0].chunk_id == "C-OPEN"
        self.catalog_risk_types = tuple(rule.id for rule in known_rules)
        return RiskDiscoveryResult(
            risks=(
                DiscoveredRisk(
                    risk_type="open_123456789abc",
                    title="独家数据授权终止风险",
                    owner=AgentRole.LEGAL,
                    severity=RiskSeverity.HIGH,
                    claim="核心训练数据依赖可能提前终止的独家授权。",
                    confidence=0.86,
                    evidence_chunk_ids=("C-OPEN",),
                    search_queries=("核心训练数据授权提前终止",),
                    anchor_terms=("独家授权",),
                    uncertainty="替代数据来源仍需人工核对。",
                ),
            ),
            batch_count=1,
            model_attempts=2,
            raw_candidate_count=2,
            consolidated_candidate_count=1,
            consolidation_mode="semantic_model",
            consolidation_attempts=1,
        )


class VectorGuidedOpenRiskDiscoverer(OpenRiskDiscoverer):
    def __init__(self) -> None:
        self.used_guided_path = False

    def discover_with_retrieval(
        self,
        chunks,
        known_rules,
        *,
        corpus_selector,
    ) -> RiskDiscoveryResult:
        self.used_guided_path = True
        selected = corpus_selector(
            ("exclusive data license termination",),
            semantic_top_k_per_query=2,
            lexical_top_k_per_query=2,
            neighbor_radius=1,
            coverage_section_count=2,
            max_chunks=20,
        )
        result = self.discover(selected, known_rules)
        return result.model_copy(
            update={
                "discovery_mode": "vector_guided_original_text",
                "source_chunk_count": len(chunks),
                "selected_chunk_count": len(selected),
                "retrieval_query_count": 1,
            }
        )


class AcceptingOpenDecisionAnalyzer:
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
                    claim=analyzed.analysis.claim,
                    severity=analyzed.analysis.severity,
                    confidence=analyzed.analysis.confidence,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty=analyzed.analysis.uncertainty,
                    rationale="决策Agent确认专业Agent证据能够支持开放风险。",
                    source_proposal_ids=(proposal.proposal_id,),
                ),
            )
        )


class DebateEmbeddingProvider:
    model_name = "debate-semantic"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "反证条款" in text else [0.0, 1.0]
            for text in texts
        ]


class TwoRoundDecisionAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-pro"
    prompt_version = "decision-adjudication-v4"

    def __init__(self) -> None:
        self.calls = 0

    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        self.calls += 1
        if self.calls == 1:
            proposal = proposals[0]
            analyzed = proposal.analyzed_risk
            return DecisionBatch(
                decisions=(
                    DecisionVerdict(
                        risk_type=analyzed.rule.id,
                        accepted=True,
                        claim=analyzed.analysis.claim,
                        severity=analyzed.analysis.severity,
                        confidence=0.62,
                        evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                        uncertainty="缓释条款尚未完成对照。",
                        rationale="需要法律Agent检索反证条款后再作终局判断。",
                        source_proposal_ids=(proposal.proposal_id,),
                        needs_debate=True,
                        challenge="现有结论是否忽略已经生效的终止或缓释条款？",
                        challenge_role=AgentRole.LEGAL,
                        challenge_queries=("反证条款 终止 缓释 已生效",),
                    ),
                )
            )
        debate = next(proposal for proposal in proposals if proposal.proposal_id.startswith("P-debate"))
        analyzed = debate.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=True,
                    claim="补证显示相关风险仍存在，但已受到终止条款缓释。",
                    severity=RiskSeverity.MEDIUM,
                    confidence=0.84,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty="结论仅适用于招股书披露的条款状态。",
                    rationale="法律Agent补充的反证条款支持收窄结论并下调严重程度。",
                    source_proposal_ids=(debate.proposal_id,),
                ),
            )
        )


def test_decision_challenge_triggers_targeted_retrieval_and_second_round() -> None:
    original = _chunk("C-ORIGINAL", "现金消耗持续增加，相关融资风险仍然存在。")
    rebuttal = _chunk("C-REBUTTAL", "反证条款：相关义务已设置终止和缓释安排。")
    rule = _rule("cash_burn", AgentRole.FINANCE, "现金消耗")
    analyzer = TwoRoundDecisionAnalyzer()
    engine = GroundedAnalysisEngine(
        [original, rebuttal],
        [rule],
        embedding_provider=DebateEmbeddingProvider(),
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=4,
        semantic_top_k=4,
        rerank_top_k=1,
    )
    events: list[tuple[str, str]] = []

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=analyzer,
    ).run(
        run_id="RUN-DEBATE",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: events.append((stage, detail)),
    )

    assert analyzer.calls == 2
    assert result.decisions[0].analysis.severity is RiskSeverity.MEDIUM
    assert result.decisions[0].analysis.evidence_chunk_ids == ("C-REBUTTAL",)
    assert any(proposal.proposal_id.startswith("P-debate-1-legal") for proposal in result.proposals)
    stages = [stage for stage, _detail in events]
    assert "agent:decision:debate_requested" in stages
    assert "agent:legal:debate_retrieval" in stages
    assert "agent:legal:debate_response" in stages
    assert "agent:decision:debate_completed" in stages
    debate_event = next(detail for stage, detail in events if stage == "debate_record")
    assert '"risk_name":"cash_burn"' in debate_event
    decision_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DECISION)
    debate_call = next(
        call for call in decision_trace.tool_calls
        if call.tool_name == "risk_debate_verification"
    )
    assert debate_call.output["challenge_role"] == "legal"
    assert debate_call.output["status"] == "completed"
    assert debate_call.output["recalled_evidence_count"] == 1
    assert debate_call.output["new_evidence_count"] == 1


class SelectiveDebateDecisionAnalyzer(TwoRoundDecisionAnalyzer):
    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        self.calls += 1
        if self.calls == 1:
            verdicts: list[DecisionVerdict] = []
            for proposal in proposals:
                analyzed = proposal.analyzed_risk
                debate = analyzed.rule.id == "cash_burn"
                verdicts.append(
                    DecisionVerdict(
                        risk_type=analyzed.rule.id,
                        accepted=True,
                        claim=analyzed.analysis.claim,
                        severity=analyzed.analysis.severity,
                        confidence=analyzed.analysis.confidence,
                        evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                        uncertainty=analyzed.analysis.uncertainty,
                        rationale=(
                            "现金消耗风险需要补证。"
                            if debate
                            else "现有证据已经支持结论，无需参与辩论。"
                        ),
                        source_proposal_ids=(proposal.proposal_id,),
                        needs_debate=debate,
                        challenge=("是否存在缓释安排？" if debate else ""),
                        challenge_role=(AgentRole.LEGAL if debate else None),
                        challenge_queries=(("反证条款 缓释",) if debate else ()),
                    )
                )
            return DecisionBatch(decisions=tuple(verdicts))
        debate = next(proposal for proposal in proposals if proposal.proposal_id.startswith("P-debate"))
        analyzed = debate.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=True,
                    claim="现金消耗风险经补证后已收窄。",
                    severity=RiskSeverity.MEDIUM,
                    confidence=0.84,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty="结论以补证披露为限。",
                    rationale="仅替换存在歧义的现金消耗风险。",
                    source_proposal_ids=(debate.proposal_id,),
                ),
            )
        )


def test_targeted_debate_replaces_only_challenged_risk() -> None:
    rules = [
        _rule("cash_burn", AgentRole.FINANCE, "现金消耗"),
        _rule("customer_concentration", AgentRole.BUSINESS, "客户集中"),
    ]
    engine = GroundedAnalysisEngine(
        [
            _chunk("C-CASH", "公司经营活动现金流持续流出，现金消耗持续增加并形成显著融资压力。"),
            _chunk("C-STABLE", "公司前五大客户收入占比较高，客户集中程度持续上升且风险明确。"),
            _chunk("C-MITIGATION", "反证条款显示现金消耗风险已经设置后续融资及明确缓释安排。"),
        ],
        rules,
        embedding_provider=DebateEmbeddingProvider(),
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
        semantic_top_k=5,
        rerank_top_k=2,
    )

    analyzer = SelectiveDebateDecisionAnalyzer()
    result = MultiAgentOrchestrator(
        engine,
        rules,
        decision_analyzer=analyzer,
    ).run(run_id="RUN-SELECTIVE-DEBATE", source_ref="prospectus.pdf")

    decisions = {decision.risk_type: decision for decision in result.decisions}
    assert set(decisions) == {"cash_burn", "customer_concentration"}
    assert decisions["cash_burn"].analysis.claim == "现金消耗风险经补证后已收窄。", (
        analyzer.calls,
        decisions["cash_burn"],
        result.conflicts,
    )
    assert decisions["customer_concentration"].analysis.claim == (
        "business agent identified customer_concentration."
    )


class MaterialReviewDecisionAnalyzer(TwoRoundDecisionAnalyzer):
    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        self.calls += 1
        if self.calls == 1:
            proposal = proposals[0]
            analyzed = proposal.analyzed_risk
            return DecisionBatch(
                decisions=(
                    DecisionVerdict(
                        risk_type=analyzed.rule.id,
                        accepted=True,
                        claim=analyzed.analysis.claim,
                        severity=RiskSeverity.HIGH,
                        confidence=0.78,
                        evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                        uncertainty="尚需检查招股书是否披露缓释或期后变化。",
                        rationale="现有证据支持重大风险，但第一轮未主动提出辩论。",
                        source_proposal_ids=(proposal.proposal_id,),
                    ),
                )
            )
        debate = next(proposal for proposal in proposals if proposal.proposal_id.startswith("P-debate"))
        analyzed = debate.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=True,
                    claim=analyzed.analysis.claim,
                    severity=RiskSeverity.MEDIUM,
                    confidence=0.86,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty="结论以补证检索范围为限。",
                    rationale="对抗性补证发现缓释安排，因此下调严重程度。",
                    source_proposal_ids=(debate.proposal_id,),
                ),
            )
        )


def test_high_risk_without_model_challenge_gets_limited_material_review() -> None:
    original = _chunk("C-MATERIAL", "现金消耗持续增加，融资风险重大。")
    rebuttal = _chunk("C-MITIGATION", "反证条款：现金消耗风险已设置缓释及期后融资安排。")
    rule = _rule("cash_burn", AgentRole.FINANCE, "现金消耗")
    analyzer = MaterialReviewDecisionAnalyzer()
    engine = GroundedAnalysisEngine(
        [original, rebuttal],
        [rule],
        embedding_provider=DebateEmbeddingProvider(),
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=4,
        semantic_top_k=4,
        rerank_top_k=1,
    )
    events: list[tuple[str, str]] = []

    MultiAgentOrchestrator(
        engine,
        [rule],
        decision_analyzer=analyzer,
        material_debate_max_items=1,
    ).run(
        run_id="RUN-MATERIAL-REVIEW",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: events.append((stage, detail)),
    )

    assert analyzer.calls == 2
    debate_event = next(detail for stage, detail in events if stage == "debate_record")
    assert '"trigger":"material_risk_review"' in debate_event
    assert any(
        stage == "agent:decision:debate_requested" and "对抗性复核" in detail
        for stage, detail in events
    )


class RecordingDynamicScopePlanner:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "company-risk-applicability-v2"

    def __init__(self) -> None:
        self.risk_types: tuple[str, ...] = ()

    def plan(self, rules, chunks) -> CompanyRiskScope:
        assert chunks
        self.risk_types = tuple(rule.id for rule in rules)
        return CompanyRiskScope(
            industry="由当前招股书识别的行业",
            business_model="由当前招股书识别的业务模式",
            revenue_model="由当前招股书识别的收入模式",
            risks=tuple(
                RiskApplicability(
                    risk_type=rule.id,
                    applicable=True,
                    rationale="候选风险由当前公司原文直接发现。",
                )
                for rule in rules
            ),
        )


def test_successful_discovery_defines_topics_without_fixed_catalog_priming() -> None:
    source = _chunk(
        "C-OPEN",
        "exclusive data license may terminate while cash burn remains material",
    )
    fixed_rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    engine = GroundedAnalysisEngine(
        [source],
        [fixed_rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )

    discoverer = OpenRiskDiscoverer()
    result = MultiAgentOrchestrator(
        engine,
        [fixed_rule],
        risk_discoverer=discoverer,
        decision_analyzer=AcceptingOpenDecisionAnalyzer(),
    ).run(run_id="RUN-DOCUMENT-FIRST", source_ref="prospectus.pdf")

    assert [proposal.analyzed_risk.rule.id for proposal in result.proposals] == [
        "open_123456789abc"
    ]
    discovery_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY
    ).tool_calls[0]
    assert discovery_call.inputs["known_risk_types"] == []
    assert discovery_call.inputs["catalog_matching_types"] == ["cash_burn"]
    assert discoverer.catalog_risk_types == ("cash_burn",)
    assert discovery_call.output["analysis_strategy"] == "document_first_dynamic"
    assert discovery_call.output["fallback_catalog_used"] is False
    assert discovery_call.output["raw_candidate_count"] == 2
    assert discovery_call.output["consolidated_candidate_count"] == 1


def test_orchestrator_prefers_vector_guided_original_text_discovery() -> None:
    source = _chunk("C-OPEN", "exclusive data license may terminate")
    fixed_rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    engine = GroundedAnalysisEngine(
        [source],
        [fixed_rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )
    discoverer = VectorGuidedOpenRiskDiscoverer()

    result = MultiAgentOrchestrator(
        engine,
        [fixed_rule],
        risk_discoverer=discoverer,
        decision_analyzer=AcceptingOpenDecisionAnalyzer(),
    ).run(run_id="RUN-VECTOR-GUIDED", source_ref="prospectus.pdf")

    discovery_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY
    ).tool_calls[0]
    assert discoverer.used_guided_path
    assert discovery_call.output["analysis_strategy"] == "vector_guided_dynamic"
    assert discovery_call.output["discovery_mode"] == "vector_guided_original_text"
    assert discovery_call.inputs["topic_source"] == (
        "vector_retrieved_original_text_with_section_coverage"
    )


def test_company_scope_checks_dynamic_topics_after_document_discovery() -> None:
    source = _chunk("C-OPEN", "exclusive data license may terminate")
    fixed_rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn")
    scope_planner = RecordingDynamicScopePlanner()
    engine = GroundedAnalysisEngine(
        [source],
        [fixed_rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )

    result = MultiAgentOrchestrator(
        engine,
        [fixed_rule],
        risk_discoverer=OpenRiskDiscoverer(),
        risk_scope_planner=scope_planner,
        decision_analyzer=AcceptingOpenDecisionAnalyzer(),
    ).run(run_id="RUN-DYNAMIC-SCOPE", source_ref="prospectus.pdf")

    assert scope_planner.risk_types == ("open_123456789abc",)
    scope_call = next(
        call for call in result.root_trace.tool_calls if call.tool_name == "company_risk_scoping"
    )
    assert scope_call.inputs["execution_order"] == "after_document_first_discovery"


def test_discovered_risk_is_rechecked_by_specialist_and_decision_agents() -> None:
    source = _chunk(
        "C-OPEN",
        "exclusive data license may terminate and disrupt the core training dataset",
    )
    engine = GroundedAnalysisEngine(
        [source],
        [],
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )
    progress_events: list[str] = []

    result = MultiAgentOrchestrator(
        engine,
        [],
        risk_discoverer=OpenRiskDiscoverer(),
        decision_analyzer=AcceptingOpenDecisionAnalyzer(),
    ).run(
        run_id="RUN-OPEN-RISK",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    assert result.proposals[0].agent is AgentRole.LEGAL
    assert result.decisions[0].risk_type == "open_123456789abc"
    assert result.decisions[0].selected.selected_chunks == (source,)
    discovery_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY)
    assert discovery_trace.tool_calls[0].tool_name == "open_risk_discovery"
    assert discovery_trace.output_refs == ("open_123456789abc",)
    assert progress_events[0] == "agent:discovery:started"
    assert "agent:discovery:completed" in progress_events


class FailingOpenEvidenceAnalyzer(OwnerAwareAnalyzer):
    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        raise ValueError("professional analysis unavailable")


class FailingOpenDecisionAnalyzer(AcceptingOpenDecisionAnalyzer):
    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        raise ValueError("decision analysis unavailable")


def test_open_risk_fails_closed_when_professional_model_fails() -> None:
    source = _chunk("C-OPEN", "独家授权可以提前终止，核心训练数据没有替代来源。")
    engine = GroundedAnalysisEngine(
        [source],
        [],
        evidence_analyzer=FailingOpenEvidenceAnalyzer(),
    )

    result = MultiAgentOrchestrator(
        engine,
        [],
        risk_discoverer=OpenRiskDiscoverer(),
        decision_analyzer=AcceptingOpenDecisionAnalyzer(),
    ).run(run_id="RUN-OPEN-PROFESSIONAL-FAIL", source_ref="prospectus.pdf")

    assert result.proposals[0].analyzed_risk.analysis.supported is False
    assert result.proposals[0].analyzed_risk.analysis_mode == "open_fail_closed"
    assert len(result.decisions) == 1
    assert result.decisions[0].analysis.supported is False
    assert (
        result.decisions[0].specialist_status
        is SpecialistStatus.EVIDENCE_INSUFFICIENT
    )
    assert result.decisions[0].decision_mode == "excluded_insufficient_evidence"


def test_open_risk_fails_closed_when_decision_model_fails() -> None:
    source = _chunk(
        "C-OPEN",
        "exclusive data license may terminate and disrupt the core training dataset",
    )
    engine = GroundedAnalysisEngine(
        [source],
        [],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )

    result = MultiAgentOrchestrator(
        engine,
        [],
        risk_discoverer=OpenRiskDiscoverer(),
        decision_analyzer=FailingOpenDecisionAnalyzer(),
    ).run(run_id="RUN-OPEN-DECISION-FAIL", source_ref="prospectus.pdf")

    assert result.proposals[0].analyzed_risk.analysis.supported is True
    assert len(result.decisions) == 1
    assert result.decisions[0].decision_mode == "model_adjudication_with_deterministic_fallback"


class FailingOpenRiskDiscoverer(OpenRiskDiscoverer):
    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        raise RuntimeError


class ExhaustedOpenRiskDiscoverer(OpenRiskDiscoverer):
    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        return RiskDiscoveryResult(
            risks=(),
            batch_count=2,
            model_attempts=6,
            batch_errors=("batch 1: timeout", "batch 2: invalid JSON"),
        )


class PartiallyFailedMatchedDiscoverer(OpenRiskDiscoverer):
    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        assert {rule.id for rule in known_rules} == {
            "cash_burn",
            "license_dependency",
        }
        return RiskDiscoveryResult(
            risks=(
                DiscoveredRisk(
                    risk_type="open_abcdef123456",
                    title="持续亏损及经营现金净流出风险",
                    owner=AgentRole.FINANCE,
                    severity=RiskSeverity.HIGH,
                    claim="公司持续亏损且经营活动现金流为负。",
                    confidence=0.9,
                    evidence_chunk_ids=("C-MIXED",),
                    search_queries=("持续亏损和经营现金净流出",),
                    anchor_terms=("cash burn",),
                    uncertainty="融资缓释效果仍需核验。",
                    source_candidate_ids=("R-0001-01", "R-0002-01"),
                    related_catalog_risk_type="cash_burn",
                ),
            ),
            batch_count=3,
            model_attempts=5,
            batch_errors=("batch 3: timeout",),
            raw_candidate_count=2,
            consolidated_candidate_count=1,
            consolidation_mode="semantic_model",
            consolidation_attempts=1,
        )


def test_partial_catalog_fallback_does_not_duplicate_a_semantically_matched_risk() -> None:
    source = _chunk("C-MIXED", "cash burn remains material and license renewal is required")
    rules = [
        _rule("cash_burn", AgentRole.FINANCE, "cash burn"),
        _rule("license_dependency", AgentRole.LEGAL, "license"),
    ]
    engine = GroundedAnalysisEngine(
        [source],
        rules,
        evidence_analyzer=OwnerAwareAnalyzer(),
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(
        engine,
        rules,
        risk_discoverer=PartiallyFailedMatchedDiscoverer(),
    ).run(run_id="RUN-PARTIAL-MERGE", source_ref="prospectus.pdf")

    proposal_risk_types = {proposal.analyzed_risk.rule.id for proposal in result.proposals}
    assert proposal_risk_types == {
        "open_abcdef123456",
        "license_dependency",
    }
    discovery_call = next(
        trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY
    ).tool_calls[0]
    assert discovery_call.output["catalog_merged_count"] == 1
    assert discovery_call.output["catalog_added_count"] == 1
    assert discovery_call.output["raw_candidate_count"] == 2
    assert discovery_call.output["consolidated_candidate_count"] == 1


def test_discovery_failure_is_audited_and_fixed_risks_continue() -> None:
    source = _chunk(
        "C-CASH",
        "公司研发投入较高，现金消耗持续增加，并预计需要继续取得外部融资。",
    )
    rule = _rule("cash_burn", AgentRole.FINANCE, "现金消耗")
    engine = GroundedAnalysisEngine(
        [source],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )
    progress_events: list[str] = []

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        risk_discoverer=FailingOpenRiskDiscoverer(),
    ).run(
        run_id="RUN-DISCOVERY-FALLBACK",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    assert [decision.risk_type for decision in result.decisions] == ["cash_burn"]
    discovery_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY)
    assert discovery_trace.status is TraceStatus.FAILED
    assert discovery_trace.error == "RuntimeError"
    assert discovery_trace.tool_calls[0].error == "RuntimeError"
    assert "agent:discovery:failed" in progress_events
    assert "agent:finance:completed" in progress_events


def test_all_failed_discovery_batches_are_not_reported_as_success() -> None:
    source = _chunk(
        "C-CASH",
        "公司研发投入较高，现金消耗持续增加，并预计需要继续取得外部融资。",
    )
    rule = _rule("cash_burn", AgentRole.FINANCE, "现金消耗")
    engine = GroundedAnalysisEngine(
        [source],
        [rule],
        evidence_analyzer=OwnerAwareAnalyzer(),
    )
    progress_events: list[str] = []

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        risk_discoverer=ExhaustedOpenRiskDiscoverer(),
    ).run(
        run_id="RUN-DISCOVERY-EXHAUSTED",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    discovery_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DISCOVERY)
    assert discovery_trace.status is TraceStatus.FAILED
    assert discovery_trace.error == "全部2个全文扫描批次失败"
    assert "agent:discovery:failed" in progress_events
    assert [decision.risk_type for decision in result.decisions] == ["cash_burn"]


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
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn").model_copy(
        update={"severity": RiskSeverity.CRITICAL}
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
        decision_analyzer=SeverityReviewingDecisionAnalyzer(),
    ).run(run_id="RUN-03696-DECISION", source_ref="prospectus.pdf")

    assert result.decisions[0].analysis.severity is RiskSeverity.MEDIUM
    assert result.decisions[0].analysis.claim.startswith("Evidence supports cash burn")
    assert result.decisions[0].decision_mode == "deepseek_full_adjudication"
    assert result.conflicts[0].kind == "severity_disagreement"
    assert "critical" in result.conflicts[0].detail
    assert "medium" in result.conflicts[0].detail
    decision_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DECISION)
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
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn").model_copy(
        update={"severity": RiskSeverity.CRITICAL}
    )
    progress_events: list[tuple[str, str]] = []
    decision_analyzer = DeepSeekDecisionAnalyzer(
        StreamingGroundedDecisionProvider(),
        temperature=0.1,
        max_output_tokens=1024,
        progress_callback=lambda stage, detail: progress_events.append((stage, detail)),
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
    ).run(
        run_id="RUN-03696-STREAM",
        source_ref="prospectus.pdf",
        progress_callback=lambda stage, detail: progress_events.append((stage, detail)),
    )

    assert (
        "agent:decision:delta",
        '[[batch:1/1]]{"decisions":[{"risk_type":"cash_burn"}]}',
    ) in progress_events
    assert any(stage == "agent:decision:structured" for stage, _detail in progress_events)


def test_deepseek_decision_agent_receives_only_grounded_role_proposals() -> None:
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn").model_copy(
        update={"severity": RiskSeverity.CRITICAL}
    )
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
    assert result.decisions[0].decision_mode == "deepseek_full_adjudication"


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
    rule = _rule("cash_burn", AgentRole.FINANCE, "cash burn").model_copy(
        update={"severity": RiskSeverity.CRITICAL}
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
        decision_analyzer=RejectingDecisionAnalyzer(),
    ).run(run_id="RUN-03696-REJECT", source_ref="prospectus.pdf")

    assert len(result.decisions) == 1
    assert result.decisions[0].risk_type == "cash_burn"
    assert result.conflicts[0].kind == "unresolved_support_conflict"
    assert "机器最终不纳入" in result.conflicts[0].resolution
    decision_trace = next(trace for trace in result.traces if trace.agent is AgentRole.DECISION)
    assert decision_trace.tool_calls[0].output["conflict_count"] == 1
