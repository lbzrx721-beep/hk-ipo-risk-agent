from datetime import date

import pytest

from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.agents.review_planning import CompanyReviewPlanner
from ipo_risk.domain import AgentRole
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


def chunk(id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=id,
        company_id="01318",
        document_id="doc",
        source_file="source.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(0, 0, 100, 100),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10,
        available_at=date(2024, 1, 1),
    )


def rule() -> RiskRule:
    return RiskRule(
        id="cash_flow",
        name="现金流",
        owner=AgentRole.FINANCE,
        query="现金流",
        required_keywords=("现金流",),
        claim="核查现金流状况",
    )


class Provider:
    provider_name = "test"
    model_name = "test"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete_json(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


def profile():
    return {
        "facts": [
            {
                "field": "business",
                "value": "化妆品销售",
                "sources": [{"chunk_id": "c1", "quote": "主营化妆品销售"}],
            }
        ],
        "followup_queries": [],
    }


def test_profile_then_questions_preserves_base_checks_without_risk_judgment():
    chunks = [chunk("c1", "业务概览：本公司主营化妆品销售，主要通过线上渠道销售产品。")]
    provider = Provider(
        [
            profile(),
            {
                "checks": [
                    {
                        "title": "渠道结构核查",
                        "owner": "business",
                        "question": "线上渠道销售占比及费用如何变化？",
                        "fact_indices": [0],
                        "queries": ["线上渠道 收入占比", "平台费用 销售费用"],
                        "anchor_terms": ["渠道"],
                    }
                ]
            },
        ]
    )
    engine = GroundedAnalysisEngine(chunks, [rule()])
    result = CompanyReviewPlanner(provider).plan(engine, [rule()])
    assert result.rules[0].id == "cash_flow"
    assert len(result.rules) == 2
    assert result.rules[1].id.startswith("check_")
    assert result.profile.facts[0].sources[0].chunk_id == "c1"
    assert "severity" not in result.model_dump_json(exclude={"rules"})
    assert "confidence" not in result.model_dump_json(exclude={"rules"})
    assert "revenue_model" in result.unknown_fields
    assert len(provider.requests) == 2


def test_rejects_invented_quote():
    data = profile()
    data["facts"][0]["sources"][0]["quote"] = "不存在的公司业务"
    planner = CompanyReviewPlanner(Provider([data]))
    engine = GroundedAnalysisEngine([chunk("c1", "业务概览：本公司主营化妆品销售。")], [rule()])
    with pytest.raises(ValueError, match="quote"):
        planner.plan(engine, [rule()])


def test_profile_retrieval_falls_back_when_vectors_unavailable():
    class OfflineVectors:
        def search(self, query, top_k=20):
            raise ConnectionError("vector unavailable")

    target = chunk("c1", "业务概览：本公司主营化妆品销售，拥有多个线上渠道。")
    engine = GroundedAnalysisEngine([target], [rule()], semantic_retriever=OfflineVectors())
    context = engine.retrieve_profile_context(["主营化妆品"], max_chars=2000)
    assert [c.chunk_id for c in context] == ["c1"]
    assert engine.profile_retrieval_mode == "bm25"


def test_bm25_uses_specialist_queries_when_vectors_unavailable():
    target = chunk("c1", "本公司品牌授权合同期限为十年，届满后需要另行协商续签。")
    other = chunk("c2", "本公司年度现金流稳定，经营业务所得现金流量保持正数。")
    engine = GroundedAnalysisEngine([target, other], [rule()])
    result = engine.analyze_rules(search_queries_by_risk={"cash_flow": ["品牌授权合同期限"]})[0]
    assert result.candidates[0].chunk.chunk_id == "c1"
    assert result.retrieval_mode == "agentic_bm25"


def test_followup_is_bounded_and_merges_facts():
    first = profile()
    first["followup_queries"] = ["品牌授权 期限"]
    second = {
        "facts": [
            {
                "field": "key_assets",
                "value": "品牌通过授权使用",
                "sources": [{"chunk_id": "c2", "quote": "核心品牌通过授权使用"}],
            }
        ],
        "followup_queries": ["不应再执行第三轮"],
    }
    provider = Provider([first, second, {"checks": []}])
    engine = GroundedAnalysisEngine(
        [
            chunk("c1", "业务概览：本公司主营化妆品销售，销售方式以线上渠道为主。"),
            chunk("c2", "本公司核心品牌通过授权使用，品牌授权期限为十年，到期协商续签。"),
        ],
        [rule()],
    )
    result = CompanyReviewPlanner(provider).plan(engine, [rule()])
    assert len(result.profile.facts) == 2
    assert len(result.retrieval_rounds) == 2
    assert len(provider.requests) == 3
    assert all("不应再执行" not in q for r in result.retrieval_rounds for q in r.queries)


def test_invalid_question_preserves_profile_and_base_checks():
    provider = Provider(
        [
            profile(),
            {
                "checks": [
                    {
                        "title": "现金流",
                        "owner": "finance",
                        "question": "经营现金流是否改善？",
                        "fact_indices": [99],
                        "queries": ["现金流"],
                        "anchor_terms": ["现金流"],
                    }
                ]
            },
        ]
    )
    engine = GroundedAnalysisEngine([chunk("c1", "业务概览：本公司主营化妆品销售。")], [rule()])
    result = CompanyReviewPlanner(provider).plan(engine, [rule()])
    assert result.profile.facts
    assert len(result.rules) == 1
    assert result.rules[0].id == rule().id
    assert result.rules[0].requires_model_verification
    assert not result.checks
    assert result.warnings


def test_context_budget_and_no_random_sampling():
    target = chunk("c1", "本公司主营化妆品销售。" * 10)
    unrelated = chunk("c2", "完全无关的文字。" * 10)
    engine = GroundedAnalysisEngine([target, unrelated], [rule()])
    assert not engine.retrieve_profile_context(["主营化妆品"], max_chars=10)
    assert not engine.retrieve_profile_context(["zxyw98765"], max_chars=2000)
    assert engine.retrieve_profile_context(["主营化妆品"], neighbor_radius=0) == (target,)


def test_review_questions_require_model_verification_not_keyword_confirmation():
    planned = RiskRule(
        id="check_aabbcc",
        name="品牌授权核查",
        owner=AgentRole.LEGAL,
        query="品牌授权",
        required_keywords=("品牌授权",),
        claim="品牌授权何时到期？",
    )
    engine = GroundedAnalysisEngine(
        [chunk("c1", "本公司品牌授权合同期限为十年，届满后需要另行协商续签。")], [planned]
    )
    result = engine.analyze_rules()[0]
    assert result.candidates
    assert result.analysis.supported is False
    assert not result.analysis.claim


def test_orchestrator_dispatches_independent_full_document_review():
    from ipo_risk.agents.analysis import EvidenceAnalysis
    from ipo_risk.agents.orchestrator import MultiAgentOrchestrator

    class Analyzer:
        provider_name = "test"
        model_name = "test"
        prompt_version = "test"

        def analyze(self, rule, candidates):
            return EvidenceAnalysis(
                supported=False,
                claim="",
                severity=rule.severity,
                confidence=0,
                evidence_chunk_ids=(),
                uncertainty="不构成风险",
            )

    engine = GroundedAnalysisEngine(
        [
            chunk("c1", "业务概览：本公司主营化妆品销售。"),
            chunk("c2", "本公司现金流充裕，经营活动现金流量净额增加，银行存款稳定。"),
        ],
        [rule()],
        evidence_analyzer=Analyzer(),
    )
    planner = CompanyReviewPlanner(Provider([profile(), {"checks": []}]))
    events = []
    result = MultiAgentOrchestrator(engine, [rule()], review_planner=planner).run(
        run_id="run-plan",
        source_ref="source.pdf",
        progress_callback=lambda s, d: events.append((s, d)),
    )
    assert result.proposals[0].analyzed_risk.rule.id == "cash_flow"
    assert "c2" in [c.chunk.chunk_id for c in result.proposals[0].analyzed_risk.candidates]
    assert "with_discovery_seed" not in result.proposals[0].analyzed_risk.retrieval_mode
    assert any(stage == "company_review_plan" for stage, _ in events)


def test_orchestrator_planning_failure_keeps_baseline_and_emits_degraded_event():
    from ipo_risk.agents.orchestrator import MultiAgentOrchestrator

    engine = GroundedAnalysisEngine([chunk("c1", "业务概览：本公司主营化妆品销售。")], [rule()])
    planner = CompanyReviewPlanner(Provider([{"bad": "schema"}]))
    events = []
    result = MultiAgentOrchestrator(engine, [rule()], review_planner=planner).run(
        run_id="run-failed",
        source_ref="source.pdf",
        progress_callback=lambda s, d: events.append((s, d)),
    )
    assert any(s == "agent:discovery:degraded" for s, _ in events)
    assert result.proposals[0].analyzed_risk.rule.id == "cash_flow"
