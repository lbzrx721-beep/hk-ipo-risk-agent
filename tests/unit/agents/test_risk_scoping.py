from __future__ import annotations

from datetime import date
import json

from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator
from ipo_risk.agents.risk_scoping import (
    CompanyRiskScope,
    DeepSeekCompanyRiskScopePlanner,
    RiskApplicability,
)
from ipo_risk.domain import AgentRole
from ipo_risk.retrieval import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryPolicy, retry_model_node_error


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="06651",
        document_id="prospectus_2025-12-18",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=10,
        printed_page_label="1",
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


def _rule(risk_id: str, name: str, claim: str) -> RiskRule:
    return RiskRule(
        id=risk_id,
        name=name,
        owner=AgentRole.FINANCE,
        query=name,
        required_keywords=(name,),
        claim=claim,
    )


class DigitalTwinRiskScopePlanner:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "company-risk-scoping-v1"

    def plan(
        self,
        rules: list[RiskRule],
        chunks: list[DocumentChunk],
    ) -> CompanyRiskScope:
        assert "数字孪生" in "".join(chunk.text for chunk in chunks)
        return CompanyRiskScope(
            industry="数字孪生软件",
            business_model="按项目交付数字孪生产品及解决方案",
            revenue_model="项目验收后确认收入",
            risks=tuple(
                RiskApplicability(
                    risk_type=rule.id,
                    applicable=rule.id == "customer_concentration",
                    rationale=(
                        "客户收入集中适用于项目制软件公司。"
                        if rule.id == "customer_concentration"
                        else "临床试验供应商只适用于生物医药公司。"
                    ),
                )
                for rule in rules
            ),
        )


def test_company_scope_recommends_exclusion_without_auto_deleting_candidate() -> None:
    customer_rule = _rule(
        "customer_concentration",
        "客户集中度",
        "前五大客户收入占比较高，需要核验集中度风险。",
    )
    contaminated_rule = _rule(
        "supplier_concentration",
        "临床试验供应商",
        "公司依赖临床前评估及临床试验供应商。",
    )
    chunks = [
        _chunk(
            "C-06651-001",
            "公司是一家数字孪生软件企业，收入主要来自项目制解决方案。客户集中度",
        )
    ]
    engine = GroundedAnalysisEngine(
        chunks,
        [customer_rule, contaminated_rule],
        lexical_top_k=5,
    )

    result = MultiAgentOrchestrator(
        engine,
        [customer_rule, contaminated_rule],
        risk_scope_planner=DigitalTwinRiskScopePlanner(),
    ).run(run_id="RUN-06651-SCOPE", source_ref="prospectus.pdf")

    assert [proposal.analyzed_risk.rule.id for proposal in result.proposals] == [
        "customer_concentration",
        "supplier_concentration",
    ]
    scope_call = next(
        call for call in result.root_trace.tool_calls if call.tool_name == "company_risk_scoping"
    )
    assert scope_call.output["industry"] == "数字孪生软件"
    assert scope_call.output["excluded_risk_types"] == ["supplier_concentration"]
    assert scope_call.output["auto_removed_risk_types"] == []


class RiskScopeProvider:
    model_name = "deepseek-v4-flash"

    def __init__(self) -> None:
        self.payload: dict[str, object] = {}

    def complete_json(self, **request: object) -> dict[str, object]:
        self.payload = json.loads(str(request["user_prompt"]))
        return {
            "industry": "数字孪生软件",
            "business_model": "按项目交付行业解决方案",
            "revenue_model": "在客户验收项目后确认收入",
            "risks": [
                {
                    "risk_type": "customer_concentration",
                    "applicable": True,
                    "rationale": "项目制业务可能依赖少数大型客户。",
                }
            ],
        }


def test_deepseek_scope_planner_uses_bounded_document_context_not_fixed_queries() -> None:
    rule = RiskRule(
        id="customer_concentration",
        name="客户集中度",
        owner=AgentRole.FINANCE,
        query="FIXED_QUERY_MUST_NOT_ENTER_SCOPE_PROMPT",
        required_keywords=("FIXED_ANCHOR",),
        claim="核验收入是否集中于少数客户，以及集中度变化可能产生的影响。",
    )
    provider = RiskScopeProvider()
    planner = DeepSeekCompanyRiskScopePlanner(
        provider,
        temperature=0.1,
        max_output_tokens=2048,
        max_context_chars=500,
    )
    chunks = [
        _chunk("C-06651-PROFILE", "公司是一家数字孪生软件企业，按项目交付解决方案。"),
        _chunk("C-06651-EXTRA", "不应超过上下文预算。" * 100),
    ]

    scope = planner.plan([rule], chunks)

    assert scope.industry == "数字孪生软件"
    assert "数字孪生软件企业" in str(provider.payload["document_context"])
    assert len(str(provider.payload["document_context"])) <= 500
    assert "FIXED_QUERY_MUST_NOT_ENTER_SCOPE_PROMPT" not in json.dumps(
        provider.payload,
        ensure_ascii=False,
    )


def test_scope_context_prioritizes_business_profile_over_front_matter_noise() -> None:
    rule = _rule(
        "customer_concentration",
        "客户集中度",
        "核验收入是否集中于少数客户。",
    )
    provider = RiskScopeProvider()
    planner = DeepSeekCompanyRiskScopePlanner(
        provider,
        temperature=0.1,
        max_output_tokens=2048,
        max_context_chars=400,
    )
    front_matter = [
        _chunk(f"C-NOISE-{page:02d}", "全球发售及重要提示。" * 12).model_copy(update={"page": page})
        for page in range(1, 12)
    ]
    profile = _chunk(
        "C-BUSINESS-PROFILE",
        "我们是一家数字孪生软件企业，主要业务是按项目交付行业解决方案，收入在客户验收后确认。",
    ).model_copy(update={"page": 20})

    planner.plan([rule], [*front_matter, profile])

    assert "C-BUSINESS-PROFILE" in str(provider.payload["document_context"])


class FlakyRiskScopePlanner(DigitalTwinRiskScopePlanner):
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, rules, chunks) -> CompanyRiskScope:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("temporary malformed model response")
        return super().plan(rules, chunks)


def test_company_scope_retries_malformed_model_output_before_fallback() -> None:
    rule = _rule(
        "customer_concentration",
        "客户集中度",
        "核验收入是否集中于少数客户。",
    )
    chunks = [_chunk("C-06651-RETRY", "数字孪生软件企业，客户集中度较高。")]
    planner = FlakyRiskScopePlanner()
    engine = GroundedAnalysisEngine(chunks, [rule], lexical_top_k=5)

    result = MultiAgentOrchestrator(
        engine,
        [rule],
        risk_scope_planner=planner,
        risk_scope_retry_policy=RetryPolicy(
            max_retries=1,
            initial_seconds=0,
            retry_if=retry_model_node_error,
        ),
    ).run(run_id="RUN-06651-SCOPE-RETRY", source_ref="prospectus.pdf")

    scope_call = next(
        call for call in result.root_trace.tool_calls if call.tool_name == "company_risk_scoping"
    )
    assert planner.calls == 2
    assert scope_call.output["mode"] == "deepseek"
    assert scope_call.output["attempts"] == 2
    assert scope_call.output["retries"] == 1
