from __future__ import annotations

from ipo_risk.agents.search_planning import DeepSeekRoleSearchPlanner
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.search import RiskRule


def _rule() -> RiskRule:
    return RiskRule(
        id="cash_burn",
        name="现金消耗与融资依赖",
        owner=AgentRole.FINANCE,
        severity=RiskSeverity.HIGH,
        query="FIXED_QUERY_MUST_NOT_DRIVE_AGENT_PLAN",
        required_keywords=("FIXED_ANCHOR",),
        claim="公司可能存在现金消耗和后续融资需求。",
    )


class SearchPlanningProvider:
    model_name = "deepseek-v4-flash"

    def __init__(self) -> None:
        self.user_prompt = ""
        self.calls = 0

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        on_text_delta: object | None = None,
    ) -> dict[str, object]:
        self.calls += 1
        self.user_prompt = user_prompt
        return {
            "role": "finance",
            "plans": [
                {
                    "risk_type": "cash_burn",
                    "queries": [
                        "经营现金净流出与研发支出的关系",
                        "现有现金能够支持运营多长时间",
                        "未来外部融资需求与持续经营压力",
                    ],
                    "coverage_note": "覆盖现金流、现金跑道和融资依赖。",
                }
            ],
        }


def test_deepseek_role_agent_generates_queries_without_fixed_search_terms() -> None:
    provider = SearchPlanningProvider()
    planner = DeepSeekRoleSearchPlanner(
        provider,
        temperature=0.1,
        max_output_tokens=2048,
    )

    plan = planner.plan([_rule()])

    assert plan.role is AgentRole.FINANCE
    assert plan.plans[0].queries[0] == "经营现金净流出与研发支出的关系"
    assert "现金消耗与融资依赖" in provider.user_prompt
    assert "FIXED_QUERY_MUST_NOT_DRIVE_AGENT_PLAN" not in provider.user_prompt
    assert "FIXED_ANCHOR" not in provider.user_prompt


def test_role_search_planner_uses_the_model_assigned_to_that_agent() -> None:
    default_provider = SearchPlanningProvider()
    finance_provider = SearchPlanningProvider()
    finance_provider.model_name = "deepseek-v4-pro"
    planner = DeepSeekRoleSearchPlanner(
        default_provider,
        temperature=0.1,
        max_output_tokens=2048,
        providers_by_role={AgentRole.FINANCE: finance_provider},
    )

    planner.plan([_rule()])

    assert default_provider.calls == 0
    assert finance_provider.calls == 1
    assert planner.model_name_for(AgentRole.FINANCE) == "deepseek-v4-pro"
