from __future__ import annotations

from threading import Thread

import pytest

from ipo_risk.agents.intervention import (
    RiskScopeCandidate,
    RiskScopeReviewController,
    RiskScopeReviewDecision,
    RiskScopeReviewRequest,
    RiskScopeSelection,
)
from ipo_risk.domain import AgentRole, RiskSeverity


def _request() -> RiskScopeReviewRequest:
    return RiskScopeReviewRequest(
        run_id="RUN-HUMAN-SCOPE",
        industry="企业软件",
        business_model="订阅服务",
        revenue_model="项目与订阅收入",
        candidates=(
            RiskScopeCandidate(
                risk_type="cash_burn",
                title="现金消耗压力",
                owner=AgentRole.FINANCE,
                severity=RiskSeverity.HIGH,
                ai_recommended=True,
                rationale="研发投入较高。",
                review_question="核验现金消耗和融资依赖。",
                source="document_discovery",
            ),
        ),
    )


def _decision() -> RiskScopeReviewDecision:
    return RiskScopeReviewDecision(
        run_id="RUN-HUMAN-SCOPE",
        reviewer="测试复核人",
        selections=(
            RiskScopeSelection(
                risk_type="cash_burn",
                included=True,
                owner=AgentRole.MARKET,
                severity=RiskSeverity.MEDIUM,
                analysis_requirement="补充关注上市前融资窗口。",
            ),
        ),
    )


def test_controller_blocks_graph_node_until_human_decision() -> None:
    controller = RiskScopeReviewController()
    result: list[RiskScopeReviewDecision] = []
    decision = _decision()
    thread = Thread(target=lambda: result.append(controller.request_review(_request())))

    thread.start()
    assert controller.wait_for_request(timeout=1)
    assert thread.is_alive()

    controller.submit(decision)
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result == [decision]
    assert controller.pending_request() is None


def test_controller_rejects_decision_that_does_not_cover_candidates() -> None:
    controller = RiskScopeReviewController()
    thread = Thread(target=lambda: controller.request_review(_request()), daemon=True)
    thread.start()
    assert controller.wait_for_request(timeout=1)

    invalid = _decision().model_copy(update={"selections": ()})
    with pytest.raises(ValueError, match="cover every candidate"):
        controller.submit(invalid)

    controller.submit(_decision())
    thread.join(timeout=1)
