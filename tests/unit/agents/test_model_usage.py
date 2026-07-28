from __future__ import annotations

import pytest

from ipo_risk.agents.usage import (
    ModelBudget,
    ModelBudgetExceededError,
    ModelPricing,
    ModelUsageLedger,
)


def test_usage_ledger_counts_tokens_cost_and_pauses_before_next_call() -> None:
    ledger = ModelUsageLedger(
        budget=ModelBudget(max_calls=3, max_input_tokens=100, max_cost_rmb=1.0),
    )
    pricing = ModelPricing(
        input_per_million_rmb=10_000,
        output_per_million_rmb=10_000,
    )

    ledger.begin_call(model_name="test", request_tag="discovery")
    ledger.complete_call(input_tokens=60, output_tokens=10, pricing=pricing)
    snapshot = ledger.snapshot()

    assert snapshot.calls == 1
    assert snapshot.input_tokens == 60
    assert snapshot.estimated_cost_rmb == 0.7
    assert snapshot.budget_exhausted is False

    ledger.begin_call(model_name="test", request_tag="specialist")
    ledger.complete_call(input_tokens=50, output_tokens=10, pricing=pricing)

    with pytest.raises(ModelBudgetExceededError, match="预算"):
        ledger.begin_call(model_name="test", request_tag="decision")
