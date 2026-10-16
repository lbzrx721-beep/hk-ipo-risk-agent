from __future__ import annotations

import pytest

from ipo_risk.agents.model_routing import (
    BALANCED_AGENT_MODELS,
    model_api_name,
    normalize_agent_models,
)


def test_complete_six_agent_assignment_is_preserved() -> None:
    assert normalize_agent_models(BALANCED_AGENT_MODELS) == BALANCED_AGENT_MODELS
    assert model_api_name(BALANCED_AGENT_MODELS["decision"]) == "kimi-k3"


def test_legacy_single_model_only_controls_discovery_and_decision() -> None:
    assert normalize_agent_models(None, legacy_reasoning_model="deepseek_pro") == {
        "discovery": "deepseek_pro",
        "finance": "deepseek_flash",
        "legal": "deepseek_flash",
        "business": "deepseek_flash",
        "market": "deepseek_flash",
        "decision": "deepseek_pro",
    }


def test_partial_or_unknown_agent_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly six roles"):
        normalize_agent_models({"discovery": "kimi"})
    with pytest.raises(ValueError, match="unsupported model"):
        normalize_agent_models({**BALANCED_AGENT_MODELS, "decision": "unknown"})
