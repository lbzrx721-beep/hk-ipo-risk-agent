"""Immutable model routing contract for the six user-facing Agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast


AgentModel = Literal["deepseek_flash", "deepseek_pro", "kimi"]
AgentModelAssignments = dict[str, AgentModel]

AGENT_ROLE_KEYS = (
    "discovery",
    "finance",
    "legal",
    "business",
    "market",
    "decision",
)

MODEL_API_NAMES: dict[AgentModel, str] = {
    "deepseek_flash": "deepseek-v4-flash",
    "deepseek_pro": "deepseek-v4-pro",
    "kimi": "kimi-k3",
}

MODEL_DISPLAY_NAMES: dict[AgentModel, str] = {
    "deepseek_flash": "DeepSeek Flash",
    "deepseek_pro": "DeepSeek Pro",
    "kimi": "Kimi K3",
}

BALANCED_AGENT_MODELS: AgentModelAssignments = {
    "discovery": "deepseek_flash",
    "finance": "deepseek_pro",
    "legal": "deepseek_pro",
    "business": "deepseek_pro",
    "market": "deepseek_flash",
    "decision": "kimi",
}


def normalize_agent_models(
    assignments: Mapping[str, str] | None,
    *,
    legacy_reasoning_model: str = "deepseek_flash",
) -> AgentModelAssignments:
    """Validate a complete mapping while preserving legacy single-model requests."""
    legacy = "deepseek_flash" if legacy_reasoning_model == "deepseek" else legacy_reasoning_model
    if legacy not in MODEL_API_NAMES:
        raise ValueError(f"unsupported reasoning model: {legacy_reasoning_model}")
    legacy_model = cast(AgentModel, legacy)
    defaults: AgentModelAssignments = {
        "discovery": legacy_model,
        "finance": "deepseek_flash",
        "legal": "deepseek_flash",
        "business": "deepseek_flash",
        "market": "deepseek_flash",
        "decision": legacy_model,
    }
    if assignments is None:
        return defaults
    unknown = sorted(set(assignments) - set(AGENT_ROLE_KEYS))
    missing = sorted(set(AGENT_ROLE_KEYS) - set(assignments))
    if unknown or missing:
        raise ValueError(
            f"agent model assignments must contain exactly six roles; "
            f"missing={missing}, unknown={unknown}"
        )
    normalized: AgentModelAssignments = {}
    for role in AGENT_ROLE_KEYS:
        model = str(assignments[role]).strip()
        if model == "deepseek":
            model = "deepseek_flash"
        if model not in MODEL_API_NAMES:
            raise ValueError(f"unsupported model for {role}: {model}")
        normalized[role] = cast(AgentModel, model)
    return normalized


def model_api_name(model: AgentModel) -> str:
    return MODEL_API_NAMES[model]
