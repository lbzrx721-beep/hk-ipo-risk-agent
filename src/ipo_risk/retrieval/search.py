"""风险规则加载与证据候选筛选。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.calculations import CalculationSpec
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.lexical import BM25Retriever, SearchHit


class RiskRule(BaseModel):
    """一条可审计的词法风险规则。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    owner: AgentRole
    evidence_fields: tuple[str, ...] = ()
    enabled: bool = True
    severity: RiskSeverity = RiskSeverity.MEDIUM
    query: str = ""
    required_keywords: tuple[str, ...] = ()
    claim: str = ""
    max_evidence: int = Field(default=1, ge=1, le=3)
    calculation: CalculationSpec | None = None

    @model_validator(mode="after")
    def enabled_rules_must_be_executable(self) -> RiskRule:
        if self.enabled and not self.query:
            raise ValueError("an enabled risk rule must include query")
        if self.enabled and not self.required_keywords:
            raise ValueError("an enabled risk rule must include required_keywords")
        if self.enabled and not self.claim:
            raise ValueError("an enabled risk rule must include claim")
        return self


def load_risk_rules(path: Path) -> list[RiskRule]:
    """从 risks.yaml 加载带查询与结论模板的风险规则。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = [RiskRule.model_validate(item) for item in raw.get("risks", [])]
    return [rule for rule in rules if rule.enabled]


def _normalized_for_match(text: str) -> str:
    return "".join(text.lower().split())


def select_rule_candidates(
    rule: RiskRule,
    retriever: BM25Retriever,
    *,
    top_k: int = 30,
) -> list[SearchHit]:
    """只保留同时包含全部锚定关键词的 BM25 候选。"""
    required = [_normalized_for_match(keyword) for keyword in rule.required_keywords]
    selected: list[SearchHit] = []
    for hit in retriever.search(rule.query, top_k=top_k):
        text = _normalized_for_match(hit.chunk.text)
        if hit.chunk.is_heading or len(text) < 20:
            continue
        if all(keyword in text for keyword in required):
            selected.append(hit)
        if len(selected) >= rule.max_evidence:
            break
    return selected


__all__ = ["RiskRule", "load_risk_rules", "select_rule_candidates"]
