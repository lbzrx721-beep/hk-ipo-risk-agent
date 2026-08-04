from __future__ import annotations

from pathlib import Path

from ipo_risk.retrieval import load_risk_rules


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_default_risk_catalog_has_no_company_or_industry_specific_facts() -> None:
    rules = load_risk_rules(_PROJECT_ROOT / "configs" / "risks.yaml")
    catalog_text = "\n".join(
        text
        for rule in rules
        for text in (
            rule.query,
            *rule.query_variants,
            *rule.required_keywords,
            rule.claim,
        )
    )

    for leaked_fact in (
        "临床",
        "臨床",
        "候选药物",
        "候選藥物",
        "患者数据",
        "健康信息",
        "37,204,500",
        "39.29%",
    ):
        assert leaked_fact not in catalog_text

    assert "commercialization_uncertainty" in {rule.id for rule in rules}
