from pathlib import Path


def test_prompt_directory_contains_only_active_runtime_templates() -> None:
    project_root = Path(__file__).resolve().parents[3]
    prompt_root = project_root / "prompts"
    actual = {path.relative_to(prompt_root).as_posix() for path in prompt_root.rglob("*.md")}

    assert actual == {
        "README.md",
        "decision/decision_adjudication_v4.md",
        "decision/evidence_analysis_v5.md",
        "discovery/company_risk_scope_v4.md",
        "discovery/document_risk_discovery_v6.md",
        "discovery/risk_consolidation_v3.md",
        "discovery/vector_query_planning_v2.md",
        "roles/business_v2.md",
        "roles/finance_v2.md",
        "roles/legal_v2.md",
        "roles/market_v2.md",
        "search/role_search_planning_v4.md",
    }


def test_active_prompts_define_evidence_quality_and_dynamic_topic_boundaries() -> None:
    project_root = Path(__file__).resolve().parents[3]
    prompt_root = project_root / "prompts"
    discovery = (prompt_root / "discovery/document_risk_discovery_v6.md").read_text(
        encoding="utf-8"
    )
    specialist = (prompt_root / "decision/evidence_analysis_v5.md").read_text(
        encoding="utf-8"
    )
    decision = (prompt_root / "decision/decision_adjudication_v4.md").read_text(
        encoding="utf-8"
    )

    assert len(discovery) > 3_000
    assert all(
        marker in discovery
        for marker in ("不是固定风险目录", "表格", "反证", "severity", "confidence")
    )
    assert all(
        marker in specialist
        for marker in ("主体", "期间", "比例分母", "needs_more_evidence", "证据冲突")
    )
    assert all(
        marker in decision
        for marker in ("全部有证据", "accepted=false", "人工", "evidence_chunk_ids")
    )


def test_each_specialist_role_has_a_detailed_distinct_charter() -> None:
    project_root = Path(__file__).resolve().parents[3]
    role_root = project_root / "prompts/roles"
    charters = {
        role: (role_root / f"{role}_v2.md").read_text(encoding="utf-8")
        for role in ("finance", "legal", "business", "market")
    }

    assert all(len(content) > 700 for content in charters.values())
    assert len(set(charters.values())) == 4
    assert "比例分母" in charters["finance"]
    assert "触发条件" in charters["legal"]
    assert "商业化阶段" in charters["business"]
    assert "上市后价格" in charters["market"]
