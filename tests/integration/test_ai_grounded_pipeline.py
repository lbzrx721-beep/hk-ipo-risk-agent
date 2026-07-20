from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import DecisionBatch, DecisionVerdict
from ipo_risk.agents.search_planning import PlannedRiskSearch, RoleSearchPlan
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.pipeline import analyze_prospectus
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


class TinyEmbeddingProvider:
    model_name = "test-semantic"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class TinyDeepSeekAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "evidence-analysis-v1"

    def analyze(
        self,
        rule: RiskRule,
        candidates: list[DocumentChunk],
    ) -> EvidenceAnalysis:
        return EvidenceAnalysis(
            supported=True,
            claim="The company has persistent research cash outflow and financing dependence.",
            severity=RiskSeverity.HIGH,
            confidence=0.88,
            evidence_chunk_ids=(candidates[0].chunk_id,),
            uncertainty="Future financing needs remain uncertain.",
        )


class TinyDeepSeekDecisionAnalyzer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "decision-adjudication-v1"

    def adjudicate(self, proposals: list[object]) -> DecisionBatch:
        proposal = proposals[0]
        analyzed = proposal.analyzed_risk
        return DecisionBatch(
            decisions=(
                DecisionVerdict(
                    risk_type=analyzed.rule.id,
                    accepted=True,
                    claim="Decision agent confirms persistent research cash outflow.",
                    severity=RiskSeverity.HIGH,
                    confidence=0.9,
                    evidence_chunk_ids=analyzed.analysis.evidence_chunk_ids,
                    uncertainty="Human review remains required.",
                    rationale="The cited passage directly supports the specialist proposal.",
                    source_proposal_ids=(proposal.proposal_id,),
                ),
            )
        )


class TinySearchPlanner:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "role-search-planning-v1"

    def plan(self, rules: list[RiskRule], *, feedback: object = None) -> RoleSearchPlan:
        return RoleSearchPlan(
            role=AgentRole.FINANCE,
            plans=(
                PlannedRiskSearch(
                    risk_type=rules[0].id,
                    queries=("Which disclosure shows persistent research cash outflow?",),
                    coverage_note="Covers cash outflow and financing dependence.",
                ),
            ),
        )


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 100),
        (
            "The company reported persistent cash outflow from research activities "
            "and expects further financing."
        ),
    )
    document.save(path)
    document.close()


def test_pipeline_records_hybrid_retrieval_and_deepseek_analysis(tmp_path: Path) -> None:
    pdf_path = tmp_path / "prospectus.pdf"
    rules_path = tmp_path / "risks.yaml"
    _write_pdf(pdf_path)
    rules_path.write_text(
        """
risks:
  - id: cash_burn
    name: Cash burn
    owner: finance
    severity: high
    query: cash outflow research financing
    required_keywords: [cash outflow, research]
    claim: The company has continuing cash outflow.
""".strip(),
        encoding="utf-8",
    )

    progress_events: list[str] = []
    artifacts = analyze_prospectus(
        pdf_path=pdf_path,
        company_id="03696",
        company_name="Test Company",
        company_folder="03696_test",
        document_id="prospectus_2025-12-18",
        available_at=date(2025, 12, 18),
        project_root=tmp_path,
        interim_root=tmp_path / "interim",
        artifacts_root=tmp_path / "artifacts",
        risk_rules_path=rules_path,
        min_findings=1,
        embedding_provider=TinyEmbeddingProvider(),
        evidence_analyzer=TinyDeepSeekAnalyzer(),
        decision_analyzer=TinyDeepSeekDecisionAnalyzer(),
        search_planner=TinySearchPlanner(),
        lexical_top_k=5,
        semantic_top_k=5,
        rerank_top_k=3,
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    finding = artifacts.report.risk_findings[0]
    assert finding.claim.startswith("Decision agent confirms")
    assert finding.confidence == 0.9
    assert finding.uncertainty == "Human review remains required."
    assert finding.created_by is AgentRole.FINANCE
    assert {trace.agent for trace in artifacts.report.agent_traces} >= {
        AgentRole.ORCHESTRATOR,
        AgentRole.FINANCE,
        AgentRole.DECISION,
    }
    calls = tuple(
        call for trace in artifacts.report.agent_traces for call in trace.tool_calls
    )
    retrieval_call = next(call for call in calls if call.tool_name == "evidence_retrieval")
    analysis_call = next(call for call in calls if call.tool_name == "evidence_analysis")
    assert retrieval_call.output["mode"] == "agentic_semantic"
    assert retrieval_call.output["embedding_retries"] == 0
    assert analysis_call.output["mode"] == "deepseek"
    assert analysis_call.output["model"] == "deepseek-v4-flash"
    assert analysis_call.output["prompt_version"] == "evidence-analysis-v1"
    assert analysis_call.output["attempts"] == 1
    assert analysis_call.output["retries"] == 0
    search_call = next(call for call in calls if call.tool_name == "agent_search_planning")
    assert search_call.output["mode"] == "deepseek"
    assert search_call.output["queries_by_risk"] == {
        "cash_burn": ["Which disclosure shows persistent research cash outflow?"]
    }
    decision_call = next(call for call in calls if call.tool_name == "risk_adjudication")
    assert decision_call.output["mode"] == "deepseek"
    assert decision_call.output["prompt_version"] == "decision-adjudication-v1"
    assert artifacts.embedding_dir is not None
    assert (artifacts.embedding_dir / "vectors.npy").is_file()
    assert (artifacts.embedding_dir / "metadata.json").is_file()
    assert (artifacts.embedding_dir / "manifest.json").is_file()
    assert progress_events == [
        "parsing_started",
        "parsing_completed",
        "agents_started",
        "agent:finance:started",
        "agent:finance:search_planned",
        "agent:finance:prompt",
        "agent:finance:progress",
        "agent:finance:completed",
        "agent:decision:started",
        "agent:decision:completed",
        "agents_completed",
        "evidence_started",
        "completed",
    ]
