from __future__ import annotations

from datetime import date
from pathlib import Path
from threading import Event

import fitz
import numpy as np

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.decision import DecisionBatch, DecisionVerdict
from ipo_risk.agents.discovery import DiscoveredRisk, RiskDiscoveryResult
from ipo_risk.agents.risk_scoping import CompanyRiskScope, RiskApplicability
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


class RecordingEmbeddingProvider(TinyEmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed(texts)


class DiscoveryAwareEmbeddingProvider(TinyEmbeddingProvider):
    def __init__(self, discovery_started: Event) -> None:
        self.discovery_started = discovery_started
        self.discovery_observed_while_indexing = False
        self._index_call_seen = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._index_call_seen:
            self._index_call_seen = True
            self.discovery_observed_while_indexing = self.discovery_started.wait(
                timeout=3
            )
        return super().embed(texts)


class RecordingChromaWriter:
    def __init__(self) -> None:
        self.document_vectors: np.ndarray | None = None

    def add_chunks(
        self,
        chunks,
        *,
        skip_existing_document=True,
        precomputed_embeddings=None,
    ) -> int:
        del skip_existing_document
        self.document_vectors = precomputed_embeddings
        return len(chunks)


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
        return DecisionBatch(
            decisions=tuple(
                DecisionVerdict(
                    risk_type=proposal.analyzed_risk.rule.id,
                    accepted=True,
                    claim=(
                        "Decision agent confirms "
                        f"{proposal.analyzed_risk.analysis.claim}"
                    ),
                    severity=RiskSeverity.HIGH,
                    confidence=0.9,
                    evidence_chunk_ids=(
                        proposal.analyzed_risk.analysis.evidence_chunk_ids
                    ),
                    uncertainty="Human review remains required.",
                    rationale=(
                        "The cited passage directly supports the specialist proposal."
                    ),
                    source_proposal_ids=(proposal.proposal_id,),
                )
                for proposal in proposals
            ),
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


class TinyRiskScopePlanner:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "company-risk-scoping-v1"

    def plan(self, rules, chunks) -> CompanyRiskScope:
        assert chunks
        return CompanyRiskScope(
            industry="Enterprise software",
            business_model="Project delivery and software services",
            revenue_model="Project acceptance revenue",
            risks=tuple(
                RiskApplicability(
                    risk_type=rule.id,
                    applicable=True,
                    rationale="This is a cross-industry financial risk.",
                )
                for rule in rules
            ),
        )


class TinyOpenRiskDiscoverer:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"
    prompt_version = "open-risk-discovery-v1"

    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        assert tuple(rule.id for rule in known_rules) == ("cash_burn",)
        source = next(chunk for chunk in chunks if "exclusive data license" in chunk.text)
        return RiskDiscoveryResult(
            risks=(
                DiscoveredRisk(
                    risk_type="open_abcdef123456",
                    title="Exclusive data license termination",
                    owner=AgentRole.LEGAL,
                    severity=RiskSeverity.HIGH,
                    claim="The core dataset depends on a terminable exclusive license.",
                    confidence=0.84,
                    evidence_chunk_ids=(source.chunk_id,),
                    search_queries=("核心数据独家授权终止",),
                    anchor_terms=("exclusive",),
                    uncertainty="Alternative datasets require verification.",
                ),
            ),
            batch_count=1,
            model_attempts=1,
        )


class SignalingOpenRiskDiscoverer(TinyOpenRiskDiscoverer):
    def __init__(self, started: Event) -> None:
        self.started = started

    def discover(self, chunks, known_rules) -> RiskDiscoveryResult:
        self.started.set()
        return super().discover(chunks, known_rules)


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
    page.insert_text(
        (72, 125),
        "The exclusive data license may terminate and disrupt model training.",
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
        risk_scope_planner=TinyRiskScopePlanner(),
        lexical_top_k=5,
        semantic_top_k=5,
        rerank_top_k=3,
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    finding = artifacts.report.risk_findings[0]
    assert finding.claim.startswith(
        "Decision agent confirms The company has persistent research cash outflow"
    )
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
    assert decision_call.output["mode"] == "deepseek_full_adjudication"
    assert decision_call.output["prompt_version"] == "decision-adjudication-v1"
    scope_call = next(call for call in calls if call.tool_name == "company_risk_scoping")
    assert scope_call.output["industry"] == "Enterprise software"
    assert scope_call.output["applicable_risk_types"] == ["cash_burn"]
    assert artifacts.embedding_dir is not None
    assert (artifacts.embedding_dir / "vectors.npy").is_file()
    assert (artifacts.embedding_dir / "metadata.json").is_file()
    assert (artifacts.embedding_dir / "manifest.json").is_file()
    assert progress_events[0:4] == [
        "parsing_started",
        "parsing_progress",
        "parsing_completed",
        "agents_started",
    ]
    assert "embedding_started" in progress_events
    assert "embedding_completed" in progress_events
    assert "agent:finance:completed" in progress_events
    assert progress_events[-1] == "completed"


def test_vector_index_and_risk_discovery_start_in_parallel(tmp_path: Path) -> None:
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
    discovery_started = Event()
    embedding = DiscoveryAwareEmbeddingProvider(discovery_started)

    analyze_prospectus(
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
        embedding_provider=embedding,
        evidence_analyzer=TinyDeepSeekAnalyzer(),
        decision_analyzer=TinyDeepSeekDecisionAnalyzer(),
        risk_discoverer=SignalingOpenRiskDiscoverer(discovery_started),
        lexical_top_k=5,
        semantic_top_k=5,
        rerank_top_k=3,
    )

    assert embedding.discovery_observed_while_indexing


def test_pipeline_adds_grounded_open_risk_and_discovery_trace(tmp_path: Path) -> None:
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
        evidence_analyzer=TinyDeepSeekAnalyzer(),
        decision_analyzer=TinyDeepSeekDecisionAnalyzer(),
        risk_discoverer=TinyOpenRiskDiscoverer(),
        lexical_top_k=5,
        rerank_top_k=3,
    )

    open_finding = next(
        finding
        for finding in artifacts.report.risk_findings
        if finding.risk_type == "open_abcdef123456"
    )
    assert open_finding.created_by is AgentRole.LEGAL
    discovery_trace = next(
        trace
        for trace in artifacts.report.agent_traces
        if trace.agent is AgentRole.DISCOVERY
    )
    assert discovery_trace.tool_calls[0].tool_name == "open_risk_discovery"
    assert discovery_trace.output_refs == ("open_abcdef123456",)


def test_hybrid_vector_backend_uses_memory_then_persists_same_vectors(
    tmp_path: Path,
) -> None:
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
    claim: Verify continuing cash outflow.
""".strip(),
        encoding="utf-8",
    )
    provider = RecordingEmbeddingProvider()
    writer = RecordingChromaWriter()
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
        embedding_provider=provider,
        evidence_analyzer=TinyDeepSeekAnalyzer(),
        decision_analyzer=TinyDeepSeekDecisionAnalyzer(),
        vector_store_backend="hybrid",
        chroma_persist_dir=tmp_path / "chroma",
        chroma_writer_factory=lambda _provider, _path, _name: writer,
        lexical_top_k=5,
        semantic_top_k=5,
        rerank_top_k=3,
        progress_callback=lambda stage, detail: progress_events.append(stage),
    )

    task = artifacts.vector_persistence_task
    assert task is not None
    assert task.wait(timeout=2)
    assert task.snapshot().status == "ready"
    assert writer.document_vectors is not None
    assert writer.document_vectors.shape == (
        artifacts.report.source_document.chunk_count,
        provider.dimensions,
    )
    document_embedding_calls = [
        call
        for call in provider.calls
        if any("persistent cash outflow" in text for text in call)
    ]
    assert len(document_embedding_calls) == 1
    assert "vector_persistence_started" in progress_events
    assert "vector_persistence_completed" in progress_events
