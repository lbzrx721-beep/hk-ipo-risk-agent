from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.agents.analysis import EvidenceAnalysis
from ipo_risk.agents.grounded import AnalyzedRisk, RetrievalCandidate
from ipo_risk.agents.orchestrator import AdjudicatedRisk, AgentProposal
from ipo_risk.agents.specialist import SpecialistStatus
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.parser.document_parser import PageResult, TextBlock
from ipo_risk.pipeline import _decision_evidence_chunks, _prepare_prospectus_source
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.storage.files import sha256_file


def _debate_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="01318",
        document_id="prospectus",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(10.0, 20.0, 300.0, 60.0),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10.0,
        available_at=date(2024, 12, 2),
    )


def _debate_analysis(rule: RiskRule, chunk: DocumentChunk, confidence: float) -> AnalyzedRisk:
    return AnalyzedRisk(
        rule=rule,
        analysis=EvidenceAnalysis(
            supported=True,
            claim=chunk.text,
            severity=RiskSeverity.HIGH,
            confidence=confidence,
            evidence_chunk_ids=(chunk.chunk_id,),
            uncertainty="以招股书当前披露为限。",
        ),
        candidates=(RetrievalCandidate(chunk=chunk, score=1.0, rank=1),),
        selected_chunks=(chunk,),
        retrieval_mode="test",
        analysis_mode="test",
        analysis_attempts=1,
        search_queries=(rule.query,),
        specialist_status=SpecialistStatus.SUPPORTED,
    )


def test_debate_evidence_uses_all_cited_proposals_without_losing_original_cards() -> None:
    rule = RiskRule(
        id="license_dependency",
        name="牌照依赖",
        owner=AgentRole.LEGAL,
        query="牌照",
        required_keywords=("牌照",),
        claim="业务依赖牌照。",
    )
    original = _debate_analysis(rule, _debate_chunk("C-ORIGINAL", "原始风险证据"), 0.95)
    revised = _debate_analysis(rule, _debate_chunk("C-01318-0025-005", "辩论补证证据"), 0.88)
    proposals = (
        AgentProposal("P-original", AgentRole.LEGAL, original),
        AgentProposal("P-debate", AgentRole.LEGAL, revised),
    )
    decision = AdjudicatedRisk(
        risk_type=rule.id,
        selected=original,
        analysis=EvidenceAnalysis(
            supported=True,
            claim="补证后收窄风险结论。",
            severity=RiskSeverity.MEDIUM,
            confidence=0.9,
            evidence_chunk_ids=("C-01318-0025-005",),
            uncertainty="以辩论补证结果为限。",
        ),
        source_proposal_ids=("P-original", "P-debate"),
        rationale="只修正存在歧义的风险。",
        decision_mode="targeted_debate",
        decision_attempts=2,
    )

    assert _decision_evidence_chunks(decision, proposals) == (revised.selected_chunks[0],)


def test_prepared_prospectus_reuses_versioned_parser_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "prospectus.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    calls = 0

    def fake_parse_document(path: Path, *, min_chars: int):
        nonlocal calls
        assert path == pdf_path
        assert min_chars == 80
        calls += 1
        yield PageResult(
            page=1,
            width=595.0,
            height=842.0,
            has_text_layer=True,
            text_blocks=[
                TextBlock(
                    page=1,
                    x0=72.0,
                    y0=100.0,
                    x1=420.0,
                    y1=120.0,
                    text="公司存在客户集中度风险，前五大客户收入占比较高。",
                    font_size=10.0,
                )
            ],
            raw_text="公司存在客户集中度风险，前五大客户收入占比较高。",
            char_count=24,
        )

    monkeypatch.setattr(
        "ipo_risk.pipeline.parse_document",
        fake_parse_document,
    )
    arguments = {
        "pdf_path": pdf_path,
        "company_id": "00001",
        "company_folder": "00001_测试公司",
        "document_id": "doc-test",
        "source_file": "data/raw/prospectus/00001_测试公司/prospectus.pdf",
        "source_hash": sha256_file(pdf_path),
        "available_at": date(2025, 1, 1),
        "interim_root": tmp_path / "interim",
        "artifacts_root": tmp_path / "artifacts",
        "min_text_chars_before_ocr": 80,
        "progress_callback": None,
    }

    first = _prepare_prospectus_source(**arguments)
    second = _prepare_prospectus_source(**arguments)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1
    assert second.chunks == first.chunks
