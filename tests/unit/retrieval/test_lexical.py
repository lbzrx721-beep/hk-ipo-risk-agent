from __future__ import annotations

from datetime import date

from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.parser.document_parser import PageResult, TextBlock
from ipo_risk.retrieval.chunker import DocumentChunk, build_page_chunks
from ipo_risk.retrieval.lexical import BM25Retriever
from ipo_risk.retrieval.search import RiskRule, select_rule_candidates


def _chunk(chunk_id: str, page: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        source_hash="a" * 64,
        page=page,
        printed_page_label=str(page - 9),
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


def test_build_page_chunks_preserves_page_label_and_excludes_footer() -> None:
    page = PageResult(
        page=70,
        width=595.0,
        height=842.0,
        has_text_layer=True,
        text_blocks=[
            TextBlock(
                page=70,
                x0=283.0,
                y0=749.0,
                x1=312.0,
                y1=761.0,
                text="– 61 –",
                font_size=10.5,
                block_no=1,
            ),
            TextBlock(
                page=70,
                x0=70.0,
                y0=70.0,
                x1=525.0,
                y1=82.0,
                text="我們的經營產生負現金流量。",
                font_size=10.5,
                block_no=2,
            ),
        ],
    )

    chunks = build_page_chunks(
        page,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        source_hash="a" * 64,
        available_at=date(2025, 12, 18),
    )

    assert len(chunks) == 1
    assert chunks[0].printed_page_label == "61"
    assert chunks[0].page == 70
    assert "– 61 –" not in chunks[0].text


def test_bm25_retrieves_relevant_traditional_chinese_evidence() -> None:
    chunks = [
        _chunk("C1", 30, "我們的經營產生負現金流量，現金流出主要來自研發活動。"),
        _chunk("C2", 108, "五大客戶佔總收入九成以上，損失主要客戶可能造成重大不利影響。"),
        _chunk("C3", 200, "董事會負責公司治理及會議安排。"),
    ]

    retriever = BM25Retriever(chunks)
    hits = retriever.search("五大客戶 收入 客戶集中 重大不利影響", top_k=2)

    assert hits[0].chunk.chunk_id == "C2"
    assert hits[0].score > hits[1].score


def test_rule_candidate_requires_grounding_keywords() -> None:
    chunks = [
        _chunk("C1", 30, "我們的經營產生負現金流量，現金流出主要來自研發活動。"),
        _chunk("C2", 31, "融資活動可能受市場情緒影響。"),
    ]
    rule = RiskRule(
        id="cash_burn",
        name="现金消耗与覆盖期",
        owner=AgentRole.FINANCE,
        severity=RiskSeverity.HIGH,
        query="經營活動所用現金 負現金流量 研發活動",
        required_keywords=("負現金流量", "研發活動"),
        claim="公司在业绩记录期存在经营现金净流出。",
    )

    selected = select_rule_candidates(rule, BM25Retriever(chunks), top_k=5)

    assert [candidate.chunk.chunk_id for candidate in selected] == ["C1"]
