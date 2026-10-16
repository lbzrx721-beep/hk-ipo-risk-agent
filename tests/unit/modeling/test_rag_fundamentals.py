from __future__ import annotations

from datetime import date

import pytest

from ipo_risk.modeling.rag_fundamentals import (
    RagFundamentalExtractor,
    RagExtractionResponse,
    RagFieldExtraction,
    _MODEL_FIELD_NAMES,
    _anchor_search,
    _cornerstone_midpoint_ratio,
    _downgrade_invalid_fields,
    _expand_hits,
    _pipeline_stage_score,
    _validate_response,
)
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import SearchHit
from ipo_risk.resilience import RetryPolicy


def _chunk(
    index: int,
    *,
    text: str,
    kind: str = "text",
    page: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"C-00001-0001-{index:03d}",
        company_id="00001",
        document_id="doc-1",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=page,
        bbox=(1.0, float(index), 100.0, float(index + 1)),
        text=text,
        font_size=10.0,
        available_at=date(2024, 1, 1),
        chunk_kind=kind,
        table_id="T-1" if kind == "table" else None,
        table_cell_refs=("T-1:R1:C1",) if kind == "table" else (),
        table_cell_values=(text,) if kind == "table" else (),
    )


def _valid_response(source_id: str) -> RagExtractionResponse:
    fields = []
    numeric_financials = {
        "fundamental_revenue_latest_native": 500.0,
        "fundamental_operating_cash_flow_latest_native": 45.0,
        "fundamental_ending_cash_latest_native": 120.0,
        "fundamental_net_profit_latest_native": -20.0,
        "fundamental_research_development_latest_native": 30.0,
    }
    for name, value in numeric_financials.items():
        fields.append(
            RagFieldExtraction(
                name=name,
                status="found",
                value=value,
                currency="RMB",
                unit="million_native_currency",
                period_end="2023-12-31",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            )
        )
    for name in (
        "customer_largest_ratio_pct",
        "customer_top5_ratio_pct",
        "supplier_largest_ratio_pct",
        "supplier_top5_ratio_pct",
        "cornerstone_ratio_disclosed_pct",
    ):
        fields.append(
            RagFieldExtraction(
                name=name,
                status="found",
                value=25.0,
                unit="percentage_points",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            )
        )
    for name in ("cornerstone_share_count", "global_offering_share_count"):
        fields.append(
            RagFieldExtraction(
                name=name,
                status="found",
                value=1000.0,
                unit="shares",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            )
        )
    for name in ("cornerstone_lockup_months", "controlling_shareholder_lockup_months"):
        fields.append(
            RagFieldExtraction(
                name=name,
                status="found",
                value=6.0,
                unit="months",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            )
        )
    fields.extend(
        [
            RagFieldExtraction(
                name="industry_primary",
                status="found",
                value="生物医药",
                unit="category",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            ),
            RagFieldExtraction(
                name="pipeline_stage_name",
                status="found",
                value="BLA已获受理",
                unit="stage_name",
                source_chunk_ids=(source_id,),
                confidence=0.9,
            ),
        ]
    )
    return RagExtractionResponse(financial_period_used="2023-12-31", fields=tuple(fields))


class _UnavailableSemanticRetriever:
    def __init__(self) -> None:
        self.search_called = False

    def activate_document(self, *_args: object, **_kwargs: object) -> bool:
        return False

    def search(self, *_args: object, **_kwargs: object) -> list[SearchHit]:
        self.search_called = True
        raise AssertionError("semantic search must not run during lexical fallback")


class _MissingFieldsProvider:
    model_name = "test-model"

    def complete_json(self, **_kwargs: object) -> dict[str, object]:
        response = RagExtractionResponse(
            fields=tuple(
                RagFieldExtraction(name=name, status="not_found")
                for name in _MODEL_FIELD_NAMES
            )
        )
        return response.model_dump(mode="json")


def test_extractor_uses_audited_lexical_fallback_when_vector_index_is_missing() -> None:
    retriever = _UnavailableSemanticRetriever()
    extractor = RagFundamentalExtractor(
        retriever=retriever,  # type: ignore[arg-type]
        provider=_MissingFieldsProvider(),  # type: ignore[arg-type]
        system_prompt="test",
        retry_policy=RetryPolicy(max_retries=0),
    )
    record = extractor.extract(
        manifest_row={
            "company_id": "00001",
            "company_name": "测试公司",
            "prospectus_date": "2024-01-01",
            "source_hash": "a" * 64,
        },
        document_id="doc-1",
        chunks=[_chunk(1, text="公司主要业务为测试服务。")],
        existing_operating_cash_flow=None,
    )

    assert record["retrieval_mode"] == "lexical_anchor_fallback"
    assert retriever.search_called is False


def test_database_connection_failure_does_not_block_lexical_fundamentals() -> None:
    class BrokenRetriever(_UnavailableSemanticRetriever):
        def activate_document(self, *_args: object, **_kwargs: object) -> bool:
            raise ConnectionError("database offline")

    for retriever in (None, BrokenRetriever()):
        extractor = RagFundamentalExtractor(
            retriever=retriever,  # type: ignore[arg-type]
            provider=_MissingFieldsProvider(),  # type: ignore[arg-type]
            system_prompt="test", retry_policy=RetryPolicy(max_retries=0),
        )
        record = extractor.extract(
            manifest_row={"company_id": "00001", "company_name": "测试公司",
                          "prospectus_date": "2024-01-01", "source_hash": "a" * 64},
            document_id="doc-1", chunks=[_chunk(1, text="公司主要业务为测试服务。")],
            existing_operating_cash_flow=None,
        )
        assert record["retrieval_mode"] == "lexical_anchor_fallback"


def test_expand_hits_keeps_seed_neighbor_and_same_table() -> None:
    chunks = [
        _chunk(1, text="标题"),
        _chunk(2, text="现金及现金等价物"),
        _chunk(3, text="120", kind="table"),
        _chunk(4, text="脚注", kind="table"),
        _chunk(5, text="其他内容"),
    ]
    selected = _expand_hits(
        chunks,
        semantic_hits=[SearchHit(chunk=chunks[2], score=0.9, rank=1)],
        lexical_hits=[],
        neighbor_radius=1,
        max_chunks=5,
        max_chars=1000,
    )
    assert [item.chunk_id for item in selected] == [
        chunks[1].chunk_id,
        chunks[2].chunk_id,
        chunks[3].chunk_id,
    ]


def test_cash_anchor_prefers_balance_table_over_definition() -> None:
    chunks = [
        _chunk(1, text="现金及现金等价物包括银行存款"),
        _chunk(2, text="现金及现金等价物增加╱减少净额 10,000", kind="table"),
        _chunk(
            3,
            text="现金及现金等价物 2022年 134,907 2023年 146,317",
            kind="table",
        ),
    ]
    hits = _anchor_search(
        chunks,
        field_name="fundamental_ending_cash_latest_native",
        top_k=3,
    )
    assert hits[0].chunk.chunk_id == chunks[2].chunk_id


def test_net_profit_anchor_prefers_issuer_income_statement_over_target_appendix() -> None:
    chunks = [
        _chunk(
            1,
            page=13,
            text="節選合併綜合收益表 本集團 2019年 年內溢利 1,761百萬元",
            kind="table",
        ),
        _chunk(
            2,
            page=450,
            text=(
                "附錄一B STEIGENBERGER HOTELS AG "
                "截至2019年12月31日止年度 年內虧損 -14,389,939"
            ),
            kind="table",
        ),
    ]
    hits = _anchor_search(
        chunks,
        field_name="fundamental_net_profit_latest_native",
        top_k=2,
    )
    assert hits[0].chunk.chunk_id == chunks[0].chunk_id


def test_net_profit_anchor_recognizes_net_profit_wording() -> None:
    chunks = [
        _chunk(
            1,
            page=13,
            text="節選合併綜合收益表 淨利潤（虧損） 1,228 727 1,761",
        ),
        _chunk(2, page=450, text="附錄一B 年內虧損 -14,389,939", kind="table"),
    ]
    hits = _anchor_search(
        chunks,
        field_name="fundamental_net_profit_latest_native",
        top_k=2,
    )
    assert hits[0].chunk.chunk_id == chunks[0].chunk_id


def test_lockup_anchor_prefers_operational_clause_over_definition() -> None:
    chunks = [
        _chunk(1, text="「控股股東」指甲公司及乙公司", kind="table"),
        _chunk(
            2,
            text=(
                "控股股東的承諾：自上市日期起首六個月期間不會出售股份，"
                "隨後六個月不得因處置而失去控股股東地位。"
            ),
        ),
    ]
    hits = _anchor_search(
        chunks,
        field_name="controlling_shareholder_lockup_months",
        top_k=2,
    )
    assert hits[0].chunk.chunk_id == chunks[1].chunk_id


def test_cornerstone_lockup_anchor_does_not_select_controlling_shareholder_clause() -> None:
    chunks = [
        _chunk(1, text="控股股東於首六個月期間不會出售股份"),
        _chunk(
            2,
            text="各基石投資者同意於上市日期後六個月禁售期內不處置發售股份",
        ),
    ]
    hits = _anchor_search(
        chunks,
        field_name="cornerstone_lockup_months",
        top_k=2,
    )
    assert hits[0].chunk.chunk_id == chunks[1].chunk_id


def test_response_rejects_evidence_not_returned_by_rag() -> None:
    response = _valid_response("C-00001-0001-001")
    with pytest.raises(ValueError, match="未返回"):
        _validate_response(response, {})


def test_response_rejects_inequality_as_exact_percentage() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    fields = list(response.fields)
    for index, field in enumerate(fields):
        if field.name == "customer_top5_ratio_pct":
            fields[index] = field.model_copy(update={"value": 30.0})
            break
    response = response.model_copy(update={"fields": tuple(fields)})
    source = _chunk(
        1,
        text=(
            "其他指標為25%。五大客戶合共佔少於我們淨收入的30%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    with pytest.raises(ValueError, match="不是精確比例"):
        _validate_response(response, {source_id: source})


def test_response_rejects_financial_value_from_separate_appendix_subject() -> None:
    source_id = "C-00001-0450-T01-01"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        page=450,
        kind="table",
        text=(
            "附錄一B STEIGENBERGER HOTELS AG "
            "截至2019年12月31日止年度 年內虧損 -14,389,939"
        ),
    )
    with pytest.raises(ValueError, match="其他主体附錄"):
        _validate_response(response, {source_id: source})


def test_response_detects_separate_appendix_subject_in_adjacent_chunk() -> None:
    source_id = "C-00001-0450-T01-01"
    response = _valid_response(source_id)
    source = _chunk(1, page=450, kind="table", text="附錄一B 年內虧損 -14,389,939")
    heading = _chunk(2, page=451, text="STEIGENBERGER HOTELS AG 歷史財務資料")
    with pytest.raises(ValueError, match="其他主体附錄"):
        _validate_response(
            response,
            {source_id: source, heading.chunk_id: heading},
        )


def test_response_allows_cross_group_evidence_when_content_supports_field() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        text=(
            "本集團淨利潤120百萬元，客戶及供應商比例25%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    allowed = {name: {source_id} for name in (field.name for field in response.fields)}
    allowed["fundamental_net_profit_latest_native"] = set()
    _validate_response(response, {source_id: source}, allowed_by_field=allowed)


def test_response_rejects_cross_group_evidence_when_content_does_not_support_field() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        text=(
            "五大供應商合共佔採購額25%。其他比例均為25%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    allowed = {name: {source_id} for name in (field.name for field in response.fields)}
    allowed["fundamental_net_profit_latest_native"] = set()
    with pytest.raises(ValueError, match="没有直接支持"):
        _validate_response(response, {source_id: source}, allowed_by_field=allowed)


def test_response_rejects_ambiguous_no_material_research_as_zero() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    fields = list(response.fields)
    for index, field in enumerate(fields):
        if field.name == "fundamental_research_development_latest_native":
            fields[index] = field.model_copy(update={"value": 0.0})
            break
    response = response.model_copy(update={"fields": tuple(fields)})
    source = _chunk(
        1,
        text=(
            "本集團並無任何重大研發活動。其他比例均為25%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    with pytest.raises(ValueError, match="不能据此认定研发费用为零"):
        _validate_response(response, {source_id: source})


def test_invalid_field_can_be_downgraded_without_discarding_company() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        text=(
            "本集團年內虧損20百萬元，其他比例均為25%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    fields = list(response.fields)
    for index, field in enumerate(fields):
        if field.name == "fundamental_research_development_latest_native":
            fields[index] = field.model_copy(update={"source_chunk_ids": (source_id,)})
            break
    response = response.model_copy(update={"fields": tuple(fields)})
    downgraded = _downgrade_invalid_fields(
        response,
        {source_id: source},
        allowed_by_field={
            name: ({source_id} if name != "fundamental_research_development_latest_native" else set())
            for name in (field.name for field in response.fields)
        },
        prospectus_date=date(2024, 1, 1),
    )
    extracted = {field.name: field for field in downgraded.fields}
    assert extracted["fundamental_research_development_latest_native"].status == "not_found"
    assert extracted["fundamental_net_profit_latest_native"].status == "found"
    assert any("fundamental_research_development_latest_native" in item for item in downgraded.warnings)


def test_response_allows_shared_evidence_for_offer_share_count() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        text=(
            "全球發售1,000股，其他比例均為25%。"
            "基石投資者同意上市日期後六個月禁售；控股股東同意上市日期後六個月不會出售。"
        ),
    )
    allowed = {name: {source_id} for name in (field.name for field in response.fields)}
    allowed["global_offering_share_count"] = set()
    _validate_response(response, {source_id: source}, allowed_by_field=allowed)


def test_response_rejects_cornerstone_lockup_without_cornerstone_subject() -> None:
    source_id = "C-00001-0001-001"
    response = _valid_response(source_id)
    source = _chunk(
        1,
        text="控股股東同意上市日期後六個月不會出售，其他比例均為25%。",
    )
    with pytest.raises(ValueError, match="证据主体不匹配"):
        _validate_response(response, {source_id: source})


def test_pipeline_stage_score_uses_highest_business_stage() -> None:
    assert _pipeline_stage_score("核心管线BLA已获受理，另有II期研究") == 5
    assert _pipeline_stage_score("临床III期") == 4
    assert _pipeline_stage_score("核心产品处于IIb期临床试验") == 3
    assert _pipeline_stage_score("早期研发") is None


def test_cornerstone_ratio_uses_middle_price_scenario() -> None:
    chunks = [
        _chunk(1, text="|總計|28,000美元|3,066,400|40.67%|3.48%|", kind="table"),
        _chunk(2, text="|總計|28,000美元|3,216,850|42.66%|3.65%|", kind="table"),
        _chunk(3, text="|總計|28,000美元|3,382,800|44.86%|3.84%|", kind="table"),
    ]
    ratio, sources = _cornerstone_midpoint_ratio(chunks)
    assert ratio == pytest.approx(42.66)
    assert sources == (chunks[1].chunk_id,)


def test_cornerstone_ratio_derives_midpoint_from_price_bounds() -> None:
    chunks = [
        _chunk(1, page=340, text="基於發售價3.60港元（發售價範圍下限）"),
        _chunk(
            2,
            page=340,
            text="|總計|百萬美元|107,638,000|43.0%|37.5%|10.9%|10.4%|",
            kind="table",
        ),
        _chunk(3, page=341, text="基於發售價4.70港元（發售價範圍上限）"),
        _chunk(
            4,
            page=341,
            text="|總計|百萬美元|82,448,000|33.0%|28.6%|8.1%|8.0%|",
            kind="table",
        ),
    ]
    ratio, sources = _cornerstone_midpoint_ratio(chunks)
    assert ratio == pytest.approx(37.3373, abs=1e-4)
    assert set(sources) == {
        chunks[0].chunk_id,
        chunks[1].chunk_id,
        chunks[2].chunk_id,
        chunks[3].chunk_id,
    }
