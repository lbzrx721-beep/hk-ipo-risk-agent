from __future__ import annotations

from datetime import date
import json

import pandas as pd

from ipo_risk.modeling.historical_cases import (
    HistoricalCaseIndexRecord,
    HistoricalCaseMatch,
    build_case_index_records,
    build_case_summary,
)
from ipo_risk.modeling.industry import NormalizedIndustry, normalize_industry
from ipo_risk.modeling.llm_prediction import select_historical_cases


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "company_id": "00300",
        "company_name": "当前公司",
        "prospectus_date": "2023-06-01",
        "target_observation_date_5d": "2023-06-12",
        "industry_primary": "生物医药",
        "offering_market_cap_mid_hkd": 5_000_000_000.0,
        "fundamental_revenue_latest_hkd_mn": 1000.0,
        "fundamental_revenue_cagr_3y_pct": 20.0,
        "fundamental_gross_margin_latest_pct": 30.0,
        "market_return_20d_pct": -2.0,
        "market_volatility_20d_pct": 1.5,
        "agent_finding_count": 6,
        "agent_has_risk_cash_burn": 1,
        "target_significant_drop_5d": 1,
        "target_return_vs_offer_pct_5d": -18.0,
    }
    row.update(overrides)
    return row


def test_industry_normalization_merges_variants_but_keeps_raw_value() -> None:
    simplified = normalize_industry("生物医药")
    traditional = normalize_industry("生物醫藥")

    assert simplified.level_1 == traditional.level_1 == "医疗健康"
    assert simplified.level_2 == traditional.level_2 == "生物医药"
    assert traditional.raw == "生物醫藥"


def test_case_embedding_summary_never_contains_post_listing_result() -> None:
    frame = pd.DataFrame([_row()])

    records = build_case_index_records(frame)

    assert len(records) == 1
    record = records[0]
    assert record.significant_drop_5d is True
    assert record.return_vs_offer_5d_pct == -18.0
    assert "-18" not in record.case_summary_text
    serialized_features = json.dumps(record.feature_payload, ensure_ascii=False)
    assert "target_" not in serialized_features
    assert "现金消耗与流动性" in record.case_summary_text


def test_case_builder_accepts_both_existing_5d_return_column_names() -> None:
    old = pd.DataFrame([_row()])
    new = old.rename(columns={"target_return_vs_offer_pct_5d": "target_return_vs_offer_5d_pct"})

    assert build_case_index_records(old)[0].return_vs_offer_5d_pct == -18.0
    assert build_case_index_records(new)[0].return_vs_offer_5d_pct == -18.0


class _FakeCaseStore:
    def __init__(self, matches: tuple[HistoricalCaseMatch, ...]) -> None:
        self.matches = matches
        self.query = ""

    def search_semantic(self, **kwargs: object) -> tuple[HistoricalCaseMatch, ...]:
        self.query = str(kwargs["case_summary_text"])
        return self.matches


def _match(
    company_id: str,
    *,
    industry: NormalizedIndustry,
    market_cap: float,
    semantic: float,
) -> HistoricalCaseMatch:
    return HistoricalCaseMatch(
        record=HistoricalCaseIndexRecord(
            case_id=f"{company_id}-20220101",
            company_id=company_id,
            company_name=f"历史{company_id}",
            prospectus_date=date(2022, 1, 1),
            outcome_available_at=date(2022, 1, 12),
            industry=industry,
            feature_payload={
                "offering_market_cap_mid_hkd": market_cap,
                "fundamental_revenue_latest_hkd_mn": 1000.0,
            },
            case_summary_text="无标签历史摘要",
            significant_drop_5d=company_id == "00100",
            return_vs_offer_5d_pct=-12.0 if company_id == "00100" else 3.0,
        ),
        semantic_similarity=semantic,
    )


def test_hybrid_retrieval_combines_industry_numeric_and_semantic_scores() -> None:
    frame = pd.DataFrame(
        [
            _row(
                company_id="00100",
                company_name="同行",
                prospectus_date="2022-01-01",
                target_observation_date_5d="2022-01-12",
                offering_market_cap_mid_hkd=4_900_000_000.0,
            ),
            _row(
                company_id="00101",
                company_name="异业",
                industry_primary="建筑工程",
                prospectus_date="2022-02-01",
                target_observation_date_5d="2022-02-12",
                offering_market_cap_mid_hkd=9_000_000_000.0,
                target_significant_drop_5d=0,
            ),
            _row(),
        ]
    )
    store = _FakeCaseStore(
        (
            _match(
                "00101",
                industry=normalize_industry("建筑工程"),
                market_cap=9_000_000_000.0,
                semantic=0.98,
            ),
            _match(
                "00100",
                industry=normalize_industry("生物醫藥"),
                market_cap=4_900_000_000.0,
                semantic=0.85,
            ),
        )
    )

    cases = select_historical_cases(
        frame,
        "00300",
        top_k=2,
        case_store=store,  # type: ignore[arg-type]
    )

    assert cases[0].company_id == "00100"
    assert cases[0].industry_level_1 == "医疗健康"
    assert "target_" not in store.query
    assert "-18" not in store.query


def test_summary_excludes_company_identity() -> None:
    row = _row(company_name="不能进入向量的公司名", company_id="09999")
    summary = build_case_summary(row)

    assert "不能进入向量的公司名" not in summary
    assert "09999" not in summary
