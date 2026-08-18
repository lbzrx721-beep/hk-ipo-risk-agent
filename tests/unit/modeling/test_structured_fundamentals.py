from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipo_risk.modeling.structured_fundamentals import (
    _offering_share_ratio,
    build_structured_fundamental_dataset,
    load_structured_feature_snapshots,
)


def test_issue_ratio_ignores_cornerstone_allocation_ratio() -> None:
    assert (
        _offering_share_ratio("發售股份將佔緊隨全球發售完成後本公司經擴大已發行股本約25%。") == 25.0
    )
    assert (
        _offering_share_ratio(
            "基石投資者認購55,386,200股發售股份，約佔緊隨全球發售完成後"
            "發售股份的57.46%，並佔已發行股本2.36%。"
        )
        is None
    )
    assert (
        _offering_share_ratio("發售股份將佔本公司緊隨全球發售完成後股本中已發行H股總數約64.39%。")
        is None
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_builds_offering_and_financial_features_from_existing_caches(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "data/processed/competition/company_manifest.jsonl"
    row = {
        "company_id": "00300",
        "company_name": "测试公司",
        "company_folder": "00300_测试公司",
        "prospectus_date": "2024-09-09",
        "source_hash": "a" * 64,
        "split": "validation",
    }
    _write_jsonl(manifest, [row])
    cache_name = "doc_00300_prospectus_2024-09-09_aaaaaaaa"
    chunk_path = tmp_path / "data/interim/chunks/00300_测试公司" / cache_name / "chunks.jsonl"
    _write_jsonl(
        chunk_path,
        [
            {
                "chunk_id": "C-00300-0002-001",
                "page": 2,
                "text": "最高發售價每股54.80港元。全球發售的發售股份數目 492,135,100股。",
            },
            {
                "chunk_id": "C-00300-0035-001",
                "page": 35,
                "text": "發售價範圍 每股發售股份52.00港元至每股發售股份54.80港元。",
            },
            {
                "chunk_id": "C-00300-0036-001",
                "page": 36,
                "text": (
                    "上市市值預期介乎389,233.8百萬港元（基於每股52.00港元）"
                    "至410,192.6百萬港元（基於每股54.80港元）。"
                ),
            },
            {
                "chunk_id": "C-00300-0033-001",
                "page": 33,
                "text": "我們估計將收取約25,972百萬港元的募集資金淨額。",
            },
            {
                "chunk_id": "C-00300-0416-001",
                "page": 416,
                "text": (
                    "基石投資者認購3,436,700股發售股份。緊隨全球發售完成後，"
                    "控股股東將擁有約69.03%的表決權。"
                ),
            },
        ],
    )
    table_path = tmp_path / "data/interim/tables/00300_测试公司" / cache_name / "tables.jsonl"
    headers = [
        "项目",
        "截至12月31日止年度 / 2021年 / 人民幣千元",
        "截至12月31日止年度 / 2022年 / 人民幣千元",
        "截至12月31日止年度 / 2023年 / 人民幣千元",
    ]
    _write_jsonl(
        table_path,
        [
            {
                "table_id": "T-P0023-001",
                "page": 23,
                "title": "歷史財務資料概要",
                "headers": headers,
                "rows": [
                    ["收入", "100,000", "121,000", "144,000"],
                    ["毛利", "30,000", "39,930", "50,400"],
                ],
                "currency": "RMB",
                "unit": None,
            },
            {
                "table_id": "T-P0028-001",
                "page": 28,
                "title": "合併現金流量表概要",
                "headers": headers,
                "rows": [
                    ["經營活動所得現金淨額", "(10,000)", "5,000", "12,000"],
                ],
                "currency": "RMB",
                "unit": None,
            },
        ],
    )

    result = build_structured_fundamental_dataset(tmp_path)
    snapshots = load_structured_feature_snapshots(result.audit_path)
    features = snapshots["00300"]

    assert features["offering_offer_price_low_hkd"].value == 52.0
    assert features["offering_offer_price_high_hkd"].value == 54.8
    assert features["offering_gross_proceeds_mid_hkd"].value == pytest.approx(492_135_100 * 53.4)
    assert features["offering_net_proceeds_hkd"].value == 25_972_000_000
    assert features["offering_market_cap_mid_hkd"].value == 399_713_200_000
    assert features["offering_share_ratio_pct"].value == pytest.approx(
        100 * 492_135_100 / (399_713_200_000 / 53.4)
    )
    assert features["fundamental_revenue_cagr_3y_pct"].value == 20.0
    assert features["fundamental_gross_margin_latest_pct"].value == 35.0
    assert features["fundamental_gross_margin_change_3y_pct_points"].value == 5.0
    assert features["fundamental_operating_cash_flow_latest_native"].value == 12_000_000
    assert features["fundamental_negative_operating_cash_flow_years_3y"].value == 1
    assert features["fundamental_operating_cash_flow_to_revenue_latest_pct"].value == pytest.approx(
        100 * 12_000 / 144_000
    )
    assert "T-P0023-001" in features["fundamental_revenue_latest_native"].source_ref


def test_missing_values_are_explicit_and_do_not_become_zero(tmp_path: Path) -> None:
    manifest = tmp_path / "data/processed/competition/company_manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "company_id": "00001",
                "company_name": "缺失公司",
                "company_folder": "00001_缺失公司",
                "prospectus_date": "2020-01-01",
                "source_hash": "b" * 64,
                "split": "train",
            }
        ],
    )

    result = build_structured_fundamental_dataset(tmp_path)
    snapshots = load_structured_feature_snapshots(result.audit_path)
    features = snapshots["00001"]

    assert features["fundamental_revenue_latest_native"].value is None
    assert features["fundamental_revenue_available"].value == 0
    assert features["offering_terms_available"].value == 0


def test_table_level_unit_applies_to_all_years_and_ignores_percentage_twins(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "data/processed/competition/company_manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "company_id": "02788",
                "company_name": "单位测试",
                "company_folder": "02788_单位测试",
                "prospectus_date": "2025-11-14",
                "source_hash": "c" * 64,
                "split": "test",
            }
        ],
    )
    cache_name = "doc_02788_prospectus_2025-11-14_cccccccc"
    table_path = tmp_path / "data/interim/tables/02788_单位测试" / cache_name / "tables.jsonl"
    _write_jsonl(
        table_path,
        [
            {
                "table_id": "T-P0023-001",
                "page": 23,
                "title": "财务资料概要",
                "headers": [
                    "项目",
                    "截至12月31日止年度 / 2022年 / %",
                    "截至12月31日止年度 / 2022年 / %",
                    "截至12月31日止年度 / 2023年 / %",
                    "截至12月31日止年度 / 2023年 / %",
                    "截至12月31日止年度 / 2024年 / 人民币百万元",
                    "截至12月31日止年度 / 2024年 / %",
                ],
                "rows": [
                    ["收入", "13,489.7", "100.0", "13,814.7", "100.0", "15,163.2", "100.0"],
                    ["毛利", "2,041.1", "15.1", "2,336.3", "16.9", "4,276.5", "28.2"],
                ],
                "currency": "RMB",
                "unit": None,
            }
        ],
    )

    result = build_structured_fundamental_dataset(tmp_path)
    features = load_structured_feature_snapshots(result.audit_path)["02788"]

    assert features["fundamental_revenue_latest_native"].value == 15_163_200_000
    assert features["fundamental_revenue_growth_latest_pct"].value == pytest.approx(
        100 * (15_163.2 / 13_814.7 - 1)
    )
    assert features["fundamental_gross_margin_latest_pct"].value == pytest.approx(
        100 * 4_276.5 / 15_163.2
    )
