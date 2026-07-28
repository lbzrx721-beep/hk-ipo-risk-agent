"""赛事官方数据接入、标准化与市场标签测试。"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

import fitz

from ipo_risk.competition import (
    CompetitionProspectus,
    MarketObservation,
    build_horizon_labels,
    import_competition_delivery,
    normalize_stock_code,
    parse_prospectus_filename,
)


def _observation(
    day: int,
    close: float,
    *,
    low: float | None = None,
    preclose: float | None = 10.0,
) -> MarketObservation:
    return MarketObservation(
        trade_date=date(2025, 1, day),
        currency="HKD",
        open_hkd=close,
        high_hkd=close,
        low_hkd=low if low is not None else close,
        close_hkd=close,
        volume_shares=1000,
        turnover_hkd=close * 1000,
        preclose_hkd=preclose,
        average_price_hkd=close,
    )


def test_normalize_stock_code_rejects_temporary_derivative_codes() -> None:
    assert normalize_stock_code("0084.HK") == "00084"
    assert normalize_stock_code("06651") == "06651"
    assert normalize_stock_code("9959!1.HK") is None


def test_parse_prospectus_filename_preserves_company_and_offering(tmp_path: Path) -> None:
    path = tmp_path / "00300_09-09-2024_美的集團_全球發售.pdf"

    parsed = parse_prospectus_filename(path)

    assert parsed.company_id == "00300"
    assert parsed.company_name == "美的集團"
    assert parsed.offering_description == "全球發售"
    assert parsed.prospectus_date == date(2024, 9, 9)


def test_build_horizon_labels_uses_offer_price_and_window() -> None:
    entry = CompetitionProspectus(
        company_id="01234",
        company_name="测试公司",
        prospectus_date=date(2024, 12, 20),
        offering_description="全球發售",
        year=2024,
        source_path=Path("01234.pdf"),
    )
    rows = (
        _observation(2, 11.0),
        _observation(3, 9.5, low=8.8, preclose=11.0),
        _observation(4, 9.0, preclose=9.5),
        _observation(5, 8.8, preclose=9.0),
        _observation(6, 8.5, preclose=8.8),
    )

    labels = build_horizon_labels(
        entry=entry,
        rows=rows,
        listing_date=date(2025, 1, 2),
        listing_date_source="official",
        offer_price_hkd=10.0,
        offer_price_source="official",
        horizons=(1, 5),
        significant_drop_threshold_pct=-10.0,
    )

    assert labels[0]["return_vs_offer_pct"] == 10.0
    assert labels[0]["significant_drop"] is False
    assert labels[1]["return_vs_offer_pct"] == -15.0
    assert labels[1]["significant_drop"] is True
    assert labels[1]["broke_issue_price"] is True
    assert labels[1]["split"] == "validation"


def _write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="gb18030", newline="") as target:
        writer = csv.writer(target, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)


def test_import_competition_delivery_builds_standard_data_and_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "delivery"
    annual = source / "2025_1份"
    pdf = annual / "01234_20-12-2024_测试公司_全球發售.pdf"
    _write_pdf(pdf)
    _write_csv(
        source / "hkcompanyinfo.csv",
        ["S_INFO_COMPCODE", "S_INFO_COMPNAME", "COMP_NAME_ENG"],
        [["COMP1", "测试公司有限公司", "Test Company Limited"]],
    )
    _write_csv(
        source / "hksharedescription.csv",
        [
            "S_INFO_WINDCODE",
            "S_INFO_CODE",
            "SECURITYTYPE",
            "S_INFO_COMPCODE",
            "S_INFO_LISTDATE",
            "S_INFO_LISTPRICE",
        ],
        [["1234.HK", "1234", "ORD", "COMP1", "20250102", "10.0"]],
    )
    headers = [
        "S_INFO_WINDCODE",
        "TRADE_DT",
        "CRNCY_CODE",
        "S_DQ_OPEN",
        "S_DQ_HIGH",
        "S_DQ_LOW",
        "S_DQ_CLOSE",
        "S_DQ_VOLUME",
        "S_DQ_AMOUNT",
        "S_DQ_PRECLOSE",
        "S_DQ_AVGPRICE",
    ]
    market_rows: list[list[object]] = []
    for day, close in zip((2, 3, 6, 7, 8), (11.0, 10.0, 9.5, 9.2, 8.8), strict=True):
        volume = 1000
        market_rows.append(
            [
                "1234.HK",
                f"202501{day:02d}",
                "HKD",
                close,
                close,
                close - 0.2,
                close,
                volume,
                close,
                10.0,
                close,
            ]
        )
    _write_csv(source / "hkshareeodprices.csv", headers, market_rows)
    project = tmp_path / "project"

    result = import_competition_delivery(
        source,
        project_root=project,
        horizons=(1, 5),
        imported_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert result.company_count == 1
    assert result.full_label_count == 1
    assert result.unavailable_count == 0
    metadata_path = project / "data/raw/company_info/01234_测试公司/公司信息.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["listing_date"] == "2025-01-02"
    assert metadata["offering"]["offer_price_hkd"] == 10.0
    assert metadata["company_profile_source"]["company_name_en"] == "Test Company Limited"
    market_path = project / "data/raw/market/01234_测试公司/日线行情.json"
    market = json.loads(market_path.read_text(encoding="utf-8"))
    assert market["row_count"] == 5
    assert market["rows"][0]["turnover_hkd"] == 11000.0
    with result.labels_path.open("r", encoding="utf-8", newline="") as source_file:
        labels = list(csv.DictReader(source_file))
    assert [row["horizon"] for row in labels] == ["1", "5"]
    assert labels[-1]["significant_drop"] == "True"

    repeated = import_competition_delivery(
        source,
        project_root=project,
        horizons=(1, 5),
        imported_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert repeated.company_count == 1
    audit_files = list(
        (project / "data/interim/audit/01234_测试公司").glob(
            "pre_competition_*"
        )
    )
    assert audit_files == []
