from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ipo_risk.modeling.dataset import build_competition_modeling_dataset


def test_modeling_dataset_uses_only_information_visible_before_listing_date(
    tmp_path: Path,
) -> None:
    root = tmp_path
    processed = root / "data" / "processed" / "competition"
    modeling = root / "data" / "processed" / "modeling"
    prospectus = root / "data" / "raw" / "prospectus" / "00002_测试公司"
    processed.mkdir(parents=True)
    modeling.mkdir(parents=True)
    prospectus.mkdir(parents=True)
    pdf_path = prospectus / "招股书.pdf"
    pdf_path.write_bytes(b"%PDF synthetic")

    manifest = {
        "company_id": "00002",
        "ticker": "00002.HK",
        "company_name": "测试公司",
        "prospectus_date": "2024-01-10",
        "offering_description": "全球發售",
        "prospectus_file": pdf_path.relative_to(root).as_posix(),
        "page_count": 321,
        "listing_date": "2024-01-20",
        "offer_price_hkd": 10.0,
        "offer_price_source": "hkshareeodprices.first_day_preclose_proxy",
        "split": "validation",
    }
    manifest_path = processed / "company_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    labels = pd.DataFrame(
        [
            {
                "company_id": "00001",
                "prospectus_date": "2023-10-01",
                "listing_date": "2023-10-10",
                "horizon": 1,
                "observation_date": "2023-10-10",
                "return_vs_offer_pct": 2.0,
                "significant_drop": False,
                "label_status": "complete",
                "offer_price_source": "official",
            },
            {
                "company_id": "00001",
                "prospectus_date": "2023-10-01",
                "listing_date": "2023-10-10",
                "horizon": 5,
                "observation_date": "2023-10-16",
                "return_vs_offer_pct": -12.0,
                "significant_drop": True,
                "label_status": "complete",
                "offer_price_source": "official",
            },
            {
                "company_id": "00002",
                "prospectus_date": "2024-01-10",
                "listing_date": "2024-01-20",
                "horizon": 5,
                "observation_date": "2024-01-26",
                "return_vs_offer_pct": -15.0,
                "significant_drop": True,
                "label_status": "complete",
                "offer_price_source": "proxy",
            },
            {
                "company_id": "00003",
                "prospectus_date": "2024-01-01",
                "listing_date": "2024-01-05",
                "horizon": 1,
                "observation_date": "2024-01-05",
                "return_vs_offer_pct": -10.0,
                "significant_drop": True,
                "label_status": "complete",
                "offer_price_source": "proxy",
            },
            {
                "company_id": "00003",
                "prospectus_date": "2024-01-01",
                "listing_date": "2024-01-05",
                "horizon": 5,
                "observation_date": "2024-01-12",
                "return_vs_offer_pct": -99.0,
                "significant_drop": True,
                "label_status": "complete",
                "offer_price_source": "proxy",
            },
        ]
    )
    labels_path = processed / "market_labels.csv"
    labels.to_csv(labels_path, index=False)
    daily = pd.DataFrame(
        [
            {
                "trade_date": f"2024-01-0{day}",
                "median_return_pct": 1.0,
                "advance_ratio": 0.6,
                "median_amplitude_pct": 2.0,
                "market_turnover_hkd": 100.0 + day,
                "median_security_turnover_hkd": 10.0,
                "active_securities": 1000,
            }
            for day in range(2, 10)
        ]
        + [
            {
                "trade_date": "2024-01-11",
                "median_return_pct": -90.0,
                "advance_ratio": 0.0,
                "median_amplitude_pct": 99.0,
                "market_turnover_hkd": 99999.0,
                "median_security_turnover_hkd": 999.0,
                "active_securities": 1000,
            },
            {
                "trade_date": "2024-01-21",
                "median_return_pct": 999.0,
                "advance_ratio": 1.0,
                "median_amplitude_pct": 999.0,
                "market_turnover_hkd": 999999.0,
                "median_security_turnover_hkd": 9999.0,
                "active_securities": 1000,
            }
        ]
    )
    daily.to_csv(modeling / "daily_market_sentiment.csv", index=False)

    result = build_competition_modeling_dataset(
        root,
        manifest_path=manifest_path,
        labels_path=labels_path,
        eod_path=root / "missing.csv",
        output_dir=modeling,
    )

    matrix = pd.read_csv(result.feature_matrix_path)
    assert len(matrix) == 1
    assert matrix.loc[0, "target_significant_drop_5d"] == 1
    assert matrix.loc[0, "market_return_5d_pct"] < 0
    assert matrix.loc[0, "recent_ipo_count_90d"] == 1
    assert matrix.loc[0, "recent_ipo_5d_observed_count_180d"] == 2
    assert matrix.loc[0, "prospectus_date"] == "2024-01-10"
    assert matrix.loc[0, "listing_date"] == "2024-01-20"
    assert matrix.loc[0, "as_of_date"] == "2024-01-20"
    assert not any("offer_price" in column for column in result.feature_columns)
    assert not any("listing_date" in column for column in result.feature_columns)

    audit = json.loads(result.feature_audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert all(
        item["available_at"] <= audit["as_of_date"]
        for item in audit["features"].values()
    )
    assert audit["target"]["available_at"] > audit["as_of_date"]
    assert audit["target"]["prediction_input_eligible"] is False
