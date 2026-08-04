from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ipo_risk.modeling.predict import load_model_bundle, predict_feature_rows
from ipo_risk.modeling.train import train_day5_baselines


def test_time_ordered_baselines_write_predictions_metrics_and_model(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    split_dates = {
        "train": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
        "validation": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "test": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
    }
    company_number = 1
    for split, dates in split_dates.items():
        for index, prospectus_date in enumerate(dates):
            target = index % 2
            rows.append(
                {
                    "company_id": f"{company_number:05d}",
                    "ticker": f"{company_number:05d}.HK",
                    "company_name": f"公司{company_number}",
                    "prospectus_date": prospectus_date,
                    "as_of_date": prospectus_date,
                    "split": split,
                    "target_significant_drop_5d": target,
                    "target_return_vs_offer_5d_pct": -15.0 if target else 5.0,
                    "target_observation_date": prospectus_date,
                    "label_offer_price_source": "official",
                    "document_page_count": 300 + target * 100,
                    "document_offering_description": "全球發售" if target else "股份發售",
                    "market_return_20d_pct": -5.0 if target else 3.0,
                    "recent_ipo_count_30d": 8 if target else 2,
                }
            )
            company_number += 1
    matrix_path = tmp_path / "matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "document_page_count",
                    "document_offering_description",
                    "market_return_20d_pct",
                    "recent_ipo_count_30d",
                ]
            }
        ),
        encoding="utf-8",
    )

    result = train_day5_baselines(
        matrix_path,
        summary_path=summary_path,
        output_dir=tmp_path / "models",
    )

    assert result.model_path.is_file()
    assert result.predictions_path.is_file()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert set(metrics["variants"]) == {"prospectus_only", "market_only", "fused"}
    assert metrics["leakage_controls"]["test_used_for_fitting_or_threshold"] is False
    assert metrics["variants"]["fused"]["splits"]["test"]["sample_count"] == 4

    bundle = load_model_bundle(result.model_path)
    predicted = predict_feature_rows(bundle, pd.DataFrame(rows))
    assert predicted["predicted_probability"].between(0, 1).all()
    assert set(predicted["predicted_significant_drop_5d"].unique()).issubset({0, 1})
