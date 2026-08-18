from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ipo_risk.modeling.agent_comparison import train_agent_feature_comparison


def _write_fixture(tmp_path: Path, *, leaking_feature: bool = False) -> tuple[Path, Path, Path]:
    rows: list[dict[str, object]] = []
    company_number = 1
    for year in (2020, 2021, 2022, 2023, 2024, 2025):
        for index in range(8):
            target = index % 2
            rows.append(
                {
                    "company_id": f"{company_number:05d}",
                    "ticker": f"{company_number:05d}.HK",
                    "company_name": f"公司{company_number}",
                    "prospectus_date": f"{year}-06-{index + 1:02d}",
                    "as_of_date": f"{year}-06-{index + 1:02d}",
                    "split": "legacy",
                    "run_id": f"RUN-{company_number}",
                    "target_significant_drop_5d": target,
                    "target_broke_issue_price_1d": target,
                    "target_return_vs_offer_20d_pct": -30 if target else 20,
                    "document_page_count": 500 if target else 200,
                    "market_return_20d_pct": -5 if target else 4,
                    "agent_finding_count": 9 if target else 1,
                    "agent_evidence_count": 20 if target else 3,
                    "agent_failed_trace_count": 0,
                }
            )
            company_number += 1
    matrix_path = tmp_path / "agent_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    base_summary = tmp_path / "base_summary.json"
    base_features = ["document_page_count", "market_return_20d_pct"]
    if leaking_feature:
        base_features.append("target_return_vs_offer_20d_pct")
    base_summary.write_text(json.dumps({"feature_columns": base_features}), encoding="utf-8")
    agent_summary = tmp_path / "agent_summary.json"
    agent_summary.write_text(
        json.dumps(
            {
                "agent_feature_columns": [
                    "agent_finding_count",
                    "agent_evidence_count",
                    "agent_failed_trace_count",
                ]
            }
        ),
        encoding="utf-8",
    )
    return matrix_path, base_summary, agent_summary


def test_agent_comparison_uses_fixed_years_and_explicit_feature_manifests(
    tmp_path: Path,
) -> None:
    matrix_path, base_summary, agent_summary = _write_fixture(tmp_path)
    result = train_agent_feature_comparison(
        matrix_path,
        base_summary_path=base_summary,
        agent_summary_path=agent_summary,
        output_dir=tmp_path / "models",
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["cohort_counts"] == {"train": 24, "validation": 8, "test": 8}
    assert metrics["external_case_count"] == 8
    assert metrics["leakage_controls"]["all_target_prefixed_columns_blocked"] is True
    assert set(metrics["variants"]) == {
        "logistic_base_64",
        "logistic_agent_augmented",
        "lightgbm_base_64",
        "lightgbm_agent_augmented",
    }
    assert metrics["variants"]["logistic_base_64"]["feature_count"] == 2
    assert metrics["variants"]["lightgbm_agent_augmented"]["feature_count"] == 4
    manifest = json.loads(result.feature_manifest_path.read_text(encoding="utf-8"))
    assert manifest["raw_agent_feature_count"] == 3
    assert manifest["agent_feature_count"] == 2
    assert manifest["excluded_agent_feature_columns"] == ["agent_failed_trace_count"]
    assert result.base_model_path.is_file()
    assert result.agent_model_path.is_file()
    assert result.lightgbm_base_model_path.is_file()
    assert result.lightgbm_agent_model_path.is_file()
    assert result.feature_importance_path.is_file()
    importance = pd.read_csv(result.feature_importance_path)
    assert set(importance["variant"]) == {
        "lightgbm_base_64",
        "lightgbm_agent_augmented",
    }
    predictions = pd.read_csv(result.predictions_path)
    assert set(predictions["split"]) == {"train", "validation", "test"}
    assert "target_return_vs_offer_20d_pct" not in predictions.columns


def test_agent_comparison_rejects_target_leakage_in_feature_manifest(tmp_path: Path) -> None:
    matrix_path, base_summary, agent_summary = _write_fixture(tmp_path, leaking_feature=True)
    with pytest.raises(ValueError, match="label leakage blocked"):
        train_agent_feature_comparison(
            matrix_path,
            base_summary_path=base_summary,
            agent_summary_path=agent_summary,
            output_dir=tmp_path / "models",
        )
