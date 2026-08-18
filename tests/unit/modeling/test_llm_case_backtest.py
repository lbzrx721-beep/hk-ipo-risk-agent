from __future__ import annotations

import pandas as pd

from ipo_risk.modeling.llm_case_backtest import (
    comparison_delta,
    compute_backtest_metrics,
    select_balanced_pilot,
)


def test_balanced_pilot_is_deterministic_and_contains_both_classes() -> None:
    frame = pd.DataFrame(
        [
            {
                "company_id": f"{index:05d}",
                "company_name": f"公司{index}",
                "prospectus_date": f"2024-01-{index:02d}",
                "split": "validation",
                "target_significant_drop_5d": index in {2, 5, 8},
            }
            for index in range(1, 10)
        ]
    )

    first = select_balanced_pilot(frame, split="validation", sample_size=5, random_state=42)
    second = select_balanced_pilot(frame, split="validation", sample_size=5, random_state=42)

    assert first["company_id"].tolist() == second["company_id"].tolist()
    assert set(first["target_significant_drop_5d"]) == {0, 1}


def test_metrics_and_delta_compare_the_two_arms() -> None:
    observations = [
        _observation("without_case_rag", True, False, 40),
        _observation("without_case_rag", False, False, 30),
        _observation("with_case_rag", True, True, 80),
        _observation("with_case_rag", False, False, 20),
    ]

    metrics = compute_backtest_metrics(observations)
    delta = comparison_delta(metrics)

    assert metrics["without_case_rag"]["accuracy"] == 0.5
    assert metrics["with_case_rag"]["accuracy"] == 1.0
    assert metrics["with_case_rag"]["input_tokens"] == 200
    assert delta["accuracy"] == 0.5
    assert delta["brier_score"] < 0


def test_metrics_can_reclassify_scores_with_business_threshold() -> None:
    observations = [
        _observation("without_case_rag", True, False, 30),
        _observation("without_case_rag", False, False, 20),
    ]

    metrics = compute_backtest_metrics(observations, alert_threshold=25)

    assert metrics["without_case_rag"]["recall"] == 1.0
    assert metrics["without_case_rag"]["accuracy"] == 1.0


def _observation(arm: str, actual: bool, predicted: bool, score: int) -> dict[str, object]:
    return {
        "arm": arm,
        "actual_significant_drop_5d": actual,
        "predicted_significant_drop_5d": predicted,
        "risk_score": score,
        "elapsed_seconds": 1.0,
        "usage": {
            "calls": 3,
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 0,
            "estimated_cost_rmb": None,
        },
    }
