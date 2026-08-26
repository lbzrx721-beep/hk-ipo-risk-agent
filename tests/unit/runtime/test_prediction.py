from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ipo_risk.runtime import prediction as prediction_module


def test_runtime_prediction_uses_current_agent_features_and_drops_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "artifacts" / "risk_analysis.json"
    model_path.write_bytes(b"test")
    report_path.parent.mkdir(parents=True)

    monkeypatch.setenv("IPO_RISK_PREDICTION_MODEL", str(model_path))
    monkeypatch.setattr(
        prediction_module,
        "_load_bundle",
        lambda *_args: {
            "model_name": "test_model",
            "features": ["agent_score", "offering_share_ratio_pct"],
            "feature_ranking": ["agent_score", "offering_share_ratio_pct"],
            "threshold": 0.4,
        },
    )
    def fake_predict(_bundle, frame: pd.DataFrame) -> pd.DataFrame:
        assert "target_significant_drop_5d" not in frame.columns
        assert frame.loc[0, "agent_score"] == 3.0
        return frame.assign(
            predicted_probability=0.72,
            predicted_significant_drop_5d=1,
            predicted_decision="high_risk",
            prediction_requires_review=False,
        )

    monkeypatch.setattr(prediction_module, "predict_feature_rows", fake_predict)

    payload, destination = prediction_module.generate_runtime_prediction(
        project_root=tmp_path,
        company_id="1",
        company_name="测试公司",
        run_id="RUN-TEST",
        features={
            "agent_score": 3.0,
            "offering_share_ratio_pct": 20.0,
            "target_return_vs_offer_pct_5d": -20.0,
        },
        report_path=report_path,
    )

    assert destination == report_path.with_name("prediction_5d.json")
    assert destination.is_file()
    assert payload["probability"] == 0.72
    assert payload["decision"] == "high_risk"
    assert payload["feature_snapshot"]["agent_score"] == 3.0
    assert payload["used_historical_feature_backfill"] is False
    assert payload["matched_reference_features"] is False
    assert payload["feature_readiness"] == "ready"
    assert payload["candidate_feature_count"] == 2
    assert payload["available_candidate_feature_count"] == 2
    assert payload["recall_guard"]["watchlist"] is False
    assert payload["recall_guard"]["caught_beyond_primary_model"] is False


def test_runtime_prediction_blocks_when_current_snapshot_is_too_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "artifacts" / "risk_analysis.json"
    model_path.write_bytes(b"test")
    report_path.parent.mkdir(parents=True)
    monkeypatch.setenv("IPO_RISK_PREDICTION_MODEL", str(model_path))
    monkeypatch.setattr(
        prediction_module,
        "_load_bundle",
        lambda *_args: {
            "model_name": "test_model",
            "features": ["a", "b", "c", "d"],
            "threshold": 0.4,
        },
    )

    called = False

    def fake_predict(_bundle, _frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal called
        called = True
        raise AssertionError("model must not run with a blocked feature projection")

    monkeypatch.setattr(prediction_module, "predict_feature_rows", fake_predict)

    with pytest.raises(
        prediction_module.RuntimePredictionUnavailable,
        match="低于最低要求",
    ):
        prediction_module.generate_runtime_prediction(
            project_root=tmp_path,
            company_id="1",
            company_name="测试公司",
            run_id="RUN-TEST",
            features={"a": 1.0},
            report_path=report_path,
        )

    assert called is False
