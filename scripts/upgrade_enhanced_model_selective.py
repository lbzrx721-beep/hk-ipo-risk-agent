"""为 enhanced_blend3_v2 增加按历史 OOF 选择的三态自动裁决策略。

策略不会改变原二分类概率和阈值；只把高置信度样本标记为 high_risk / low_risk，
中间区间标记为 uncertain，交给后续 Agent 或人工复核。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import typer


app = typer.Typer(add_completion=False)


def _load_finalizer(root: Path):
    path = root / "scripts" / "finalize_enhanced_model.py"
    spec = importlib.util.spec_from_file_location("finalize_enhanced_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load finalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_decided_accuracy: float,
) -> dict[str, float]:
    candidates: list[tuple[float, float, float, float, float]] = []
    low_candidates = np.unique(np.r_[np.linspace(0.05, 0.55, 251), probabilities])
    high_candidates = np.unique(np.r_[np.linspace(0.35, 0.90, 276), probabilities])
    for low in low_candidates:
        for high in high_candidates:
            if low >= high:
                continue
            decided = (probabilities <= low) | (probabilities >= high)
            if not decided.any():
                continue
            predicted_high = probabilities >= high
            accuracy = float((predicted_high[decided] == labels[decided]).mean())
            if accuracy < minimum_decided_accuracy:
                continue
            coverage = float(decided.mean())
            high_risk_recall = float(
                ((predicted_high == 1) & decided & (labels == 1)).sum()
                / max(1, int((labels == 1).sum()))
            )
            candidates.append((coverage, high_risk_recall, accuracy, float(low), float(high)))
    if not candidates:
        raise RuntimeError("rolling OOF cannot satisfy selective accuracy target")
    coverage, recall, accuracy, low, high = sorted(candidates, reverse=True)[0]
    return {
        "low_probability_max": low,
        "high_probability_min": high,
        "selection_source": "rolling_oof_2022_2023",
        "minimum_decided_accuracy": minimum_decided_accuracy,
        "oof_coverage": coverage,
        "oof_accuracy_on_decided": accuracy,
        "oof_high_risk_recall": recall,
    }


def _evaluate(labels: np.ndarray, probabilities: np.ndarray, policy: dict) -> dict:
    low = float(policy["low_probability_max"])
    high = float(policy["high_probability_min"])
    decided = (probabilities <= low) | (probabilities >= high)
    predicted_high = probabilities >= high
    return {
        "sample_count": int(len(labels)),
        "decided_count": int(decided.sum()),
        "uncertain_count": int((~decided).sum()),
        "coverage": float(decided.mean()),
        "accuracy_on_decided": float(
            (predicted_high[decided] == labels[decided]).mean()
        ),
        "high_risk_recall_overall": float(
            ((predicted_high == 1) & decided & (labels == 1)).sum()
            / max(1, int((labels == 1).sum()))
        ),
    }


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), "--project-root"),
    source_model: Path = typer.Option(
        Path("artifacts/models/enhanced_blend3_v2"), "--source-model"
    ),
    destination: Path = typer.Option(
        Path("artifacts/models/enhanced_blend3_v2_selective"), "--destination"
    ),
    minimum_decided_accuracy: float = typer.Option(0.85, "--minimum-decided-accuracy"),
) -> None:
    root = project_root.resolve()
    source = (root / source_model).resolve()
    target = (root / destination).resolve()
    finalizer = _load_finalizer(root)
    bundle = joblib.load(source / "model.joblib")
    frame = pd.read_csv(
        root / "data/processed/modeling/agent_augmented_feature_matrix_multihorizon.csv",
        dtype={"company_id": str},
    )
    frame["modeling_year"] = pd.to_datetime(frame["prospectus_date"]).dt.year.astype(int)
    dev = frame.loc[frame["modeling_year"].isin(finalizer.DEV_YEARS)]
    frequency = dev["industry_primary"].fillna("unknown").value_counts(normalize=True)
    frame["industry_freq"] = (
        frame["industry_primary"].fillna("unknown").map(frequency).fillna(0.0)
    )
    features = bundle["features"]
    seeds = bundle["seeds"]

    oof_labels: list[np.ndarray] = []
    oof_probabilities: list[np.ndarray] = []
    for validation_year in finalizer.ROLLING_VALIDATION_YEARS:
        train = frame.loc[
            frame["modeling_year"].between(finalizer.DEV_YEARS[0], validation_year - 1)
        ]
        validation = frame.loc[frame["modeling_year"] == validation_year]
        weight = finalizer._sample_weight(train)
        per_seed: list[np.ndarray] = []
        for seed in seeds:
            lgb_model, _ = finalizer._fit_lgb_with_early_stopping(
                train, validation, features, seed=seed, weight=weight
            )
            hgb_model = finalizer._fit_hgb(train, features, seed=seed, weight=weight)
            log_model = finalizer._fit_logistic(train, features, seed=seed, weight=weight)
            per_seed.append(
                np.mean(
                    [
                        lgb_model.predict_proba(validation[features])[:, 1],
                        hgb_model.predict_proba(validation[features])[:, 1],
                        log_model.predict_proba(validation[features])[:, 1],
                    ],
                    axis=0,
                )
            )
        oof_labels.append(validation[finalizer.TARGET].astype(int).to_numpy())
        oof_probabilities.append(np.mean(per_seed, axis=0))
    policy = _select_policy(
        np.concatenate(oof_labels),
        np.concatenate(oof_probabilities),
        minimum_decided_accuracy=minimum_decided_accuracy,
    )

    test = frame.loc[frame["modeling_year"] == finalizer.TEST_YEAR]
    test_per_seed = []
    for index in range(len(seeds)):
        test_per_seed.append(
            np.mean(
                [
                    bundle["lgb_models"][index].predict_proba(test[features])[:, 1],
                    bundle["hgb_models"][index].predict_proba(test[features])[:, 1],
                    bundle["log_models"][index].predict_proba(test[features])[:, 1],
                ],
                axis=0,
            )
        )
    test_probability = np.mean(test_per_seed, axis=0)
    evaluation = _evaluate(
        test[finalizer.TARGET].astype(int).to_numpy(), test_probability, policy
    )

    target.mkdir(parents=True, exist_ok=True)
    upgraded = dict(bundle)
    upgraded["model_name"] = "enhanced_blend3_v2_selective"
    upgraded["selective_policy"] = policy
    joblib.dump(upgraded, target / "model.joblib")
    for filename in ("metrics.json", "predictions_2024.csv"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)
    (target / "selective_metrics.json").write_text(
        json.dumps(
            {
                "policy": policy,
                "test_2024": evaluation,
                "note": "Accuracy is measured only on auto-decided rows; uncertain rows require review.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(json.dumps({"policy": policy, "test_2024": evaluation}, ensure_ascii=False, indent=2))
    typer.echo(f"已写入：{target}")


if __name__ == "__main__":
    app()
