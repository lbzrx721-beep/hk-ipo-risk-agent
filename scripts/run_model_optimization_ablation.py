"""分阶段验证派生特征、Agent交互、时间权重和多种子集成。"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling.stable_agent_model import (
    AGENT_SEMANTIC_FEATURES,
    BUSINESS_BASE_FEATURES,
    RAW_BUSINESS_BASE_FEATURES,
    train_stable_agent_model,
)
from ipo_risk.storage.files import write_json


app = typer.Typer(add_completion=False)


def _without_agent_interactions() -> tuple[str, ...]:
    return tuple(
        feature
        for feature in AGENT_SEMANTIC_FEATURES
        if not feature.startswith("agent_interaction_")
    )


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    reuse_existing: bool = typer.Option(True, help="复用已有同阶段指标"),
) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    modeling = root / "data" / "processed" / "modeling"
    destination = root / "artifacts" / "models" / "day5_optimization_ablation"
    destination.mkdir(parents=True, exist_ok=True)
    no_interactions = _without_agent_interactions()
    stages: list[dict[str, Any]] = [
        {
            "name": "01_legacy_raw",
            "base": RAW_BUSINESS_BASE_FEATURES,
            "agent": no_interactions,
            "time_decay": False,
            "ensemble_seed_count": 1,
            "stability_seed_count": 1,
        },
        {
            "name": "02_engineered_ratios",
            "base": BUSINESS_BASE_FEATURES,
            "agent": no_interactions,
            "time_decay": False,
            "ensemble_seed_count": 1,
            "stability_seed_count": 1,
        },
        {
            "name": "03_agent_interactions",
            "base": BUSINESS_BASE_FEATURES,
            "agent": AGENT_SEMANTIC_FEATURES,
            "time_decay": False,
            "ensemble_seed_count": 1,
            "stability_seed_count": 1,
        },
        {
            "name": "04_time_decay",
            "base": BUSINESS_BASE_FEATURES,
            "agent": AGENT_SEMANTIC_FEATURES,
            "time_decay": True,
            "ensemble_seed_count": 1,
            "stability_seed_count": 1,
        },
        {
            "name": "05_multi_seed",
            "base": BUSINESS_BASE_FEATURES,
            "agent": AGENT_SEMANTIC_FEATURES,
            "time_decay": True,
            "ensemble_seed_count": 5,
            "stability_seed_count": 3,
        },
    ]
    rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_dir = destination / str(stage["name"])
        metrics_path = stage_dir / "metrics.json"
        if not (reuse_existing and metrics_path.is_file()):
            typer.echo(f"训练阶段：{stage['name']}")
            train_stable_agent_model(
                modeling / "agent_augmented_feature_matrix.csv",
                base_summary_path=modeling / "feature_matrix_summary.json",
                agent_summary_path=modeling / "agent_augmented_feature_summary.json",
                output_dir=stage_dir,
                random_state=config.prediction.random_state,
                top_k_fraction=config.prediction.top_k_fraction,
                base_feature_whitelist=tuple(stage["base"]),
                agent_feature_whitelist=tuple(stage["agent"]),
                use_time_decay=bool(stage["time_decay"]),
                ensemble_seed_count=int(stage["ensemble_seed_count"]),
                stability_seed_count=int(stage["stability_seed_count"]),
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for model_variant, key in (
            ("business_base", "test_2024_base"),
            ("agent_augmented", "test_2024"),
        ):
            score = metrics[key]
            rolling_key = (
                "base_historical_tuning_performance"
                if model_variant == "business_base"
                else "historical_tuning_performance"
            )
            rows.append(
                {
                    "stage": stage["name"],
                    "model_variant": model_variant,
                    "time_decay": bool(stage["time_decay"]),
                    "ensemble_seed_count": int(stage["ensemble_seed_count"]),
                    "rolling_pr_auc_2022_2023": metrics[rolling_key]["pr_auc"],
                    **{
                        metric: score[metric]
                        for metric in (
                            "pr_auc",
                            "roc_auc",
                            "precision",
                            "recall",
                            "f1",
                            "f2",
                            "accuracy",
                            "brier_score",
                            "alert_count",
                        )
                    },
                }
            )
    ranked = sorted(rows, key=lambda row: float(row["pr_auc"]), reverse=True)
    agent_ranked = [row for row in ranked if row["model_variant"] == "agent_augmented"]
    historical_candidate = max(
        rows, key=lambda row: float(row["rolling_pr_auc_2022_2023"])
    )
    summary_path = destination / "summary.json"
    summary_csv_path = destination / "summary.csv"
    write_json(
        {
            "schema_version": 1,
            "selection_note": (
                "所有特征、权重、模型数量和阈值均只用2020-2023确定；"
                "2024仅用于同口径独立评测，不参与拟合。"
            ),
            "stages": rows,
            "historical_only_deployment_candidate": historical_candidate,
            "best_2024_research_result": ranked[0],
            "best_2024_agent_inclusive_result": agent_ranked[0],
        },
        summary_path,
    )
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"完成：{summary_path}")
    typer.echo(f"表格：{summary_csv_path}")
    typer.echo(
        f"当前最高PR-AUC={ranked[0]['pr_auc']:.4f}，"
        f"阶段={ranked[0]['stage']}，模型={ranked[0]['model_variant']}。"
    )


if __name__ == "__main__":
    app()
