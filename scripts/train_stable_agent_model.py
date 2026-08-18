"""训练滚动验证筛选后的稳定 Agent 融合模型。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling import train_stable_agent_model


app = typer.Typer(add_completion=False)


@app.command()
def main(project_root: Path = typer.Option(Path.cwd(), help="项目根目录")) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    modeling = root / "data" / "processed" / "modeling"
    result = train_stable_agent_model(
        modeling / "agent_augmented_feature_matrix.csv",
        base_summary_path=modeling / "feature_matrix_summary.json",
        agent_summary_path=modeling / "agent_augmented_feature_summary.json",
        output_dir=(root / "artifacts" / "models" / "day5_stable_agent_v5_strict_ablation"),
        random_state=config.prediction.random_state,
        top_k_fraction=config.prediction.top_k_fraction,
        ensemble_seed_count=5,
        stability_seed_count=3,
        use_time_decay=True,
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    test = metrics["test_2024"]
    base_test = metrics["test_2024_base"]
    delta = metrics["agent_incremental_value_2024"]
    manifest = json.loads(result.feature_manifest_path.read_text(encoding="utf-8"))
    typer.echo(
        f"完成：稳定特征={manifest['selected_feature_count']}，"
        f"2024准确率={test['accuracy']:.4f}，PR-AUC={test['pr_auc']:.4f}，"
        f"召回率={test['recall']:.4f}，预警准确率={test['precision']:.4f}。"
    )
    typer.echo(
        f"消融对比：基础模型PR-AUC={base_test['pr_auc']:.4f}，"
        f"Agent增强PR-AUC={test['pr_auc']:.4f}，"
        f"增量={delta['pr_auc_delta']:+.4f}；"
        f"基础召回率={base_test['recall']:.4f}，"
        f"Agent增强召回率={test['recall']:.4f}。"
    )
    typer.echo(f"完整指标：{result.metrics_path}")
    typer.echo(f"稳定特征：{result.feature_stability_path}")
    typer.echo(f"逐公司预测：{result.predictions_path}")


if __name__ == "__main__":
    app()
