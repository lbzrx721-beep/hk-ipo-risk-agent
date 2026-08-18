"""比较基础特征模型和 Agent 增强特征模型。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling import train_agent_feature_comparison


app = typer.Typer(add_completion=False)


@app.command()
def main(project_root: Path = typer.Option(Path.cwd(), help="项目根目录")) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    modeling_dir = root / "data" / "processed" / "modeling"
    result = train_agent_feature_comparison(
        modeling_dir / "agent_augmented_feature_matrix.csv",
        base_summary_path=modeling_dir / "feature_matrix_summary.json",
        agent_summary_path=modeling_dir / "agent_augmented_feature_summary.json",
        output_dir=root / "artifacts" / "models" / "day5_agent_comparison_v1",
        random_state=config.prediction.random_state,
        top_k_fraction=config.prediction.top_k_fraction,
        max_alert_fraction=config.prediction.max_alert_fraction,
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    counts = metrics["cohort_counts"]
    typer.echo(
        f"完成：训练={counts['train']}，验证={counts['validation']}，"
        f"测试={counts['test']}，外部案例={metrics['external_case_count']}"
    )
    for variant in (
        "logistic_base_64",
        "logistic_agent_augmented",
        "lightgbm_base_64",
        "lightgbm_agent_augmented",
    ):
        validation = metrics["variants"][variant]["splits"]["validation"]
        test = metrics["variants"][variant]["splits"]["test"]
        typer.echo(
            f"{variant}: 验证PR-AUC={validation['pr_auc']:.4f}、"
            f"测试PR-AUC={test['pr_auc']:.4f}、测试F2={test['f2']:.4f}"
        )
    typer.echo(f"完整指标：{result.metrics_path}")
    typer.echo(f"逐公司预测：{result.predictions_path}")
    typer.echo(f"LightGBM特征重要性：{result.feature_importance_path}")


if __name__ == "__main__":
    app()
