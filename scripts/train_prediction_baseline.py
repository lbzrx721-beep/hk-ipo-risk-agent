"""训练首个五日显著下跌预测基线并生成时间回放结果。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling import train_day5_baselines


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    modeling_dir = root / "data" / "processed" / "modeling"
    output_dir = root / "artifacts" / "models" / "day5_significant_drop_v1"
    result = train_day5_baselines(
        modeling_dir / "prelisting_feature_matrix.csv",
        summary_path=modeling_dir / "feature_matrix_summary.json",
        output_dir=output_dir,
        random_state=config.prediction.random_state,
        top_k_fraction=config.prediction.top_k_fraction,
        max_alert_fraction=config.prediction.max_alert_fraction,
    )
    typer.echo(
        f"完成：主模型={result.primary_variant}，"
        f"验证集阈值={result.selected_threshold:.4f}"
    )
    typer.echo(f"模型：{result.model_path}")
    typer.echo(f"时间切分指标：{result.metrics_path}")
    typer.echo(f"逐公司预测：{result.predictions_path}")
    typer.echo(f"解释：{result.explanations_path}")


if __name__ == "__main__":
    app()
