"""一条命令构建特征并训练五日显著下跌预测基线。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling import (
    build_competition_modeling_dataset,
    train_day5_baselines,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    force_market_rebuild: bool = typer.Option(
        False,
        "--force-market-rebuild",
        help="重新扫描赛事全市场行情",
    ),
) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    typer.echo("第1/2步：生成严格上市前可用的特征矩阵")
    dataset = build_competition_modeling_dataset(
        root,
        market_windows=tuple(config.prediction.market_feature_windows),
        recent_ipo_lookback_days=config.prediction.recent_ipo_lookback_days,
        force_market_rebuild=force_market_rebuild,
    )
    typer.echo(
        f"特征完成：{dataset.row_count}家公司，"
        f"{len(dataset.feature_columns)}个特征"
    )
    typer.echo("第2/2步：训练招股书、市场和融合三组时间切分基线")
    training = train_day5_baselines(
        dataset.feature_matrix_path,
        summary_path=dataset.summary_path,
        output_dir=root / "artifacts" / "models" / "day5_significant_drop_v1",
        random_state=config.prediction.random_state,
        top_k_fraction=config.prediction.top_k_fraction,
        max_alert_fraction=config.prediction.max_alert_fraction,
    )
    typer.echo(
        f"全部完成：验证集阈值={training.selected_threshold:.4f}，"
        f"评测={training.metrics_path}"
    )


if __name__ == "__main__":
    app()
