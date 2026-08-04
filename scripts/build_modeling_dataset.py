"""生成赛事公司的 Point-in-Time 特征矩阵。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.modeling import build_competition_modeling_dataset


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    force_market_rebuild: bool = typer.Option(
        False,
        "--force-market-rebuild",
        help="忽略每日市场情绪缓存并重新扫描1GB赛事行情",
    ),
) -> None:
    root = project_root.resolve()
    config = load_config(root / "configs" / "base.yaml")
    windows = tuple(config.prediction.market_feature_windows)
    typer.echo("正在构建上市前特征；首次运行会扫描赛事全市场行情……")
    result = build_competition_modeling_dataset(
        root,
        market_windows=windows,
        recent_ipo_lookback_days=config.prediction.recent_ipo_lookback_days,
        force_market_rebuild=force_market_rebuild,
    )
    typer.echo(
        f"完成：{result.row_count}家公司，{len(result.feature_columns)}个特征。"
    )
    typer.echo(f"特征矩阵：{result.feature_matrix_path}")
    typer.echo(f"逐特征来源：{result.feature_audit_path}")
    typer.echo(f"构建摘要：{result.summary_path}")


if __name__ == "__main__":
    app()
