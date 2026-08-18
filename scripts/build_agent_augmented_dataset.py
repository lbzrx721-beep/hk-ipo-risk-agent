"""重建历史Agent语义特征，并构建四期限标签的统一建模数据集。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.experiments import (
    build_consolidated_agent_dataset,
    normalize_historical_agent_features,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    normalized = normalize_historical_agent_features(project_root)
    typer.echo(
        f"语义标准化：{normalized.completed_count}家公司；"
        f"未重新调用PDF、向量模型或Agent。"
    )
    result = build_consolidated_agent_dataset(project_root)
    typer.echo(
        f"完成：{result.company_count}家公司，"
        f"清理{result.duplicate_record_count}条重复记录。"
    )
    typer.echo(f"Agent特征：{result.agent_feature_matrix_path}")
    typer.echo(f"融合数据集：{result.augmented_feature_matrix_path}")
    typer.echo(f"来源审计：{result.audit_path}")
    typer.echo(f"质量报告：{result.summary_path}")


if __name__ == "__main__":
    app()
