"""复用现有解析缓存，构建发行与财务特征并刷新统一训练表。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.experiments import build_consolidated_agent_dataset
from ipo_risk.modeling import (
    build_competition_modeling_dataset,
    build_structured_fundamental_dataset,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    company_id: list[str] | None = typer.Option(
        None,
        "--company-id",
        help="只处理指定股票代码；可重复传入。默认处理全部公司。",
    ),
    rebuild_modeling_tables: bool = typer.Option(
        True,
        "--rebuild-modeling-tables/--no-rebuild-modeling-tables",
        help="提取后刷新基础特征表和Agent融合表。",
    ),
) -> None:
    root = project_root.resolve()
    result = build_structured_fundamental_dataset(
        root,
        company_ids=company_id or (),
    )
    typer.echo(
        f"结构化提取完成：{result.company_count}家公司，"
        f"{len(result.feature_columns)}个候选特征。"
    )
    typer.echo(f"特征矩阵：{result.feature_matrix_path}")
    typer.echo(f"来源审计：{result.audit_path}")
    typer.echo(f"覆盖率报告：{result.summary_path}")

    if rebuild_modeling_tables:
        base = build_competition_modeling_dataset(root)
        typer.echo(
            f"基础特征表已刷新：{base.row_count}家公司，"
            f"{len(base.feature_columns)}个特征。"
        )
        augmented = build_consolidated_agent_dataset(root)
        typer.echo(f"Agent融合表已刷新：{augmented.company_count}家公司。")


if __name__ == "__main__":
    app()
