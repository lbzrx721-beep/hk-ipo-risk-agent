"""合并2025年外部核验数据，并刷新基础/RAG建模矩阵。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.experiments import build_consolidated_agent_dataset
from ipo_risk.modeling import build_competition_modeling_dataset
from ipo_risk.modeling.supplements import merge_fundamental_supplements


app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: list[Path] = typer.Option(
        ...,
        "--source",
        help="补充数据文件；可重复传入，后传入的记录覆盖前面的同公司记录。",
    ),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    year: int = typer.Option(2025, help="补充数据所属年份"),
    rebuild_modeling_tables: bool = typer.Option(
        True,
        "--rebuild-modeling-tables/--no-rebuild-modeling-tables",
        help="导入后刷新上市前特征表和Agent融合表。",
    ),
) -> None:
    root = project_root.resolve()
    result = merge_fundamental_supplements(
        root,
        source_paths=[path.resolve() for path in source],
        year=year,
    )
    typer.echo(f"{year}年补充数据已去重合并：{result.company_count}家公司。")
    typer.echo(f"正式补充文件：{result.supplement_path}")
    typer.echo(f"结构化矩阵：{result.structured_matrix_path}")
    typer.echo(f"RAG矩阵：{result.rag_matrix_path}")
    if rebuild_modeling_tables:
        base = build_competition_modeling_dataset(root)
        typer.echo(f"上市前特征表已刷新：{base.row_count}家公司。")
        augmented = build_consolidated_agent_dataset(root)
        typer.echo(f"Agent融合表已刷新：{augmented.company_count}家公司。")


if __name__ == "__main__":
    app()
