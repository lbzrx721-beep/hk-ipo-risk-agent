"""把一家公司或全部公司原始资料登记到SQLite目录。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.storage import (
    CatalogRepository,
    CompanyIngestionResult,
    register_all_companies,
    register_company_sources,
)


app = typer.Typer(add_completion=False)


def _show_result(result: CompanyIngestionResult) -> None:
    typer.echo(
        f"{result.company.company_id} {result.company.company_name_zh}: "
        f"文档{len(result.documents)}份，结构化数据{len(result.datasets)}份"
    )
    for document in result.documents:
        typer.echo(
            f"  [文档] {document.document_type} | {document.document_id} | "
            f"可用日期 {document.available_at}"
        )
    for dataset in result.datasets:
        typer.echo(
            f"  [数据] {dataset.dataset_type} | {dataset.dataset_id} | "
            f"上市前可用={dataset.pre_listing_eligible}"
        )
    for warning in result.warnings:
        typer.echo(f"  [待处理] {warning}", err=True)


@app.command()
def main(
    company_folder: str = typer.Option(
        "03696_英矽智能",
        help="data/raw下统一使用的 股票代码_公司名称 目录名",
    ),
    all_companies: bool = typer.Option(
        False,
        "--all",
        help="扫描并登记company_info目录中的全部公司",
    ),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    root = project_root.resolve()
    if all_companies:
        results = register_all_companies(project_root=root)
    else:
        results = (register_company_sources(company_folder, project_root=root),)
    for result in results:
        _show_result(result)

    config = load_config()
    repository = CatalogRepository(root / config.paths.database)
    counts = repository.counts()
    typer.echo(
        "数据库合计: "
        f"公司{counts.companies}，文档{counts.documents}，"
        f"数据集{counts.datasets}，分析任务{counts.analysis_runs}"
    )
    typer.echo(f"SQLite: {repository.database_path}")


if __name__ == "__main__":
    app()
