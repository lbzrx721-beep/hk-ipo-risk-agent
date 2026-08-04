"""接入东吴证券赛事官方数据并生成标准公司资料和市场标签。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.competition import (
    import_competition_delivery,
    reconcile_backed_up_market_dataset,
)
from ipo_risk.storage import CatalogRepository, register_company_sources


app = typer.Typer(add_completion=False)


@app.command()
def main(
    source_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="赛事交付目录；可以是下载外层目录或实际数据目录",
    ),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    delivery_target: Path | None = typer.Option(
        None,
        help="项目内官方数据快照目录；默认使用标准competition目录",
    ),
    significant_drop_threshold_pct: float = typer.Option(
        -10.0,
        help="显著下跌标签阈值；-10表示相对发行价跌幅达到10%",
    ),
    register_catalog: bool = typer.Option(
        True,
        "--register-catalog/--no-register-catalog",
        help="完成后把565家公司、文档和行情登记到SQLite目录",
    ),
) -> None:
    root = project_root.resolve()
    target = delivery_target.resolve() if delivery_target else None
    result = import_competition_delivery(
        source_root,
        project_root=root,
        delivery_target=target,
        significant_drop_threshold_pct=significant_drop_threshold_pct,
    )
    typer.echo(
        f"已接入{result.company_count}家公司："
        f"完整标签{result.full_label_count}，部分标签{result.partial_label_count}，"
        f"不可用{result.unavailable_count}。"
    )
    typer.echo(
        f"原始数据硬链接{result.hardlinked_files}个，"
        f"实际复制{result.copied_files}个。"
    )
    typer.echo(f"公司清单：{result.manifest_path}")
    typer.echo(f"市场标签：{result.labels_path}")
    typer.echo(f"质量报告：{result.quality_report_path}")

    if not register_catalog:
        return
    repository = CatalogRepository(root / "data" / "app.db")
    repository.initialize()
    failures: list[str] = []
    for index, company_folder in enumerate(result.company_folders, start=1):
        try:
            reconcile_backed_up_market_dataset(
                repository,
                project_root=root,
                company_folder=company_folder,
            )
            register_company_sources(
                company_folder,
                project_root=root,
                repository=repository,
            )
        except Exception as exc:  # CLI必须汇总全部失败公司后再退出
            failures.append(f"{company_folder}: {type(exc).__name__}: {exc}")
        if index % 50 == 0 or index == len(result.company_folders):
            typer.echo(f"目录登记进度：{index}/{len(result.company_folders)}")
    if failures:
        for failure in failures:
            typer.echo(f"[登记失败] {failure}", err=True)
        raise typer.Exit(code=1)
    counts = repository.counts()
    typer.echo(
        f"SQLite登记完成：公司{counts.companies}，文档{counts.documents}，"
        f"数据集{counts.datasets}。"
    )


if __name__ == "__main__":
    app()
