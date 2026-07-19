"""运行单份公司招股书风险分析。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.pipeline import analyze_company


app = typer.Typer(add_completion=False)


@app.command()
def main(
    company_folder: str = typer.Option(
        "03696_英矽智能",
        help="data/raw 下统一使用的 股票代码_公司名称 目录名",
    ),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    document_id: str | None = typer.Option(
        None,
        help="可选的已登记招股书document_id；省略时选择最新正式版本",
    ),
    min_findings: int = typer.Option(5, min=1, help="最少有证据风险数量"),
) -> None:
    artifacts = analyze_company(
        company_folder,
        project_root=project_root,
        document_id=document_id,
        min_findings=min_findings,
    )
    typer.echo(f"风险结论: {artifacts.report.summary.finding_count}")
    typer.echo(f"证据截图: {artifacts.report.summary.screenshot_count}")
    typer.echo(f"风险 JSON: {artifacts.report_path}")
    typer.echo(f"页面数据: {artifacts.pages_path}")
    typer.echo(f"段落切片: {artifacts.chunks_path}")


if __name__ == "__main__":
    app()
