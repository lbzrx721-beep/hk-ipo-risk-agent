"""只读检查东吴证券赛事官方数据的完整性和标签覆盖率。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ipo_risk.competition import audit_competition_delivery


app = typer.Typer(add_completion=False)


@app.command()
def main(
    source_root: Path = typer.Option(
        Path("data/raw/competition/dongwu_hk_ipo_2020_2025"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="含年度招股书目录及三张CSV的赛事数据目录",
    ),
    output: Path | None = typer.Option(
        None,
        help="可选的UTF-8 JSON审计报告输出路径",
    ),
) -> None:
    report = audit_competition_delivery(source_root)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"审计报告：{output.resolve()}")
    typer.echo(text)


if __name__ == "__main__":
    app()
