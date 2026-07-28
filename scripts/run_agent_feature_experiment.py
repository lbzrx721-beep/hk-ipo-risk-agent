"""批量生成历史招股书Agent预测特征。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.experiments import run_agent_feature_experiment


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    experiment_name: str = typer.Option(
        "agent_feature_pilot_v1",
        help="实验名称；模型或提示词变化后必须使用新名称",
    ),
    sample_size: int = typer.Option(20, min=1, help="自动抽样公司数量"),
    split: list[str] | None = typer.Option(
        None,
        "--split",
        help="可重复填写，例如 --split train --split validation",
    ),
    company_folder: list[str] | None = typer.Option(
        None,
        "--company-folder",
        help="显式指定公司时可重复填写，并忽略自动抽样",
    ),
    min_findings: int = typer.Option(1, min=1, help="最少候选风险数量"),
    stop_on_error: bool = typer.Option(
        False,
        "--stop-on-error",
        help="任一公司失败后立即停止；默认记录失败并继续",
    ),
) -> None:
    result = run_agent_feature_experiment(
        project_root,
        experiment_name=experiment_name,
        sample_size=sample_size,
        splits=tuple(split or ("train", "validation")),
        company_folders=tuple(company_folder or ()),
        min_findings=min_findings,
        continue_on_error=not stop_on_error,
    )
    typer.echo(
        f"实验完成：选择{result.selected_count}家，"
        f"成功{result.completed_count}家，失败{result.failed_count}家。"
    )
    typer.echo(f"状态：{result.status_path}")
    typer.echo(f"摘要：{result.summary_path}")
    typer.echo(f"Agent特征：{result.agent_feature_matrix_path}")
    if result.augmented_feature_matrix_path is not None:
        typer.echo(f"融合矩阵：{result.augmented_feature_matrix_path}")


if __name__ == "__main__":
    app()
