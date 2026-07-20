"""核验、复核并导出一条分析任务中的风险结论。"""

from __future__ import annotations

from pathlib import Path

import typer

from ipo_risk.config import load_config
from ipo_risk.domain import ReviewDecision, RiskFinding
from ipo_risk.reviewing import IncompleteReviewError, ReviewBlockedError, ReviewService
from ipo_risk.storage.files import read_json
from ipo_risk.storage.repositories import CatalogRepository


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _service(project_root: Path) -> ReviewService:
    root = project_root.resolve()
    config = load_config(
        config_path=root / "configs" / "base.yaml",
        providers_path=root / "configs" / "providers.local.yaml",
    )
    repository = CatalogRepository(root / config.paths.database)
    repository.initialize()
    return ReviewService(repository=repository, project_root=root)


@app.command("list")
def list_findings(
    run_id: str = typer.Option(..., help="分析任务run_id"),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    """列出任务中的全部风险及当前复核状态。"""
    service = _service(project_root)
    findings = service.repository.list_findings(run_id)
    if not findings:
        typer.echo(f"未找到风险结论: {run_id}", err=True)
        raise typer.Exit(1)
    for finding in findings:
        typer.echo(
            f"{finding.finding_id}\t{finding.status.value}\t"
            f"{finding.severity.value}\t{finding.title}"
        )


@app.command("verify")
def verify(
    run_id: str = typer.Option(..., help="分析任务run_id"),
    finding_id: str = typer.Option(..., help="风险finding_id"),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    """执行不改变状态的确定性核验。"""
    result = _service(project_root).verify_finding(run_id, finding_id)
    typer.echo(f"{finding_id}: {'PASS' if result.passed else 'FAIL'}")
    for check in result.checks:
        typer.echo(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    for evidence in result.evidence_verifications:
        typer.echo(
            f"  [{'PASS' if evidence.passed else 'FAIL'}] evidence:{evidence.evidence_id}"
        )
        for check in evidence.checks:
            typer.echo(
                f"    [{'PASS' if check.passed else 'FAIL'}] "
                f"{check.name}: {check.detail}"
            )
    if not result.passed:
        raise typer.Exit(2)


@app.command("review")
def review(
    run_id: str = typer.Option(..., help="分析任务run_id"),
    finding_id: str = typer.Option(..., help="风险finding_id"),
    decision: ReviewDecision = typer.Option(..., help="confirmed/rejected/edited"),
    reviewer: str = typer.Option(..., help="复核人姓名或账号"),
    comment: str | None = typer.Option(None, help="复核意见"),
    edited_finding: Path | None = typer.Option(None, help="edited决定使用的RiskFinding JSON"),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    """确认、驳回或编辑一条候选风险并写入审计记录。"""
    edited = (
        RiskFinding.model_validate(read_json(edited_finding))
        if edited_finding is not None
        else None
    )
    try:
        result = _service(project_root).review_finding(
            run_id=run_id,
            finding_id=finding_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
            edited_finding=edited,
        )
    except ReviewBlockedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"{result.review_id}: {result.finding_id} -> "
        f"{result.resulting_finding.status.value}"
    )


@app.command("export")
def export(
    run_id: str = typer.Option(..., help="分析任务run_id"),
    output_path: Path | None = typer.Option(None, help="最终报告输出路径"),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
) -> None:
    """在全部风险完成复核后导出verified风险报告。"""
    try:
        path = _service(project_root).export_verified_report(
            run_id,
            output_path=output_path,
        )
    except IncompleteReviewError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"最终报告: {path}")


if __name__ == "__main__":
    app()
