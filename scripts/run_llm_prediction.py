"""对单家公司运行LLM历史先例委员会预测。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from ipo_risk.agents import OpenAICompatibleJsonChatProvider
from ipo_risk.config import load_config
from ipo_risk.modeling.llm_prediction import (
    LLMPredictionCommittee,
    LLMPredictionSingleJudge,
    build_llm_prediction_payload,
    load_latest_risk_report,
    select_historical_cases,
)
from ipo_risk.modeling.historical_cases import PgVectorHistoricalCaseStore
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.storage.files import write_json


app = typer.Typer(add_completion=False)


def _progress(event: str, detail: str) -> None:
    typer.echo(f"[{event}] {detail}")


@app.command()
def main(
    company_id: str = typer.Option(..., "--company-id", help="五位港股代码"),
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    matrix_path: Path | None = typer.Option(None, help="特征矩阵CSV"),
    top_k: int = typer.Option(8, min=1, max=20, help="历史相似案例数"),
    case_retrieval: str = typer.Option(
        "none",
        help="历史案例检索：none（当前默认）、hybrid、numeric或auto（实验兼容）",
    ),
    prediction_mode: str = typer.Option(
        "committee",
        "--prediction-mode",
        help="预测模式：committee（三Agent）或single（单一决策Agent）",
    ),
    dry_run: bool = typer.Option(False, help="仅生成模型输入，不调用付费模型"),
) -> None:
    root = project_root.resolve()
    normalized_id = company_id.strip().zfill(5)
    retrieval_mode = case_retrieval.strip().lower()
    mode = prediction_mode.strip().lower()
    if mode not in {"committee", "single"}:
        raise typer.BadParameter("prediction-mode must be committee or single")
    if retrieval_mode not in {"none", "auto", "hybrid", "numeric"}:
        raise typer.BadParameter("case-retrieval must be none, auto, hybrid or numeric")
    if retrieval_mode == "auto":
        retrieval_mode = "numeric" if dry_run else "hybrid"
    matrix = (
        matrix_path.resolve()
        if matrix_path is not None
        else root / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix.csv"
    )
    frame = pd.read_csv(matrix, dtype={"company_id": str})
    ids = frame["company_id"].astype(str).str.zfill(5)
    matches = frame.loc[ids == normalized_id]
    if matches.empty:
        raise typer.BadParameter(f"特征矩阵中找不到公司：{normalized_id}")
    row = matches.sort_values("prospectus_date").iloc[-1]
    config = load_config(root / "configs" / "base.yaml")
    case_store: PgVectorHistoricalCaseStore | None = None
    if retrieval_mode == "hybrid":
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            model_name=config.embedding.model_name,
            dimensions=config.embedding.dimensions,
            batch_size=1,
            request_concurrency=1,
            timeout_seconds=config.embedding.timeout_seconds,
            api_key=config.embedding.require_api_key(),
            base_url=config.embedding.base_url,
            max_retries=config.embedding.max_retries,
            retry_initial_seconds=config.embedding.retry_initial_seconds,
            retry_backoff_factor=config.embedding.retry_backoff_factor,
            retry_max_seconds=config.embedding.retry_max_seconds,
        )
        case_store = PgVectorHistoricalCaseStore(
            embedding_provider,
            dsn=config.vector_store.require_postgres_dsn(),
            schema=config.vector_store.postgres_schema,
            connect_timeout_seconds=config.vector_store.postgres_connect_timeout_seconds,
        )
        try:
            case_store.ensure_schema()
            if case_store.count == 0:
                raise RuntimeError(
                    "历史案例向量库为空，请先执行 python scripts/build_historical_case_index.py"
                )
            cases = select_historical_cases(
                frame,
                normalized_id,
                top_k=top_k,
                case_store=case_store,
            )
        finally:
            case_store.close()
    elif retrieval_mode == "numeric":
        cases = select_historical_cases(frame, normalized_id, top_k=top_k)
    else:
        cases = ()
    report = load_latest_risk_report(root, normalized_id)
    payload = build_llm_prediction_payload(row, report=report, historical_cases=cases)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        root
        / "artifacts"
        / "predictions"
        / ("llm_single_judge" if mode == "single" else "llm_committee")
        / normalized_id
    )

    if dry_run:
        output = output_dir / f"{stamp}_input.json"
        write_json(payload, output, pretty=True)
        typer.echo(
            f"输入已生成：基础字段={sum(len(payload[key]) for key in ('offering', 'fundamentals', 'derived_ratios', 'prelisting_market'))}项，"
            f"Agent风险={len(payload['agent_risks'])}条，历史案例={len(cases)}家，"
            f"检索模式={retrieval_mode}。"
        )
        typer.echo(f"文件：{output}")
        return

    if not config.llm.is_configured:
        raise RuntimeError("LLM未配置，请检查configs/providers.local.yaml")
    provider = OpenAICompatibleJsonChatProvider(
        api_key=config.llm.require_api_key(),
        base_url=config.llm.base_url,
        model_name=config.llm.model_name,
        provider_name=config.llm.provider,
        request_extra_body=(None if config.llm.disable_thinking else {}),
        supports_json_object=config.llm.supports_json_object,
        reasoning_effort=config.llm.reasoning_effort,
    )
    prompt_dir = root / "prompts" / "prediction"
    retry_policy = RetryPolicy(
        max_retries=config.agents.max_retries,
        initial_seconds=config.agents.retry_initial_seconds,
        backoff_factor=config.agents.retry_backoff_factor,
        max_seconds=config.agents.retry_max_seconds,
        retry_if=retry_model_node_error,
    )
    max_tokens = min(config.llm.max_output_tokens, 4_096)
    predictor: LLMPredictionCommittee | LLMPredictionSingleJudge
    if mode == "single":
        predictor = LLMPredictionSingleJudge(
            provider,
            prompt=(prompt_dir / "single_judge_v1.md").read_text(encoding="utf-8"),
            temperature=0.0,
            max_output_tokens=max_tokens,
            retry_policy=retry_policy,
            progress_callback=_progress,
        )
    else:
        predictor = LLMPredictionCommittee(
            provider,
            bull_prompt=(prompt_dir / "bull_v2.md").read_text(encoding="utf-8"),
            bear_prompt=(prompt_dir / "bear_v2.md").read_text(encoding="utf-8"),
            decision_prompt=(prompt_dir / "decision_v2.md").read_text(encoding="utf-8"),
            temperature=0.0,
            max_output_tokens=max_tokens,
            retry_policy=retry_policy,
            progress_callback=_progress,
        )
    result = predictor.predict(payload)
    output = output_dir / f"{stamp}_prediction.json"
    write_json(result.model_dump(mode="json"), output, pretty=True)
    typer.echo(
        f"完成：风险评分={result.risk_score}/100，等级={result.risk_level}，"
        f"五日显著下跌预警={'是' if result.predicted_significant_drop_5d else '否'}，"
        f"业务阈值={result.alert_threshold}，置信度={result.confidence:.0%}。"
    )
    typer.echo(f"结果：{output}")


if __name__ == "__main__":
    app()
