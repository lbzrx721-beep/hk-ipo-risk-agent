"""对比LLM委员会在无案例RAG和历史案例RAG下的五日风险预测。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from ipo_risk.agents import OpenAICompatibleJsonChatProvider
from ipo_risk.agents.usage import ModelPricing, ModelUsageLedger
from ipo_risk.config import load_config
from ipo_risk.modeling.historical_cases import PgVectorHistoricalCaseStore
from ipo_risk.modeling.llm_case_backtest import (
    BACKTEST_ARMS,
    comparison_delta,
    compute_backtest_metrics,
    select_balanced_pilot,
)
from ipo_risk.modeling.llm_prediction import (
    BUSINESS_ALERT_THRESHOLD,
    LLMPredictionCommittee,
    LLMPredictionSingleJudge,
    PROMPT_VERSION,
    SINGLE_JUDGE_PROMPT_VERSION,
    build_llm_prediction_payload,
    load_latest_risk_report,
    select_historical_cases,
)
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider
from ipo_risk.storage.files import read_json, write_json


app = typer.Typer(add_completion=False)


def _progress(event: str, detail: str) -> None:
    typer.echo(f"[{event}] {detail}")


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    matrix_path: Path | None = typer.Option(None, help="Agent增强特征矩阵CSV"),
    split: str = typer.Option("validation", help="回测数据分组"),
    sample_size: int = typer.Option(5, min=2, help="小样本公司数"),
    random_state: int = typer.Option(42, help="固定抽样种子"),
    top_k: int = typer.Option(8, min=1, max=20, help="每家公司历史案例数"),
    max_output_tokens: int = typer.Option(2_048, min=512, max=8_192),
    output_dir: Path | None = typer.Option(None, help="回测输出目录"),
    arm: str = typer.Option(
        "both",
        help="运行组：both、without_case_rag或with_case_rag",
    ),
    prediction_mode: str = typer.Option(
        "committee",
        "--prediction-mode",
        help="预测模式：committee（三Agent）或single（单一决策Agent）",
    ),
    alert_threshold: int = typer.Option(
        BUSINESS_ALERT_THRESHOLD,
        min=1,
        max=99,
        help="业务预警阈值；概率评分本身不变",
    ),
    exclude_backtest: Path | None = typer.Option(
        None,
        help="排除另一份backtest.json中的样本，用于不重叠留出验证",
    ),
) -> None:
    root = project_root.resolve()
    matrix = (
        matrix_path.resolve()
        if matrix_path is not None
        else root / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix.csv"
    )
    frame = pd.read_csv(matrix, dtype={"company_id": str})
    run_arms = _resolve_arms(arm)
    mode = prediction_mode.strip().lower()
    if mode not in {"committee", "single"}:
        raise typer.BadParameter("prediction-mode must be committee or single")
    prompt_version = (
        SINGLE_JUDGE_PROMPT_VERSION if mode == "single" else PROMPT_VERSION
    )
    excluded_ids = _load_excluded_ids(exclude_backtest)
    selection_frame = frame.loc[
        ~frame["company_id"].astype(str).str.zfill(5).isin(excluded_ids)
    ]
    selected = select_balanced_pilot(
        selection_frame,
        split=split,
        sample_size=sample_size,
        random_state=random_state,
    )
    config = load_config(root / "configs" / "base.yaml")
    if not config.llm.is_configured:
        raise RuntimeError("LLM未配置，请检查configs/providers.local.yaml")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_dir.resolve()
        if output_dir is not None
        else root / "artifacts" / "evaluations" / "llm_case_rag" / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "backtest.json"
    sample_rows = [
        {
            "company_id": str(row["company_id"]).zfill(5),
            "company_name": str(row["company_name"]),
            "prospectus_date": str(row["prospectus_date"]),
            "actual_significant_drop_5d": bool(int(row["target_significant_drop_5d"])),
        }
        for _, row in selected.iterrows()
    ]
    if output_path.exists():
        state = read_json(output_path)
        _validate_resume_state(
            state,
            sample_rows=sample_rows,
            model_name=config.llm.model_name,
            split=split,
            top_k=top_k,
            run_arms=run_arms,
            alert_threshold=alert_threshold,
            excluded_ids=excluded_ids,
            prediction_mode=mode,
            prompt_version=prompt_version,
        )
        state["status"] = "running"
        typer.echo(f"断点续跑：已读取{len(state['observations'])}条历史执行记录。")
    else:
        state = {
            "schema_version": "llm-case-rag-backtest-v1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": config.llm.model_name,
            "prompt_version": prompt_version,
            "prediction_mode": mode,
            "arms": list(run_arms),
            "alert_threshold": alert_threshold,
            "excluded_company_ids": sorted(excluded_ids),
            "split": split,
            "sample_size": sample_size,
            "random_state": random_state,
            "top_k": top_k,
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
            "sample": sample_rows,
            "observations": [],
            "metrics": {},
            "comparison_delta": {},
        }
    write_json(state, output_path, pretty=True)
    typer.echo(f"试验目录：{run_dir}")
    completed_keys = {
        (str(item.get("company_id")), str(item.get("arm")))
        for item in state["observations"]
        if not item.get("error")
    }
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        company_id = str(row["company_id"]).zfill(5)
        typer.echo(
            f"[{index}/{sample_size}] {company_id} · {row['company_name']} · "
            f"真实标签={int(row['target_significant_drop_5d'])}"
        )
        if all((company_id, arm_name) in completed_keys for arm_name in run_arms):
            typer.echo("  两组均已完成，断点续跑跳过该公司。")
            continue
        report = load_latest_risk_report(root, company_id)
        cases = (
            _select_hybrid_cases(frame, company_id, top_k=top_k, config=config)
            if "with_case_rag" in run_arms
            else ()
        )
        for arm_name in run_arms:
            if (company_id, arm_name) in completed_keys:
                typer.echo(f"  {arm_name}: 已完成，断点续跑跳过。")
                continue
            arm_cases = () if arm_name == "without_case_rag" else cases
            payload = build_llm_prediction_payload(
                row,
                report=report,
                historical_cases=arm_cases,
            )
            ledger = ModelUsageLedger(pricing=_pricing(config))
            predictor = _predictor(
                root,
                config=config,
                ledger=ledger,
                max_output_tokens=max_output_tokens,
                alert_threshold=alert_threshold,
                prediction_mode=mode,
            )
            started = time.perf_counter()
            observation: dict[str, Any] = {
                "arm": arm_name,
                "company_id": company_id,
                "company_name": str(row["company_name"]),
                "prospectus_date": str(row["prospectus_date"]),
                "actual_significant_drop_5d": bool(int(row["target_significant_drop_5d"])),
                "historical_case_count": len(arm_cases),
            }
            try:
                result = predictor.predict(payload)
                observation.update(result.model_dump(mode="json"))
                typer.echo(
                    f"  {arm_name}: 预测="
                    f"{int(result.predicted_significant_drop_5d)} · "
                    f"评分={result.risk_score} · 置信度={result.confidence:.0%}"
                )
            except Exception as exc:  # noqa: BLE001 - 保存失败公司后继续
                observation["error"] = f"{type(exc).__name__}: {exc}"
                typer.echo(f"  {arm_name}: 失败 · {observation['error']}", err=True)
            observation["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            observation["usage"] = ledger.snapshot().as_dict()
            state["observations"].append(observation)
            state["metrics"] = compute_backtest_metrics(
                state["observations"], alert_threshold=alert_threshold
            )
            state["comparison_delta"] = comparison_delta(state["metrics"])
            write_json(state, output_path, pretty=True)

    _archive_recovered_errors(state)
    state["status"] = "completed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state["metrics"] = compute_backtest_metrics(
        state["observations"], alert_threshold=alert_threshold
    )
    state["comparison_delta"] = comparison_delta(state["metrics"])
    write_json(state, output_path, pretty=True)
    _print_summary(state, run_arms=run_arms)
    typer.echo(f"结果：{output_path}")


def _validate_resume_state(
    state: dict[str, Any],
    *,
    sample_rows: list[dict[str, Any]],
    model_name: str,
    split: str,
    top_k: int,
    run_arms: tuple[str, ...],
    alert_threshold: int,
    excluded_ids: set[str],
    prediction_mode: str,
    prompt_version: str,
) -> None:
    expected = {
        "schema_version": "llm-case-rag-backtest-v1",
        "model_name": model_name,
        "prompt_version": prompt_version,
        "prediction_mode": prediction_mode,
        "split": split,
        "top_k": top_k,
        "arms": list(run_arms),
        "alert_threshold": alert_threshold,
        "excluded_company_ids": sorted(excluded_ids),
        "sample": sample_rows,
    }
    mismatched = [name for name, value in expected.items() if state.get(name) != value]
    if mismatched:
        raise ValueError(f"断点目录与本次参数不一致：{', '.join(mismatched)}")


def _archive_recovered_errors(state: dict[str, Any]) -> None:
    """成功重试后把旧失败尝试移入审计区，保持主结果一组一条。"""
    successful_keys = {
        (str(item.get("company_id")), str(item.get("arm")))
        for item in state["observations"]
        if not item.get("error")
    }
    recovered = [
        item
        for item in state["observations"]
        if item.get("error")
        and (str(item.get("company_id")), str(item.get("arm"))) in successful_keys
    ]
    if recovered:
        state.setdefault("attempt_errors", []).extend(recovered)
        recovered_ids = {id(item) for item in recovered}
        state["observations"] = [
            item for item in state["observations"] if id(item) not in recovered_ids
        ]


def _resolve_arms(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "both":
        return BACKTEST_ARMS
    if normalized in BACKTEST_ARMS:
        return (normalized,)
    raise typer.BadParameter("arm must be both, without_case_rag or with_case_rag")


def _load_excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = read_json(path.resolve())
    return {
        str(item["company_id"]).zfill(5)
        for item in payload.get("sample", [])
        if item.get("company_id") is not None
    }


def _select_hybrid_cases(
    frame: pd.DataFrame,
    company_id: str,
    *,
    top_k: int,
    config: Any,
) -> tuple[Any, ...]:
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
    store = PgVectorHistoricalCaseStore(
        embedding_provider,
        dsn=config.vector_store.require_postgres_dsn(),
        schema=config.vector_store.postgres_schema,
        connect_timeout_seconds=config.vector_store.postgres_connect_timeout_seconds,
    )
    try:
        store.ensure_schema()
        if store.count == 0:
            raise RuntimeError(
                "历史案例库为空，请先执行 python scripts/build_historical_case_index.py"
            )
        return select_historical_cases(
            frame,
            company_id,
            top_k=top_k,
            case_store=store,
        )
    finally:
        store.close()


def _pricing(config: Any) -> ModelPricing:
    return ModelPricing(
        input_per_million_rmb=config.llm.input_price_per_million_rmb,
        output_per_million_rmb=config.llm.output_price_per_million_rmb,
        cached_input_per_million_rmb=config.llm.cached_input_price_per_million_rmb,
    )


def _predictor(
    root: Path,
    *,
    config: Any,
    ledger: ModelUsageLedger,
    max_output_tokens: int,
    alert_threshold: int,
    prediction_mode: str,
) -> LLMPredictionCommittee | LLMPredictionSingleJudge:
    provider = OpenAICompatibleJsonChatProvider(
        api_key=config.llm.require_api_key(),
        base_url=config.llm.base_url,
        model_name=config.llm.model_name,
        provider_name=config.llm.provider,
        request_extra_body=(None if config.llm.disable_thinking else {}),
        supports_json_object=config.llm.supports_json_object,
        reasoning_effort=config.llm.reasoning_effort,
        usage_ledger=ledger,
        pricing=_pricing(config),
    )
    prompt_dir = root / "prompts" / "prediction"
    retry_policy = RetryPolicy(
        max_retries=config.agents.max_retries,
        initial_seconds=config.agents.retry_initial_seconds,
        backoff_factor=config.agents.retry_backoff_factor,
        max_seconds=config.agents.retry_max_seconds,
        retry_if=retry_model_node_error,
    )
    if prediction_mode == "single":
        return LLMPredictionSingleJudge(
            provider,
            prompt=(prompt_dir / "single_judge_v1.md").read_text(encoding="utf-8"),
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            retry_policy=retry_policy,
            progress_callback=_progress,
            alert_threshold=alert_threshold,
        )
    return LLMPredictionCommittee(
        provider,
        bull_prompt=(prompt_dir / "bull_v2.md").read_text(encoding="utf-8"),
        bear_prompt=(prompt_dir / "bear_v2.md").read_text(encoding="utf-8"),
        decision_prompt=(prompt_dir / "decision_v2.md").read_text(encoding="utf-8"),
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        retry_policy=retry_policy,
        progress_callback=_progress,
        alert_threshold=alert_threshold,
    )


def _print_summary(state: dict[str, Any], *, run_arms: tuple[str, ...]) -> None:
    typer.echo("\n对照结果：")
    for arm in run_arms:
        metrics = state["metrics"].get(arm, {})
        typer.echo(
            f"  {arm}: 完成={metrics.get('completed', 0)} · "
            f"准确率={metrics.get('accuracy')} · 召回率={metrics.get('recall')} · "
            f"精确率={metrics.get('precision')} · F1={metrics.get('f1')} · "
            f"PR-AUC={metrics.get('pr_auc')} · Token="
            f"{metrics.get('input_tokens', 0)}入/{metrics.get('output_tokens', 0)}出"
        )
    if len(run_arms) == 2:
        typer.echo(f"  RAG增量：{state['comparison_delta']}")


if __name__ == "__main__":
    app()
