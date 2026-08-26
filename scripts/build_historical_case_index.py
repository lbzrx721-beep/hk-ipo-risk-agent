"""构建无未来泄漏的历史IPO案例pgvector索引。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import typer

from ipo_risk.config import load_config
from ipo_risk.modeling.historical_cases import (
    PgVectorHistoricalCaseStore,
    build_case_index_records,
)
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    matrix_path: Path | None = typer.Option(None, help="Agent增强特征矩阵"),
    dry_run: bool = typer.Option(False, help="只校验和预览，不调用向量模型或数据库"),
) -> None:
    root = project_root.resolve()
    matrix = (
        matrix_path.resolve()
        if matrix_path is not None
        else root / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix.csv"
    )
    frame = pd.read_csv(matrix, dtype={"company_id": str})
    records = build_case_index_records(frame)
    industries = Counter(record.industry.level_1 for record in records)
    typer.echo(f"可入库历史案例：{len(records)}家。")
    typer.echo(
        "一级行业分布：" + "、".join(f"{name}{count}家" for name, count in industries.most_common())
    )
    if records:
        typer.echo(f"示例摘要：{records[0].case_summary_text[:300]}")
    if dry_run:
        typer.echo("dry-run完成：未调用Embedding、未连接数据库。")
        return

    config = load_config(root / "configs" / "base.yaml")
    provider = OpenAICompatibleEmbeddingProvider(
        model_name=config.embedding.model_name,
        api_model_name=config.embedding.request_model_name,
        dimensions=config.embedding.dimensions,
        batch_size=config.embedding.batch_size,
        request_concurrency=config.embedding.request_concurrency,
        timeout_seconds=config.embedding.timeout_seconds,
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        max_retries=config.embedding.max_retries,
        retry_initial_seconds=config.embedding.retry_initial_seconds,
        retry_backoff_factor=config.embedding.retry_backoff_factor,
        retry_max_seconds=config.embedding.retry_max_seconds,
    )
    store = PgVectorHistoricalCaseStore(
        provider,
        dsn=config.vector_store.require_postgres_dsn(),
        schema=config.vector_store.postgres_schema,
        connect_timeout_seconds=config.vector_store.postgres_connect_timeout_seconds,
    )
    try:
        store.ensure_schema()
        count = store.upsert(records)
    finally:
        store.close()
    typer.echo(
        f"历史案例库已写入：{count}家；模型={provider.model_name}；维度={provider.dimensions}。"
    )


if __name__ == "__main__":
    app()
