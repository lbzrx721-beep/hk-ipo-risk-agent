"""从既有 pgvector 招股书索引提取新增财务、集中度和锁定期特征。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import typer
from openai import OpenAI

from ipo_risk.agents.analysis import OpenAICompatibleJsonChatProvider
from ipo_risk.config import load_config
from ipo_risk.modeling.rag_fundamentals import (
    RagFundamentalExtractor,
    build_rag_fundamental_dataset,
)
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.pgvector import PgVectorRetriever
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider
from ipo_risk.resilience import RetryPolicy, retry_model_node_error
from ipo_risk.storage.files import read_json, read_jsonl, write_jsonl
from ipo_risk.storage.ingestion import make_document_id
from ipo_risk.storage.paths import DocumentPaths


app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    company_id: list[str] | None = typer.Option(
        None,
        "--company-id",
        help="只处理指定股票代码；可重复传入。",
    ),
    year_from: int = typer.Option(2020, min=2000, max=2100),
    year_to: int = typer.Option(2024, min=2000, max=2100),
    output_dir: Path | None = typer.Option(
        None,
        help="输出目录；试跑建议指向tmp，正式批量默认写入modeling目录。",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="复用输出目录中已成功的公司结果。",
    ),
) -> None:
    """逐家公司检索并提取；不重新解析 PDF，也不重新生成文档向量。"""
    if year_from > year_to:
        raise typer.BadParameter("year-from不能晚于year-to")
    root = project_root.resolve()
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else root / "data" / "processed" / "modeling"
    )
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config()
    manifest_path = root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    selected_ids = {value.zfill(5) for value in (company_id or [])}
    selected = [
        row
        for row in manifest_rows
        if year_from <= int(row.get("prospectus_year") or str(row["prospectus_date"])[:4]) <= year_to
        and (
            not selected_ids
            or str(row.get("company_id") or "").zfill(5) in selected_ids
        )
    ]
    if selected_ids:
        found = {str(row.get("company_id") or "").zfill(5) for row in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise typer.BadParameter(f"公司未在指定年份清单中：{', '.join(missing)}")
    if not selected:
        raise typer.BadParameter("没有匹配的公司")

    audit_path = destination / "rag_fundamental_feature_audit.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if resume and audit_path.is_file():
        completed = {
            str(row.get("company_id") or "").zfill(5): row
            for row in read_jsonl(audit_path)
        }
    failures_path = destination / "rag_fundamental_feature_failures.jsonl"
    failures = read_jsonl(failures_path) if resume and failures_path.is_file() else []
    failure_by_company = {
        str(row.get("company_id") or "").zfill(5): row for row in failures
    }
    ocf_by_company = _load_operating_cash_flow(root)

    embedding_http = httpx.Client(trust_env=False)
    embedding_client = OpenAI(
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        max_retries=0,
        timeout=config.embedding.timeout_seconds,
        http_client=embedding_http,
    )
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        model_name=config.embedding.model_name,
        api_model_name=config.embedding.request_model_name,
        dimensions=config.embedding.dimensions,
        batch_size=1,
        request_concurrency=1,
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        client=embedding_client,
        max_retries=config.embedding.max_retries,
        retry_initial_seconds=config.embedding.retry_initial_seconds,
        retry_backoff_factor=config.embedding.retry_backoff_factor,
        retry_max_seconds=config.embedding.retry_max_seconds,
    )
    llm_config = (
        config.specialist_llm
        if config.specialist_llm is not None and config.specialist_llm.is_configured
        else config.llm
    )
    llm_client = OpenAI(
        api_key=llm_config.require_api_key(),
        base_url=llm_config.base_url,
        max_retries=0,
        timeout=300,
    )
    chat_provider = OpenAICompatibleJsonChatProvider(
        model_name=llm_config.model_name,
        client=llm_client,
        provider_name=llm_config.provider,
        request_extra_body=(None if llm_config.disable_thinking else {}),
        supports_json_object=llm_config.supports_json_object,
        reasoning_effort=llm_config.reasoning_effort,
    )
    system_prompt = (
        root / "prompts" / "extraction" / "rag_fundamentals_v1.md"
    ).read_text(encoding="utf-8")
    try:
        for position, row in enumerate(selected, start=1):
            code = str(row.get("company_id") or "").zfill(5)
            if code in completed:
                typer.echo(f"[{position}/{len(selected)}] {code} 已完成，断点跳过。")
                continue
            typer.echo(f"[{position}/{len(selected)}] {code} · {row.get('company_name')}")
            store: PgVectorRetriever | None = None
            try:
                document_date = date.fromisoformat(str(row["prospectus_date"])[:10])
                document_id = make_document_id(
                    code,
                    "prospectus",
                    document_date,
                    str(row["source_hash"]),
                )
                paths = DocumentPaths.build(
                    interim_root=root / config.paths.interim_dir,
                    artifacts_root=root / config.paths.artifacts_dir,
                    company_folder=str(row["company_folder"]),
                    document_id=document_id,
                )
                chunks = [
                    DocumentChunk.model_validate(item)
                    for item in read_jsonl(paths.chunks_path)
                ]
                store = PgVectorRetriever(
                    embedding_provider,
                    dsn=config.vector_store.require_postgres_dsn(),
                    schema=config.vector_store.postgres_schema,
                    connect_timeout_seconds=(
                        config.vector_store.postgres_connect_timeout_seconds
                    ),
                )
                vector_status = _ensure_pgvector_document(
                    store=store,
                    config=config,
                    paths=paths,
                    chunks=chunks,
                    row=row,
                    document_id=document_id,
                )
                if vector_status == "repaired":
                    typer.echo("  pgvector文档不完整，已从本地.npy缓存修复。")
                elif vector_status == "lexical_fallback":
                    typer.echo(
                        "  pgvector文档不完整且无可复用向量缓存，"
                        "本次使用BM25与字段锚点检索。"
                    )
                extractor = RagFundamentalExtractor(
                    retriever=store,
                    provider=chat_provider,
                    system_prompt=system_prompt,
                    max_output_tokens=min(llm_config.max_output_tokens, 12_000),
                    retry_policy=RetryPolicy(
                        max_retries=config.agents.max_retries,
                        initial_seconds=config.agents.retry_initial_seconds,
                        backoff_factor=config.agents.retry_backoff_factor,
                        max_seconds=config.agents.retry_max_seconds,
                        retry_if=retry_model_node_error,
                    ),
                )
                record = extractor.extract(
                    manifest_row=row,
                    document_id=document_id,
                    chunks=chunks,
                    existing_operating_cash_flow=ocf_by_company.get(code),
                    progress=lambda detail: typer.echo(f"  {detail}"),
                )
                completed[code] = record
                failure_by_company.pop(code, None)
                typer.echo("  完成：已保存字段与证据审计。")
                _checkpoint(
                    root,
                    destination,
                    manifest_rows,
                    completed,
                    failure_by_company,
                )
            except Exception as error:
                failure_by_company[code] = {
                    "company_id": code,
                    "company_name": row.get("company_name"),
                    "prospectus_date": row.get("prospectus_date"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                typer.echo(f"  失败：{type(error).__name__}: {error}", err=True)
                _checkpoint(
                    root,
                    destination,
                    manifest_rows,
                    completed,
                    failure_by_company,
                )
            finally:
                if store is not None:
                    store.close()
    finally:
        llm_client.close()
        embedding_client.close()

    if not completed:
        raise typer.Exit(code=1)
    result = build_rag_fundamental_dataset(
        root,
        manifest_rows=manifest_rows,
        records=list(completed.values()),
        output_dir=destination,
    )
    typer.echo(
        f"完成：{result.company_count}家公司，{len(result.feature_columns)}个新增特征。"
    )
    typer.echo(f"特征矩阵：{result.feature_matrix_path}")
    typer.echo(f"审计记录：{result.audit_path}")
    typer.echo(f"失败记录：{failures_path}")


def _load_operating_cash_flow(root: Path) -> dict[str, float]:
    path = root / "data" / "processed" / "modeling" / "structured_fundamental_feature_matrix.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={"company_id": str})
    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        hkd_mn = row.get("fundamental_operating_cash_flow_latest_hkd_mn")
        fx_to_hkd = row.get("fundamental_fx_to_hkd")
        if pd.notna(hkd_mn) and pd.notna(fx_to_hkd) and float(fx_to_hkd) > 0:
            native_mn = float(hkd_mn) / float(fx_to_hkd)
            result[str(row["company_id"]).zfill(5)] = native_mn
            continue
        raw = row.get("fundamental_operating_cash_flow_latest_native")
        if pd.notna(raw):
            value = float(raw)
            # 旧数据部分以“元”保存；RAG财务字段统一使用“百万元”。
            if abs(value) >= 100_000:
                value /= 1_000_000
            result[str(row["company_id"]).zfill(5)] = value
    return result


def _ensure_pgvector_document(
    *,
    store: PgVectorRetriever,
    config: Any,
    paths: DocumentPaths,
    chunks: list[DocumentChunk],
    row: dict[str, Any],
    document_id: str,
) -> str:
    """确认向量索引可用，或用缓存修复；无缓存时返回可审计的词法降级。"""
    source_hash = str(row["source_hash"])
    if store.activate_document(
        document_id,
        source_hash=source_hash,
        expected_chunk_count=len(chunks),
    ):
        return "ready"

    embedding_dir = paths.embedding_dir(config.embedding.model_name)
    vectors_path = embedding_dir / "vectors.npy"
    metadata_path = embedding_dir / "metadata.json"
    manifest_path = embedding_dir / "manifest.json"
    required = (vectors_path, metadata_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return "lexical_fallback"

    metadata = read_json(metadata_path)
    manifest = read_json(manifest_path)
    vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if {chunk.document_id for chunk in chunks} != {document_id}:
        raise ValueError("本地切片document_id与当前招股书不一致")
    if {chunk.source_hash for chunk in chunks} != {source_hash}:
        raise ValueError("本地切片source_hash与当前招股书不一致")
    if str(metadata.get("model_name")) != config.embedding.model_name:
        raise ValueError("本地向量模型名称与当前配置不一致")
    if int(metadata.get("dimensions", 0)) != config.embedding.dimensions:
        raise ValueError("本地向量维度与当前配置不一致")
    if int(metadata.get("chunk_count", 0)) != len(chunks):
        raise ValueError("本地向量元数据切片数不一致")
    if list(manifest.get("chunk_ids") or []) != chunk_ids:
        raise ValueError("本地向量清单与切片顺序不一致")
    if vectors.shape != (len(chunks), config.embedding.dimensions):
        raise ValueError(f"本地vectors.npy形状异常：{vectors.shape}")

    store.add_chunks(
        chunks,
        skip_existing_document=False,
        precomputed_embeddings=vectors,
        company_name=str(row.get("company_name") or ""),
    )
    verification = store.verify_document(
        document_id,
        expected_chunk_count=len(chunks),
    )
    if not verification.complete:
        raise RuntimeError(
            "pgvector自动修复后完整性校验仍失败："
            f"{verification.stored_chunk_count}/{verification.expected_chunk_count}"
        )
    return "repaired"


def _checkpoint(
    root: Path,
    destination: Path,
    manifest_rows: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    failures: dict[str, dict[str, Any]],
) -> None:
    if completed:
        build_rag_fundamental_dataset(
            root,
            manifest_rows=manifest_rows,
            records=list(completed.values()),
            output_dir=destination,
        )
    write_jsonl(
        [failures[key] for key in sorted(failures)],
        destination / "rag_fundamental_feature_failures.jsonl",
    )


if __name__ == "__main__":
    app()
