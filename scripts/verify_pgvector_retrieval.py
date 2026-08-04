"""用真实查询向量验证pgvector能够返回可回链的招股书原文。"""

from __future__ import annotations

import argparse
from datetime import date

from ipo_risk.config import load_config
from ipo_risk.retrieval.pgvector import PgVectorRetriever
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider
from ipo_risk.storage.files import read_jsonl
from ipo_risk.storage.ingestion import make_document_id
from ipo_risk.storage.paths import DocumentPaths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--query", default="客户集中度、供应商依赖与现金流风险")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    manifest = (
        config.project_root
        / "data"
        / "processed"
        / "competition"
        / "company_manifest.jsonl"
    )
    matches = [
        row
        for row in read_jsonl(manifest)
        if str(row["company_id"]) == args.company_id
        and row.get("label_status") == "complete"
    ]
    if len(matches) != 1:
        raise SystemExit(f"当前清单未唯一匹配公司：{args.company_id}")
    row = matches[0]
    document_date = date.fromisoformat(str(row["prospectus_date"]))
    document_id = make_document_id(
        str(row["company_id"]),
        "prospectus",
        document_date,
        str(row["source_hash"]),
    )
    paths = DocumentPaths.build(
        interim_root=config.project_root / config.paths.interim_dir,
        artifacts_root=config.project_root / config.paths.artifacts_dir,
        company_folder=str(row["company_folder"]),
        document_id=document_id,
    )
    chunk_count = len(read_jsonl(paths.chunks_path))
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        model_name=config.embedding.model_name,
        dimensions=config.embedding.dimensions,
        batch_size=config.embedding.batch_size,
        request_concurrency=1,
        timeout_seconds=config.embedding.timeout_seconds,
        max_retries=config.embedding.max_retries,
        retry_initial_seconds=config.embedding.retry_initial_seconds,
        retry_backoff_factor=config.embedding.retry_backoff_factor,
        retry_max_seconds=config.embedding.retry_max_seconds,
    )
    store = PgVectorRetriever(
        provider,
        dsn=config.vector_store.require_postgres_dsn(),
        schema=config.vector_store.postgres_schema,
        connect_timeout_seconds=config.vector_store.postgres_connect_timeout_seconds,
    )
    try:
        if not store.activate_document(
            document_id,
            source_hash=str(row["source_hash"]),
            expected_chunk_count=chunk_count,
        ):
            raise RuntimeError("数据库中的当前文档不完整")
        hits = store.search(args.query, top_k=args.top_k)
    finally:
        store.close()
    print(
        f"检索成功：{row['company_folder']}，查询={args.query!r}，命中{len(hits)}条。"
    )
    for hit in hits:
        preview = " ".join(hit.chunk.text.split())[:120]
        print(
            f"{hit.rank}. score={hit.score:.4f} "
            f"chunk={hit.chunk.chunk_id} page={hit.chunk.page} {preview}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
