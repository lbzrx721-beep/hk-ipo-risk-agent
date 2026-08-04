"""把当前比赛文档的本地向量缓存可恢复地导入pgvector。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
from psycopg import InterfaceError, OperationalError

from ipo_risk.config import AppConfig, load_config
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.pgvector import PgVectorRetriever
from ipo_risk.storage.files import read_json, read_jsonl, write_json
from ipo_risk.storage.ingestion import make_document_id
from ipo_risk.storage.paths import DocumentPaths


@dataclass(frozen=True)
class CacheOnlyProvider:
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("导入脚本只允许使用已经生成的本地向量")


@dataclass(frozen=True)
class ImportCandidate:
    row: dict[str, Any]
    document_id: str
    paths: DocumentPaths
    vectors_path: Path
    metadata_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ImportOutcome:
    company_folder: str
    document_id: str
    added_count: int
    chunk_count: int
    attempts: int = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _competition_rows(config: AppConfig) -> list[dict[str, Any]]:
    manifest_path = (
        config.project_root
        / "data"
        / "processed"
        / "competition"
        / "company_manifest.jsonl"
    )
    rows = [
        row for row in read_jsonl(manifest_path) if row.get("label_status") == "complete"
    ]
    split_order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            split_order.get(str(row.get("split")), 9),
            int(row.get("sequence", 0)),
        ),
    )


def _candidate(config: AppConfig, row: dict[str, Any]) -> ImportCandidate:
    company_id = str(row["company_id"])
    source_hash = str(row["source_hash"])
    document_date = datetime.fromisoformat(str(row["prospectus_date"])).date()
    document_id = make_document_id(
        company_id,
        "prospectus",
        document_date,
        source_hash,
    )
    paths = DocumentPaths.build(
        interim_root=config.project_root / config.paths.interim_dir,
        artifacts_root=config.project_root / config.paths.artifacts_dir,
        company_folder=str(row["company_folder"]),
        document_id=document_id,
    )
    embedding_dir = paths.embedding_dir(config.embedding.model_name)
    return ImportCandidate(
        row=row,
        document_id=document_id,
        paths=paths,
        vectors_path=embedding_dir / "vectors.npy",
        metadata_path=embedding_dir / "metadata.json",
        manifest_path=embedding_dir / "manifest.json",
    )


def _load_verified_cache(
    config: AppConfig,
    candidate: ImportCandidate,
) -> tuple[list[DocumentChunk], np.ndarray]:
    required = (
        candidate.paths.chunks_path,
        candidate.vectors_path,
        candidate.metadata_path,
        candidate.manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缓存文件缺失：" + "；".join(missing))

    chunks = [
        DocumentChunk.model_validate(item)
        for item in read_jsonl(candidate.paths.chunks_path)
    ]
    metadata = read_json(candidate.metadata_path)
    manifest = read_json(candidate.manifest_path)
    vectors = np.load(candidate.vectors_path, allow_pickle=False, mmap_mode="r")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if not chunks:
        raise ValueError("切片缓存为空")
    if {chunk.document_id for chunk in chunks} != {candidate.document_id}:
        raise ValueError("切片document_id与比赛清单不一致")
    if {chunk.source_hash for chunk in chunks} != {str(candidate.row["source_hash"])}:
        raise ValueError("切片source_hash与比赛清单不一致")
    if str(metadata.get("model_name")) != config.embedding.model_name:
        raise ValueError("向量模型名称不一致")
    if int(metadata.get("dimensions", 0)) != config.embedding.dimensions:
        raise ValueError("向量维度元数据不一致")
    if int(metadata.get("chunk_count", 0)) != len(chunks):
        raise ValueError("向量元数据切片数不一致")
    if list(manifest.get("chunk_ids") or []) != chunk_ids:
        raise ValueError("向量清单与切片顺序不一致")
    if vectors.shape != (len(chunks), config.embedding.dimensions):
        raise ValueError(f"vectors.npy形状异常：{vectors.shape}")
    return chunks, vectors


def _expected_chunk_count(
    config: AppConfig,
    candidate: ImportCandidate,
) -> int:
    """只读取轻量元数据，供已导入文档快速执行断点续传核验。"""
    if not candidate.metadata_path.is_file():
        raise FileNotFoundError(f"向量元数据缺失：{candidate.metadata_path}")
    metadata = read_json(candidate.metadata_path)
    if str(metadata.get("model_name")) != config.embedding.model_name:
        raise ValueError("向量模型名称不一致")
    if int(metadata.get("dimensions", 0)) != config.embedding.dimensions:
        raise ValueError("向量维度元数据不一致")
    chunk_count = int(metadata.get("chunk_count", 0))
    if chunk_count < 1:
        raise ValueError("向量元数据切片数无效")
    return chunk_count


def _status_path(config: AppConfig, *, verify_only: bool = False) -> Path:
    return (
        config.project_root
        / "artifacts"
        / "runtime"
        / "pgvector_import"
        / ("verification_status.json" if verify_only else "status.json")
    )


def _process_candidate_once(
    config: AppConfig,
    provider: CacheOnlyProvider,
    row: dict[str, Any],
    *,
    batch_size: int,
    verify_only: bool,
) -> ImportOutcome:
    candidate = _candidate(config, row)
    expected_chunk_count = _expected_chunk_count(config, candidate)
    store = PgVectorRetriever(
        provider,
        dsn=config.vector_store.require_postgres_dsn(),
        schema=config.vector_store.postgres_schema,
        connect_timeout_seconds=config.vector_store.postgres_connect_timeout_seconds,
    )
    try:
        source_hash = str(candidate.row["source_hash"])
        verification = store.verify_document(
            candidate.document_id,
            expected_chunk_count=expected_chunk_count,
        )
        reusable = verification.complete and store.activate_document(
            candidate.document_id,
            source_hash=source_hash,
            expected_chunk_count=expected_chunk_count,
        )
        if verify_only:
            if not reusable:
                raise RuntimeError(
                    "数据库不完整或来源版本不匹配："
                    f"{verification.stored_chunk_count}/{expected_chunk_count}"
                )
            added = 0
        elif reusable:
            added = 0
        else:
            chunks, vectors = _load_verified_cache(config, candidate)
            added = store.add_chunks(
                chunks,
                skip_existing_document=True,
                precomputed_embeddings=vectors,
                company_name=str(row.get("company_name") or ""),
                batch_size=batch_size,
            )
            verification = store.verify_document(
                candidate.document_id,
                expected_chunk_count=len(chunks),
            )
            if not verification.complete:
                raise RuntimeError("写入完成后数据库完整性校验失败")
    finally:
        store.close()
    return ImportOutcome(
        company_folder=str(row["company_folder"]),
        document_id=candidate.document_id,
        added_count=added,
        chunk_count=expected_chunk_count,
    )


def _process_candidate(
    config: AppConfig,
    provider: CacheOnlyProvider,
    row: dict[str, Any],
    *,
    batch_size: int,
    verify_only: bool,
    max_retries: int,
    retry_initial_seconds: float,
) -> ImportOutcome:
    company = str(row["company_folder"])
    for attempt in range(1, max_retries + 2):
        try:
            outcome = _process_candidate_once(
                config,
                provider,
                row,
                batch_size=batch_size,
                verify_only=verify_only,
            )
            return ImportOutcome(
                company_folder=outcome.company_folder,
                document_id=outcome.document_id,
                added_count=outcome.added_count,
                chunk_count=outcome.chunk_count,
                attempts=attempt,
            )
        except (OperationalError, InterfaceError) as error:
            if attempt > max_retries:
                raise
            delay = retry_initial_seconds * (2 ** (attempt - 1))
            print(
                f"{company} 数据库连接暂时不可用，{delay:g}秒后进行第{attempt + 1}次尝试："
                f"{error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--company-id", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must not be negative")
    if args.retry_initial_seconds <= 0:
        raise SystemExit("--retry-initial-seconds must be positive")

    config = load_config()
    if config.embedding.dimensions != 1024:
        raise SystemExit("当前数据库结构要求embedding维度为1024")
    rows = _competition_rows(config)
    if args.company_id:
        rows = [row for row in rows if str(row["company_id"]) == args.company_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没有匹配的比赛文档")

    provider = CacheOnlyProvider(
        model_name=config.embedding.model_name,
        dimensions=config.embedding.dimensions,
    )
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "verifying" if args.verify_only else "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "embedding_model": provider.model_name,
        "embedding_dimensions": provider.dimensions,
        "total": len(rows),
        "completed": 0,
        "imported": 0,
        "reused": 0,
        "failed": 0,
        "chunks_imported": 0,
        "retry_attempts": 0,
        "workers": min(args.workers, len(rows)),
        "failures": [],
    }
    path = _status_path(config, verify_only=args.verify_only)
    write_json(status, path)
    with ThreadPoolExecutor(
        max_workers=min(args.workers, len(rows)),
        thread_name_prefix="pgvector-import",
    ) as executor:
        future_rows = {
            executor.submit(
                _process_candidate,
                config,
                provider,
                row,
                batch_size=args.batch_size,
                verify_only=args.verify_only,
                max_retries=args.max_retries,
                retry_initial_seconds=args.retry_initial_seconds,
            ): row
            for row in rows
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            company = str(row["company_folder"])
            try:
                outcome = future.result()
                if outcome.added_count:
                    status["imported"] += 1
                    status["chunks_imported"] += outcome.added_count
                else:
                    status["reused"] += 1
                status["retry_attempts"] += outcome.attempts - 1
                print(
                    f"[{status['completed'] + 1}/{len(rows)}] {company} "
                    f"{'新增' + str(outcome.added_count) + '条' if outcome.added_count else '已完整复用'}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - 单公司失败不阻断可恢复批任务
                status["failed"] += 1
                candidate = _candidate(config, row)
                status["failures"].append(
                    {
                        "company_folder": company,
                        "document_id": candidate.document_id,
                        "error": str(error),
                    }
                )
                print(
                    f"[{status['completed'] + 1}/{len(rows)}] {company} 失败：{error}",
                    flush=True,
                )
            status["completed"] += 1
            status["updated_at"] = _utc_now()
            write_json(status, path)

    status["state"] = "completed" if status["failed"] == 0 else "completed_with_failures"
    status["ended_at"] = _utc_now()
    status["updated_at"] = _utc_now()
    write_json(status, path)
    print(
        "导入结束："
        f"新增{status['imported']}份，复用{status['reused']}份，"
        f"失败{status['failed']}份，新增向量{status['chunks_imported']}条。",
        flush=True,
    )
    return 0 if status["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
