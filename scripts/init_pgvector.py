"""初始化PostgreSQL + pgvector表结构，并可在批量导入后创建HNSW索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ipo_risk.config import load_config


def _execute_sql(dsn: str, path: Path) -> None:
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("请先安装项目retrieval依赖") from error
    statement = path.read_text(encoding="utf-8")
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement, prepare=False)


def _counts(dsn: str) -> tuple[int, int]:
    import psycopg

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM ipo_risk.vector_documents")
            documents = int(cursor.fetchone()[0])
            cursor.execute("SELECT count(*) FROM ipo_risk.prospectus_chunks")
            chunks = int(cursor.fetchone()[0])
    return documents, chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-hnsw",
        action="store_true",
        help="仅在计划跨公司近似检索时创建全局HNSW索引",
    )
    args = parser.parse_args()

    config = load_config()
    dsn = config.vector_store.require_postgres_dsn()
    root = config.project_root
    _execute_sql(dsn, root / "scripts" / "sql" / "init_pgvector.sql")
    print("pgvector表结构已就绪。", flush=True)
    if args.create_hnsw:
        print("开始创建跨公司HNSW索引；历史数据较多时可能耗时较长。", flush=True)
        _execute_sql(dsn, root / "scripts" / "sql" / "create_pgvector_hnsw.sql")
        print("HNSW索引已就绪。", flush=True)
    documents, chunks = _counts(dsn)
    print(f"当前数据库：{documents}份文档，{chunks}条向量。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
