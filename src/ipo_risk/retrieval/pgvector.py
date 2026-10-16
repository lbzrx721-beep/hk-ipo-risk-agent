"""PostgreSQL + pgvector持久化检索后端。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from threading import RLock
from typing import Any

import numpy as np

from ipo_risk.observability.trace import retrieval_outputs, trace_step
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import SearchHit
from ipo_risk.retrieval.semantic import EmbeddingProvider


_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class PgVectorDocumentStatus:
    """数据库内一份文档向量的完整性快照。"""

    document_id: str
    embedding_model: str
    expected_chunk_count: int
    stored_chunk_count: int
    dimensions: int | None
    complete: bool


def _require_safe_identifier(value: str, field_name: str) -> str:
    candidate = value.strip()
    if not _SAFE_IDENTIFIER.fullmatch(candidate):
        raise ValueError(f"{field_name} must be a lowercase SQL identifier")
    return candidate


def _load_pg_driver() -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from psycopg import sql
        from pgvector.psycopg import register_vector
    except ImportError as error:  # pragma: no cover - 取决于可选运行环境
        raise RuntimeError(
            "pgvector后端未安装，请安装项目的retrieval依赖"
        ) from error
    return psycopg, sql, register_vector


class PgVectorRetriever:
    """把招股书切片持久化到pgvector，并限定在当前文档内检索。"""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        dsn: str,
        schema: str = "ipo_risk",
        connect_timeout_seconds: float = 10.0,
        connection: Any | None = None,
    ) -> None:
        if not dsn.strip() and connection is None:
            raise ValueError("PostgreSQL DSN must not be empty")
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if provider.dimensions != 1024:
            raise ValueError("当前pgvector表只接受1024维向量")
        self._provider = provider
        self._schema = _require_safe_identifier(schema, "postgres_schema")
        self._psycopg, self._sql, register_vector = _load_pg_driver()
        self._connection = connection or self._psycopg.connect(
            dsn,
            connect_timeout=max(1, int(connect_timeout_seconds)),
            application_name="hk-ipo-risk-agent",
        )
        register_vector(self._connection)
        self._connection.autocommit = True
        self._owns_connection = connection is None
        self._active_document_id: str | None = None
        self._lock = RLock()

    def _table(self, name: str) -> Any:
        return self._sql.SQL("{}.{}").format(
            self._sql.Identifier(self._schema),
            self._sql.Identifier(name),
        )

    @property
    def count(self) -> int:
        """返回当前向量模型在数据库中的切片总数。"""
        query = self._sql.SQL(
            "SELECT count(*) FROM {} WHERE embedding_model = %s"
        ).format(self._table("prospectus_chunks"))
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(query, (self._provider.model_name,))
            return int(cursor.fetchone()[0])

    def close(self) -> None:
        if self._owns_connection and not self._connection.closed:
            self._connection.close()

    def __del__(self) -> None:  # pragma: no cover - 仅作异常路径的连接兜底
        try:
            self.close()
        except Exception:
            pass

    def activate_document(
        self,
        document_id: str,
        *,
        source_hash: str,
        expected_chunk_count: int,
    ) -> bool:
        """仅在数据库声明该版本完整时激活，避免重复加载本地向量。"""
        query = self._sql.SQL(
            """
            SELECT is_complete, chunk_count
            FROM {}
            WHERE document_id = %s
              AND embedding_model = %s
              AND source_hash = %s
              AND embedding_dimensions = %s
            """
        ).format(self._table("vector_documents"))
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    document_id,
                    self._provider.model_name,
                    source_hash,
                    self._provider.dimensions,
                ),
            )
            row = cursor.fetchone()
        active = bool(row and row[0] and int(row[1]) == expected_chunk_count)
        self._active_document_id = document_id if active else None
        return active

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        skip_existing_document: bool = True,
        precomputed_embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
        company_name: str = "",
        batch_size: int = 256,
    ) -> int:
        """以单文档事务写入切片；失败时不会留下半份文档。"""
        if not chunks:
            return 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        document_ids = {chunk.document_id for chunk in chunks}
        company_ids = {chunk.company_id for chunk in chunks}
        source_hashes = {chunk.source_hash for chunk in chunks}
        if len(document_ids) != 1 or len(company_ids) != 1 or len(source_hashes) != 1:
            raise ValueError("add_chunks requires one consistent document")
        document_id = next(iter(document_ids))
        company_id = next(iter(company_ids))
        source_hash = next(iter(source_hashes))
        if skip_existing_document and self.activate_document(
            document_id,
            source_hash=source_hash,
            expected_chunk_count=len(chunks),
        ):
            return 0

        embeddings = precomputed_embeddings
        if embeddings is None:
            embeddings = self._provider.embed([chunk.text for chunk in chunks])
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.shape != (len(chunks), self._provider.dimensions):
            raise ValueError("document vectors do not match chunks and dimensions")

        upsert_document = self._sql.SQL(
            """
            INSERT INTO {} (
                document_id, embedding_model, embedding_dimensions,
                company_id, company_name, source_file, source_hash,
                available_at, chunk_count, is_current, is_complete
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, true, false)
            ON CONFLICT (document_id, embedding_model) DO UPDATE SET
                embedding_dimensions = EXCLUDED.embedding_dimensions,
                company_id = EXCLUDED.company_id,
                company_name = EXCLUDED.company_name,
                source_file = EXCLUDED.source_file,
                source_hash = EXCLUDED.source_hash,
                available_at = EXCLUDED.available_at,
                chunk_count = EXCLUDED.chunk_count,
                is_current = true,
                is_complete = false,
                updated_at = now()
            """
        ).format(self._table("vector_documents"))
        mark_old_versions = self._sql.SQL(
            """
            UPDATE {} SET is_current = false, updated_at = now()
            WHERE company_id = %s AND embedding_model = %s AND document_id <> %s
            """
        ).format(self._table("vector_documents"))
        delete_chunks = self._sql.SQL(
            "DELETE FROM {} WHERE document_id = %s AND embedding_model = %s"
        ).format(self._table("prospectus_chunks"))
        copy_chunks = self._sql.SQL(
            """
            COPY {} (
                document_id, embedding_model, chunk_id, company_id,
                source_file, source_hash, page, printed_page_label, bbox,
                content, is_heading, is_ocr, font_size, available_at,
                chunk_kind, table_id, table_cell_refs, table_cell_values, embedding
            ) FROM STDIN (FORMAT BINARY)
            """
        ).format(self._table("prospectus_chunks"))
        complete_document = self._sql.SQL(
            """
            UPDATE {} SET is_complete = true, updated_at = now()
            WHERE document_id = %s AND embedding_model = %s
            """
        ).format(self._table("vector_documents"))

        records = [
            (
                chunk.document_id,
                self._provider.model_name,
                chunk.chunk_id,
                chunk.company_id,
                chunk.source_file,
                chunk.source_hash,
                chunk.page,
                chunk.printed_page_label,
                list(chunk.bbox),
                chunk.text,
                chunk.is_heading,
                chunk.is_ocr,
                chunk.font_size,
                chunk.available_at,
                chunk.chunk_kind,
                chunk.table_id,
                list(chunk.table_cell_refs),
                list(chunk.table_cell_values),
                vectors[index],
            )
            for index, chunk in enumerate(chunks)
        ]
        first = chunks[0]
        with self._lock, self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    mark_old_versions,
                    (company_id, self._provider.model_name, document_id),
                )
                cursor.execute(
                    upsert_document,
                    (
                        document_id,
                        self._provider.model_name,
                        self._provider.dimensions,
                        company_id,
                        company_name,
                        first.source_file,
                        source_hash,
                        first.available_at,
                        len(chunks),
                    ),
                )
                cursor.execute(
                    delete_chunks,
                    (document_id, self._provider.model_name),
                )
                with cursor.copy(copy_chunks) as copy:
                    copy.set_types(
                        [
                            "text",
                            "text",
                            "text",
                            "text",
                            "text",
                            "text",
                            "int4",
                            "text",
                            "float4[]",
                            "text",
                            "bool",
                            "bool",
                            "float4",
                            "date",
                            "text",
                            "text",
                            "text[]",
                            "text[]",
                            "vector",
                        ]
                    )
                    for record in records:
                        copy.write_row(record)
                cursor.execute(
                    complete_document,
                    (document_id, self._provider.model_name),
                )
        self._active_document_id = document_id
        return len(chunks)

    def verify_document(
        self,
        document_id: str,
        *,
        expected_chunk_count: int,
    ) -> PgVectorDocumentStatus:
        query = self._sql.SQL(
            """
            SELECT d.is_complete, d.chunk_count, count(c.chunk_id),
                   min(vector_dims(c.embedding)), max(vector_dims(c.embedding))
            FROM {} d
            LEFT JOIN {} c
              ON c.document_id = d.document_id
             AND c.embedding_model = d.embedding_model
            WHERE d.document_id = %s AND d.embedding_model = %s
            GROUP BY d.is_complete, d.chunk_count
            """
        ).format(
            self._table("vector_documents"),
            self._table("prospectus_chunks"),
        )
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(query, (document_id, self._provider.model_name))
            row = cursor.fetchone()
        if row is None:
            return PgVectorDocumentStatus(
                document_id=document_id,
                embedding_model=self._provider.model_name,
                expected_chunk_count=expected_chunk_count,
                stored_chunk_count=0,
                dimensions=None,
                complete=False,
            )
        stored = int(row[2])
        min_dimensions = int(row[3]) if row[3] is not None else None
        max_dimensions = int(row[4]) if row[4] is not None else None
        complete = bool(
            row[0]
            and int(row[1]) == expected_chunk_count
            and stored == expected_chunk_count
            and min_dimensions == self._provider.dimensions
            and max_dimensions == self._provider.dimensions
        )
        return PgVectorDocumentStatus(
            document_id=document_id,
            embedding_model=self._provider.model_name,
            expected_chunk_count=expected_chunk_count,
            stored_chunk_count=stored,
            dimensions=min_dimensions if min_dimensions == max_dimensions else None,
            complete=complete,
        )

    @trace_step("pgvector.search", run_type="retriever", inputs=("query", "top_k"), process_outputs=retrieval_outputs)
    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """在当前文档内按余弦距离检索，并还原完整DocumentChunk。"""
        if top_k < 1 or not query.strip():
            return []
        if self._active_document_id is None:
            raise RuntimeError("请先激活或写入一份文档")
        query_vectors = self._provider.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        vector = np.asarray(query_vectors[0], dtype=np.float32)
        if vector.shape != (self._provider.dimensions,):
            raise ValueError("embedding provider returned an unexpected query dimension")
        statement = self._sql.SQL(
            """
            WITH query_vector AS (SELECT %s::vector AS value)
            SELECT c.chunk_id, c.company_id, c.document_id, c.source_file,
                   c.source_hash, c.page, c.printed_page_label, c.bbox,
                   c.content, c.is_heading, c.is_ocr, c.font_size,
                   c.available_at, c.chunk_kind, c.table_id,
                   c.table_cell_refs, c.table_cell_values,
                   1 - (c.embedding <=> q.value) AS score
            FROM {} c CROSS JOIN query_vector q
            WHERE c.document_id = %s AND c.embedding_model = %s
            ORDER BY c.embedding <=> q.value, c.chunk_id
            LIMIT %s
            """
        ).format(self._table("prospectus_chunks"))
        with self._lock, self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                cursor.execute("SET LOCAL hnsw.ef_search = 100")
                cursor.execute(
                    statement,
                    (
                        vector,
                        self._active_document_id,
                        self._provider.model_name,
                        top_k,
                    ),
                )
                rows = cursor.fetchall()
        return [
            SearchHit(
                chunk=DocumentChunk(
                    chunk_id=row[0],
                    company_id=row[1],
                    document_id=row[2],
                    source_file=row[3],
                    source_hash=row[4],
                    page=row[5],
                    printed_page_label=row[6],
                    bbox=tuple(float(value) for value in row[7]),
                    text=row[8],
                    is_heading=row[9],
                    is_ocr=row[10],
                    font_size=row[11],
                    available_at=row[12],
                    chunk_kind=row[13],
                    table_id=row[14],
                    table_cell_refs=tuple(row[15] or ()),
                    table_cell_values=tuple(row[16] or ()),
                ),
                score=float(row[17]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]


__all__ = ["PgVectorDocumentStatus", "PgVectorRetriever"]
