"""ChromaDB 持久化语义检索，并保持招股书证据的完整定位信息。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import SearchHit
from ipo_risk.retrieval.semantic import EmbeddingProvider


_SAFE_CHROMA_BATCH_SIZE = 5_000
_GENERATION_FILE = "chroma_collection_generations.json"
_HNSW_CORRUPTION_MARKERS = (
    "hnsw segment reader",
    "error loading hnsw index",
    "error creating hnsw",
    "error constructing hnsw",
    "backfill request to compactor",
)


class ChromaRetriever:
    """把文档向量增量写入 Chroma，并只检索本次激活的文档。"""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        persist_dir: Path,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if client is None:
            try:
                import chromadb
            except ImportError as error:  # pragma: no cover - 取决于可选运行环境
                raise RuntimeError(
                    "Chroma 后端未安装，请安装项目的 chromadb 依赖"
                ) from error
            client = chromadb.PersistentClient(path=str(persist_dir))
        self._provider = provider
        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._collection_name = collection_name
        self._model_fingerprint = sha256(
            f"{provider.model_name}\0{provider.dimensions}".encode("utf-8")
        ).hexdigest()[:12]
        self._generation = self._read_generation()
        self._collection: Any = self._open_collection()
        self._active_document_id: str | None = None

    @property
    def count(self) -> int:
        """返回持久化集合中的切片总数。"""
        return int(self._collection.count())

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        skip_existing_document: bool = True,
        precomputed_embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> int:
        """写入一份招股书的切片；损坏的HNSW代际会被隔离并安全重建。"""
        try:
            return self._add_chunks_once(
                chunks,
                skip_existing_document=skip_existing_document,
                precomputed_embeddings=precomputed_embeddings,
            )
        except Exception as exc:
            if not _is_hnsw_corruption(exc):
                raise
            self._recover_collection()
            return self._add_chunks_once(
                chunks,
                skip_existing_document=False,
                precomputed_embeddings=precomputed_embeddings,
            )

    def _add_chunks_once(
        self,
        chunks: list[DocumentChunk],
        *,
        skip_existing_document: bool,
        precomputed_embeddings: Sequence[Sequence[float]] | np.ndarray | None,
    ) -> int:
        if not chunks:
            return 0
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("add_chunks requires chunks from exactly one document")
        document_id = next(iter(document_ids))
        self._active_document_id = document_id

        expected_record_ids = [_record_id(chunk) for chunk in chunks]
        existing_ids: set[str] = set()
        if skip_existing_document:
            existing = self._collection.get(
                where={"document_id": document_id},
                include=[],
            )
            existing_ids = {
                str(record_id) for record_id in existing.get("ids", [])
            }
            if set(expected_record_ids).issubset(existing_ids):
                return 0

        embeddings = precomputed_embeddings
        if embeddings is None:
            embeddings = self._provider.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedding provider returned an unexpected vector count")
        if any(len(vector) != self._provider.dimensions for vector in embeddings):
            raise ValueError("embedding provider returned an unexpected dimension")

        pending_indexes = [
            index
            for index, record_id in enumerate(expected_record_ids)
            if record_id not in existing_ids
        ]
        batch_size = self._safe_batch_size()
        for start in range(0, len(pending_indexes), batch_size):
            batch_indexes = pending_indexes[start : start + batch_size]
            self._collection.add(
                ids=[expected_record_ids[index] for index in batch_indexes],
                embeddings=[embeddings[index] for index in batch_indexes],
                documents=[chunks[index].text for index in batch_indexes],
                metadatas=[
                    _chunk_metadata(chunks[index]) for index in batch_indexes
                ],
            )
        return len(pending_indexes)

    def _generation_path(self) -> Path:
        return self._persist_dir / _GENERATION_FILE

    def _generation_key(self) -> str:
        return (
            f"{self._collection_name}\0{self._provider.model_name}\0"
            f"{self._provider.dimensions}"
        )

    def _read_generation(self) -> int:
        path = self._generation_path()
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return max(0, int(payload.get(self._generation_key(), 0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 0

    def _write_generation(self) -> None:
        path = self._generation_path()
        try:
            payload = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.is_file()
                else {}
            )
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[self._generation_key()] = self._generation
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _physical_name(self) -> str:
        # v4通过代际指针隔离异常中断留下的HNSW文件；旧集合不删除。
        return (
            f"{self._collection_name[:34]}-v4-"
            f"{self._model_fingerprint}-g{self._generation}"
        )

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._physical_name(),
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self._provider.model_name,
                "embedding_dimensions": str(self._provider.dimensions),
                "generation": str(self._generation),
            },
        )

    def _recover_collection(self) -> None:
        self._generation += 1
        self._write_generation()
        self._collection = self._open_collection()

    def _safe_batch_size(self) -> int:
        """使用Chroma运行时上限与保守上限中的较小值。"""
        client = getattr(self._collection, "_client", None)
        get_max_batch_size = getattr(client, "get_max_batch_size", None)
        if callable(get_max_batch_size):
            try:
                runtime_limit = int(get_max_batch_size())
            except (TypeError, ValueError):
                runtime_limit = _SAFE_CHROMA_BATCH_SIZE
            if runtime_limit > 0:
                return min(_SAFE_CHROMA_BATCH_SIZE, runtime_limit)
        return _SAFE_CHROMA_BATCH_SIZE

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """只在当前分析文档内执行余弦向量检索。"""
        if top_k < 1 or not query.strip():
            return []
        if self._active_document_id is None:
            raise RuntimeError("请先通过 add_chunks 激活一份文档")
        query_vectors = self._provider.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        result = self._collection.query(
            query_embeddings=query_vectors,
            n_results=top_k,
            where={"document_id": self._active_document_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_list(result, "ids")
        documents = _first_result_list(result, "documents")
        metadatas = _first_result_list(result, "metadatas")
        distances = _first_result_list(result, "distances")
        hits: list[SearchHit] = []
        for rank, (_stored_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            hits.append(
                SearchHit(
                    chunk=_metadata_to_chunk(str(text), metadata),
                    score=1.0 - float(distance),
                    rank=rank,
                )
            )
        return hits


def _first_result_list(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key) or []
    return list(values[0]) if values else []


def _is_hnsw_corruption(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _HNSW_CORRUPTION_MARKERS)


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    return {
        "chunk_id": chunk.chunk_id,
        "company_id": chunk.company_id,
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "source_hash": chunk.source_hash,
        "page": chunk.page,
        "printed_page_label": chunk.printed_page_label or "",
        "bbox": json.dumps(chunk.bbox),
        "is_heading": chunk.is_heading,
        "is_ocr": chunk.is_ocr,
        "font_size": chunk.font_size,
        "available_at": chunk.available_at.isoformat(),
        "chunk_kind": chunk.chunk_kind,
        "table_id": chunk.table_id or "",
        "table_cell_refs": json.dumps(
            chunk.table_cell_refs,
            ensure_ascii=False,
        ),
        "table_cell_values": json.dumps(
            chunk.table_cell_values,
            ensure_ascii=False,
        ),
    }


def _record_id(chunk: DocumentChunk) -> str:
    return sha256(
        f"{chunk.document_id}\0{chunk.chunk_id}\0{chunk.source_hash}".encode("utf-8")
    ).hexdigest()


def _metadata_to_chunk(text: str, metadata: dict[str, Any]) -> DocumentChunk:
    printed_page_label = str(metadata.get("printed_page_label") or "").strip()
    raw_bbox = [float(value) for value in json.loads(str(metadata["bbox"]))]
    if len(raw_bbox) != 4:
        raise ValueError("stored chunk bbox must contain four coordinates")
    bbox = (raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3])
    chunk_kind = str(metadata.get("chunk_kind") or "text")
    if chunk_kind not in {"text", "table"}:
        raise ValueError(f"stored chunk kind is invalid: {chunk_kind}")
    return DocumentChunk(
        chunk_id=str(metadata["chunk_id"]),
        company_id=str(metadata["company_id"]),
        document_id=str(metadata["document_id"]),
        source_file=str(metadata["source_file"]),
        source_hash=str(metadata["source_hash"]),
        page=int(metadata["page"]),
        printed_page_label=printed_page_label or None,
        bbox=bbox,
        text=text,
        is_heading=bool(metadata["is_heading"]),
        is_ocr=bool(metadata["is_ocr"]),
        font_size=float(metadata["font_size"]),
        available_at=date.fromisoformat(str(metadata["available_at"])),
        chunk_kind=cast(Literal["text", "table"], chunk_kind),
        table_id=str(metadata.get("table_id") or "").strip() or None,
        table_cell_refs=tuple(
            str(value)
            for value in json.loads(str(metadata.get("table_cell_refs") or "[]"))
        ),
        table_cell_values=tuple(
            str(value)
            for value in json.loads(str(metadata.get("table_cell_values") or "[]"))
        ),
    )


__all__ = ["ChromaRetriever"]
