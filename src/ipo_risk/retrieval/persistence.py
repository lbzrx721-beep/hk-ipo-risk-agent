"""把内存文档向量异步持久化到 Chroma，且不阻塞当前分析。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock, Thread
from typing import Literal, Protocol

import numpy as np

from ipo_risk.retrieval.chroma import ChromaRetriever
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.semantic import EmbeddingProvider


PersistenceStatus = Literal["building", "ready", "failed"]
ProgressCallback = Callable[[str, str], None]
_LOGGER = logging.getLogger(__name__)


class ChromaWriter(Protocol):
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        skip_existing_document: bool = True,
        precomputed_embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> int: ...


ChromaWriterFactory = Callable[
    [EmbeddingProvider, Path, str],
    ChromaWriter,
]


@dataclass(frozen=True)
class PersistenceSnapshot:
    status: PersistenceStatus
    chunk_count: int
    added_count: int = 0
    error: str | None = None


_LOCKS_GUARD = Lock()
_PERSISTENCE_LOCKS: dict[tuple[str, str, str], Lock] = {}


def _default_writer_factory(
    provider: EmbeddingProvider,
    persist_dir: Path,
    collection_name: str,
) -> ChromaWriter:
    return ChromaRetriever(
        provider,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


def _persistence_lock(
    *,
    persist_dir: Path,
    collection_name: str,
    provider: EmbeddingProvider,
) -> Lock:
    key = (
        str(persist_dir.resolve()),
        collection_name,
        f"{provider.model_name}:{provider.dimensions}",
    )
    with _LOCKS_GUARD:
        return _PERSISTENCE_LOCKS.setdefault(key, Lock())


class ChromaPersistenceTask:
    """后台复用内存向量写入 Chroma，并提供线程安全状态。"""

    def __init__(
        self,
        *,
        chunks: list[DocumentChunk],
        document_vectors: np.ndarray,
        provider: EmbeddingProvider,
        persist_dir: Path,
        collection_name: str,
        writer_factory: ChromaWriterFactory | None = None,
        status_callback: Callable[[PersistenceSnapshot], None] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("Chroma persistence requires document chunks")
        if document_vectors.shape != (len(chunks), provider.dimensions):
            raise ValueError("document vectors do not match chunks and dimensions")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("Chroma persistence requires exactly one document")
        if not collection_name.strip():
            raise ValueError("Chroma collection name must not be empty")
        document_vectors.flags.writeable = False
        self._chunks = list(chunks)
        self._document_vectors = document_vectors
        self._provider = provider
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._writer_factory = writer_factory or _default_writer_factory
        self._status_callback = status_callback
        self._progress_callback = progress_callback
        self._document_id = next(iter(document_ids))
        self._lock = Lock()
        self._snapshot = PersistenceSnapshot(
            status="building",
            chunk_count=len(self._chunks),
        )
        self._thread: Thread | None = None

    def start(self) -> None:
        """启动且只启动一次后台持久化。"""
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Chroma persistence task has already started")
            self._thread = Thread(
                target=self._run,
                name=f"chroma-persist-{self._document_id}",
                daemon=True,
            )
            thread = self._thread
        self._publish(
            PersistenceSnapshot(
                status="building",
                chunk_count=len(self._chunks),
            )
        )
        self._emit(
            "vector_persistence_started",
            "内存向量已可用于分析，正在后台持久化到Chroma。",
        )
        thread.start()

    def _run(self) -> None:
        lock = _persistence_lock(
            persist_dir=self._persist_dir,
            collection_name=self._collection_name,
            provider=self._provider,
        )
        try:
            with lock:
                writer = self._writer_factory(
                    self._provider,
                    self._persist_dir,
                    self._collection_name,
                )
                added = writer.add_chunks(
                    self._chunks,
                    skip_existing_document=True,
                    precomputed_embeddings=self._document_vectors,
                )
        except Exception as exc:  # noqa: BLE001 - 后台失败必须转为状态
            error = str(exc).strip() or type(exc).__name__
            self._publish(
                PersistenceSnapshot(
                    status="failed",
                    chunk_count=len(self._chunks),
                    error=error,
                )
            )
            self._emit(
                "vector_persistence_failed",
                (
                    "内存向量索引仍可用于本次分析；Chroma后台持久化失败，"
                    f"可稍后重试：{error}"
                ),
            )
            return
        self._publish(
            PersistenceSnapshot(
                status="ready",
                chunk_count=len(self._chunks),
                added_count=added,
            )
        )
        action = "复用已有索引" if added == 0 else f"新增{added}条向量"
        self._emit(
            "vector_persistence_completed",
            f"Chroma后台持久化完成：{action}。",
        )

    def _publish(self, snapshot: PersistenceSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
        if self._status_callback is not None:
            try:
                self._status_callback(snapshot)
            except Exception:  # noqa: BLE001 - 状态记录不能破坏向量持久化
                _LOGGER.warning(
                    "failed to publish Chroma persistence status",
                    exc_info=True,
                )

    def _emit(self, stage: str, detail: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, detail)

    def snapshot(self) -> PersistenceSnapshot:
        with self._lock:
            return self._snapshot

    def wait(self, timeout: float | None = None) -> bool:
        """仅供测试和受控关闭使用；分析主流程不调用。"""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()


__all__ = [
    "ChromaPersistenceTask",
    "ChromaWriter",
    "ChromaWriterFactory",
    "PersistenceSnapshot",
    "PersistenceStatus",
]
