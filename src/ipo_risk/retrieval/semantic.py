"""可替换向量供应商上的轻量语义检索。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from typing import Any, Protocol
from threading import Lock
from uuid import uuid4

import numpy as np

from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import BM25Retriever, SearchHit
from ipo_risk.resilience import RetryExhaustedError, RetryPolicy
from ipo_risk.storage.files import ensure_dir, read_json, write_json


class EmbeddingProvider(Protocol):
    """外部或本地Embedding模型需要实现的最小接口。"""

    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序返回稠密向量。"""
        ...


class SemanticSearchProvider(Protocol):
    """可参与混合检索的最小语义搜索接口。"""

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]: ...


# 向量化进度回调：(已完成条数, 总条数)。首次向量化可能耗时较长，
# 通过逐批回调让上层能够展示百分比，避免界面看似卡死。
EmbeddingProgressCallback = Callable[[int, int], None]


class OpenAICompatibleEmbeddingProvider:
    """通过OpenAI兼容Embedding接口批量生成稠密向量。"""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        batch_size: int,
        request_concurrency: int = 1,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        max_retries: int = 0,
        retry_initial_seconds: float = 1.0,
        retry_backoff_factor: float = 2.0,
        retry_max_seconds: float = 8.0,
        sleep: Callable[[float], None] | None = None,
        progress_callback: EmbeddingProgressCallback | None = None,
        batch_cache_dir: Path | None = None,
    ) -> None:
        if dimensions < 1 or batch_size < 1 or request_concurrency < 1:
            raise ValueError(
                "embedding dimensions, batch_size and request_concurrency must be positive"
            )
        if timeout_seconds <= 0:
            raise ValueError("embedding timeout_seconds must be positive")
        if client is None:
            if not api_key or not base_url:
                raise ValueError("api_key and base_url are required without a client")
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=timeout_seconds,
            )
        self.model_name = model_name
        self.dimensions = dimensions
        self._batch_size = batch_size
        self._request_concurrency = request_concurrency
        self._client = client
        self._retry_policy = (
            RetryPolicy(
                max_retries=max_retries,
                initial_seconds=retry_initial_seconds,
                backoff_factor=retry_backoff_factor,
                max_seconds=retry_max_seconds,
            )
            if sleep is None
            else RetryPolicy(
                max_retries=max_retries,
                initial_seconds=retry_initial_seconds,
                backoff_factor=retry_backoff_factor,
                max_seconds=retry_max_seconds,
                sleep=sleep,
            )
        )
        self.max_retries = max_retries
        self.retry_count = 0
        self.request_count = 0
        self.last_attempts = 0
        self._progress_callback = progress_callback
        self._counter_lock = Lock()
        self._batch_cache_root = (
            Path(batch_cache_dir).resolve() if batch_cache_dir is not None else None
        )
        self._active_batch_cache_dir: Path | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """分批调用兼容接口并保持与输入一致的向量顺序。"""
        if not texts:
            return []
        if self._batch_cache_root is not None:
            return self._embed_with_batch_cache(texts)
        return self._embed_without_batch_cache(texts)

    def _embed_without_batch_cache(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        total = len(texts)
        batches = [
            (start, texts[start : start + self._batch_size])
            for start in range(0, total, self._batch_size)
        ]
        if len(batches) == 1 or self._request_concurrency == 1:
            vectors: list[list[float]] = []
            completed = 0
            for _, batch in batches:
                vectors.extend(self._embed_batch(batch))
                completed += len(batch)
                self._report_progress(completed, total)
            return vectors

        # 多连接并发只改变请求调度，不改变结果顺序。每批结果按其原始
        # start 位置写回，避免服务端响应先后顺序污染 chunk_id 对应关系。
        results: dict[int, list[list[float]]] = {}
        completed = 0
        with ThreadPoolExecutor(
            max_workers=min(self._request_concurrency, len(batches)),
            thread_name_prefix="embedding-request",
        ) as executor:
            future_to_batch = {
                executor.submit(self._embed_batch, batch): (start, len(batch))
                for start, batch in batches
            }
            for future in as_completed(future_to_batch):
                start, batch_count = future_to_batch[future]
                results[start] = future.result()
                completed += batch_count
                self._report_progress(completed, total)
        return [vector for start, _ in batches for vector in results[start]]

    def _embed_with_batch_cache(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """逐批持久化向量；临时失败后只补未完成批次。"""
        total = len(texts)
        batches = [
            (start, texts[start : start + self._batch_size])
            for start in range(0, total, self._batch_size)
        ]
        cache_dir = self._batch_checkpoint_dir(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._active_batch_cache_dir = cache_dir
        results: dict[int, list[list[float]]] = {}
        missing: list[tuple[int, list[str], Path]] = []
        completed = 0
        for index, (start, batch) in enumerate(batches):
            path = cache_dir / f"batch-{index:06d}.npy"
            cached = self._load_batch_checkpoint(path, len(batch))
            if cached is None:
                missing.append((start, batch, path))
                continue
            results[start] = cached
            completed += len(batch)
        if completed:
            self._report_progress(completed, total)
        if missing:
            executor = ThreadPoolExecutor(
                max_workers=min(self._request_concurrency, len(missing)),
                thread_name_prefix="embedding-checkpoint",
            )
            future_to_batch = {
                executor.submit(
                    self._embed_and_checkpoint_batch,
                    batch,
                    path,
                ): (start, len(batch))
                for start, batch, path in missing
            }
            try:
                for future in as_completed(future_to_batch):
                    start, batch_count = future_to_batch[future]
                    results[start] = future.result()
                    completed += batch_count
                    self._report_progress(completed, total)
            except Exception:
                for future in future_to_batch:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        return [vector for start, _ in batches for vector in results[start]]

    def clear_batch_checkpoint(self) -> None:
        """最终文档向量缓存写入成功后清理临时分批检查点。"""
        cache_dir = self._active_batch_cache_dir
        if cache_dir is None:
            return
        shutil.rmtree(cache_dir, ignore_errors=True)
        self._active_batch_cache_dir = None

    def _batch_checkpoint_dir(self, texts: list[str]) -> Path:
        assert self._batch_cache_root is not None
        digest = sha256()
        digest.update(
            (f"{self.model_name}\0{self.dimensions}\0{self._batch_size}\0").encode("utf-8")
        )
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return self._batch_cache_root / digest.hexdigest()

    def _load_batch_checkpoint(
        self,
        path: Path,
        expected_count: int,
    ) -> list[list[float]] | None:
        if not path.is_file():
            return None
        try:
            vectors = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if vectors.shape != (expected_count, self.dimensions):
            path.unlink(missing_ok=True)
            return None
        return vectors.astype(np.float32, copy=False).tolist()

    def _embed_and_checkpoint_batch(
        self,
        batch: list[str],
        path: Path,
    ) -> list[list[float]]:
        vectors = self._embed_batch(batch)
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(batch), self.dimensions):
            raise ValueError("embedding API returned an unexpected batch shape")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """执行一批向量请求；该方法可由多个工作线程并发调用。"""
        try:
            outcome = self._retry_policy.run(
                lambda: self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
            )
        except RetryExhaustedError as error:
            self._record_attempts(error.attempts)
            raise error.last_error from error
        self._record_attempts(outcome.attempts)
        ordered = sorted(outcome.value.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise ValueError("embedding API returned an unexpected vector count")
        vectors: list[list[float]] = []
        for item in ordered:
            vector = [float(value) for value in item.embedding]
            if len(vector) != self.dimensions:
                raise ValueError("embedding API returned an unexpected dimension")
            vectors.append(vector)
        return vectors

    def _record_attempts(self, attempts: int) -> None:
        with self._counter_lock:
            self.last_attempts = attempts
            self.request_count += attempts
            self.retry_count += attempts - 1

    def _report_progress(self, completed: int, total: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(min(completed, total), total)


class SemanticRetriever:
    """在DocumentChunk列表上建立内存向量索引并执行余弦检索。"""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        provider: EmbeddingProvider,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("SemanticRetriever requires at least one chunk")
        self._chunks = list(chunks)
        self._provider = provider
        fingerprint = _index_fingerprint(chunks, provider)
        cached = (
            _load_cached_vectors(
                cache_dir,
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                dimensions=provider.dimensions,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
            if cache_dir is not None
            else None
        )
        vectors = cached
        if vectors is None:
            vectors = np.asarray(
                provider.embed([chunk.text for chunk in chunks]),
                dtype=np.float32,
            )
            if cache_dir is not None:
                _write_cached_vectors(
                    cache_dir,
                    vectors,
                    fingerprint=fingerprint,
                    provider=provider,
                    chunks=chunks,
                )
        if vectors.ndim != 2 or vectors.shape[0] != len(self._chunks):
            raise ValueError("embedding provider returned an unexpected vector count")
        if vectors.shape[1] != provider.dimensions:
            raise ValueError("embedding provider returned an unexpected dimension")
        vectors.flags.writeable = False
        self._vectors = vectors

    @property
    def document_vectors(self) -> np.ndarray:
        """返回当前文档只读向量，供后台持久化复用而不再次调用模型。"""
        return self._vectors

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """返回按余弦相似度降序排列的语义候选段落。"""
        if top_k < 1 or not query.strip():
            return []
        query_vectors = self._provider.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        query_vector = np.asarray(query_vectors[0], dtype=np.float32)
        if len(query_vector) != self._provider.dimensions:
            raise ValueError("embedding provider returned an unexpected query dimension")

        numerator = self._vectors @ query_vector
        denominator = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        ranked_indexes = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )[: min(top_k, len(self._chunks))]
        return [
            SearchHit(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indexes, start=1)
        ]


def _index_fingerprint(
    chunks: list[DocumentChunk],
    provider: EmbeddingProvider,
) -> str:
    digest = sha256()
    digest.update(f"{provider.model_name}\0{provider.dimensions}\0".encode())
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.source_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cached_vectors(
    cache_dir: Path,
    *,
    fingerprint: str,
    chunk_count: int,
    dimensions: int,
    chunk_ids: list[str],
) -> np.ndarray | None:
    metadata_path = cache_dir / "metadata.json"
    manifest_path = cache_dir / "manifest.json"
    vectors_path = cache_dir / "vectors.npy"
    if not all(path.is_file() for path in (metadata_path, manifest_path, vectors_path)):
        return None
    try:
        metadata = read_json(metadata_path)
        manifest = read_json(manifest_path)
        if metadata.get("fingerprint") != fingerprint:
            return None
        if manifest.get("fingerprint") != fingerprint:
            return None
        if manifest.get("chunk_ids") != chunk_ids:
            return None
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError, AttributeError):
        return None
    if vectors.shape != (chunk_count, dimensions):
        return None
    return np.asarray(vectors, dtype=np.float32)


def _write_cached_vectors(
    cache_dir: Path,
    vectors: np.ndarray,
    *,
    fingerprint: str,
    provider: EmbeddingProvider,
    chunks: list[DocumentChunk],
) -> None:
    ensure_dir(cache_dir)
    np.save(cache_dir / "vectors.npy", vectors, allow_pickle=False)
    write_json(
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "model_name": provider.model_name,
            "dimensions": provider.dimensions,
            "chunk_count": int(vectors.shape[0]),
        },
        cache_dir / "metadata.json",
    )
    write_json(
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        },
        cache_dir / "manifest.json",
    )


@dataclass(frozen=True)
class HybridSearchHit:
    """一条保留词法与语义来源的融合检索命中。"""

    chunk: DocumentChunk
    score: float
    rank: int
    lexical_score: float | None
    lexical_rank: int | None
    semantic_score: float | None
    semantic_rank: int | None


@dataclass
class _CombinedHit:
    chunk: DocumentChunk
    score: float = 0.0
    lexical_score: float | None = None
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None


class HybridRetriever:
    """使用倒数排名融合合并BM25和语义候选。"""

    def __init__(
        self,
        lexical: BM25Retriever,
        semantic: SemanticSearchProvider,
        *,
        lexical_top_k: int = 20,
        semantic_top_k: int = 20,
        rrf_constant: int = 60,
    ) -> None:
        if lexical_top_k < 1 or semantic_top_k < 1:
            raise ValueError("hybrid candidate limits must be positive")
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be positive")
        self._lexical = lexical
        self._semantic = semantic
        self._lexical_top_k = lexical_top_k
        self._semantic_top_k = semantic_top_k
        self._rrf_constant = rrf_constant

    def search(self, query: str, top_k: int = 20) -> list[HybridSearchHit]:
        """返回两路候选的并集，并给出可解释的融合排名。"""
        if top_k < 1 or not query.strip():
            return []
        lexical_hits = self._lexical.search(query, top_k=self._lexical_top_k)
        semantic_hits = self._semantic.search(query, top_k=self._semantic_top_k)
        combined: dict[str, _CombinedHit] = {}

        for source, hits in (("lexical", lexical_hits), ("semantic", semantic_hits)):
            for hit in hits:
                record = combined.setdefault(
                    hit.chunk.chunk_id,
                    _CombinedHit(chunk=hit.chunk),
                )
                record.score += 1.0 / (self._rrf_constant + hit.rank)
                if source == "lexical":
                    record.lexical_score = hit.score
                    record.lexical_rank = hit.rank
                else:
                    record.semantic_score = hit.score
                    record.semantic_rank = hit.rank

        ranked = sorted(
            combined.values(),
            key=lambda record: (-record.score, record.chunk.chunk_id),
        )[:top_k]
        return [
            HybridSearchHit(
                chunk=record.chunk,
                score=record.score,
                rank=rank,
                lexical_score=record.lexical_score,
                lexical_rank=record.lexical_rank,
                semantic_score=record.semantic_score,
                semantic_rank=record.semantic_rank,
            )
            for rank, record in enumerate(ranked, start=1)
        ]


__all__ = [
    "EmbeddingProvider",
    "HybridRetriever",
    "HybridSearchHit",
    "OpenAICompatibleEmbeddingProvider",
    "SemanticSearchProvider",
    "SemanticRetriever",
]
