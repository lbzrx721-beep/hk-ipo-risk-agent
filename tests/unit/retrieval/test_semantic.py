from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from threading import Event, Lock
from time import sleep

import pytest

from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import BM25Retriever
from ipo_risk.retrieval.semantic import (
    HybridRetriever,
    OpenAICompatibleEmbeddingProvider,
    SemanticRetriever,
)


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id="03696",
        document_id="prospectus_2025-12-18",
        source_file="data/raw/prospectus/03696_英矽智能/招股章程.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(70.0, 100.0, 525.0, 180.0),
        text=text,
        is_heading=False,
        is_ocr=False,
        font_size=10.5,
        available_at=date(2025, 12, 18),
    )


class StaticEmbeddingProvider:
    model_name = "test-embedding"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "公司多年没有盈利，研发开支持续增加。": [1.0, 0.0],
            "董事会负责召开会议并审议议案。": [0.0, 1.0],
            "盈利能力受到亏损及研发投入影响": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def test_semantic_retriever_ranks_conceptual_match_first() -> None:
    chunks = [
        _chunk("C1", "公司多年没有盈利，研发开支持续增加。"),
        _chunk("C2", "董事会负责召开会议并审议议案。"),
    ]

    retriever = SemanticRetriever(chunks, StaticEmbeddingProvider())
    hits = retriever.search("盈利能力受到亏损及研发投入影响", top_k=2)

    assert [hit.chunk.chunk_id for hit in hits] == ["C1", "C2"]
    assert hits[0].score > hits[1].score


class HybridEmbeddingProvider:
    model_name = "test-hybrid-embedding"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "公司多年亏损且研发投入较高。": [1.0, 0.0],
            "董事会负责公司治理。": [0.0, 1.0],
            "公司需要取得监管牌照及经营许可。": [0.6, 0.4],
            "牌照 许可": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def test_hybrid_retriever_unions_lexical_and_semantic_candidates() -> None:
    chunks = [
        _chunk("C1", "公司多年亏损且研发投入较高。"),
        _chunk("C2", "董事会负责公司治理。"),
        _chunk("C3", "公司需要取得监管牌照及经营许可。"),
    ]
    hybrid = HybridRetriever(
        BM25Retriever(chunks),
        SemanticRetriever(chunks, HybridEmbeddingProvider()),
        lexical_top_k=1,
        semantic_top_k=1,
    )

    hits = hybrid.search("牌照 许可", top_k=3)

    by_id = {hit.chunk.chunk_id: hit for hit in hits}
    assert set(by_id) == {"C1", "C3"}
    assert by_id["C1"].semantic_rank == 1
    assert by_id["C1"].lexical_rank is None
    assert by_id["C3"].lexical_rank == 1
    assert by_id["C3"].semantic_rank is None


class CountingEmbeddingProvider:
    model_name = "cached-test-embedding"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


def test_semantic_retriever_reuses_document_embedding_cache(tmp_path: Path) -> None:
    chunks = [
        _chunk("C1", "公司多年亏损且研发投入较高。"),
        _chunk("C2", "董事会负责公司治理。"),
    ]
    first_provider = CountingEmbeddingProvider()
    second_provider = CountingEmbeddingProvider()

    SemanticRetriever(chunks, first_provider, cache_dir=tmp_path)
    SemanticRetriever(chunks, second_provider, cache_dir=tmp_path)

    assert first_provider.calls == [[chunk.text for chunk in chunks]]
    assert second_provider.calls == []


def test_semantic_retriever_exposes_read_only_document_vectors_without_reembedding() -> None:
    chunks = [
        _chunk("C1", "公司多年亏损且研发投入较高。"),
        _chunk("C2", "董事会负责公司治理。"),
    ]
    provider = CountingEmbeddingProvider()

    retriever = SemanticRetriever(chunks, provider)
    vectors = retriever.document_vectors

    assert provider.calls == [[chunk.text for chunk in chunks]]
    assert vectors.shape == (2, 2)
    assert vectors.flags.writeable is False


class FakeEmbeddingEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        data = [
            SimpleNamespace(index=index, embedding=[float(len(text)), 1.0])
            for index, text in enumerate(inputs)
        ]
        return SimpleNamespace(data=data)


def test_openai_compatible_embedding_provider_batches_requests_in_order() -> None:
    endpoint = FakeEmbeddingEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    provider = OpenAICompatibleEmbeddingProvider(
        client=client,
        model_name="qwen3.7-text-embedding",
        dimensions=2,
        batch_size=2,
    )

    vectors = provider.embed(["甲", "乙乙", "丙丙丙"])

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert [call["input"] for call in endpoint.calls] == [["甲", "乙乙"], ["丙丙丙"]]


class ConcurrentEmbeddingEndpoint(FakeEmbeddingEndpoint):
    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self._release = Event()
        self.active = 0
        self.max_active = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        with self._lock:
            self.calls.append(kwargs)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.max_active >= 2:
                self._release.set()
        self._release.wait(timeout=1)
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        # 让较早批次更晚返回，验证最终结果不依赖响应完成顺序。
        if inputs[0] == "甲":
            sleep(0.02)
        data = [
            SimpleNamespace(index=index, embedding=[float(len(text)), 1.0])
            for index, text in enumerate(inputs)
        ]
        with self._lock:
            self.active -= 1
        return SimpleNamespace(data=data)


def test_embedding_provider_runs_batches_concurrently_and_preserves_order() -> None:
    endpoint = ConcurrentEmbeddingEndpoint()
    progress: list[tuple[int, int]] = []
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=endpoint),
        model_name="qwen3-embedding:4b",
        dimensions=2,
        batch_size=1,
        request_concurrency=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    vectors = provider.embed(["甲", "乙乙", "丙丙丙"])

    assert endpoint.max_active == 2
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert progress[-1] == (3, 3)
    assert [completed for completed, _ in progress] == sorted(
        completed for completed, _ in progress
    )


def test_embedding_provider_preserves_ollama_model_tag() -> None:
    endpoint = FakeEmbeddingEndpoint()
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=endpoint),
        model_name="qwen3-embedding:4b",
        dimensions=2,
        batch_size=8,
    )

    provider.embed(["招股书文本"])

    assert endpoint.calls[0]["model"] == "qwen3-embedding:4b"
    assert endpoint.calls[0]["dimensions"] == 2
    assert endpoint.calls[0]["encoding_format"] == "float"


class FlakyEmbeddingEndpoint(FakeEmbeddingEndpoint):
    def create(self, **kwargs: object) -> SimpleNamespace:
        if len(self.calls) < 2:
            self.calls.append(kwargs)
            raise ConnectionError("temporary embedding connection failure")
        return super().create(**kwargs)


def test_embedding_provider_retries_transient_batch_failure() -> None:
    endpoint = FlakyEmbeddingEndpoint()
    delays: list[float] = []
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=endpoint),
        model_name="qwen3.7-text-embedding",
        dimensions=2,
        batch_size=20,
        max_retries=2,
        retry_initial_seconds=0.25,
        sleep=delays.append,
    )

    vectors = provider.embed(["现金流风险"])

    assert vectors == [[5.0, 1.0]]
    assert len(endpoint.calls) == 3
    assert delays == [0.25, 0.5]
    assert provider.retry_count == 2


class AuthenticationError(RuntimeError):
    status_code = 401


class RejectedEmbeddingEndpoint(FakeEmbeddingEndpoint):
    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        raise AuthenticationError("invalid API key")


def test_embedding_provider_does_not_retry_permanent_authentication_failure() -> None:
    endpoint = RejectedEmbeddingEndpoint()
    delays: list[float] = []
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=endpoint),
        model_name="qwen3.7-text-embedding",
        dimensions=2,
        batch_size=20,
        max_retries=2,
        sleep=delays.append,
    )

    with pytest.raises(AuthenticationError, match="invalid API key"):
        provider.embed(["现金流风险"])

    assert len(endpoint.calls) == 1
    assert delays == []
    assert provider.retry_count == 0


class SelectivelyFailingEmbeddingEndpoint(FakeEmbeddingEndpoint):
    def __init__(self, failing_text: str | None) -> None:
        super().__init__()
        self.failing_text = failing_text

    def create(self, **kwargs: object) -> SimpleNamespace:
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        if self.failing_text in inputs:
            self.calls.append(kwargs)
            raise ConnectionError("temporary batch failure")
        return super().create(**kwargs)


def test_embedding_provider_resumes_from_completed_batch_checkpoints(
    tmp_path: Path,
) -> None:
    texts = ["甲", "乙乙", "丙丙丙"]
    first_endpoint = SelectivelyFailingEmbeddingEndpoint("乙乙")
    first_provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=first_endpoint),
        model_name="qwen3-embedding:4b",
        dimensions=2,
        batch_size=1,
        request_concurrency=1,
        batch_cache_dir=tmp_path,
    )

    with pytest.raises(ConnectionError, match="temporary batch failure"):
        first_provider.embed(texts)

    second_endpoint = SelectivelyFailingEmbeddingEndpoint(None)
    second_provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=second_endpoint),
        model_name="qwen3-embedding:4b",
        dimensions=2,
        batch_size=1,
        request_concurrency=1,
        batch_cache_dir=tmp_path,
    )

    vectors = second_provider.embed(texts)

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    requested_texts = [text for call in second_endpoint.calls for text in call["input"]]
    assert "甲" not in requested_texts
    assert "乙乙" in requested_texts
