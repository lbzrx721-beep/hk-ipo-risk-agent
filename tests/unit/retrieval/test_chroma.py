from __future__ import annotations

from datetime import date
import logging

import numpy as np
import pytest
from ipo_risk.retrieval import ChromaRetriever, DocumentChunk


class _FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.add_batch_sizes: list[int] = []
        self._client = None

    def get(self, *, where: dict[str, object], include: list[str] | None = None):
        del include
        document_id = where["document_id"]
        ids = [
            record_id
            for record_id, record in self.records.items()
            if record["metadata"]["document_id"] == document_id
        ]
        return {"ids": ids}

    def add(self, *, ids, embeddings, documents, metadatas) -> None:
        self.add_batch_sizes.append(len(ids))
        for record_id, vector, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.records[record_id] = {
                "embedding": vector,
                "document": document,
                "metadata": metadata,
            }

    def query(self, *, query_embeddings, n_results, where, include):
        del query_embeddings, include
        document_id = where["document_id"]
        matching = [
            (record_id, record)
            for record_id, record in self.records.items()
            if record["metadata"]["document_id"] == document_id
        ][:n_results]
        return {
            "ids": [[record_id for record_id, _ in matching]],
            "documents": [[record["document"] for _, record in matching]],
            "metadatas": [[record["metadata"] for _, record in matching]],
            "distances": [[0.1 for _ in matching]],
        }

    def count(self) -> int:
        return len(self.records)


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, *, name: str, metadata: dict[str, str]):
        del metadata
        collection = self.collections.setdefault(name, _FakeCollection())
        collection._client = self
        return collection

    def get_max_batch_size(self) -> int:
        return 2


class _CorruptCollection(_FakeCollection):
    def get(self, *, where: dict[str, object], include: list[str] | None = None):
        del where, include
        raise RuntimeError(
            "Error constructing hnsw segment reader: Error loading hnsw index"
        )


class _RecoveringClient(_FakeClient):
    def get_or_create_collection(self, *, name: str, metadata: dict[str, str]):
        del metadata
        if not self.collections:
            collection: _FakeCollection = _CorruptCollection()
        else:
            collection = self.collections.get(name) or _FakeCollection()
        self.collections[name] = collection
        collection._client = self
        return collection


class _Provider:
    model_name = "test-embedding"
    dimensions = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def _chunk(*, chunk_id: str, document_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id=document_id,
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(1.0, 2.0, 3.0, 4.0),
        text=text,
        font_size=10.0,
        available_at=date(2025, 1, 1),
    )


def test_chroma_retriever_limits_search_to_the_active_document(tmp_path) -> None:
    client = _FakeClient()
    provider = _Provider()

    foreign = ChromaRetriever(
        provider,
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    foreign.add_chunks([_chunk(chunk_id="foreign-1", document_id="foreign", text="关联交易")])

    current = ChromaRetriever(
        provider,
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    current.add_chunks([_chunk(chunk_id="current-1", document_id="current", text="现金流风险")])

    hits = current.search("风险", top_k=10)

    assert [hit.chunk.chunk_id for hit in hits] == ["current-1"]
    assert hits[0].rank == 1
    assert current.count == 2


def test_chroma_retriever_reuses_an_existing_document_without_embedding(tmp_path) -> None:
    client = _FakeClient()
    provider = _Provider()
    retriever = ChromaRetriever(
        provider,
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    chunk = _chunk(chunk_id="current-1", document_id="current", text="现金流风险")

    assert retriever.add_chunks([chunk]) == 1
    assert retriever.add_chunks([chunk], skip_existing_document=True) == 0


def test_chroma_round_trip_preserves_table_semantics(tmp_path) -> None:
    client = _FakeClient()
    retriever = ChromaRetriever(
        _Provider(),
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    table_chunk = _chunk(
        chunk_id="C-00501-0068-T01-01",
        document_id="current",
        text="[表格证据]\n|客户|收入|\n|---|---|\n|客户A|120|",
    ).model_copy(
        update={
            "chunk_kind": "table",
            "table_id": "T-P0068-001",
            "table_cell_refs": (
                "T-P0068-001:R1:C1",
                "T-P0068-001:R1:C2",
            ),
            "table_cell_values": ("客户A", "120"),
        }
    )

    retriever.add_chunks([table_chunk])
    hit = retriever.search("客户收入", top_k=1)[0].chunk

    assert hit.chunk_kind == "table"
    assert hit.table_id == "T-P0068-001"
    assert hit.table_cell_refs == (
        "T-P0068-001:R1:C1",
        "T-P0068-001:R1:C2",
    )
    assert hit.table_cell_values == ("客户A", "120")


def test_chroma_splits_large_documents_and_resumes_partial_writes(tmp_path) -> None:
    client = _FakeClient()
    retriever = ChromaRetriever(
        _Provider(),
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    chunks = [
        _chunk(
            chunk_id=f"current-{index}",
            document_id="current",
            text=f"风险原文{index}",
        )
        for index in range(5)
    ]

    retriever.add_chunks(chunks[:2])
    collection = next(iter(client.collections.values()))
    collection.add_batch_sizes.clear()
    added = retriever.add_chunks(chunks, skip_existing_document=True)

    assert added == 3
    assert collection.add_batch_sizes == [2, 1]
    assert retriever.count == 5


def test_chroma_isolates_a_corrupt_hnsw_generation_and_rebuilds_safely(
    tmp_path,
) -> None:
    client = _RecoveringClient()
    retriever = ChromaRetriever(
        _Provider(),
        persist_dir=tmp_path,
        collection_name="chunks",
        client=client,
    )
    chunk = _chunk(
        chunk_id="current-1",
        document_id="current",
        text="现金流风险",
    )

    assert retriever.add_chunks([chunk]) == 1
    assert len(client.collections) == 2
    assert retriever.count == 1
    assert (tmp_path / "chroma_collection_generations.json").is_file()


def test_chroma_allows_the_same_chunk_id_in_two_document_versions(tmp_path) -> None:
    client = _FakeClient()
    provider = _Provider()
    first = ChromaRetriever(
        provider, persist_dir=tmp_path, collection_name="chunks", client=client
    )
    second = ChromaRetriever(
        provider, persist_dir=tmp_path, collection_name="chunks", client=client
    )

    first.add_chunks([_chunk(chunk_id="C-company-0001-001", document_id="v1", text="旧版")])
    second.add_chunks([_chunk(chunk_id="C-company-0001-001", document_id="v2", text="新版")])

    assert [hit.chunk.text for hit in second.search("新版")] == ["新版"]
    assert second.count == 2


def test_chroma_uses_a_separate_collection_when_embedding_model_changes(tmp_path) -> None:
    client = _FakeClient()
    first_provider = _Provider()
    second_provider = _Provider()
    second_provider.model_name = "another-model"

    ChromaRetriever(
        first_provider, persist_dir=tmp_path, collection_name="chunks", client=client
    )
    ChromaRetriever(
        second_provider, persist_dir=tmp_path, collection_name="chunks", client=client
    )

    assert len(client.collections) == 2


def test_chroma_retriever_works_with_the_installed_persistent_client(tmp_path) -> None:
    pytest.importorskip("chromadb")
    provider = _Provider()
    retriever = ChromaRetriever(
        provider,
        persist_dir=tmp_path / "chroma",
        collection_name="chunks",
    )
    retriever.add_chunks(
        [_chunk(chunk_id="current-1", document_id="current", text="现金流风险")]
    )

    hits = retriever.search("现金流", top_k=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["current-1"]


def test_installed_chroma_accepts_precomputed_numpy_vectors(tmp_path) -> None:
    pytest.importorskip("chromadb")

    class _NoEmbeddingProvider(_Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            del texts
            raise AssertionError("document embeddings must not be recomputed")

    retriever = ChromaRetriever(
        _NoEmbeddingProvider(),
        persist_dir=tmp_path / "chroma-precomputed",
        collection_name="chunks",
    )

    added = retriever.add_chunks(
        [_chunk(chunk_id="current-1", document_id="current", text="现金流风险")],
        precomputed_embeddings=np.asarray([[0.8, 0.2]], dtype=np.float32),
    )

    assert added == 1
    assert retriever.count == 1


def test_installed_chroma_does_not_emit_product_telemetry_errors(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("chromadb")
    caplog.set_level(logging.ERROR)
    retriever = ChromaRetriever(
        _Provider(),
        persist_dir=tmp_path / "chroma-no-telemetry",
        collection_name="chunks",
    )

    retriever.add_chunks(
        [_chunk(chunk_id="current-1", document_id="current", text="现金流风险")]
    )
    retriever.search("现金流", top_k=1)

    assert "Failed to send telemetry event" not in caplog.text
