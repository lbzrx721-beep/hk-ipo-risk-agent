from __future__ import annotations

from datetime import date
from pathlib import Path
from threading import Event, Lock

import numpy as np

from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.persistence import ChromaPersistenceTask


class NoEmbeddingProvider:
    model_name = "test-embedding"
    dimensions = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise AssertionError("Chroma persistence must reuse document vectors")


class RecordingWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.embeddings: np.ndarray | None = None

    def add_chunks(
        self,
        chunks,
        *,
        skip_existing_document=True,
        precomputed_embeddings=None,
    ) -> int:
        del skip_existing_document
        if self.fail:
            raise OSError("chroma disk unavailable")
        assert len(chunks) == 1
        self.embeddings = precomputed_embeddings
        return len(chunks)


def _chunk(document_id: str = "prospectus-2025-12-18") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="C-1",
        company_id="06651",
        document_id=document_id,
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=1,
        printed_page_label="1",
        bbox=(1.0, 2.0, 3.0, 4.0),
        text="数字孪生业务风险",
        font_size=10.0,
        available_at=date(2025, 12, 18),
    )


def test_chroma_persistence_reuses_the_same_document_vectors_in_background(
    tmp_path: Path,
) -> None:
    provider = NoEmbeddingProvider()
    writer = RecordingWriter()
    vectors = np.asarray([[0.8, 0.2]], dtype=np.float32)
    vectors.flags.writeable = False
    statuses = []
    events: list[tuple[str, str]] = []
    task = ChromaPersistenceTask(
        chunks=[_chunk()],
        document_vectors=vectors,
        provider=provider,
        persist_dir=tmp_path / "chroma",
        collection_name="prospectus_chunks",
        writer_factory=lambda _provider, _path, _name: writer,
        status_callback=statuses.append,
        progress_callback=lambda stage, detail: events.append((stage, detail)),
    )

    task.start()
    assert task.wait(timeout=2)

    assert provider.calls == 0
    assert writer.embeddings is vectors
    assert [snapshot.status for snapshot in statuses] == ["building", "ready"]
    assert task.snapshot().status == "ready"
    assert [stage for stage, _detail in events] == [
        "vector_persistence_started",
        "vector_persistence_completed",
    ]


def test_chroma_persistence_failure_is_recorded_without_raising_to_analysis(
    tmp_path: Path,
) -> None:
    provider = NoEmbeddingProvider()
    task = ChromaPersistenceTask(
        chunks=[_chunk()],
        document_vectors=np.asarray([[0.8, 0.2]], dtype=np.float32),
        provider=provider,
        persist_dir=tmp_path / "chroma",
        collection_name="prospectus_chunks",
        writer_factory=lambda _provider, _path, _name: RecordingWriter(fail=True),
    )

    task.start()
    assert task.wait(timeout=2)

    snapshot = task.snapshot()
    assert snapshot.status == "failed"
    assert snapshot.error == "chroma disk unavailable"


def test_chroma_writes_to_one_collection_are_serialized_across_documents(
    tmp_path: Path,
) -> None:
    first_entered = Event()
    second_entered = Event()
    release_first = Event()
    calls_lock = Lock()
    calls = 0

    class BlockingWriter(RecordingWriter):
        def add_chunks(self, chunks, **kwargs) -> int:
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                second_entered.set()
            return super().add_chunks(chunks, **kwargs)

    writer = BlockingWriter()
    provider = NoEmbeddingProvider()
    tasks = [
        ChromaPersistenceTask(
            chunks=[_chunk(document_id)],
            document_vectors=np.asarray([[0.8, 0.2]], dtype=np.float32),
            provider=provider,
            persist_dir=tmp_path / "chroma",
            collection_name="prospectus_chunks",
            writer_factory=lambda _provider, _path, _name: writer,
        )
        for document_id in ("doc-1", "doc-2")
    ]

    tasks[0].start()
    assert first_entered.wait(timeout=2)
    tasks[1].start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    assert all(task.wait(timeout=2) for task in tasks)
    assert calls == 2
