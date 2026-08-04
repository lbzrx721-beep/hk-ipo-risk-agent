from __future__ import annotations

from typing import Any

from psycopg import OperationalError

from scripts import import_embeddings_to_pgvector as migration


def test_transient_database_connection_failure_is_retried(monkeypatch) -> None:
    calls = 0

    def fake_process_once(*args: Any, **kwargs: Any) -> migration.ImportOutcome:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            raise OperationalError("connection timeout expired")
        return migration.ImportOutcome(
            company_folder="00001_测试公司",
            document_id="doc-test",
            added_count=10,
            chunk_count=10,
        )

    monkeypatch.setattr(migration, "_process_candidate_once", fake_process_once)
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)

    outcome = migration._process_candidate(
        config=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        row={"company_folder": "00001_测试公司"},
        batch_size=256,
        verify_only=False,
        max_retries=2,
        retry_initial_seconds=1,
    )

    assert calls == 2
    assert outcome.attempts == 2
    assert outcome.added_count == 10


def test_non_database_failure_is_not_retried(monkeypatch) -> None:
    calls = 0

    def fake_process_once(*args: Any, **kwargs: Any) -> migration.ImportOutcome:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ValueError("向量缓存不完整")

    monkeypatch.setattr(migration, "_process_candidate_once", fake_process_once)

    try:
        migration._process_candidate(
            config=None,  # type: ignore[arg-type]
            provider=None,  # type: ignore[arg-type]
            row={"company_folder": "00001_测试公司"},
            batch_size=256,
            verify_only=False,
            max_retries=3,
            retry_initial_seconds=1,
        )
    except ValueError as error:
        assert str(error) == "向量缓存不完整"
    else:  # pragma: no cover - 明确要求异常不能被吞掉
        raise AssertionError("expected ValueError")

    assert calls == 1
