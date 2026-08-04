import json
from pathlib import Path

from scripts.vectorize_competition_with_timeout import (
    _build_schedule,
    _is_retryable_company_error,
    _parse_timeout_reason,
    _parsing_finished,
    _resume_state,
)


def test_embedding_stage_disables_pdf_parse_timeout() -> None:
    assert _parsing_finished("embedding_started")
    assert _parsing_finished("embedding_progress")


def test_parser_cache_hit_disables_pdf_parse_timeout() -> None:
    assert _parsing_finished("parsing_cache_hit")


def test_active_pdf_parsing_remains_subject_to_timeout() -> None:
    assert not _parsing_finished("started")
    assert not _parsing_finished("parsing_started")
    assert not _parsing_finished("parsing_progress")


def test_active_parser_is_not_stopped_while_page_progress_continues() -> None:
    assert (
        _parse_timeout_reason(
            parsing_finished=False,
            elapsed_seconds=180,
            idle_seconds=5,
            stall_timeout_seconds=60,
            max_parse_seconds=300,
        )
        is None
    )


def test_parser_without_progress_is_stopped_after_stall_timeout() -> None:
    assert (
        _parse_timeout_reason(
            parsing_finished=False,
            elapsed_seconds=80,
            idle_seconds=61,
            stall_timeout_seconds=60,
            max_parse_seconds=300,
        )
        == "no_progress"
    )


def test_parser_is_stopped_at_absolute_maximum() -> None:
    assert (
        _parse_timeout_reason(
            parsing_finished=False,
            elapsed_seconds=301,
            idle_seconds=5,
            stall_timeout_seconds=60,
            max_parse_seconds=300,
        )
        == "max_duration"
    )


def test_embedding_is_never_subject_to_pdf_parse_timeout() -> None:
    assert (
        _parse_timeout_reason(
            parsing_finished=True,
            elapsed_seconds=600,
            idle_seconds=600,
            stall_timeout_seconds=60,
            max_parse_seconds=300,
        )
        is None
    )


def test_embedding_gateway_error_is_retryable() -> None:
    assert _is_retryable_company_error("Error code: 502")
    assert _is_retryable_company_error("APITimeoutError")


def test_document_validation_error_is_not_retryable() -> None:
    assert not _is_retryable_company_error("chunk bbox must satisfy x0 < x1 and y0 < y1")


def test_resume_state_runs_remaining_companies_before_historical_failures(
    tmp_path: Path,
) -> None:
    rows = [
        {"company_id": "1", "company_folder": "00001_A"},
        {"company_id": "2", "company_folder": "00002_B"},
        {"company_id": "3", "company_folder": "00003_C"},
    ]
    status_path = tmp_path / "status.json"
    failures_path = tmp_path / "failures.json"
    skipped_path = tmp_path / "skipped.json"
    retry_path = tmp_path / "retry.json"
    status_path.write_text(
        json.dumps({"total": 3, "completed": 2}),
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps({"failures": [{"company_id": "00002"}]}),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps({"skipped": []}),
        encoding="utf-8",
    )

    _, failures, _, _, finalized_ids = _resume_state(
        rows=rows,
        status_path=status_path,
        failures_path=failures_path,
        skipped_path=skipped_path,
        retry_events_path=retry_path,
    )
    schedule = list(
        _build_schedule(
            rows=rows,
            finalized_ids=finalized_ids,
            failures=failures,
        )
    )

    assert finalized_ids == {"00001", "00002"}
    assert [
        (row["company_id"], retry_round, already_finalized)
        for row, retry_round, already_finalized, _ in schedule
    ] == [
        ("3", 0, False),
        ("2", 1, True),
    ]


def test_retry_skipped_schedule_contains_only_previous_parse_timeouts() -> None:
    rows = [
        {"company_id": "1", "company_folder": "00001_A"},
        {"company_id": "2", "company_folder": "00002_B"},
        {"company_id": "3", "company_folder": "00003_C"},
    ]

    schedule = list(
        _build_schedule(
            rows=rows,
            finalized_ids={"00001", "00002", "00003"},
            failures=[{"company_id": "00002"}],
            skipped=[{"company_id": "00003"}],
            retry_skipped=True,
        )
    )

    assert [
        (row["company_id"], retry_round, already_finalized, source)
        for row, retry_round, already_finalized, source in schedule
    ] == [("3", 1, True, "skipped")]
