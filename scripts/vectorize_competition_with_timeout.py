"""顺序向量化赛事招股书，首次PDF解析超时后跳过该公司。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import get_ident
import time
import traceback
from typing import Any
from uuid import uuid4

import numpy as np

from ipo_risk.config import AppConfig, load_config
from ipo_risk.pipeline import prepare_company_semantic_index
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.semantic import OpenAICompatibleEmbeddingProvider
from ipo_risk.storage.files import read_jsonl
from ipo_risk.storage.paths import DocumentPaths


PARSING_FINISHED_STAGES = frozenset(
    {
        "parsing_cache_hit",
        "embedding_started",
        "embedding_progress",
        "embedding_completed",
        "completed",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, attempts: int = 10) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)
    raise RuntimeError("unreachable")


def _try_read_json(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path, attempts=3)
    except (OSError, json.JSONDecodeError):
        return {}


def _provider(config: AppConfig) -> OpenAICompatibleEmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider(
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        model_name=config.embedding.model_name,
        api_model_name=config.embedding.request_model_name,
        dimensions=config.embedding.dimensions,
        batch_size=config.embedding.batch_size,
        request_concurrency=config.embedding.request_concurrency,
        timeout_seconds=config.embedding.timeout_seconds,
        max_retries=config.embedding.max_retries,
        retry_initial_seconds=config.embedding.retry_initial_seconds,
        retry_backoff_factor=config.embedding.retry_backoff_factor,
        retry_max_seconds=config.embedding.retry_max_seconds,
    )


def _competition_rows(root: Path, limit: int | None) -> list[dict[str, Any]]:
    manifest = root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    rows = [row for row in read_jsonl(manifest) if row.get("label_status") == "complete"]
    split_order = {"train": 0, "validation": 1, "test": 2}
    rows.sort(
        key=lambda row: (
            split_order.get(str(row.get("split")), 9),
            int(row.get("sequence", 0)),
        )
    )
    return rows[:limit] if limit is not None else rows


def _child_progress_writer(progress_path: Path):
    def emit(stage: str, detail: str) -> None:
        try:
            _write_json(
                progress_path,
                {
                    "stage": stage,
                    "detail": detail,
                    "updated_at": _utc_now(),
                },
            )
        except OSError as exc:
            # 进度展示不能反过来中断PDF解析或向量化主任务。
            print(
                f"[{datetime.now():%H:%M:%S}] progress_warning | {exc}",
                file=sys.stderr,
                flush=True,
            )
        print(f"[{datetime.now():%H:%M:%S}] {stage} | {detail}", flush=True)

    return emit


def _prepare_one_company(
    *,
    company_folder: str,
    progress_path: Path,
    result_path: Path,
    embedding_request_concurrency: int,
    embedding_max_retries: int,
    embedding_batch_cache_dir: Path,
) -> int:
    config = load_config()
    root = config.project_root.resolve()
    emit = _child_progress_writer(progress_path)
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=config.embedding.require_api_key(),
        base_url=config.embedding.base_url,
        model_name=config.embedding.model_name,
        api_model_name=config.embedding.request_model_name,
        dimensions=config.embedding.dimensions,
        batch_size=config.embedding.batch_size,
        request_concurrency=embedding_request_concurrency,
        timeout_seconds=config.embedding.timeout_seconds,
        max_retries=embedding_max_retries,
        retry_initial_seconds=config.embedding.retry_initial_seconds,
        retry_backoff_factor=config.embedding.retry_backoff_factor,
        retry_max_seconds=config.embedding.retry_max_seconds,
        progress_callback=lambda completed, total: emit(
            "embedding_progress",
            (
                f"预向量化 [{completed}/{total}] · "
                f"{round(completed / total * 100) if total else 100}%"
            ),
        ),
        batch_cache_dir=embedding_batch_cache_dir,
    )
    try:
        emit("started", f"开始处理 {company_folder}")
        result = prepare_company_semantic_index(
            company_folder,
            project_root=root,
            embedding_provider=provider,
            progress_callback=emit,
        )
        provider.clear_batch_checkpoint()
        payload = {
            "state": "completed",
            "company_folder": result.company_folder,
            "document_id": result.document_id,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
            "embedding_ready": result.embedding_ready,
            "embedding_cache_hit": result.embedding_cache_hit,
            "parser_cache_hit": result.parser_cache_hit,
        }
        _write_json(result_path, payload)
        emit("completed", f"{company_folder} 解析与向量化完成")
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "state": "failed",
            "company_folder": company_folder,
            "error": str(exc).strip() or type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        _write_json(result_path, payload)
        emit("failed", payload["error"])
        return 1


def _parsing_finished(stage: str | None) -> bool:
    return bool(stage and stage in PARSING_FINISHED_STAGES)


def _parse_timeout_reason(
    *,
    parsing_finished: bool,
    elapsed_seconds: float,
    idle_seconds: float,
    stall_timeout_seconds: float,
    max_parse_seconds: float,
) -> str | None:
    if parsing_finished:
        return None
    if idle_seconds > stall_timeout_seconds:
        return "no_progress"
    if elapsed_seconds > max_parse_seconds:
        return "max_duration"
    return None


def _is_retryable_company_error(message: str) -> bool:
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "error code: 502",
            "error code: 503",
            "error code: 504",
            "timed out",
            "timeout",
            "connection reset",
            "connectionerror",
            "apiconnectionerror",
            "temporarily unavailable",
            "rate limit",
        )
    )


def _company_id(row: dict[str, Any]) -> str:
    return str(row["company_id"]).zfill(5)


def _resume_state(
    *,
    rows: list[dict[str, Any]],
    status_path: Path,
    failures_path: Path,
    skipped_path: Path,
    retry_events_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[str],
]:
    status = _read_json(status_path)
    if int(status.get("total", -1)) != len(rows):
        raise RuntimeError("cannot resume because the manifest size differs from the saved run")
    completed = int(status.get("completed", 0))
    finalized_ids = {str(value).zfill(5) for value in status.get("finalized_company_ids", [])}
    if not finalized_ids:
        finalized_ids = {_company_id(row) for row in rows[:completed]}
    if len(finalized_ids) != completed:
        raise RuntimeError(
            "cannot resume because finalized company IDs do not match the completed count"
        )
    failures_payload = _read_json(failures_path) if failures_path.is_file() else {}
    skipped_payload = _read_json(skipped_path) if skipped_path.is_file() else {}
    retry_payload = _read_json(retry_events_path) if retry_events_path.is_file() else {}
    failures = list(failures_payload.get("failures", []))
    skipped = list(skipped_payload.get("skipped", []))
    retry_events = list(retry_payload.get("retry_events", []))
    return (
        status,
        failures,
        skipped,
        retry_events,
        finalized_ids,
    )


def _remove_failure(
    failures: list[dict[str, Any]],
    company_id: str,
) -> None:
    failures[:] = [
        item for item in failures if str(item.get("company_id", "")).zfill(5) != company_id
    ]


def _remove_skipped(
    skipped: list[dict[str, Any]],
    company_id: str,
) -> None:
    skipped[:] = [
        item for item in skipped if str(item.get("company_id", "")).zfill(5) != company_id
    ]


def _build_schedule(
    *,
    rows: list[dict[str, Any]],
    finalized_ids: set[str],
    failures: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
    retry_skipped: bool = False,
) -> deque[tuple[dict[str, Any], int, bool, str | None]]:
    if retry_skipped:
        skipped_ids = {
            str(item.get("company_id", "")).zfill(5)
            for item in (skipped or [])
        }
        return deque(
            (row, 1, True, "skipped")
            for row in rows
            if _company_id(row) in skipped_ids
        )
    remaining_rows = [row for row in rows if _company_id(row) not in finalized_ids]
    failed_ids = {str(item.get("company_id", "")).zfill(5) for item in failures}
    historical_retry_rows = [row for row in rows if _company_id(row) in failed_ids]
    return deque(
        [(row, 0, False, None) for row in remaining_rows]
        + [(row, 1, True, "failure") for row in historical_retry_rows]
    )


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )


def _run_batch(
    *,
    limit: int | None,
    parse_stall_timeout_seconds: float,
    parse_max_seconds: float,
    embedding_request_concurrency: int,
    embedding_max_retries: int,
    company_retry_rounds: int,
    resume: bool,
    retry_skipped: bool,
) -> int:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    config = load_config()
    root = config.project_root.resolve()
    rows = _competition_rows(root, limit)
    runtime = root / "artifacts" / "runtime" / "vectorize_competition_timeout"
    company_runtime = runtime / "companies"
    embedding_batch_runtime = runtime / "embedding_batch_checkpoints"
    company_runtime.mkdir(parents=True, exist_ok=True)
    embedding_batch_runtime.mkdir(parents=True, exist_ok=True)
    status_path = runtime / "status.json"
    failures_path = runtime / "failures.json"
    skipped_path = runtime / "skipped_parse_timeout.json"
    retry_events_path = runtime / "retry_events.json"
    provider = _provider(config)
    if resume:
        (
            status,
            failures,
            skipped,
            retry_events,
            finalized_ids,
        ) = _resume_state(
            rows=rows,
            status_path=status_path,
            failures_path=failures_path,
            skipped_path=skipped_path,
            retry_events_path=retry_events_path,
        )
        status["state"] = "running"
        status["resumed_at"] = _utc_now()
        status["updated_at"] = _utc_now()
        status["current_company"] = None
        status["current_stage"] = None
        status["current_detail"] = "已从保存状态继续"
    else:
        status = {
            "schema_version": 2,
            "state": "running",
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "total": len(rows),
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_parse_timeout": 0,
            "parser_cache_hits": 0,
            "embedding_cache_hits": 0,
            "attempted": 0,
            "retry_scheduled": 0,
            "retry_recovered": 0,
            "current_company": None,
            "current_stage": None,
            "current_detail": None,
            "last_company": None,
        }
        failures = []
        skipped = []
        retry_events = []
        finalized_ids = set()
    status.setdefault("attempted", int(status.get("completed", 0)))
    status.setdefault("retry_scheduled", 0)
    status.setdefault("retry_recovered", 0)
    status["company_retry_rounds"] = company_retry_rounds
    status["embedding_request_concurrency"] = embedding_request_concurrency
    status["embedding_max_retries"] = embedding_max_retries
    status["parse_stall_timeout_seconds"] = parse_stall_timeout_seconds
    status["parse_max_seconds"] = parse_max_seconds
    status["run_mode"] = "embedding_cache_only"
    status["retry_skipped"] = retry_skipped
    status.setdefault("embedding_vectors_cached", 0)
    status["finalized_company_ids"] = [
        _company_id(row) for row in rows if _company_id(row) in finalized_ids
    ]
    _write_json(status_path, status)
    _write_json(failures_path, {"failures": failures})
    _write_json(skipped_path, {"skipped": skipped})
    _write_json(retry_events_path, {"retry_events": retry_events})

    schedule = _build_schedule(
        rows=rows,
        finalized_ids=finalized_ids,
        failures=failures,
        skipped=skipped,
        retry_skipped=retry_skipped,
    )
    while schedule:
        row, retry_round, already_finalized, recovery_source = schedule.popleft()
        company_folder = str(row["company_folder"])
        company_id = _company_id(row)
        attempt_suffix = "" if retry_round == 0 else f".retry{retry_round}"
        progress_path = company_runtime / f"{company_id}.progress.json"
        result_path = company_runtime / f"{company_id}.result.json"
        stdout_path = company_runtime / f"{company_id}{attempt_suffix}.stdout.log"
        stderr_path = company_runtime / f"{company_id}{attempt_suffix}.stderr.log"
        for path in (progress_path, result_path):
            if path.exists():
                path.unlink()

        status["current_company"] = company_folder
        status["current_stage"] = "starting"
        status["current_detail"] = (
            "正在启动独立PDF处理进程"
            if retry_round == 0
            else f"正在执行公司级重试 {retry_round}/{company_retry_rounds}"
        )
        status["current_retry_round"] = retry_round
        status["attempted"] += 1
        status["updated_at"] = _utc_now()
        _write_json(status_path, status)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--prepare-company",
            company_folder,
            "--progress-path",
            str(progress_path),
            "--result-path",
            str(result_path),
            "--embedding-request-concurrency",
            str(embedding_request_concurrency),
            "--embedding-max-retries",
            str(embedding_max_retries),
            "--embedding-batch-cache-dir",
            str(embedding_batch_runtime / company_id),
        ]
        started = time.monotonic()
        last_progress_at = started
        last_progress_signature: tuple[str, str, str] | None = None
        parse_finished = False
        timed_out = False
        timeout_reason: str | None = None
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            next_status_write = 0.0
            while process.poll() is None:
                progress = _try_read_json(progress_path) if progress_path.is_file() else {}
                stage = str(progress.get("stage") or "starting")
                detail = str(progress.get("detail") or "等待子进程进度")
                signature = (
                    stage,
                    detail,
                    str(progress.get("updated_at") or ""),
                )
                if progress and signature != last_progress_signature:
                    last_progress_signature = signature
                    last_progress_at = time.monotonic()
                parse_finished = parse_finished or _parsing_finished(stage)
                elapsed = time.monotonic() - started
                idle = time.monotonic() - last_progress_at
                timeout_reason = _parse_timeout_reason(
                    parsing_finished=parse_finished,
                    elapsed_seconds=elapsed,
                    idle_seconds=idle,
                    stall_timeout_seconds=parse_stall_timeout_seconds,
                    max_parse_seconds=parse_max_seconds,
                )
                if timeout_reason is not None:
                    timed_out = True
                    _terminate(process)
                    break
                if time.monotonic() >= next_status_write:
                    status["current_stage"] = stage
                    status["current_detail"] = detail
                    status["current_elapsed_seconds"] = round(elapsed, 1)
                    status["current_idle_seconds"] = round(idle, 1)
                    status["updated_at"] = _utc_now()
                    _write_json(status_path, status)
                    next_status_write = time.monotonic() + 1.0
                time.sleep(0.25)

        if timed_out:
            skipped_item = {
                "company_folder": company_folder,
                "company_id": company_id,
                "page_count": row.get("page_count"),
                "timeout_type": timeout_reason,
                "reason": (
                    f"PDF解析连续{parse_stall_timeout_seconds:g}秒无进展，已跳过"
                    if timeout_reason == "no_progress"
                    else f"PDF解析总时长超过{parse_max_seconds:g}秒，已跳过"
                ),
                "skipped_at": _utc_now(),
            }
            if recovery_source == "skipped":
                _remove_skipped(skipped, company_id)
            else:
                status["skipped_parse_timeout"] += 1
            skipped.append(skipped_item)
            if recovery_source == "failure":
                _remove_failure(failures, company_id)
                status["failed"] = max(0, int(status["failed"]) - 1)
                _write_json(failures_path, {"failures": failures})
            _write_json(skipped_path, {"skipped": skipped})
            print(
                f"[{status['completed'] + 1}/{status['total']}] {company_folder} 解析超时，已跳过",
                flush=True,
            )
        elif process.returncode != 0 or not result_path.is_file():
            result = _read_json(result_path) if result_path.is_file() else {}
            error_message = str(result.get("error") or f"子进程退出码 {process.returncode}")
            if _is_retryable_company_error(error_message) and retry_round < company_retry_rounds:
                next_round = retry_round + 1
                retry_event = {
                    "company_folder": company_folder,
                    "company_id": company_id,
                    "retry_round": next_round,
                    "error": error_message,
                    "scheduled_at": _utc_now(),
                }
                retry_events.append(retry_event)
                schedule.append((row, next_round, already_finalized, recovery_source))
                status["retry_scheduled"] += 1
                status["current_company"] = None
                status["current_stage"] = "retry_scheduled"
                status["current_detail"] = (
                    f"{company_folder} 遇到临时接口错误，已安排第"
                    f"{next_round}/{company_retry_rounds}轮重试"
                )
                status["updated_at"] = _utc_now()
                _write_json(
                    retry_events_path,
                    {"retry_events": retry_events},
                )
                _write_json(status_path, status)
                print(
                    f"{company_folder} 临时失败，已安排第"
                    f"{next_round}/{company_retry_rounds}轮重试："
                    f"{error_message}",
                    flush=True,
                )
                continue
            failure = {
                "company_folder": company_folder,
                "company_id": company_id,
                "retry_round": retry_round,
                "error": error_message,
                "failed_at": _utc_now(),
            }
            _remove_failure(failures, company_id)
            failures.append(failure)
            if recovery_source == "skipped":
                _remove_skipped(skipped, company_id)
                status["skipped_parse_timeout"] = max(
                    0,
                    int(status["skipped_parse_timeout"]) - 1,
                )
                status["failed"] += 1
                _write_json(skipped_path, {"skipped": skipped})
            elif not already_finalized:
                status["failed"] += 1
            _write_json(failures_path, {"failures": failures})
            print(
                f"[{status['completed'] + 1}/{status['total']}] "
                f"{company_folder} 失败：{failure['error']}",
                flush=True,
            )
        else:
            result = _read_json(result_path)
            paths = DocumentPaths.build(
                interim_root=root / config.paths.interim_dir,
                artifacts_root=root / config.paths.artifacts_dir,
                company_folder=company_folder,
                document_id=str(result["document_id"]),
            )
            chunks = [DocumentChunk.model_validate(item) for item in read_jsonl(paths.chunks_path)]
            vectors = np.load(
                paths.embedding_dir(provider.model_name) / "vectors.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
            if vectors.ndim != 2 or vectors.shape != (
                len(chunks),
                provider.dimensions,
            ):
                raise RuntimeError(
                    f"{company_folder} embedding缓存不完整："
                    f"期望({len(chunks)}, {provider.dimensions})，实际{vectors.shape}"
                )
            status["succeeded"] += 1
            if recovery_source == "failure":
                _remove_failure(failures, company_id)
                status["failed"] = max(0, int(status["failed"]) - 1)
                _write_json(failures_path, {"failures": failures})
            elif recovery_source == "skipped":
                _remove_skipped(skipped, company_id)
                status["skipped_parse_timeout"] = max(
                    0,
                    int(status["skipped_parse_timeout"]) - 1,
                )
                _write_json(skipped_path, {"skipped": skipped})
            if retry_round > 0:
                status["retry_recovered"] += 1
            status["parser_cache_hits"] += int(result["parser_cache_hit"])
            status["embedding_cache_hits"] += int(result["embedding_cache_hit"])
            status["embedding_vectors_cached"] += len(chunks)
            print(
                f"[{status['completed'] + 1}/{status['total']}] "
                f"{company_folder} embedding缓存完成：{len(chunks)}条",
                flush=True,
            )

        if not already_finalized:
            finalized_ids.add(company_id)
            status["completed"] = len(finalized_ids)
            status["finalized_company_ids"] = [
                _company_id(item) for item in rows if _company_id(item) in finalized_ids
            ]
        status["last_company"] = company_folder
        status["current_stage"] = None
        status["current_detail"] = None
        status["current_retry_round"] = 0
        status["current_elapsed_seconds"] = 0
        status["current_idle_seconds"] = 0
        status["updated_at"] = _utc_now()
        _write_json(status_path, status)

    if status["failed"]:
        status["state"] = "completed_with_failures"
    elif status["skipped_parse_timeout"]:
        status["state"] = "completed_with_skips"
    else:
        status["state"] = "completed"
    status["current_company"] = None
    status["current_stage"] = None
    status["current_detail"] = "所有可处理公司的embedding缓存任务已结束"
    status["ended_at"] = _utc_now()
    status["updated_at"] = _utc_now()
    _write_json(status_path, status)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--parse-stall-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--parse-max-seconds",
        type=float,
        default=300.0,
    )
    parser.add_argument("--prepare-company", default=None)
    parser.add_argument("--progress-path", type=Path, default=None)
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument(
        "--embedding-request-concurrency",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--embedding-max-retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--company-retry-rounds",
        type=int,
        default=2,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="与--resume配合，仅重新处理上次因PDF解析超时跳过的公司",
    )
    parser.add_argument("--embedding-batch-cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prepare_company:
        if (
            args.progress_path is None
            or args.result_path is None
            or args.embedding_batch_cache_dir is None
        ):
            raise SystemExit(
                "--prepare-company requires --progress-path, --result-path "
                "and --embedding-batch-cache-dir"
            )
        return _prepare_one_company(
            company_folder=args.prepare_company,
            progress_path=args.progress_path,
            result_path=args.result_path,
            embedding_request_concurrency=args.embedding_request_concurrency,
            embedding_max_retries=args.embedding_max_retries,
            embedding_batch_cache_dir=args.embedding_batch_cache_dir,
        )
    if args.parse_stall_timeout_seconds <= 0:
        raise SystemExit("--parse-stall-timeout-seconds must be positive")
    if args.parse_max_seconds <= 0:
        raise SystemExit("--parse-max-seconds must be positive")
    if args.parse_max_seconds < args.parse_stall_timeout_seconds:
        raise SystemExit(
            "--parse-max-seconds must be greater than or equal to --parse-stall-timeout-seconds"
        )
    if args.embedding_request_concurrency <= 0:
        raise SystemExit("--embedding-request-concurrency must be positive")
    if args.embedding_max_retries < 0:
        raise SystemExit("--embedding-max-retries must be non-negative")
    if args.company_retry_rounds < 0:
        raise SystemExit("--company-retry-rounds must be non-negative")
    if args.retry_skipped and not args.resume:
        raise SystemExit("--retry-skipped requires --resume")
    return _run_batch(
        limit=args.limit,
        parse_stall_timeout_seconds=args.parse_stall_timeout_seconds,
        parse_max_seconds=args.parse_max_seconds,
        embedding_request_concurrency=args.embedding_request_concurrency,
        embedding_max_retries=args.embedding_max_retries,
        company_retry_rounds=args.company_retry_rounds,
        resume=args.resume,
        retry_skipped=args.retry_skipped,
    )


if __name__ == "__main__":
    raise SystemExit(main())
