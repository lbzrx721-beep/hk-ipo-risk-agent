"""上传招股书、创建分析任务和查询实时状态。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter, ValidationError

from ipo_risk.agents.intervention import (
    LlmFailureRecoveryDecision,
    RiskScopeReviewDecision,
)
from ipo_risk.api.batch_service import BatchWorkspaceRegistry
from ipo_risk.api.schemas import CatalogBatchRequest, PredictionRequest, UploadMetadata
from ipo_risk.runtime.batch_analysis_job import BatchUploadSpec
from ipo_risk.uploads import ProspectusUploadService


_UPLOAD_METADATA_ADAPTER = TypeAdapter(list[UploadMetadata])


def _sse(event: str, payload: object) -> str:
    body = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _parse_metadata(raw: str) -> list[UploadMetadata]:
    try:
        return _UPLOAD_METADATA_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "prospectus.pdf", await upload.read()))
    return payloads


def _build_upload_specs(
    project_root: Path,
    payloads: list[tuple[str, bytes]],
    metadata: list[UploadMetadata],
) -> tuple[BatchUploadSpec, ...]:
    if len(payloads) != len(metadata):
        raise ValueError("上传文件数量与公司信息数量不一致。")
    service = ProspectusUploadService(project_root=project_root)
    specs: list[BatchUploadSpec] = []
    seen_hashes: set[str] = set()
    for (file_name, content), item in zip(payloads, metadata, strict=True):
        if item.file_name != file_name:
            raise ValueError(f"文件顺序不一致：预期{item.file_name}，收到{file_name}。")
        suggestion = service.inspect(file_name=file_name, content=content)
        completed_runs = service.repository.list_analysis_runs_for_source_hash(
            suggestion.source_hash,
            status="succeeded",
        )
        if completed_runs and not item.force_reanalysis:
            raise ValueError(
                f"{file_name}已有完整分析（{completed_runs[0].run_id}），"
                "默认不会重复调用模型；如需重跑请明确勾选重新分析。"
            )
        if suggestion.source_hash in seen_hashes:
            raise ValueError(f"批次中存在重复PDF：{file_name}。")
        seen_hashes.add(suggestion.source_hash)
        specs.append(
            BatchUploadSpec(
                file_name=file_name,
                content=content,
                company_id=item.company_id,
                company_name=item.company_name,
                document_date=item.document_date,
                source_hash=suggestion.source_hash,
                page_count=suggestion.page_count,
            )
        )
    return tuple(specs)


def create_analysis_router(registry: BatchWorkspaceRegistry) -> APIRouter:
    router = APIRouter(tags=["analysis"])

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/companies")
    def companies(split: str | None = None, q: str | None = None) -> dict[str, Any]:
        rows = registry.catalog()
        if split:
            rows = tuple(item for item in rows if item.split == split)
        if q:
            query = q.strip().lower()
            rows = tuple(
                item
                for item in rows
                if query in item.company_id.lower() or query in item.company_name.lower()
            )
        return {
            "count": len(rows),
            "companies": [registry.company_to_dict(item) for item in rows[:200]],
        }

    @router.post("/uploads/inspect")
    async def inspect_uploads(
        files: Annotated[list[UploadFile], File(description="一份或多份PDF")],
    ) -> dict[str, Any]:
        payloads = await _read_uploads(files)

        def inspect_all() -> list[dict[str, Any]]:
            service = ProspectusUploadService(project_root=registry.project_root)
            results: list[dict[str, Any]] = []
            for file_name, content in payloads:
                suggestion = service.inspect(file_name=file_name, content=content)
                completed_runs = service.repository.list_analysis_runs_for_source_hash(
                    suggestion.source_hash,
                    status="succeeded",
                )
                latest = completed_runs[0] if completed_runs else None
                results.append(
                    {
                        "file_name": file_name,
                        "company_id": suggestion.company_id,
                        "company_name": suggestion.company_name,
                        "document_date": (
                            suggestion.document_date.isoformat()
                            if suggestion.document_date
                            else None
                        ),
                        "source": suggestion.source,
                        "page_count": suggestion.page_count,
                        "source_hash": suggestion.source_hash,
                        "existing_run_id": latest.run_id if latest else None,
                        "existing_completed_at": (
                            (latest.ended_at or latest.created_at).isoformat() if latest else None
                        ),
                    }
                )
            return results

        try:
            return {"files": await run_in_threadpool(inspect_all)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/batches", status_code=201)
    async def create_batch(
        files: Annotated[list[UploadFile], File()],
        metadata: Annotated[str, Form()],
    ) -> dict[str, Any]:
        payloads = await _read_uploads(files)
        parsed = _parse_metadata(metadata)
        try:
            specs = await run_in_threadpool(
                _build_upload_specs,
                registry.project_root,
                payloads,
                parsed,
            )
            workspace = registry.create_from_uploads(specs)
            return workspace.snapshot_payload()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/batches/from-catalog", status_code=201)
    def create_catalog_batch(request: CatalogBatchRequest) -> dict[str, Any]:
        try:
            workspace = registry.create_from_catalog(request.company_folders)
            return workspace.snapshot_payload()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/batches")
    def list_batches() -> dict[str, Any]:
        batches = registry.list_snapshots()
        return {"count": len(batches), "batches": batches}

    @router.get("/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict[str, Any]:
        return registry.require(batch_id).snapshot_payload()

    @router.post("/batches/{batch_id}/append", status_code=201)
    async def append_batch(
        batch_id: str,
        files: Annotated[list[UploadFile], File()],
        metadata: Annotated[str, Form()],
    ) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        payloads = await _read_uploads(files)
        parsed = _parse_metadata(metadata)
        try:
            specs = await run_in_threadpool(
                _build_upload_specs,
                registry.project_root,
                payloads,
                parsed,
            )
            workspace.append_uploads(specs)
            return workspace.snapshot_payload()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/batches/{batch_id}/pause")
    def pause_batch(batch_id: str) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        workspace.pause_after_current()
        return workspace.snapshot_payload()

    @router.post("/batches/{batch_id}/resume")
    def resume_batch(batch_id: str) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        workspace.resume()
        return workspace.snapshot_payload()

    @router.post("/batches/{batch_id}/companies/{company_key}/risk-scope")
    def submit_risk_scope(
        batch_id: str,
        company_key: str,
        decision: RiskScopeReviewDecision,
    ) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        try:
            workspace.submit_risk_scope_review(company_key, decision)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return workspace.snapshot_payload()

    @router.post("/batches/{batch_id}/companies/{company_key}/llm-recovery")
    def submit_llm_recovery(
        batch_id: str,
        company_key: str,
        decision: LlmFailureRecoveryDecision,
    ) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        try:
            workspace.submit_llm_failure_recovery(company_key, decision)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return workspace.snapshot_payload()

    @router.post("/batches/{batch_id}/companies/{company_key}/prediction")
    def rerun_prediction(
        batch_id: str,
        company_key: str,
        request: PredictionRequest,
    ) -> dict[str, Any]:
        workspace = registry.require(batch_id)
        try:
            prediction = workspace.rerun_prediction(
                company_key,
                variant=request.variant,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "prediction": prediction,
            "batch": workspace.snapshot_payload(),
        }

    @router.get("/batches/{batch_id}/stream")
    async def batch_stream(batch_id: str, request: Request) -> StreamingResponse:
        workspace = registry.require(batch_id)

        async def events() -> AsyncIterator[str]:
            last_snapshot = ""
            last_events: dict[str, str] = {}
            last_structured_events: dict[str, str] = {}
            heartbeat = 0
            while not await request.is_disconnected():
                snapshot = workspace.snapshot_payload()
                serialized = json.dumps(snapshot, ensure_ascii=False, default=str)
                if serialized != last_snapshot:
                    yield _sse("snapshot", snapshot)
                    last_snapshot = serialized
                for company in snapshot["companies"]:
                    company_key = str(company["company_key"])
                    recent = workspace.events_payload(company_key)[-120:]
                    event_json = json.dumps(recent, ensure_ascii=False, default=str)
                    if event_json != last_events.get(company_key, ""):
                        yield _sse(
                            "company-events",
                            {"company_key": company_key, "events": recent},
                        )
                        last_events[company_key] = event_json
                    durable = workspace.structured_events_payload(company_key)
                    durable_json = json.dumps(
                        durable,
                        ensure_ascii=False,
                        default=str,
                    )
                    if durable_json != last_structured_events.get(company_key, ""):
                        yield _sse(
                            "company-structured-events",
                            {"company_key": company_key, "events": durable},
                        )
                        last_structured_events[company_key] = durable_json
                heartbeat += 1
                if heartbeat >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat = 0
                await asyncio.sleep(1)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return router


__all__ = ["create_analysis_router"]
