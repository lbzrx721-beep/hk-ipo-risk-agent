"""FastAPI应用入口。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ipo_risk.api.batch_service import BatchWorkspaceRegistry
from ipo_risk.api.routers.analysis import create_analysis_router
from ipo_risk.api.routers.reports import create_reports_router
from ipo_risk.config import load_config
from ipo_risk.reviewing import ReviewService
from ipo_risk.retrying import FindingRetryService
from ipo_risk.storage.repositories import CatalogRepository


def create_app(*, project_root: Path | None = None) -> FastAPI:
    root = (project_root or Path.cwd()).resolve()
    app = FastAPI(
        title="港股IPO风险分析API",
        version="0.2.0",
        description="为React工作台提供批量上传、任务控制和实时事件流。",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    registry = BatchWorkspaceRegistry(project_root=root)
    config = load_config(
        config_path=root / "configs" / "base.yaml",
        providers_path=root / "configs" / "providers.local.yaml",
    )
    repository = CatalogRepository(root / config.paths.database)
    repository.initialize()
    review_service = ReviewService(repository=repository, project_root=root)
    retry_service = FindingRetryService(repository=repository, project_root=root)
    app.state.project_root = root
    app.state.batch_registry = registry
    app.state.review_service = review_service
    app.state.retry_service = retry_service
    app.include_router(create_analysis_router(registry), prefix="/api")
    app.include_router(create_reports_router(review_service, retry_service), prefix="/api")

    frontend_dist = root / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=frontend_dist, html=True),
            name="frontend",
        )
    return app


app = create_app()


__all__ = ["app", "create_app"]
