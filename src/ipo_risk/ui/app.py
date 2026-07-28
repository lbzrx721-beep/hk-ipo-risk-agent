"""Streamlit人工复核工作台启动入口。"""

from __future__ import annotations

from pathlib import Path

from ipo_risk.config import load_config
from ipo_risk.reviewing import ReviewService
from ipo_risk.storage.repositories import CatalogRepository
from ipo_risk.ui.workspace_app import render_workspace_app
from ipo_risk.uploads import ProspectusUploadService


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_review_service(project_root: Path | None = None) -> ReviewService:
    """按指定项目根目录连接SQLite复核服务。"""
    root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(
        config_path=root / "configs" / "base.yaml",
        providers_path=root / "configs" / "providers.local.yaml",
    )
    repository = CatalogRepository(root / config.paths.database)
    repository.initialize()
    return ReviewService(
        repository=repository,
        project_root=root,
    )


def main() -> None:
    review_service = create_review_service()
    render_workspace_app(
        upload_service=ProspectusUploadService(
            project_root=review_service.project_root,
            repository=review_service.repository,
        ),
        review_service=review_service,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_review_service", "main"]
