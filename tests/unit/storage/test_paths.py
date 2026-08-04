"""文档级产物路径规则测试。"""

from pathlib import Path

import pytest

from ipo_risk.storage.paths import DocumentPaths


def test_document_paths_isolate_company_document_and_model(tmp_path: Path) -> None:
    paths = DocumentPaths.build(
        interim_root=tmp_path / "interim",
        artifacts_root=tmp_path / "artifacts",
        company_folder="03696_英矽智能",
        document_id="doc_03696_prospectus_2025-12-18_c44e3c20",
    )

    assert paths.pages_path.name == "pages.jsonl"
    assert paths.pages_path.parent.name == "doc_03696_prospectus_2025-12-18_c44e3c20"
    assert paths.tables_path == (
        tmp_path
        / "interim"
        / "tables"
        / "03696_英矽智能"
        / "doc_03696_prospectus_2025-12-18_c44e3c20"
        / "tables.jsonl"
    )
    assert paths.chunks_path == (
        tmp_path
        / "interim/chunks/03696_英矽智能"
        / "doc_03696_prospectus_2025-12-18_c44e3c20/chunks.jsonl"
    )
    assert paths.embedding_dir("qwen3.7-text-embedding").name == (
        "qwen3.7-text-embedding"
    )
    assert paths.embedding_dir("qwen3-embedding:4b").name == (
        "qwen3-embedding%3A4b"
    )
    assert paths.prelisting_features_path.name == "prelisting_features.json"
    assert paths.prelisting_features_path.parent == paths.report_path.parent


def test_document_paths_reject_directory_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="document_id"):
        DocumentPaths.build(
            interim_root=tmp_path / "interim",
            artifacts_root=tmp_path / "artifacts",
            company_folder="03696_英矽智能",
            document_id="../another-document",
        )
