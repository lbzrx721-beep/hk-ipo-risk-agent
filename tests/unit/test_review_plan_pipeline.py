from datetime import date
from pathlib import Path
from types import SimpleNamespace

from ipo_risk import pipeline
from ipo_risk.agents.review_planning import CompanyReviewPlanner
from ipo_risk.config import AppConfig


def configure(monkeypatch):
    config = AppConfig.model_validate(
        {
            "retrieval": {"mode": "bm25"},
            "embedding": {"api_key": "test-only-not-a-real-key"},
            "llm": {"api_key": "test-only-not-a-real-key"},
            "agents": {"discovery_strategy": "company_review_plan"},
        }
    )
    document = SimpleNamespace(
        document_type="prospectus",
        media_type="application/pdf",
        document_id="test-doc",
        available_at=date(2024, 1, 1),
        document_date=date(2024, 1, 1),
        local_file="unused.pdf",
        source_hash="a" * 64,
    )
    ingestion = SimpleNamespace(
        documents=[document],
        company=SimpleNamespace(
            company_id="01318", company_name_zh="测试公司", metadata_file="unused.json"
        ),
    )
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(pipeline, "register_company_sources", lambda *a, **kw: ingestion)
    monkeypatch.setattr(
        pipeline,
        "CatalogRepository",
        lambda *a: SimpleNamespace(save_analysis_report=lambda *a: None),
    )
    monkeypatch.setattr(pipeline, "read_json", lambda *a: {"listing_date": "2024-01-01"})

    def forbidden(*args, **kwargs):
        raise AssertionError("BM25 mode must not initialize embeddings or PostgreSQL")

    monkeypatch.setattr(pipeline, "OpenAICompatibleEmbeddingProvider", forbidden)
    monkeypatch.setattr(pipeline, "PgVectorRetriever", forbidden)
    monkeypatch.setattr(pipeline, "SemanticRetriever", forbidden)
    return config


def test_bm25_analysis_wires_planner_and_lexical_fundamentals_without_vector_connection(
    monkeypatch,
):
    configure(monkeypatch)
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        pipeline,
        "OpenAICompatibleJsonChatProvider",
        lambda **kw: SimpleNamespace(model_name=kw["model_name"], provider_name="test"),
    )
    seen = {}

    def analyze(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            report=object(),
            report_path=root / "tmp/test-not-written.json",
            prediction_path=None,
            embedding_dir=None,
        )

    monkeypatch.setattr(pipeline, "analyze_prospectus", analyze)
    pipeline.analyze_company("01318_测试公司", project_root=root)
    assert isinstance(seen["review_planner"], CompanyReviewPlanner)
    assert seen["risk_discoverer"] is None
    assert seen["embedding_provider"] is None
    assert seen["postgres_dsn"] is None
    assert seen["rag_fundamental_extractor"].retriever is None


def test_bm25_preparation_skips_vectorization_even_with_configured_embedding_key(
    monkeypatch, tmp_path
):
    configure(monkeypatch)
    monkeypatch.setattr(pipeline, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        pipeline,
        "_prepare_prospectus_source",
        lambda **kw: SimpleNamespace(chunks=[object()], pages=[object()], cache_hit=True),
    )
    events = []
    result = pipeline.prepare_company_semantic_index(
        "01318_测试公司",
        project_root=tmp_path,
        progress_callback=lambda s, d: events.append((s, d)),
    )
    assert result.embedding_ready is False
    assert result.chunk_count == 1
    assert any(stage == "embedding_skipped" for stage, _ in events)
