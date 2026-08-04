"""招股书切片与检索公共接口。"""

from .chunker import DocumentChunk, build_page_chunks
from .lexical import BM25Retriever, SearchHit, tokenize_text
from .pgvector import PgVectorDocumentStatus, PgVectorRetriever
from .search import RiskRule, load_risk_rules, select_rule_candidates
from .semantic import (
    EmbeddingProvider,
    HybridRetriever,
    HybridSearchHit,
    OpenAICompatibleEmbeddingProvider,
    SemanticRetriever,
)

__all__ = [
    "BM25Retriever",
    "DocumentChunk",
    "EmbeddingProvider",
    "HybridRetriever",
    "HybridSearchHit",
    "OpenAICompatibleEmbeddingProvider",
    "PgVectorDocumentStatus",
    "PgVectorRetriever",
    "RiskRule",
    "SearchHit",
    "SemanticRetriever",
    "build_page_chunks",
    "load_risk_rules",
    "select_rule_candidates",
    "tokenize_text",
]
