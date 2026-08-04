-- 仅在需要跨公司近似检索时执行。
-- 当前按document_id隔离的RAG使用精确余弦排序，不要求这个全局索引。

CREATE INDEX IF NOT EXISTS prospectus_chunks_embedding_hnsw_idx
    ON ipo_risk.prospectus_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

ANALYZE ipo_risk.vector_documents;
ANALYZE ipo_risk.prospectus_chunks;
