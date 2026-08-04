-- 港股IPO招股书向量库基础结构。
-- HNSW索引单独创建，避免在约310万条历史向量导入期间反复维护大索引。

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS ipo_risk;

CREATE TABLE IF NOT EXISTS ipo_risk.vector_documents (
    document_id text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL CHECK (embedding_dimensions = 1024),
    company_id text NOT NULL,
    company_name text NOT NULL DEFAULT '',
    source_file text NOT NULL,
    source_hash text NOT NULL,
    available_at date NOT NULL,
    chunk_count integer NOT NULL CHECK (chunk_count >= 0),
    is_current boolean NOT NULL DEFAULT true,
    is_complete boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, embedding_model)
);

CREATE INDEX IF NOT EXISTS vector_documents_company_idx
    ON ipo_risk.vector_documents (company_id, embedding_model, is_current);
CREATE INDEX IF NOT EXISTS vector_documents_source_idx
    ON ipo_risk.vector_documents (source_hash, embedding_model);

CREATE TABLE IF NOT EXISTS ipo_risk.prospectus_chunks (
    document_id text NOT NULL,
    embedding_model text NOT NULL,
    chunk_id text NOT NULL,
    company_id text NOT NULL,
    source_file text NOT NULL,
    source_hash text NOT NULL,
    page integer NOT NULL CHECK (page >= 1),
    printed_page_label text,
    bbox real[] NOT NULL CHECK (cardinality(bbox) = 4),
    content text NOT NULL,
    is_heading boolean NOT NULL DEFAULT false,
    is_ocr boolean NOT NULL DEFAULT false,
    font_size real NOT NULL DEFAULT 0 CHECK (font_size >= 0),
    available_at date NOT NULL,
    chunk_kind text NOT NULL CHECK (chunk_kind IN ('text', 'table')),
    table_id text,
    table_cell_refs text[] NOT NULL DEFAULT '{}',
    table_cell_values text[] NOT NULL DEFAULT '{}',
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, embedding_model, chunk_id),
    FOREIGN KEY (document_id, embedding_model)
        REFERENCES ipo_risk.vector_documents (document_id, embedding_model)
        ON DELETE CASCADE,
    CHECK (cardinality(table_cell_refs) = cardinality(table_cell_values))
);

CREATE INDEX IF NOT EXISTS prospectus_chunks_document_idx
    ON ipo_risk.prospectus_chunks (document_id, embedding_model);
CREATE INDEX IF NOT EXISTS prospectus_chunks_company_page_idx
    ON ipo_risk.prospectus_chunks (company_id, page);
CREATE INDEX IF NOT EXISTS prospectus_chunks_kind_idx
    ON ipo_risk.prospectus_chunks (chunk_kind)
    WHERE chunk_kind = 'table';

-- 兼容早期试运行创建的char(64)字段，便于驱动使用高吞吐二进制COPY。
ALTER TABLE ipo_risk.vector_documents
    ALTER COLUMN source_hash TYPE text;
ALTER TABLE ipo_risk.prospectus_chunks
    ALTER COLUMN source_hash TYPE text;
