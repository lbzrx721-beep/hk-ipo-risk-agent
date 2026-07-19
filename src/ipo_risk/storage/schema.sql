PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    schema_name TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS companies (
    company_id TEXT PRIMARY KEY,
    company_folder TEXT NOT NULL UNIQUE,
    ticker TEXT,
    exchange TEXT,
    market TEXT,
    company_name_zh TEXT NOT NULL,
    company_name_en TEXT,
    listing_date TEXT,
    metadata_file TEXT NOT NULL,
    metadata_hash TEXT NOT NULL CHECK (length(metadata_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    document_type TEXT NOT NULL,
    title TEXT NOT NULL,
    local_file TEXT NOT NULL,
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    document_date TEXT NOT NULL,
    available_at TEXT NOT NULL,
    language TEXT,
    source_kind TEXT,
    source_url TEXT,
    media_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL CHECK (file_size_bytes > 0),
    page_count INTEGER CHECK (page_count IS NULL OR page_count > 0),
    text_layer_status TEXT,
    metadata_status TEXT NOT NULL CHECK (metadata_status IN ('declared', 'inferred')),
    metadata_json TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT,
    UNIQUE (company_id, document_type, source_hash),
    UNIQUE (company_id, local_file)
);

CREATE INDEX IF NOT EXISTS idx_documents_company_available
    ON documents(company_id, available_at, document_type);
CREATE INDEX IF NOT EXISTS idx_documents_source_hash
    ON documents(source_hash);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    title TEXT NOT NULL,
    local_file TEXT NOT NULL,
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    available_at TEXT NOT NULL,
    file_format TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    row_count INTEGER CHECK (row_count IS NULL OR row_count >= 0),
    usage_scope TEXT,
    pre_listing_eligible INTEGER NOT NULL CHECK (pre_listing_eligible IN (0, 1)),
    metadata_json TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT,
    UNIQUE (company_id, dataset_type, source_hash),
    UNIQUE (company_id, local_file)
);

CREATE INDEX IF NOT EXISTS idx_datasets_company_available
    ON datasets(company_id, available_at, dataset_type);

CREATE TABLE IF NOT EXISTS vector_indexes (
    index_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
    vectors_path TEXT NOT NULL,
    metadata_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building', 'ready', 'stale', 'failed')),
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE RESTRICT,
    UNIQUE (document_id, embedding_model, dimensions)
);

CREATE INDEX IF NOT EXISTS idx_vector_indexes_document
    ON vector_indexes(document_id, status);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    report_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_company
    ON analysis_runs(company_id, as_of_date, created_at);

CREATE TABLE IF NOT EXISTS analysis_run_documents (
    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    PRIMARY KEY (run_id, document_id),
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS evidence (
    run_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page INTEGER NOT NULL CHECK (page > 0),
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    screenshot_path TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, evidence_id),
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_evidence_document_page
    ON evidence(document_id, page);

CREATE TABLE IF NOT EXISTS risk_findings (
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    risk_type TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id),
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_findings_company_type
    ON risk_findings(company_id, risk_type, status);

CREATE TABLE IF NOT EXISTS finding_evidence (
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id, evidence_id),
    FOREIGN KEY (run_id, finding_id)
        REFERENCES risk_findings(run_id, finding_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, evidence_id)
        REFERENCES evidence(run_id, evidence_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_traces (
    run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    PRIMARY KEY (run_id, trace_id),
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('confirmed', 'rejected', 'edited')),
    reviewer TEXT NOT NULL,
    comment TEXT,
    edited_payload_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id, finding_id)
        REFERENCES risk_findings(run_id, finding_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_reviews_run_finding
    ON reviews(run_id, finding_id, created_at);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    run_id TEXT,
    company_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    horizon INTEGER NOT NULL CHECK (horizon IN (1, 5, 20, 60)),
    target TEXT NOT NULL,
    risk_probability REAL NOT NULL CHECK (
        risk_probability >= 0.0 AND risk_probability <= 1.0
    ),
    model_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE SET NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_predictions_company_asof
    ON predictions(company_id, as_of_date, horizon);
