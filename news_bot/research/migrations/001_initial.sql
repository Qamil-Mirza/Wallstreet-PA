-- Research-store timestamps use normalized UTC ISO-8601 text:
-- YYYY-MM-DDTHH:MM:SS.ffffffZ.
-- JSON metadata uses UTF-8 JSON with sorted keys and compact separators.
-- Exact decimal, money, and quantity values use canonical fixed-point TEXT,
-- never SQLite REAL.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL
);

CREATE TABLE portfolio_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    nav TEXT NOT NULL CHECK (typeof(nav) = 'text'),
    cash TEXT NOT NULL CHECK (typeof(cash) = 'text'),
    is_stale INTEGER NOT NULL CHECK (is_stale IN (0, 1)),
    account_ref TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE positions (
    position_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity TEXT NOT NULL CHECK (typeof(quantity) = 'text'),
    market_value TEXT NOT NULL CHECK (typeof(market_value) = 'text'),
    currency TEXT NOT NULL,
    cost_basis TEXT CHECK (cost_basis IS NULL OR typeof(cost_basis) = 'text'),
    security_id TEXT,
    metadata_json TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots(snapshot_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (security_id) REFERENCES securities(security_id)
        ON DELETE RESTRICT
);

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE securities (
    security_id TEXT PRIMARY KEY,
    entity_id TEXT,
    symbol TEXT NOT NULL,
    security_type TEXT NOT NULL,
    exchange TEXT,
    currency TEXT,
    identifiers_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (symbol, exchange),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE TABLE relationships (
    relationship_id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    as_of TEXT NOT NULL,
    confidence TEXT CHECK (confidence IS NULL OR typeof(confidence) = 'text'),
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (source_entity_id, target_entity_id, kind, as_of),
    FOREIGN KEY (source_entity_id) REFERENCES entities(entity_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (target_entity_id) REFERENCES entities(entity_id)
        ON DELETE RESTRICT
);

CREATE TABLE source_documents (
    document_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    publisher TEXT NOT NULL,
    published_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    raw_content_path TEXT,
    extraction_status TEXT NOT NULL,
    title TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE document_passages (
    passage_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    locator_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (document_id, ordinal),
    UNIQUE (document_id, content_hash),
    FOREIGN KEY (document_id) REFERENCES source_documents(document_id)
        ON DELETE RESTRICT
);

CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    entity_id TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'guidance', 'estimate', 'inference')),
    text TEXT NOT NULL,
    as_of TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK (typeof(confidence) = 'text'),
    status TEXT NOT NULL CHECK (status IN ('active', 'contradicted', 'superseded')),
    created_at TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE TABLE claim_evidence (
    claim_id TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    stance TEXT NOT NULL CHECK (stance IN ('supports', 'contradicts')),
    PRIMARY KEY (claim_id, passage_id),
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT
);

CREATE TABLE industry_theses (
    thesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    industry_key TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    thesis_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'archived')),
    created_at TEXT NOT NULL,
    UNIQUE (industry_key, version)
);

CREATE TABLE thesis_evidence (
    thesis_id INTEGER NOT NULL,
    passage_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (thesis_id, passage_id),
    UNIQUE (thesis_id, ordinal),
    FOREIGN KEY (thesis_id) REFERENCES industry_theses(thesis_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT
);

CREATE TABLE scenarios (
    scenario_id TEXT PRIMARY KEY,
    scenario_key TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    industry_key TEXT,
    title TEXT NOT NULL,
    narrative TEXT NOT NULL,
    probability TEXT CHECK (probability IS NULL OR typeof(probability) = 'text'),
    horizon TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'archived')),
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (scenario_key, version)
);

CREATE TABLE recommendations (
    recommendation_id TEXT PRIMARY KEY,
    recommendation_key TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    entity_id TEXT,
    security_id TEXT,
    rating TEXT NOT NULL
        CHECK (rating IN ('buy', 'hold', 'sell_reduce', 'no_rating')),
    rationale TEXT NOT NULL,
    target_price TEXT CHECK (target_price IS NULL OR typeof(target_price) = 'text'),
    currency TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'archived')),
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (recommendation_key, version),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT,
    FOREIGN KEY (security_id) REFERENCES securities(security_id) ON DELETE RESTRICT
);

CREATE TABLE research_tasks (
    task_id TEXT PRIMARY KEY,
    parent_task_id TEXT,
    task_kind TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK (state IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_text TEXT,
    FOREIGN KEY (parent_task_id) REFERENCES research_tasks(task_id)
        ON DELETE RESTRICT
);

CREATE TABLE agent_runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    output_text TEXT,
    error_text TEXT,
    provider TEXT,
    model TEXT,
    metadata_json TEXT,
    FOREIGN KEY (task_id) REFERENCES research_tasks(task_id) ON DELETE RESTRICT
);

CREATE TABLE reports (
    report_id TEXT PRIMARY KEY,
    report_key TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('draft', 'reviewed', 'published', 'archived')),
    created_by_run_id TEXT,
    created_at TEXT NOT NULL,
    published_at TEXT,
    metadata_json TEXT,
    UNIQUE (report_key, version),
    FOREIGN KEY (created_by_run_id) REFERENCES agent_runs(run_id)
        ON DELETE RESTRICT
);

CREATE TABLE model_usage (
    usage_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    cost_usd TEXT NOT NULL CHECK (typeof(cost_usd) = 'text'),
    recorded_at TEXT NOT NULL,
    provider_request_id TEXT,
    metadata_json TEXT,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE RESTRICT
);

CREATE TABLE budget_reservations (
    reservation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT,
    amount_usd TEXT NOT NULL CHECK (typeof(amount_usd) = 'text'),
    state TEXT NOT NULL
        CHECK (state IN ('reserved', 'consumed', 'released', 'expired')),
    reserved_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (task_id) REFERENCES research_tasks(task_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX idx_source_documents_canonical_url
    ON source_documents(canonical_url);
CREATE INDEX idx_entities_canonical_name ON entities(canonical_name);
CREATE INDEX idx_securities_symbol ON securities(symbol);
CREATE INDEX idx_claims_status ON claims(status);
CREATE INDEX idx_research_tasks_state ON research_tasks(state);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_industry_theses_key_version
    ON industry_theses(industry_key, version);
