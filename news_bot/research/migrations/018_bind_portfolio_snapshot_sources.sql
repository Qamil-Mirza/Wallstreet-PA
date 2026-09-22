-- Bind each portfolio snapshot to the hashed Flex query identity that created it.
-- This prevents outage fallback from crossing account/query boundaries.

CREATE TABLE portfolio_snapshot_sources (
    snapshot_id TEXT PRIMARY KEY,
    source_key TEXT NOT NULL CHECK (length(source_key) = 64),
    bound_at TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_portfolio_snapshot_sources_source
    ON portfolio_snapshot_sources(source_key, snapshot_id);

CREATE TRIGGER portfolio_snapshot_sources_no_update
BEFORE UPDATE ON portfolio_snapshot_sources
BEGIN
    SELECT RAISE(ABORT, 'portfolio snapshot source binding is immutable');
END;

CREATE TRIGGER portfolio_snapshot_sources_no_delete
BEFORE DELETE ON portfolio_snapshot_sources
BEGIN
    SELECT RAISE(ABORT, 'portfolio snapshot source binding is durable');
END;
