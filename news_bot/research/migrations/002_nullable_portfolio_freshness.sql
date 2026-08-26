-- Portfolio source snapshots may have no consumption-time freshness evaluation.
-- Rebuild both sides of the foreign key so already-applied v1 databases retain
-- their portfolio data while is_stale becomes nullable.

PRAGMA defer_foreign_keys = ON;

ALTER TABLE positions RENAME TO positions_v1_old;
ALTER TABLE portfolio_snapshots RENAME TO portfolio_snapshots_v1_old;

CREATE TABLE portfolio_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    nav TEXT NOT NULL CHECK (typeof(nav) = 'text'),
    cash TEXT NOT NULL CHECK (typeof(cash) = 'text'),
    is_stale INTEGER CHECK (is_stale IS NULL OR is_stale IN (0, 1)),
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

INSERT INTO portfolio_snapshots (
    snapshot_id, as_of, base_currency, nav, cash, is_stale, account_ref,
    metadata_json, created_at
)
SELECT
    snapshot_id, as_of, base_currency, nav, cash, is_stale, account_ref,
    metadata_json, created_at
FROM portfolio_snapshots_v1_old;

INSERT INTO positions (
    position_id, snapshot_id, symbol, quantity, market_value, currency,
    cost_basis, security_id, metadata_json
)
SELECT
    position_id, snapshot_id, symbol, quantity, market_value, currency,
    cost_basis, security_id, metadata_json
FROM positions_v1_old;

DROP TABLE positions_v1_old;
DROP TABLE portfolio_snapshots_v1_old;

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_snapshot_id ON positions(snapshot_id);
CREATE INDEX idx_positions_security_id ON positions(security_id);
