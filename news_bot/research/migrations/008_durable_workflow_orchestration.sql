-- Durable workflow DAGs, task claims, safe results, and renewable named leases.

CREATE TABLE workflow_runs (
    workflow_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE CHECK (length(idempotency_key) = 64),
    workflow_kind TEXT NOT NULL CHECK (
        workflow_kind IN ('daily', 'weekly', 'monthly', 'backfill')
    ),
    period_key TEXT NOT NULL,
    as_of TEXT NOT NULL,
    portfolio_date TEXT,
    source_hashes_json TEXT NOT NULL CHECK (json_valid(source_hashes_json) = 1),
    definition_hash TEXT NOT NULL CHECK (length(definition_hash) = 64),
    state TEXT NOT NULL CHECK (
        state IN ('running', 'completed', 'partial', 'blocked', 'failed', 'deferred')
    ),
    dry_run INTEGER NOT NULL CHECK (dry_run IN (0, 1)),
    authorize_analysis INTEGER NOT NULL CHECK (authorize_analysis IN (0, 1)),
    industry_key TEXT,
    max_documents INTEGER CHECK (max_documents IS NULL OR max_documents > 0),
    completed_stages_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(completed_stages_json) = 1),
    report_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(report_ids_json) = 1),
    omissions_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(omissions_json) = 1),
    new_agent_runs INTEGER NOT NULL DEFAULT 0 CHECK (new_agent_runs >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (
        (state = 'running' AND completed_at IS NULL)
        OR (state <> 'running' AND completed_at IS NOT NULL)
    )
);

CREATE INDEX idx_workflow_runs_kind_period
    ON workflow_runs(workflow_kind, period_key);
CREATE INDEX idx_workflow_runs_state_updated
    ON workflow_runs(state, updated_at);

CREATE TABLE workflow_tasks (
    task_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE CHECK (length(idempotency_key) = 64),
    stage TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'running', 'completed', 'failed', 'deferred')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    assigned_role TEXT,
    originating_role TEXT,
    defer_reason TEXT,
    result_ref TEXT,
    result_hash TEXT CHECK (result_hash IS NULL OR length(result_hash) = 64),
    outcome_json TEXT CHECK (outcome_json IS NULL OR json_valid(outcome_json) = 1),
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 64),
    lease_expires_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (workflow_id, task_id),
    UNIQUE (workflow_id, ordinal),
    CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL)),
    CHECK (state = 'running' OR (lease_token IS NULL AND lease_expires_at IS NULL)),
    FOREIGN KEY (workflow_id) REFERENCES workflow_runs(workflow_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_workflow_tasks_workflow_ordinal
    ON workflow_tasks(workflow_id, ordinal);
CREATE INDEX idx_workflow_tasks_runnable
    ON workflow_tasks(workflow_id, state, ordinal, attempt_count);
CREATE INDEX idx_workflow_tasks_lease_expiry
    ON workflow_tasks(state, lease_expires_at);

CREATE TABLE workflow_task_dependencies (
    workflow_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    dependency_task_id TEXT NOT NULL,
    PRIMARY KEY (workflow_id, task_id, dependency_task_id),
    CHECK (task_id <> dependency_task_id),
    FOREIGN KEY (workflow_id) REFERENCES workflow_runs(workflow_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (workflow_id, task_id)
        REFERENCES workflow_tasks(workflow_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY (workflow_id, dependency_task_id)
        REFERENCES workflow_tasks(workflow_id, task_id) ON DELETE RESTRICT
);

CREATE INDEX idx_workflow_task_dependencies_dependency
    ON workflow_task_dependencies(workflow_id, dependency_task_id);

CREATE TABLE workflow_leases (
    lease_name TEXT PRIMARY KEY,
    workflow_id TEXT,
    owner_id TEXT NOT NULL,
    lease_token TEXT NOT NULL UNIQUE CHECK (length(lease_token) = 64),
    expires_at TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    renewed_at TEXT NOT NULL,
    FOREIGN KEY (workflow_id) REFERENCES workflow_runs(workflow_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_workflow_leases_expiry ON workflow_leases(expires_at);

CREATE TRIGGER workflow_runs_identity_immutable
BEFORE UPDATE OF workflow_id, idempotency_key, workflow_kind, period_key, as_of,
    portfolio_date, source_hashes_json, definition_hash, dry_run,
    authorize_analysis, industry_key, max_documents, created_at
ON workflow_runs
BEGIN
    SELECT RAISE(ABORT, 'workflow identity is immutable');
END;

CREATE TRIGGER workflow_tasks_identity_immutable
BEFORE UPDATE OF task_id, workflow_id, idempotency_key, stage, ordinal,
    max_attempts, created_at
ON workflow_tasks
BEGIN
    SELECT RAISE(ABORT, 'workflow task identity is immutable');
END;

CREATE TRIGGER workflow_runs_no_delete
BEFORE DELETE ON workflow_runs
BEGIN
    SELECT RAISE(ABORT, 'workflow runs are durable');
END;

CREATE TRIGGER workflow_tasks_no_delete
BEFORE DELETE ON workflow_tasks
BEGIN
    SELECT RAISE(ABORT, 'workflow tasks are durable');
END;

CREATE TRIGGER workflow_task_dependencies_no_update
BEFORE UPDATE ON workflow_task_dependencies
BEGIN
    SELECT RAISE(ABORT, 'workflow dependencies are immutable');
END;

CREATE TRIGGER workflow_task_dependencies_no_delete
BEFORE DELETE ON workflow_task_dependencies
BEGIN
    SELECT RAISE(ABORT, 'workflow dependencies are immutable');
END;
