-- Durable publication intents/receipts and guarded orchestration transitions.

CREATE TABLE publication_effects (
    publication_effect_id TEXT PRIMARY KEY,
    effect_key TEXT NOT NULL UNIQUE CHECK (length(effect_key) = 64),
    workflow_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('claimed', 'confirmed', 'outcome_unknown', 'reconciled')
    ),
    claim_token TEXT CHECK (claim_token IS NULL OR length(claim_token) = 64),
    claim_expires_at TEXT,
    report_id TEXT,
    receipt_hash TEXT CHECK (receipt_hash IS NULL OR length(receipt_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (workflow_id, task_id),
    CHECK (
        (state = 'claimed' AND claim_token IS NOT NULL
            AND claim_expires_at IS NOT NULL AND report_id IS NULL
            AND receipt_hash IS NULL)
        OR
        (state = 'outcome_unknown' AND claim_token IS NULL
            AND claim_expires_at IS NULL AND report_id IS NULL
            AND receipt_hash IS NULL)
        OR
        (state IN ('confirmed', 'reconciled') AND claim_token IS NULL
            AND claim_expires_at IS NULL AND report_id IS NOT NULL
            AND receipt_hash IS NOT NULL)
    ),
    FOREIGN KEY (workflow_id) REFERENCES workflow_runs(workflow_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (workflow_id, task_id)
        REFERENCES workflow_tasks(workflow_id, task_id) ON DELETE RESTRICT
);

CREATE INDEX idx_publication_effects_workflow_state
    ON publication_effects(workflow_id, state);
CREATE INDEX idx_publication_effects_state_updated
    ON publication_effects(state, updated_at);

CREATE TRIGGER publication_effects_identity_immutable
BEFORE UPDATE OF publication_effect_id, effect_key, workflow_id, task_id, created_at
ON publication_effects
BEGIN
    SELECT RAISE(ABORT, 'publication effect identity is immutable');
END;

CREATE TRIGGER publication_effects_transition_guard
BEFORE UPDATE OF state ON publication_effects
WHEN NOT (
    (OLD.state = 'claimed' AND NEW.state IN ('confirmed', 'outcome_unknown'))
    OR (OLD.state = 'outcome_unknown' AND NEW.state = 'reconciled')
)
BEGIN
    SELECT RAISE(ABORT, 'publication effect transition is invalid');
END;

CREATE TRIGGER publication_effects_terminal_immutable
BEFORE UPDATE ON publication_effects
WHEN OLD.state IN ('confirmed', 'reconciled')
BEGIN
    SELECT RAISE(ABORT, 'publication effect receipt is immutable');
END;

CREATE TRIGGER publication_effects_no_delete
BEFORE DELETE ON publication_effects
BEGIN
    SELECT RAISE(ABORT, 'publication effects are durable');
END;

CREATE TRIGGER workflow_runs_terminal_immutable
BEFORE UPDATE ON workflow_runs
WHEN OLD.state IN ('completed', 'partial', 'blocked', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'terminal workflow run is immutable');
END;

CREATE TRIGGER workflow_runs_transition_guard
BEFORE UPDATE OF state ON workflow_runs
WHEN NOT (
    (OLD.state = 'running' AND NEW.state IN (
        'completed', 'partial', 'blocked', 'failed', 'deferred'
    ))
    OR (OLD.state = 'deferred' AND NEW.state = 'running')
)
BEGIN
    SELECT RAISE(ABORT, 'workflow run transition is invalid');
END;

CREATE TRIGGER workflow_tasks_role_immutable
BEFORE UPDATE OF assigned_role, originating_role ON workflow_tasks
BEGIN
    SELECT RAISE(ABORT, 'workflow task roles are immutable');
END;

CREATE TRIGGER workflow_tasks_completed_immutable
BEFORE UPDATE ON workflow_tasks
WHEN OLD.state = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed workflow task is immutable');
END;

CREATE TRIGGER workflow_tasks_transition_guard
BEFORE UPDATE OF state ON workflow_tasks
WHEN OLD.state <> NEW.state AND NOT (
    (OLD.state = 'pending' AND NEW.state IN ('running', 'deferred'))
    OR (OLD.state = 'running' AND NEW.state IN ('completed', 'failed', 'deferred'))
    OR (OLD.state IN ('failed', 'deferred') AND NEW.state = 'pending')
)
BEGIN
    SELECT RAISE(ABORT, 'workflow task transition is invalid');
END;

CREATE TRIGGER workflow_tasks_attempt_progression
BEFORE UPDATE OF attempt_count ON workflow_tasks
WHEN NEW.attempt_count <> OLD.attempt_count + 1
    OR NEW.state <> 'running'
    OR OLD.state <> 'pending'
BEGIN
    SELECT RAISE(ABORT, 'workflow task attempt progression is invalid');
END;

CREATE TRIGGER workflow_tasks_terminal_output_guard
BEFORE UPDATE OF result_ref, result_hash, outcome_json ON workflow_tasks
WHEN OLD.state <> 'running'
    OR NEW.state NOT IN ('completed', 'deferred')
BEGIN
    SELECT RAISE(ABORT, 'workflow task output transition is invalid');
END;

CREATE TRIGGER workflow_tasks_no_result_rewrite
BEFORE UPDATE OF result_ref, result_hash, outcome_json ON workflow_tasks
WHEN OLD.result_ref IS NOT NULL OR OLD.result_hash IS NOT NULL
    OR OLD.outcome_json IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'workflow task output is immutable');
END;

CREATE TRIGGER workflow_tasks_terminal_reason_immutable
BEFORE UPDATE OF defer_reason, completed_at ON workflow_tasks
WHEN OLD.state IN ('failed', 'deferred') AND NEW.state <> 'pending'
BEGIN
    SELECT RAISE(ABORT, 'workflow task terminal reason is immutable');
END;
