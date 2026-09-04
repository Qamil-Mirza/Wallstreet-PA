-- Add durable typed-output replay, bounded lease recovery, and explicit
-- unknown-usage accounting while preserving populated v5 audit rows.

DROP TRIGGER provider_attempts_no_update;
DROP TRIGGER provider_attempts_require_running_execution;
DROP TRIGGER provider_attempts_no_delete;
DROP TRIGGER agent_executions_identity_immutable;
DROP TRIGGER agent_executions_evidence_hash_set_once;
DROP TRIGGER agent_executions_terminal_immutable;
DROP TRIGGER agent_executions_transition_once;
DROP TRIGGER agent_executions_no_delete;

DROP INDEX idx_provider_attempts_attempt_id;
DROP INDEX idx_provider_attempts_provider_recorded_at;
DROP INDEX idx_agent_executions_workflow_run_id;
DROP INDEX idx_agent_executions_task_id;
DROP INDEX idx_agent_executions_state_started_at;

ALTER TABLE provider_attempts RENAME TO provider_attempts_v5_old;
ALTER TABLE agent_executions RENAME TO agent_executions_v5_old;

CREATE TABLE agent_executions (
    attempt_id TEXT PRIMARY KEY,
    workflow_run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64),
    prompt_hash TEXT NOT NULL CHECK (length(prompt_hash) = 64),
    evidence_hash TEXT CHECK (evidence_hash IS NULL OR length(evidence_hash) = 64),
    state TEXT NOT NULL CHECK (state IN ('running', 'succeeded', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    safe_failure_code TEXT,
    output_hash TEXT CHECK (output_hash IS NULL OR length(output_hash) = 64),
    output_json TEXT CHECK (output_json IS NULL OR json_valid(output_json) = 1),
    provider TEXT,
    model TEXT,
    inference_mode TEXT CHECK (
        inference_mode IS NULL OR inference_mode IN ('external', 'local_only')
    ),
    fallback_reason TEXT,
    provider_attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (provider_attempt_count >= 0),
    input_tokens INTEGER DEFAULT 0 CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER DEFAULT 0 CHECK (output_tokens IS NULL OR output_tokens >= 0),
    reasoning_tokens INTEGER DEFAULT 0 CHECK (
        reasoning_tokens IS NULL OR reasoning_tokens >= 0
    ),
    usage_known INTEGER NOT NULL DEFAULT 1 CHECK (usage_known IN (0, 1)),
    lease_token TEXT,
    lease_expires_at TEXT,
    UNIQUE (workflow_run_id, task_id, role, schema_version),
    CHECK (
        (usage_known = 1 AND input_tokens IS NOT NULL
            AND output_tokens IS NOT NULL AND reasoning_tokens IS NOT NULL)
        OR
        (usage_known = 0 AND input_tokens IS NULL
            AND output_tokens IS NULL AND reasoning_tokens IS NULL)
    ),
    CHECK (
        (state = 'running' AND completed_at IS NULL
            AND safe_failure_code IS NULL AND output_hash IS NULL
            AND output_json IS NULL)
        OR
        (state = 'succeeded' AND completed_at IS NOT NULL
            AND evidence_hash IS NOT NULL AND safe_failure_code IS NULL
            AND output_hash IS NOT NULL)
        OR
        (state = 'failed' AND completed_at IS NOT NULL
            AND safe_failure_code IS NOT NULL AND output_hash IS NULL
            AND output_json IS NULL)
    ),
    FOREIGN KEY (task_id) REFERENCES research_tasks(task_id) ON DELETE RESTRICT
);

INSERT INTO agent_executions (
    attempt_id, workflow_run_id, task_id, role, schema_version, input_hash,
    prompt_hash, evidence_hash, state, started_at, completed_at,
    safe_failure_code, output_hash, provider, model, inference_mode,
    fallback_reason, provider_attempt_count, input_tokens, output_tokens,
    reasoning_tokens, usage_known
)
SELECT
    attempt_id, workflow_run_id, task_id, role, schema_version, input_hash,
    prompt_hash, evidence_hash, state, started_at, completed_at,
    safe_failure_code, output_hash, provider, model, inference_mode,
    fallback_reason, provider_attempt_count, input_tokens, output_tokens,
    reasoning_tokens, 1
FROM agent_executions_v5_old;

CREATE TABLE provider_attempts (
    provider_attempt_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    reasoning_tokens INTEGER CHECK (
        reasoning_tokens IS NULL OR reasoning_tokens >= 0
    ),
    usage_known INTEGER NOT NULL CHECK (usage_known IN (0, 1)),
    inference_mode TEXT NOT NULL CHECK (
        inference_mode IN ('external', 'local_only')
    ),
    fallback_reason TEXT,
    response_hash TEXT CHECK (response_hash IS NULL OR length(response_hash) = 64),
    failure_code TEXT,
    reservation_id TEXT,
    reservation_state TEXT CHECK (
        reservation_state IS NULL OR reservation_state IN (
            'reserved', 'usage_unknown', 'reconciled', 'released',
            'consumed', 'expired'
        )
    ),
    reserved_cost_usd TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (attempt_id, ordinal),
    CHECK (
        (status = 'succeeded' AND failure_code IS NULL)
        OR (status = 'failed' AND failure_code IS NOT NULL)
    ),
    CHECK (
        (usage_known = 1 AND input_tokens IS NOT NULL
            AND output_tokens IS NOT NULL AND reasoning_tokens IS NOT NULL)
        OR
        (usage_known = 0 AND input_tokens IS NULL
            AND output_tokens IS NULL AND reasoning_tokens IS NULL)
    ),
    CHECK (
        (reservation_id IS NULL AND reservation_state IS NULL
            AND reserved_cost_usd IS NULL)
        OR
        (reservation_id IS NOT NULL AND reservation_state IS NOT NULL
            AND reserved_cost_usd IS NOT NULL)
    ),
    FOREIGN KEY (attempt_id) REFERENCES agent_executions(attempt_id)
        ON DELETE RESTRICT
);

INSERT INTO provider_attempts (
    provider_attempt_id, attempt_id, ordinal, status, provider, model,
    latency_ms, input_tokens, output_tokens, reasoning_tokens, usage_known,
    inference_mode, fallback_reason, response_hash, failure_code, recorded_at
)
SELECT
    provider_attempt_id, attempt_id, ordinal, status, provider, model,
    latency_ms, input_tokens, output_tokens, reasoning_tokens, 1,
    inference_mode, fallback_reason, response_hash, failure_code, recorded_at
FROM provider_attempts_v5_old;

DROP TABLE provider_attempts_v5_old;
DROP TABLE agent_executions_v5_old;

CREATE INDEX idx_agent_executions_workflow_run_id
    ON agent_executions(workflow_run_id);
CREATE INDEX idx_agent_executions_task_id ON agent_executions(task_id);
CREATE INDEX idx_agent_executions_state_started_at
    ON agent_executions(state, started_at);
CREATE INDEX idx_provider_attempts_attempt_id
    ON provider_attempts(attempt_id);
CREATE INDEX idx_provider_attempts_provider_recorded_at
    ON provider_attempts(provider, recorded_at);

CREATE TRIGGER agent_executions_identity_immutable
BEFORE UPDATE OF attempt_id, workflow_run_id, task_id, role, schema_version,
    input_hash, prompt_hash, started_at
ON agent_executions
BEGIN
    SELECT RAISE(ABORT, 'agent execution identity is immutable');
END;

CREATE TRIGGER agent_executions_evidence_hash_set_once
BEFORE UPDATE OF evidence_hash ON agent_executions
WHEN NOT (
    OLD.state = 'running' AND NEW.state = 'running'
    AND OLD.evidence_hash IS NULL AND NEW.evidence_hash IS NOT NULL
    AND length(NEW.evidence_hash) = 64
)
BEGIN
    SELECT RAISE(ABORT, 'agent execution evidence hash is immutable');
END;

CREATE TRIGGER agent_executions_terminal_immutable
BEFORE UPDATE ON agent_executions
WHEN OLD.state IN ('succeeded', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'agent execution terminal row is immutable');
END;

CREATE TRIGGER agent_executions_transition_once
BEFORE UPDATE OF state ON agent_executions
WHEN OLD.state <> 'running' OR NEW.state NOT IN ('succeeded', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'agent execution state transition is invalid');
END;

CREATE TRIGGER agent_executions_success_requires_output_json
BEFORE UPDATE OF state ON agent_executions
WHEN NEW.state = 'succeeded' AND (
    NEW.output_json IS NULL OR json_valid(NEW.output_json) <> 1
)
BEGIN
    SELECT RAISE(ABORT, 'successful agent execution requires typed output JSON');
END;

CREATE TRIGGER agent_executions_output_json_set_on_success
BEFORE UPDATE OF output_json ON agent_executions
WHEN NOT (
    OLD.state = 'running' AND (
        (NEW.state = 'succeeded' AND OLD.output_json IS NULL
            AND NEW.output_json IS NOT NULL AND json_valid(NEW.output_json) = 1)
        OR
        (NEW.state = 'failed' AND OLD.output_json IS NULL
            AND NEW.output_json IS NULL)
    )
)
BEGIN
    SELECT RAISE(ABORT, 'agent execution output JSON is immutable');
END;

CREATE TRIGGER agent_executions_lease_progression
BEFORE UPDATE OF lease_token, lease_expires_at ON agent_executions
WHEN NOT (
    OLD.state = 'running' AND NEW.state = 'running'
    AND NEW.lease_token IS NOT NULL AND NEW.lease_expires_at IS NOT NULL
    AND (OLD.lease_expires_at IS NULL OR NEW.lease_expires_at > OLD.lease_expires_at)
)
BEGIN
    SELECT RAISE(ABORT, 'agent execution lease is immutable');
END;

CREATE TRIGGER agent_executions_new_running_requires_lease
BEFORE INSERT ON agent_executions
WHEN NEW.state = 'running' AND (
    NEW.lease_token IS NULL OR NEW.lease_expires_at IS NULL
)
BEGIN
    SELECT RAISE(ABORT, 'running agent execution requires a lease');
END;

CREATE TRIGGER agent_executions_no_delete
BEFORE DELETE ON agent_executions
BEGIN
    SELECT RAISE(ABORT, 'agent executions are immutable');
END;

CREATE TRIGGER provider_attempts_no_update
BEFORE UPDATE ON provider_attempts
BEGIN
    SELECT RAISE(ABORT, 'provider attempts are immutable');
END;

CREATE TRIGGER provider_attempts_require_running_execution
BEFORE INSERT ON provider_attempts
WHEN NOT EXISTS (
    SELECT 1 FROM agent_executions
    WHERE attempt_id = NEW.attempt_id AND state = 'running'
)
BEGIN
    SELECT RAISE(ABORT, 'provider attempt requires running agent execution');
END;

CREATE TRIGGER provider_attempts_no_delete
BEFORE DELETE ON provider_attempts
BEGIN
    SELECT RAISE(ABORT, 'provider attempts are immutable');
END;
