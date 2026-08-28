-- Claim logical agent executions before inference and retain immutable,
-- hash-only provider-attempt telemetry without rewriting legacy audit rows.

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
    provider TEXT,
    model TEXT,
    inference_mode TEXT CHECK (
        inference_mode IS NULL OR inference_mode IN ('external', 'local_only')
    ),
    fallback_reason TEXT,
    provider_attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (provider_attempt_count >= 0),
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    reasoning_tokens INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
    UNIQUE (workflow_run_id, task_id, role, schema_version),
    CHECK (
        (state = 'running' AND completed_at IS NULL
            AND safe_failure_code IS NULL AND output_hash IS NULL)
        OR
        (state = 'succeeded' AND completed_at IS NOT NULL
            AND evidence_hash IS NOT NULL
            AND safe_failure_code IS NULL AND output_hash IS NOT NULL)
        OR
        (state = 'failed' AND completed_at IS NOT NULL
            AND safe_failure_code IS NOT NULL AND output_hash IS NULL)
    ),
    FOREIGN KEY (task_id) REFERENCES research_tasks(task_id) ON DELETE RESTRICT
);

CREATE INDEX idx_agent_executions_workflow_run_id
    ON agent_executions(workflow_run_id);
CREATE INDEX idx_agent_executions_task_id ON agent_executions(task_id);
CREATE INDEX idx_agent_executions_state_started_at
    ON agent_executions(state, started_at);

CREATE TABLE provider_attempts (
    provider_attempt_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    reasoning_tokens INTEGER NOT NULL CHECK (reasoning_tokens >= 0),
    inference_mode TEXT NOT NULL CHECK (
        inference_mode IN ('external', 'local_only')
    ),
    fallback_reason TEXT,
    response_hash TEXT CHECK (response_hash IS NULL OR length(response_hash) = 64),
    failure_code TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (attempt_id, ordinal),
    CHECK (
        (status = 'succeeded' AND failure_code IS NULL)
        OR (status = 'failed' AND failure_code IS NOT NULL)
    ),
    FOREIGN KEY (attempt_id) REFERENCES agent_executions(attempt_id)
        ON DELETE RESTRICT
);

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
    OLD.state = 'running'
    AND OLD.evidence_hash IS NULL
    AND NEW.evidence_hash IS NOT NULL
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
