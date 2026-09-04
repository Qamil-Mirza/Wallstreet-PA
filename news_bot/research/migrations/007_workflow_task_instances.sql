-- Normalize logical tasks into workflow-scoped state instances without
-- changing the legacy research_tasks API or its dependent foreign keys.

CREATE TABLE workflow_task_instances (
    task_instance_id TEXT PRIMARY KEY,
    workflow_run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64),
    prompt_hash TEXT NOT NULL CHECK (length(prompt_hash) = 64),
    state TEXT NOT NULL CHECK (state IN ('running', 'completed', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    safe_failure_code TEXT,
    owns_research_task_state INTEGER NOT NULL DEFAULT 0
        CHECK (owns_research_task_state IN (0, 1)),
    UNIQUE (workflow_run_id, task_id, role, schema_version),
    CHECK (
        (state = 'running' AND completed_at IS NULL
            AND safe_failure_code IS NULL)
        OR
        (state = 'completed' AND completed_at IS NOT NULL
            AND safe_failure_code IS NULL)
        OR
        (state = 'failed' AND completed_at IS NOT NULL
            AND safe_failure_code IS NOT NULL)
    ),
    FOREIGN KEY (task_id) REFERENCES research_tasks(task_id) ON DELETE RESTRICT
);

INSERT INTO workflow_task_instances (
    task_instance_id, workflow_run_id, task_id, role, schema_version,
    input_hash, prompt_hash, state, started_at, completed_at,
    safe_failure_code, owns_research_task_state
)
SELECT
    'task_instance_' || substr(execution.attempt_id, 15),
    execution.workflow_run_id,
    execution.task_id,
    execution.role,
    execution.schema_version,
    execution.input_hash,
    execution.prompt_hash,
    CASE execution.state
        WHEN 'succeeded' THEN 'completed'
        ELSE execution.state
    END,
    execution.started_at,
    execution.completed_at,
    execution.safe_failure_code,
    CASE WHEN ROW_NUMBER() OVER (
        PARTITION BY execution.task_id
        ORDER BY execution.started_at, execution.attempt_id
    ) = 1 THEN 1 ELSE 0 END
FROM agent_executions AS execution;

CREATE UNIQUE INDEX idx_workflow_task_instances_owner
    ON workflow_task_instances(task_id)
    WHERE owns_research_task_state = 1;
CREATE INDEX idx_workflow_task_instances_workflow
    ON workflow_task_instances(workflow_run_id, task_id);
CREATE INDEX idx_workflow_task_instances_task_id
    ON workflow_task_instances(task_id);
CREATE INDEX idx_workflow_task_instances_state_started
    ON workflow_task_instances(state, started_at);

CREATE TRIGGER workflow_task_instances_identity_immutable
BEFORE UPDATE OF task_instance_id, workflow_run_id, task_id, role,
    schema_version, input_hash, prompt_hash, started_at,
    owns_research_task_state
ON workflow_task_instances
BEGIN
    SELECT RAISE(ABORT, 'workflow task instance identity is immutable');
END;

CREATE TRIGGER workflow_task_instances_terminal_immutable
BEFORE UPDATE ON workflow_task_instances
WHEN OLD.state IN ('completed', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'workflow task instance terminal row is immutable');
END;

CREATE TRIGGER workflow_task_instances_transition_once
BEFORE UPDATE OF state ON workflow_task_instances
WHEN OLD.state <> 'running' OR NEW.state NOT IN ('completed', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'workflow task instance transition is invalid');
END;

CREATE TRIGGER workflow_task_instances_no_delete
BEFORE DELETE ON workflow_task_instances
BEGIN
    SELECT RAISE(ABORT, 'workflow task instances are immutable');
END;

DROP TRIGGER agent_executions_identity_immutable;
DROP TRIGGER agent_executions_terminal_immutable;

ALTER TABLE agent_executions ADD COLUMN task_instance_id TEXT
    REFERENCES workflow_task_instances(task_instance_id) ON DELETE RESTRICT;
ALTER TABLE agent_executions ADD COLUMN reservation_state TEXT CHECK (
    reservation_state IS NULL OR reservation_state IN (
        'reserved', 'usage_unknown', 'reconciled', 'released',
        'consumed', 'expired'
    )
);
ALTER TABLE agent_executions ADD COLUMN reserved_cost_usd TEXT;

UPDATE agent_executions
SET task_instance_id = 'task_instance_' || substr(attempt_id, 15);

UPDATE agent_executions AS execution
SET
    reservation_state = CASE
        WHEN EXISTS (
            SELECT 1
            FROM provider_attempts AS attempt
            WHERE attempt.attempt_id = execution.attempt_id
                AND attempt.reservation_state = 'usage_unknown'
        ) THEN 'usage_unknown'
        ELSE (
            SELECT attempt.reservation_state
            FROM provider_attempts AS attempt
            WHERE attempt.attempt_id = execution.attempt_id
                AND attempt.reservation_state IS NOT NULL
            ORDER BY attempt.ordinal DESC
            LIMIT 1
        )
    END,
    reserved_cost_usd = (
        SELECT decimal_sum_exact(attempt.reserved_cost_usd)
        FROM provider_attempts AS attempt
        WHERE attempt.attempt_id = execution.attempt_id
            AND attempt.reserved_cost_usd IS NOT NULL
    )
WHERE EXISTS (
    SELECT 1
    FROM provider_attempts AS attempt
    WHERE attempt.attempt_id = execution.attempt_id
        AND attempt.reservation_state IS NOT NULL
);

CREATE INDEX idx_agent_executions_task_instance_id
    ON agent_executions(task_instance_id);

CREATE TRIGGER agent_executions_identity_immutable
BEFORE UPDATE OF attempt_id, workflow_run_id, task_id, task_instance_id,
    role, schema_version, input_hash, prompt_hash, started_at
ON agent_executions
BEGIN
    SELECT RAISE(ABORT, 'agent execution identity is immutable');
END;

CREATE TRIGGER agent_executions_terminal_immutable
BEFORE UPDATE ON agent_executions
WHEN OLD.state IN ('succeeded', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'agent execution terminal row is immutable');
END;

CREATE TRIGGER agent_executions_require_task_instance
BEFORE INSERT ON agent_executions
WHEN NEW.task_instance_id IS NULL OR NOT EXISTS (
    SELECT 1 FROM workflow_task_instances AS instance
    WHERE instance.task_instance_id = NEW.task_instance_id
        AND instance.workflow_run_id = NEW.workflow_run_id
        AND instance.task_id = NEW.task_id
        AND instance.role = NEW.role
        AND instance.schema_version = NEW.schema_version
        AND instance.input_hash = NEW.input_hash
        AND instance.prompt_hash = NEW.prompt_hash
        AND instance.state = 'running'
)
BEGIN
    SELECT RAISE(ABORT, 'agent execution requires an exact running task instance');
END;
