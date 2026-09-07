-- Permit a verified publication receipt to complete an uncertain publish task
-- without consuming another external-attempt budget.

DROP TRIGGER workflow_tasks_transition_guard;
DROP TRIGGER workflow_tasks_terminal_output_guard;
DROP TRIGGER workflow_tasks_terminal_reason_immutable;

CREATE TRIGGER workflow_tasks_transition_guard
BEFORE UPDATE OF state ON workflow_tasks
WHEN OLD.state <> NEW.state AND NOT (
    (OLD.state = 'pending' AND NEW.state IN ('running', 'deferred'))
    OR (OLD.state = 'running' AND NEW.state IN ('completed', 'failed', 'deferred'))
    OR (OLD.state IN ('failed', 'deferred') AND NEW.state = 'pending')
    OR (
        OLD.stage = 'publish' AND NEW.state = 'completed'
        AND OLD.state IN ('pending', 'running', 'failed', 'deferred')
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'workflow task transition is invalid');
END;

CREATE TRIGGER workflow_tasks_terminal_output_guard
BEFORE UPDATE OF result_ref, result_hash, outcome_json ON workflow_tasks
WHEN NOT (
    (OLD.state = 'running' AND NEW.state IN ('completed', 'deferred'))
    OR (
        OLD.stage = 'publish' AND NEW.state = 'completed'
        AND OLD.state IN ('pending', 'running', 'failed', 'deferred')
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'workflow task output transition is invalid');
END;

CREATE TRIGGER workflow_tasks_terminal_reason_immutable
BEFORE UPDATE OF defer_reason, completed_at ON workflow_tasks
WHEN OLD.state IN ('failed', 'deferred')
    AND NEW.state <> 'pending'
    AND NOT (
        OLD.stage = 'publish' AND NEW.state = 'completed'
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'workflow task terminal reason is immutable');
END;
