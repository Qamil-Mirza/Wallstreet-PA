-- Bind publish task JSON, report reference, and result hash to the exact
-- confirmed/reconciled receipt. The probe makes missing/wrong hash functions
-- fail the migration atomically rather than creating unusable triggers.

CREATE TABLE publication_hash_function_probe (
    digest TEXT NOT NULL CHECK (digest = sha256_hex(''))
);
INSERT INTO publication_hash_function_probe (digest) VALUES (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
);
DROP TABLE publication_hash_function_probe;

DROP TRIGGER workflow_tasks_transition_guard;
DROP TRIGGER workflow_tasks_terminal_output_guard;
DROP TRIGGER workflow_tasks_terminal_reason_immutable;

CREATE TRIGGER workflow_tasks_transition_guard
BEFORE UPDATE OF state ON workflow_tasks
WHEN OLD.state <> NEW.state AND NOT (
    (OLD.state = 'pending' AND NEW.state IN ('running', 'deferred'))
    OR (OLD.state = 'running' AND NEW.state = 'failed')
    OR (OLD.state = 'running' AND NEW.state = 'deferred')
    OR (
        OLD.state = 'running' AND NEW.state = 'completed'
        AND OLD.stage <> 'publish'
    )
    OR (OLD.state IN ('failed', 'deferred') AND NEW.state = 'pending')
    OR (
        OLD.stage = 'publish' AND NEW.state = 'completed'
        AND OLD.state IN ('pending', 'running', 'failed', 'deferred')
        AND NEW.outcome_json IS NOT NULL
        AND json_valid(NEW.outcome_json) = 1
        AND NEW.result_hash IS NOT NULL
        AND length(NEW.result_hash) = 64
        AND NEW.result_hash NOT GLOB '*[^0-9a-f]*'
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.report_id = json_extract(
                    NEW.outcome_json, '$.report_id'
                )
                AND effect.receipt_hash = json_extract(
                    NEW.outcome_json, '$.publication_receipt_hash'
                )
                AND length(effect.receipt_hash) = 64
                AND effect.receipt_hash NOT GLOB '*[^0-9a-f]*'
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'workflow task transition is invalid');
END;

CREATE TRIGGER workflow_tasks_terminal_output_guard
BEFORE UPDATE OF result_ref, result_hash, outcome_json ON workflow_tasks
WHEN NOT (
    (OLD.state = 'running' AND NEW.state = 'deferred')
    OR (
        OLD.state = 'running' AND NEW.state = 'completed'
        AND OLD.stage <> 'publish'
    )
    OR (
        OLD.stage = 'publish' AND NEW.state = 'completed'
        AND OLD.state IN ('pending', 'running', 'failed', 'deferred')
        AND NEW.outcome_json IS NOT NULL
        AND json_valid(NEW.outcome_json) = 1
        AND NEW.result_hash IS NOT NULL
        AND length(NEW.result_hash) = 64
        AND NEW.result_hash NOT GLOB '*[^0-9a-f]*'
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.report_id = json_extract(
                    NEW.outcome_json, '$.report_id'
                )
                AND effect.receipt_hash = json_extract(
                    NEW.outcome_json, '$.publication_receipt_hash'
                )
                AND length(effect.receipt_hash) = 64
                AND effect.receipt_hash NOT GLOB '*[^0-9a-f]*'
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
        AND NEW.outcome_json IS NOT NULL
        AND json_valid(NEW.outcome_json) = 1
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.report_id = json_extract(
                    NEW.outcome_json, '$.report_id'
                )
                AND effect.receipt_hash = json_extract(
                    NEW.outcome_json, '$.publication_receipt_hash'
                )
                AND length(effect.receipt_hash) = 64
                AND effect.receipt_hash NOT GLOB '*[^0-9a-f]*'
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'workflow task terminal reason is immutable');
END;
