-- Require the exact safe StageOutcome serialization for every publish task.
-- The probe keeps upgrades atomic when either deterministic scalar is absent.

CREATE TABLE canonical_publication_function_probe (
    outcome_valid INTEGER NOT NULL CHECK (
        outcome_valid = is_canonical_publication_outcome(
            '{"defer_reason":null,"material_event":null,"new_agent_runs":0,"omissions":[],"originating_role":null,"publication_receipt_hash":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","published_claim_ids":[],"quality_gate":null,"report_id":"report-probe","result_hash":null,"result_ref":"report-probe","reviewer_verdict":null}',
            'report-probe',
            'report-probe',
            'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        )
    ),
    digest TEXT NOT NULL CHECK (
        digest = sha256_hex(
            '{"defer_reason":null,"material_event":null,"new_agent_runs":0,"omissions":[],"originating_role":null,"publication_receipt_hash":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","published_claim_ids":[],"quality_gate":null,"report_id":"report-probe","result_hash":null,"result_ref":"report-probe","reviewer_verdict":null}'
        )
    )
);
INSERT INTO canonical_publication_function_probe (outcome_valid, digest)
VALUES (
    1,
    '2ddcb7bf2e6f71708b3fce3e074dc2364b8695855f00578057cce48e531ddb7c'
);
DROP TABLE canonical_publication_function_probe;

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
        AND NEW.result_hash IS NOT NULL
        AND length(NEW.result_hash) = 64
        AND NEW.result_hash NOT GLOB '*[^0-9a-f]*'
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND is_canonical_publication_outcome(
                    NEW.outcome_json,
                    NEW.result_ref,
                    effect.report_id,
                    effect.receipt_hash
                ) = 1
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
        AND NEW.result_hash IS NOT NULL
        AND length(NEW.result_hash) = 64
        AND NEW.result_hash NOT GLOB '*[^0-9a-f]*'
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND is_canonical_publication_outcome(
                    NEW.outcome_json,
                    NEW.result_ref,
                    effect.report_id,
                    effect.receipt_hash
                ) = 1
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
        AND NEW.result_hash IS NOT NULL
        AND length(NEW.result_hash) = 64
        AND NEW.result_hash NOT GLOB '*[^0-9a-f]*'
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND is_canonical_publication_outcome(
                    NEW.outcome_json,
                    NEW.result_ref,
                    effect.report_id,
                    effect.receipt_hash
                ) = 1
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'workflow task terminal reason is immutable');
END;
