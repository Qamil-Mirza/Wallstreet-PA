-- Make the confirmed/reconciled publication effect the authoritative source
-- for the exact safe StageOutcome bytes used to complete and recover its task.

CREATE TABLE publication_effect_outcome_function_probe (
    outcome_valid INTEGER NOT NULL CHECK (
        outcome_valid = is_canonical_publication_effect_outcome(
            '{"defer_reason":null,"material_event":null,"new_agent_runs":2,"omissions":["missing_claim_lineage"],"originating_role":null,"publication_receipt_hash":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","published_claim_ids":["claim-1"],"quality_gate":null,"report_id":"report-probe","result_hash":null,"result_ref":"report-probe","reviewer_verdict":null}',
            'report-probe',
            'report-probe',
            'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        )
    ),
    digest TEXT NOT NULL CHECK (
        digest = sha256_hex(
            '{"defer_reason":null,"material_event":null,"new_agent_runs":2,"omissions":["missing_claim_lineage"],"originating_role":null,"publication_receipt_hash":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","published_claim_ids":["claim-1"],"quality_gate":null,"report_id":"report-probe","result_hash":null,"result_ref":"report-probe","reviewer_verdict":null}'
        )
    )
);
INSERT INTO publication_effect_outcome_function_probe (outcome_valid, digest)
VALUES (
    1,
    '1fadbd9f8530a2fc1c489883b52ad6cf8072a93f1f2ec6d5c378c47c4be600e8'
);
DROP TABLE publication_effect_outcome_function_probe;

DROP TRIGGER publication_effects_transition_guard;
DROP TRIGGER publication_effects_terminal_immutable;

ALTER TABLE publication_effects ADD COLUMN outcome_json TEXT;
ALTER TABLE publication_effects ADD COLUMN result_hash TEXT CHECK (
    result_hash IS NULL OR (
        length(result_hash) = 64
        AND result_hash NOT GLOB '*[^0-9a-f]*'
    )
);

UPDATE publication_effects
SET outcome_json = json_object(
    'defer_reason', NULL,
    'material_event', NULL,
    'new_agent_runs', 0,
    'omissions', json('[]'),
    'originating_role', NULL,
    'publication_receipt_hash', receipt_hash,
    'published_claim_ids', json('[]'),
    'quality_gate', NULL,
    'report_id', report_id,
    'result_hash', NULL,
    'result_ref', report_id,
    'reviewer_verdict', NULL
)
WHERE state IN ('confirmed', 'reconciled');

UPDATE publication_effects
SET result_hash = sha256_hex(outcome_json)
WHERE state IN ('confirmed', 'reconciled');

CREATE TRIGGER publication_effects_transition_guard
BEFORE UPDATE OF state ON publication_effects
WHEN NOT (
    (
        OLD.state = 'claimed'
        AND NEW.state = 'outcome_unknown'
        AND NEW.outcome_json IS NULL
        AND NEW.result_hash IS NULL
    )
    OR (
        (
            (OLD.state = 'claimed' AND NEW.state = 'confirmed')
            OR (OLD.state = 'outcome_unknown' AND NEW.state = 'reconciled')
        )
        AND NEW.outcome_json IS NOT NULL
        AND NEW.result_hash IS NOT NULL
        AND NEW.result_hash = sha256_hex(NEW.outcome_json)
        AND is_canonical_publication_effect_outcome(
            NEW.outcome_json,
            NEW.report_id,
            NEW.report_id,
            NEW.receipt_hash
        ) = 1
        AND NOT EXISTS (
            SELECT 1
            FROM json_each(
                NEW.outcome_json, '$.published_claim_ids'
            ) AS published
            WHERE NOT EXISTS (
                SELECT 1
                FROM workflow_tasks AS review
                JOIN json_each(
                    review.outcome_json,
                    '$.quality_gate.allowed_claim_ids'
                ) AS allowed
                WHERE review.workflow_id = NEW.workflow_id
                    AND review.stage = 'review'
                    AND review.state = 'completed'
                    AND json_valid(review.outcome_json) = 1
                    AND json_extract(
                        review.outcome_json,
                        '$.quality_gate.review_verdict'
                    ) = 'pass'
                    AND json_extract(
                        review.outcome_json,
                        '$.quality_gate.publication_verdict'
                    ) IN ('final', 'partial')
                    AND allowed.type = 'text'
                    AND allowed.value = published.value
            )
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'publication effect transition is invalid');
END;

CREATE TRIGGER publication_effects_outcome_transition_guard
BEFORE UPDATE OF report_id, receipt_hash, outcome_json, result_hash
ON publication_effects
WHEN NOT (
    (OLD.state = 'claimed' AND NEW.state = 'confirmed')
    OR (OLD.state = 'outcome_unknown' AND NEW.state = 'reconciled')
)
BEGIN
    SELECT RAISE(ABORT, 'publication effect outcome transition is invalid');
END;

CREATE TRIGGER publication_effects_terminal_immutable
BEFORE UPDATE ON publication_effects
WHEN OLD.state IN ('confirmed', 'reconciled')
BEGIN
    SELECT RAISE(ABORT, 'publication effect receipt is immutable');
END;

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
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.outcome_json = NEW.outcome_json
                AND effect.result_hash = NEW.result_hash
                AND NEW.result_hash = sha256_hex(NEW.outcome_json)
                AND is_canonical_publication_effect_outcome(
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
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.outcome_json = NEW.outcome_json
                AND effect.result_hash = NEW.result_hash
                AND NEW.result_hash = sha256_hex(NEW.outcome_json)
                AND is_canonical_publication_effect_outcome(
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
        AND EXISTS (
            SELECT 1 FROM publication_effects AS effect
            WHERE effect.workflow_id = OLD.workflow_id
                AND effect.task_id = OLD.task_id
                AND effect.state IN ('confirmed', 'reconciled')
                AND effect.report_id = NEW.result_ref
                AND effect.outcome_json = NEW.outcome_json
                AND effect.result_hash = NEW.result_hash
                AND NEW.result_hash = sha256_hex(NEW.outcome_json)
                AND is_canonical_publication_effect_outcome(
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
