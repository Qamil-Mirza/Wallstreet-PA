-- Admit only publication intents created by the currently leased publish task,
-- and require an immutable completed review authorization at every terminal edge.

CREATE TABLE publication_effect_key_function_probe (
    effect_key TEXT NOT NULL CHECK (
        effect_key = publication_effect_key('wf_probe', 'wft_probe')
    )
);
INSERT INTO publication_effect_key_function_probe (effect_key) VALUES (
    '2b084334b569d7a43951ad3465a633c9dcfeac215d457cc786d41efaa071e9ba'
);
DROP TABLE publication_effect_key_function_probe;

CREATE TRIGGER publication_effects_insert_guard
BEFORE INSERT ON publication_effects
WHEN COALESCE(
    NEW.state = 'claimed'
    AND NEW.effect_key = publication_effect_key(NEW.workflow_id, NEW.task_id)
    AND NEW.publication_effect_id = 'pub_' || substr(NEW.effect_key, 1, 40)
    AND length(NEW.claim_token) = 64
    AND NEW.claim_token NOT GLOB '*[^0-9a-f]*'
    AND NEW.claim_expires_at IS NOT NULL
    AND NEW.created_at = NEW.updated_at
    AND NEW.report_id IS NULL
    AND NEW.receipt_hash IS NULL
    AND NEW.outcome_json IS NULL
    AND NEW.result_hash IS NULL
    AND EXISTS (
        SELECT 1
        FROM workflow_tasks AS publish
        WHERE publish.workflow_id = NEW.workflow_id
            AND publish.task_id = NEW.task_id
            AND publish.stage = 'publish'
            AND publish.state = 'running'
            AND publish.lease_token = NEW.claim_token
            AND publish.lease_expires_at = NEW.claim_expires_at
    ),
    0
) = 0
BEGIN
    SELECT RAISE(ABORT, 'publication effect insert is invalid');
END;

DROP TRIGGER publication_effects_transition_guard;

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
        AND EXISTS (
            SELECT 1
            FROM workflow_tasks AS review
            WHERE review.workflow_id = NEW.workflow_id
                AND review.stage = 'review'
                AND review.state = 'completed'
                AND review.outcome_json IS NOT NULL
                AND json_valid(review.outcome_json) = 1
                AND json_type(
                    review.outcome_json, '$.quality_gate'
                ) = 'object'
                AND json_extract(
                    review.outcome_json, '$.quality_gate.review_verdict'
                ) = 'pass'
                AND json_extract(
                    review.outcome_json, '$.quality_gate.publication_verdict'
                ) IN ('final', 'partial')
                AND json_type(
                    review.outcome_json, '$.quality_gate.allowed_claim_ids'
                ) = 'array'
        )
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
                    AND review.outcome_json IS NOT NULL
                    AND json_valid(review.outcome_json) = 1
                    AND json_type(
                        review.outcome_json, '$.quality_gate'
                    ) = 'object'
                    AND json_extract(
                        review.outcome_json,
                        '$.quality_gate.review_verdict'
                    ) = 'pass'
                    AND json_extract(
                        review.outcome_json,
                        '$.quality_gate.publication_verdict'
                    ) IN ('final', 'partial')
                    AND json_type(
                        review.outcome_json,
                        '$.quality_gate.allowed_claim_ids'
                    ) = 'array'
                    AND allowed.type = 'text'
                    AND allowed.value = published.value
            )
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'publication effect transition is invalid');
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
        AND EXISTS (
            SELECT 1 FROM workflow_tasks AS review
            WHERE review.workflow_id = OLD.workflow_id
                AND review.stage = 'review'
                AND review.state = 'completed'
                AND review.outcome_json IS NOT NULL
                AND json_valid(review.outcome_json) = 1
                AND json_extract(
                    review.outcome_json, '$.quality_gate.review_verdict'
                ) = 'pass'
                AND json_extract(
                    review.outcome_json, '$.quality_gate.publication_verdict'
                ) IN ('final', 'partial')
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
        AND EXISTS (
            SELECT 1 FROM workflow_tasks AS review
            WHERE review.workflow_id = OLD.workflow_id
                AND review.stage = 'review'
                AND review.state = 'completed'
                AND review.outcome_json IS NOT NULL
                AND json_valid(review.outcome_json) = 1
                AND json_extract(
                    review.outcome_json, '$.quality_gate.review_verdict'
                ) = 'pass'
                AND json_extract(
                    review.outcome_json, '$.quality_gate.publication_verdict'
                ) IN ('final', 'partial')
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
        AND EXISTS (
            SELECT 1 FROM workflow_tasks AS review
            WHERE review.workflow_id = OLD.workflow_id
                AND review.stage = 'review'
                AND review.state = 'completed'
                AND review.outcome_json IS NOT NULL
                AND json_valid(review.outcome_json) = 1
                AND json_extract(
                    review.outcome_json, '$.quality_gate.review_verdict'
                ) = 'pass'
                AND json_extract(
                    review.outcome_json, '$.quality_gate.publication_verdict'
                ) IN ('final', 'partial')
        )
    )
BEGIN
    SELECT RAISE(ABORT, 'workflow task terminal reason is immutable');
END;
