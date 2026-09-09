-- Recheck the authoritative task/effect lease at the instant a claimed
-- publication effect is confirmed. Explicit receipt reconciliation remains
-- available after a claim expires because it follows outcome_unknown.

CREATE TABLE research_utc_now_function_probe (
    valid INTEGER NOT NULL CHECK (
        length(research_utc_now()) = 27
        AND research_utc_now() GLOB
            '????-??-??T??:??:??.??????Z'
    )
);
INSERT INTO research_utc_now_function_probe (valid) VALUES (1);
DROP TABLE research_utc_now_function_probe;

DROP TRIGGER publication_effects_transition_guard;

CREATE TRIGGER publication_effects_transition_guard
BEFORE UPDATE OF state ON publication_effects
WHEN NOT (
    (
        OLD.state = 'claimed'
        AND NEW.state = 'outcome_unknown'
        AND NEW.claim_token IS NULL
        AND NEW.claim_expires_at IS NULL
        AND NEW.outcome_json IS NULL
        AND NEW.result_hash IS NULL
    )
    OR (
        (
            (
                OLD.state = 'claimed'
                AND NEW.state = 'confirmed'
                AND OLD.claim_token IS NOT NULL
                AND length(OLD.claim_token) = 64
                AND OLD.claim_token NOT GLOB '*[^0-9a-f]*'
                AND OLD.claim_expires_at IS NOT NULL
                AND OLD.claim_expires_at > research_utc_now()
                AND NEW.claim_token IS NULL
                AND NEW.claim_expires_at IS NULL
                AND EXISTS (
                    SELECT 1
                    FROM workflow_tasks AS claimed_publish
                    WHERE claimed_publish.workflow_id = OLD.workflow_id
                        AND claimed_publish.task_id = OLD.task_id
                        AND claimed_publish.stage = 'publish'
                        AND claimed_publish.state = 'running'
                        AND claimed_publish.lease_token = OLD.claim_token
                        AND claimed_publish.lease_expires_at = OLD.claim_expires_at
                        AND claimed_publish.lease_expires_at > research_utc_now()
                )
            )
            OR (
                OLD.state = 'outcome_unknown'
                AND NEW.state = 'reconciled'
                AND OLD.claim_token IS NULL
                AND OLD.claim_expires_at IS NULL
                AND NEW.claim_token IS NULL
                AND NEW.claim_expires_at IS NULL
            )
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
            FROM workflow_task_dependencies AS dependency
            JOIN workflow_tasks AS review
                ON review.workflow_id = dependency.workflow_id
                AND review.task_id = dependency.dependency_task_id
            JOIN workflow_tasks AS publish
                ON publish.workflow_id = dependency.workflow_id
                AND publish.task_id = dependency.task_id
            WHERE dependency.workflow_id = NEW.workflow_id
                AND dependency.task_id = NEW.task_id
                AND publish.stage = 'publish'
                AND review.stage = 'review'
                AND review.ordinal + 1 = publish.ordinal
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
                FROM workflow_task_dependencies AS dependency
                JOIN workflow_tasks AS review
                    ON review.workflow_id = dependency.workflow_id
                    AND review.task_id = dependency.dependency_task_id
                JOIN workflow_tasks AS publish
                    ON publish.workflow_id = dependency.workflow_id
                    AND publish.task_id = dependency.task_id
                JOIN json_each(
                    review.outcome_json,
                    '$.quality_gate.allowed_claim_ids'
                ) AS allowed
                WHERE dependency.workflow_id = NEW.workflow_id
                    AND dependency.task_id = NEW.task_id
                    AND publish.stage = 'publish'
                    AND review.stage = 'review'
                    AND review.ordinal + 1 = publish.ordinal
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
