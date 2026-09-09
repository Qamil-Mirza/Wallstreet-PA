-- Workflow definitions and read models require exactly one task per stage.
-- Index creation intentionally fails atomically if a legacy database is corrupt;
-- migration must never choose a duplicate row or destroy durable history.

CREATE UNIQUE INDEX idx_workflow_tasks_workflow_stage
    ON workflow_tasks(workflow_id, stage);
