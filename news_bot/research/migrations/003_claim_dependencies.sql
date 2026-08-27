-- Make one concrete evidence anchor part of every claim row so the database,
-- not only repository code, rejects uncited material claims. Existing v2
-- claims already have passage evidence and retain their complete lineage.

PRAGMA defer_foreign_keys = ON;

DROP TRIGGER claims_immutable_fields;
DROP TRIGGER claims_no_delete;
DROP TRIGGER claim_evidence_no_update;
DROP TRIGGER claim_evidence_no_delete;
DROP INDEX idx_claims_status;
DROP INDEX idx_claims_entity_as_of;
DROP INDEX idx_claim_evidence_passage_id;

ALTER TABLE claim_evidence RENAME TO claim_evidence_v2_old;
ALTER TABLE claims RENAME TO claims_v2_old;

CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    entity_id TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'guidance', 'estimate', 'inference')),
    text TEXT NOT NULL,
    as_of TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK (typeof(confidence) = 'text'),
    status TEXT NOT NULL CHECK (status IN ('active', 'contradicted', 'superseded')),
    primary_passage_id TEXT,
    primary_supporting_claim_id TEXT,
    created_at TEXT NOT NULL,
    CHECK (
        (kind IN ('fact', 'guidance', 'estimate') AND primary_passage_id IS NOT NULL)
        OR
        (kind = 'inference' AND (
            primary_passage_id IS NOT NULL OR primary_supporting_claim_id IS NOT NULL
        ))
    ),
    CHECK (
        primary_supporting_claim_id IS NULL
        OR primary_supporting_claim_id <> claim_id
    ),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT,
    FOREIGN KEY (primary_passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (primary_supporting_claim_id) REFERENCES claims(claim_id)
        ON DELETE RESTRICT
);

CREATE TABLE claim_evidence (
    claim_id TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    stance TEXT NOT NULL CHECK (stance IN ('supports', 'contradicts')),
    PRIMARY KEY (claim_id, passage_id),
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT
);

INSERT INTO claims (
    claim_id, entity_id, kind, text, as_of, confidence, status,
    primary_passage_id, primary_supporting_claim_id, created_at
)
SELECT
    old.claim_id,
    old.entity_id,
    old.kind,
    old.text,
    old.as_of,
    old.confidence,
    old.status,
    (
        SELECT evidence.passage_id
        FROM claim_evidence_v2_old AS evidence
        WHERE evidence.claim_id = old.claim_id AND evidence.stance = 'supports'
        ORDER BY evidence.passage_id
        LIMIT 1
    ),
    NULL,
    old.created_at
FROM claims_v2_old AS old;

INSERT INTO claim_evidence (claim_id, passage_id, stance)
SELECT claim_id, passage_id, stance FROM claim_evidence_v2_old;

DROP TABLE claim_evidence_v2_old;
DROP TABLE claims_v2_old;

CREATE INDEX idx_claims_status ON claims(status);
CREATE INDEX idx_claims_entity_as_of ON claims(entity_id, as_of);
CREATE INDEX idx_claim_evidence_passage_id ON claim_evidence(passage_id);

CREATE TRIGGER claims_immutable_fields
BEFORE UPDATE OF claim_id, entity_id, kind, text, as_of, confidence,
    primary_passage_id, primary_supporting_claim_id, created_at
ON claims
BEGIN
    SELECT RAISE(ABORT, 'claim immutable fields cannot be updated');
END;

CREATE TRIGGER claims_no_delete
BEFORE DELETE ON claims
BEGIN
    SELECT RAISE(ABORT, 'claims cannot be deleted');
END;

CREATE TRIGGER claim_evidence_no_update
BEFORE UPDATE ON claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim evidence links are immutable');
END;

CREATE TRIGGER claim_evidence_no_delete
BEFORE DELETE ON claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim evidence links are immutable');
END;

CREATE TABLE claim_dependencies (
    claim_id TEXT NOT NULL,
    supporting_claim_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, supporting_claim_id),
    CHECK (claim_id <> supporting_claim_id),
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (supporting_claim_id) REFERENCES claims(claim_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_claim_dependencies_supporting_claim_id
    ON claim_dependencies(supporting_claim_id);

CREATE TRIGGER claim_dependencies_no_update
BEFORE UPDATE ON claim_dependencies
BEGIN
    SELECT RAISE(ABORT, 'claim dependency links are immutable');
END;

CREATE TRIGGER claim_dependencies_no_delete
BEFORE DELETE ON claim_dependencies
BEGIN
    SELECT RAISE(ABORT, 'claim dependency links are immutable');
END;

CREATE TRIGGER source_documents_no_update
BEFORE UPDATE ON source_documents
BEGIN
    SELECT RAISE(ABORT, 'source documents are immutable');
END;

CREATE TRIGGER source_documents_no_delete
BEFORE DELETE ON source_documents
BEGIN
    SELECT RAISE(ABORT, 'source documents are immutable');
END;

CREATE TRIGGER document_passages_no_update
BEFORE UPDATE ON document_passages
BEGIN
    SELECT RAISE(ABORT, 'document passages are immutable');
END;

CREATE TRIGGER document_passages_no_delete
BEFORE DELETE ON document_passages
BEGIN
    SELECT RAISE(ABORT, 'document passages are immutable');
END;
