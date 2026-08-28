-- Persist deterministic entity aliases and resolution provenance, and make
-- supporting and contradicting relationship assertions independently durable.

PRAGMA defer_foreign_keys = ON;

DROP INDEX idx_relationships_source_entity_id;
DROP INDEX idx_relationships_target_entity_id;

ALTER TABLE relationships RENAME TO relationships_v3_old;

CREATE TABLE relationships (
    relationship_id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (
        kind IN ('supplier', 'customer', 'competitor', 'complement', 'substitute')
    ),
    as_of TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK (typeof(confidence) = 'text'),
    stance TEXT NOT NULL CHECK (stance IN ('supports', 'contradicts')),
    provenance TEXT NOT NULL,
    primary_passage_id TEXT,
    primary_claim_id TEXT,
    lineage_sealed INTEGER NOT NULL DEFAULT 0 CHECK (lineage_sealed IN (0, 1)),
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    CHECK (source_entity_id <> target_entity_id),
    CHECK (
        lineage_sealed = 0
        OR primary_passage_id IS NOT NULL
        OR primary_claim_id IS NOT NULL
    ),
    FOREIGN KEY (source_entity_id) REFERENCES entities(entity_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (target_entity_id) REFERENCES entities(entity_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (primary_passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (primary_claim_id) REFERENCES claims(claim_id) ON DELETE RESTRICT
);

INSERT INTO relationships (
    relationship_id, source_entity_id, target_entity_id, kind, as_of,
    confidence, stance, provenance, primary_passage_id, primary_claim_id,
    lineage_sealed, metadata_json, created_at
)
SELECT
    relationship_id, source_entity_id, target_entity_id, kind, as_of,
    COALESCE(confidence, '0'), 'supports', 'legacy', NULL, NULL, 0,
    metadata_json, created_at
FROM relationships_v3_old;

DROP TABLE relationships_v3_old;

CREATE INDEX idx_relationships_source_entity_id
    ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target_entity_id
    ON relationships(target_entity_id);
CREATE INDEX idx_relationships_kind_as_of
    ON relationships(kind, as_of);

CREATE TABLE entity_aliases (
    alias_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    value TEXT NOT NULL,
    alias_type TEXT NOT NULL CHECK (alias_type IN ('name', 'symbol_market')),
    market TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (entity_id, alias_type, value, market),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE INDEX idx_entity_aliases_lookup
    ON entity_aliases(alias_type, value, market);
CREATE INDEX idx_entity_aliases_entity_id
    ON entity_aliases(entity_id);

CREATE TABLE entity_resolution_provenance (
    provenance_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    method TEXT NOT NULL,
    source TEXT NOT NULL,
    matched_identifier TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (entity_id, method, source, matched_identifier),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE INDEX idx_entity_resolution_provenance_entity_id
    ON entity_resolution_provenance(entity_id);

CREATE TABLE relationship_evidence (
    relationship_id TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    PRIMARY KEY (relationship_id, passage_id),
    FOREIGN KEY (relationship_id) REFERENCES relationships(relationship_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (passage_id) REFERENCES document_passages(passage_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_relationship_evidence_passage_id
    ON relationship_evidence(passage_id);

CREATE TABLE relationship_claim_evidence (
    relationship_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    PRIMARY KEY (relationship_id, claim_id),
    FOREIGN KEY (relationship_id) REFERENCES relationships(relationship_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE RESTRICT
);

CREATE INDEX idx_relationship_claim_evidence_claim_id
    ON relationship_claim_evidence(claim_id);

CREATE TRIGGER entities_immutable
BEFORE UPDATE ON entities
BEGIN
    SELECT RAISE(ABORT, 'entities are immutable');
END;

CREATE TRIGGER entities_no_delete
BEFORE DELETE ON entities
BEGIN
    SELECT RAISE(ABORT, 'entities cannot be deleted');
END;

CREATE TRIGGER entity_aliases_immutable_update
BEFORE UPDATE ON entity_aliases
BEGIN
    SELECT RAISE(ABORT, 'entity aliases are immutable');
END;

CREATE TRIGGER entity_aliases_immutable_delete
BEFORE DELETE ON entity_aliases
BEGIN
    SELECT RAISE(ABORT, 'entity aliases cannot be deleted');
END;

CREATE TRIGGER entity_resolution_provenance_immutable_update
BEFORE UPDATE ON entity_resolution_provenance
BEGIN
    SELECT RAISE(ABORT, 'entity resolution provenance is immutable');
END;

CREATE TRIGGER entity_resolution_provenance_immutable_delete
BEFORE DELETE ON entity_resolution_provenance
BEGIN
    SELECT RAISE(ABORT, 'entity resolution provenance cannot be deleted');
END;

CREATE TRIGGER relationships_immutable_update
BEFORE UPDATE OF relationship_id, source_entity_id, target_entity_id, kind,
    as_of, confidence, stance, provenance, primary_passage_id,
    primary_claim_id, metadata_json, created_at
ON relationships
BEGIN
    SELECT RAISE(ABORT, 'relationships are immutable');
END;

CREATE TRIGGER relationships_immutable_delete
BEFORE DELETE ON relationships
BEGIN
    SELECT RAISE(ABORT, 'relationships cannot be deleted');
END;

CREATE TRIGGER relationships_must_begin_unsealed
BEFORE INSERT ON relationships
WHEN NEW.lineage_sealed <> 0
BEGIN
    SELECT RAISE(ABORT, 'relationship lineage must begin unsealed');
END;

CREATE TRIGGER relationships_lineage_seal_one_way
BEFORE UPDATE OF lineage_sealed ON relationships
WHEN NOT (OLD.lineage_sealed = 0 AND NEW.lineage_sealed = 1)
BEGIN
    SELECT RAISE(ABORT, 'relationship lineage seal is immutable');
END;

CREATE TRIGGER relationships_lineage_seal_requires_evidence
BEFORE UPDATE OF lineage_sealed ON relationships
WHEN NEW.lineage_sealed = 1 AND NOT (
    (
        NEW.primary_passage_id IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM relationship_evidence AS evidence
            WHERE evidence.relationship_id = NEW.relationship_id
                AND evidence.passage_id = NEW.primary_passage_id
        )
    )
    OR
    (
        NEW.primary_claim_id IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM relationship_claim_evidence AS evidence
            JOIN claims AS claim ON claim.claim_id = evidence.claim_id
            WHERE evidence.relationship_id = NEW.relationship_id
                AND evidence.claim_id = NEW.primary_claim_id
                AND claim.lineage_sealed = 1
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'relationship lineage seal requires matching evidence');
END;

CREATE TRIGGER relationship_evidence_no_insert_after_seal
BEFORE INSERT ON relationship_evidence
WHEN (
    SELECT lineage_sealed FROM relationships
    WHERE relationship_id = NEW.relationship_id
) <> 0
BEGIN
    SELECT RAISE(ABORT, 'relationship evidence is immutable');
END;

CREATE TRIGGER relationship_evidence_immutable_update
BEFORE UPDATE ON relationship_evidence
BEGIN
    SELECT RAISE(ABORT, 'relationship evidence is immutable');
END;

CREATE TRIGGER relationship_evidence_immutable_delete
BEFORE DELETE ON relationship_evidence
BEGIN
    SELECT RAISE(ABORT, 'relationship evidence cannot be deleted');
END;

CREATE TRIGGER relationship_claim_evidence_immutable_update
BEFORE UPDATE ON relationship_claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'relationship claim evidence is immutable');
END;

CREATE TRIGGER relationship_claim_evidence_no_insert_after_seal
BEFORE INSERT ON relationship_claim_evidence
WHEN (
    SELECT lineage_sealed FROM relationships
    WHERE relationship_id = NEW.relationship_id
) <> 0
BEGIN
    SELECT RAISE(ABORT, 'relationship claim evidence is immutable');
END;

CREATE TRIGGER relationship_claim_evidence_requires_sealed_claim
BEFORE INSERT ON relationship_claim_evidence
WHEN NOT EXISTS (
    SELECT 1 FROM claims
    WHERE claim_id = NEW.claim_id AND lineage_sealed = 1
)
BEGIN
    SELECT RAISE(ABORT, 'relationship claim evidence must be sealed');
END;

CREATE TRIGGER relationship_claim_evidence_immutable_delete
BEFORE DELETE ON relationship_claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'relationship claim evidence cannot be deleted');
END;
