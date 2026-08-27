"""Traceability tests for canonical research evidence ingestion."""

import dataclasses
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

import news_bot.research.evidence as evidence_module
from news_bot.research.evidence import (
    ClaimLineage,
    DocumentInput,
    EvidenceCacheError,
    EvidenceError,
    EvidenceIngestor,
    EvidencePassage,
    EvidencePersistenceError,
    EvidencePolicyError,
    EvidenceValidationError,
    IngestedDocument,
    canonicalize_content,
    canonicalize_url,
)
from news_bot.research.models import ClaimKind, SourceDocument


NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def make_document(url: str, text: str) -> DocumentInput:
    return DocumentInput(
        source_type="issuer_release",
        url=url,
        publisher="Example Issuer",
        published_at=NOW,
        retrieved_at=NOW,
        content=text,
    )


def test_canonical_content_hash_deduplicates_tracking_urls(migrated_store):
    first = make_document(
        "https://issuer.test/release?utm_source=email", "Revenue grew 20%."
    )
    second = make_document("https://issuer.test/release", "Revenue grew 20%.")
    ingestor = EvidenceIngestor(migrated_store)

    assert ingestor.ingest(first).document_id == ingestor.ingest(second).document_id


def test_claim_lineage_returns_exact_supporting_passage(migrated_store):
    ingestor = EvidenceIngestor(migrated_store)
    document = ingestor.ingest(
        make_document(
            "https://issuer.test/10q", "Data center revenue was $30 billion."
        )
    )
    passage = document.passages[0]

    claim = ingestor.add_claim(
        "Data center revenue was $30 billion.",
        ClaimKind.FACT,
        [passage.passage_id],
    )

    assert ingestor.lineage(claim.claim_id)[0].text == passage.text


def test_url_canonicalization_strips_credentials_defaults_and_tracking():
    canonical = canonicalize_url(
        "HTTPS://user:secret@BÜCHER.Example.:443/release?b=2&utm_source=x&a=1#frag"
    )

    assert canonical == "https://xn--bcher-kva.example/release?a=1&b=2"


def test_url_canonicalization_preserves_business_parameters_and_blanks():
    assert canonicalize_url("https://issuer.test/data?form=10-Q&filter=") == (
        "https://issuer.test/data?filter=&form=10-Q"
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/report",
        "https://./report",
        "https://issuer.test:70000/report",
        "https://bad_host.test/report",
        "https://bad host.test/report",
        "https://-leading.test/report",
        "https://trailing-.test/report",
        "https://empty..label.test/report",
        "https://issuer.test/report%",
        "https://issuer.test/report%2",
        "https://issuer.test/report%GG",
        "https://issuer.test/report%0Ainjected",
        "https://issuer.test/report?q=%00unsafe",
    ],
)
def test_url_canonicalization_rejects_unsafe_values_without_leaking_input(url):
    with pytest.raises(EvidenceValidationError) as error:
        canonicalize_url(url)

    assert url not in str(error.value)


def test_content_canonicalization_is_utf8_nfc_and_paragraph_preserving():
    decomposed = "Cafe\N{COMBINING ACUTE ACCENT}\tgrew\r\n\r\n  Margin   expanded.  "

    canonical = canonicalize_content(decomposed)

    assert canonical == (
        "Caf\N{LATIN SMALL LETTER E WITH ACUTE} grew\n\nMargin expanded."
    )


def test_content_validation_rejects_invalid_utf8_without_leaking_content():
    unsafe = b"private portfolio \xff"

    with pytest.raises(EvidenceValidationError) as error:
        canonicalize_content(unsafe)

    assert "private portfolio" not in str(error.value)


def test_ingestion_caches_only_canonical_bytes_by_sha256(migrated_store, tmp_path):
    cache_dir = tmp_path / "cache"
    ingested = EvidenceIngestor(migrated_store, cache_dir).ingest(
        make_document("https://issuer.test/release", " Revenue\tgrew. \r\n")
    )
    expected = b"Revenue grew."

    assert ingested.raw_content_path == cache_dir / hashlib.sha256(expected).hexdigest()
    assert ingested.raw_content_path.read_bytes() == expected


def test_cache_mismatch_fails_closed_without_overwrite(migrated_store, tmp_path):
    cache_dir = tmp_path / "cache"
    ingestor = EvidenceIngestor(migrated_store, cache_dir)
    document = make_document("https://issuer.test/release", "Revenue grew.")
    target = cache_dir / hashlib.sha256(b"Revenue grew.").hexdigest()
    cache_dir.mkdir()
    target.write_bytes(b"tampered")

    with pytest.raises(EvidenceCacheError, match="integrity"):
        ingestor.ingest(document)

    assert target.read_bytes() == b"tampered"


def test_cache_filesystem_failure_uses_typed_error(migrated_store, tmp_path):
    cache_dir = tmp_path / "not-a-directory"
    cache_dir.write_text("occupied", encoding="utf-8")

    with pytest.raises(EvidenceCacheError):
        EvidenceIngestor(migrated_store, cache_dir).ingest(
            make_document("https://issuer.test/release", "Revenue grew.")
        )


def test_persistence_conflict_uses_typed_redacted_error(migrated_store):
    private_text = "Private portfolio revenue grew."
    digest = hashlib.sha256(private_text.encode("utf-8")).hexdigest()
    migrated_store.insert_source_document(
        SourceDocument(
            document_id="document_conflict",
            source_type="filing",
            canonical_url="https://issuer.test/preexisting",
            publisher="Issuer",
            published_at=NOW,
            retrieved_at=NOW,
            content_hash=digest,
            raw_content_path=None,
            extraction_status="complete",
        )
    )

    with pytest.raises(EvidencePersistenceError) as error:
        EvidenceIngestor(migrated_store).ingest(
            make_document("https://private.test/secret", private_text)
        )

    assert private_text not in str(error.value)
    assert "private.test" not in str(error.value)


def test_concurrent_ingestion_is_idempotent_and_leaves_no_temp_files(
    migrated_store, tmp_path
):
    ingestor = EvidenceIngestor(migrated_store, tmp_path / "cache")
    document = make_document("https://issuer.test/release", "Revenue grew.")

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = tuple(pool.map(lambda _: ingestor.ingest(document).document_id, range(8)))

    assert len(set(ids)) == 1
    assert {path.name for path in ingestor.cache_dir.iterdir()} == {
        hashlib.sha256(b"Revenue grew.").hexdigest()
    }


def test_cache_publication_fsyncs_directory_after_link(
    migrated_store, tmp_path, monkeypatch
):
    events = []
    real_link = evidence_module.os.link
    real_fsync = evidence_module.os.fsync

    def recording_link(source, target):
        events.append("link")
        return real_link(source, target)

    def recording_fsync(descriptor):
        events.append("fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "link", recording_link)
    monkeypatch.setattr(evidence_module.os, "fsync", recording_fsync)

    EvidenceIngestor(migrated_store, tmp_path / "cache").ingest(
        make_document("https://issuer.test/release", "Revenue grew.")
    )

    link_index = events.index("link")
    assert "fsync" in events[link_index + 1 :]


def test_cache_directory_fsync_failure_prevents_database_insert(
    migrated_store, tmp_path, monkeypatch
):
    real_fsync = evidence_module.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("private durability path")
        return real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", fail_directory_fsync)
    content = "Private revenue grew."
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    with pytest.raises(EvidenceCacheError) as error:
        EvidenceIngestor(migrated_store, tmp_path / "cache").ingest(
            make_document("https://private.test/secret", content)
        )

    assert migrated_store.get_document_by_content_hash(digest) is None
    assert not (tmp_path / "cache" / digest).exists()
    assert "Private revenue" not in str(error.value)
    assert "private.test" not in str(error.value)


def test_claim_identity_hashes_structured_lineage_roles_without_collisions():
    support_and_dependency = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("passage_a",),
        ("claim_b",),
        (),
    )
    flattened_support = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("claim_b", "passage_a"),
        (),
        (),
    )
    support_and_contradiction = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("passage_a",),
        (),
        ("claim_b",),
    )
    reordered = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("passage_a",),
        ("claim_b",),
        (),
    )
    multi_role = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("passage_z", "passage_a"),
        ("claim_z", "claim_a"),
        ("contradiction_z", "contradiction_a"),
    )
    multi_role_reordered = EvidenceIngestor._claim_id(
        ClaimKind.INFERENCE,
        "Capacity may tighten.",
        ("passage_a", "passage_z"),
        ("claim_a", "claim_z"),
        ("contradiction_a", "contradiction_z"),
    )

    assert len(
        {support_and_dependency, flattened_support, support_and_contradiction}
    ) == 3
    assert reordered == support_and_dependency
    assert multi_role_reordered == multi_role


def test_passages_have_stable_ids_and_exact_canonical_offsets(migrated_store):
    content = "First paragraph.\n\n" + ("word " * 300).strip() + "\n\nLast."
    ingestor = EvidenceIngestor(migrated_store)
    first = ingestor.ingest(make_document("https://issuer.test/a", content))
    second = ingestor.ingest(make_document("https://issuer.test/b", content))
    canonical = canonicalize_content(content)

    assert len(first.passages) > 3
    assert [item.passage_id for item in first.passages] == [
        item.passage_id for item in second.passages
    ]
    for ordinal, passage in enumerate(first.passages):
        assert passage.ordinal == ordinal
        assert canonical[passage.start_offset : passage.end_offset] == passage.text


@pytest.mark.parametrize(
    "kind", [ClaimKind.FACT, ClaimKind.GUIDANCE, ClaimKind.ESTIMATE]
)
def test_non_inference_claims_require_passage_evidence(migrated_store, kind):
    with pytest.raises(EvidencePolicyError, match="passage evidence"):
        EvidenceIngestor(migrated_store).add_claim("Material claim.", kind, [])


def test_inference_can_depend_on_an_existing_claim(migrated_store):
    ingestor = EvidenceIngestor(migrated_store)
    document = ingestor.ingest(
        make_document("https://issuer.test/release", "Orders increased 20%.")
    )
    fact = ingestor.add_claim(
        "Orders increased 20%.", ClaimKind.FACT, [document.passages[0].passage_id]
    )

    inference = ingestor.add_claim(
        "Capacity may tighten.",
        ClaimKind.INFERENCE,
        [],
        supporting_claim_ids=[fact.claim_id],
    )

    assert ingestor.supporting_claim_ids(inference.claim_id) == (fact.claim_id,)


def test_inference_requires_a_passage_or_existing_claim(migrated_store):
    ingestor = EvidenceIngestor(migrated_store)

    with pytest.raises(EvidencePolicyError, match="supporting claim or passage"):
        ingestor.add_claim("Capacity may tighten.", ClaimKind.INFERENCE, [])


def test_missing_claim_dependency_rolls_back_claim(migrated_store, monkeypatch):
    ingestor = EvidenceIngestor(migrated_store)
    monkeypatch.setattr(ingestor, "_claim_id", lambda *args, **kwargs: "claim_failed")

    with pytest.raises(EvidencePolicyError, match="lineage references"):
        ingestor.add_claim(
            "Capacity may tighten.",
            ClaimKind.INFERENCE,
            [],
            supporting_claim_ids=["claim_missing"],
        )

    assert ingestor.store.get_claim("claim_failed") is None


def test_self_referential_claim_dependency_is_rejected_atomically(
    migrated_store, monkeypatch
):
    ingestor = EvidenceIngestor(migrated_store)
    monkeypatch.setattr(ingestor, "_claim_id", lambda *args, **kwargs: "claim_self")

    with pytest.raises(EvidencePolicyError, match="itself"):
        ingestor.add_claim(
            "Circular inference.",
            ClaimKind.INFERENCE,
            [],
            supporting_claim_ids=["claim_self"],
        )

    assert ingestor.store.get_claim("claim_self") is None


def test_contradicting_passage_remains_linked_without_replacing_support(
    migrated_store,
):
    ingestor = EvidenceIngestor(migrated_store)
    support = ingestor.ingest(
        make_document("https://issuer.test/guide", "Demand will grow 20%.")
    ).passages[0]
    contradiction = ingestor.ingest(
        make_document("https://analyst.test/note", "Demand will decline 5%.")
    ).passages[0]

    claim = ingestor.add_claim(
        "Demand will grow 20%.",
        ClaimKind.GUIDANCE,
        [support.passage_id],
        contradicting_passage_ids=[contradiction.passage_id],
    )

    assert {
        (item.passage_id, item.stance) for item in ingestor.lineage(claim.claim_id)
    } == {
        (support.passage_id, "supports"),
        (contradiction.passage_id, "contradicts"),
    }


@pytest.mark.parametrize(
    "record",
    [
        DocumentInput("filing", "https://issuer.test/a", "Issuer", NOW, NOW, "text"),
        EvidencePassage(
            "p", "d", 0, "text", hashlib.sha256(b"text").hexdigest(), 0, 4
        ),
        IngestedDocument("d", "https://issuer.test/a", "a" * 64, Path(__file__), ()),
        ClaimLineage("c", "p", "d", "text", 0, 4, "supports"),
    ],
)
def test_evidence_domain_records_are_frozen(record):
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.__setattr__(next(iter(record.__dataclass_fields__)), "changed")


def test_ingested_document_rejects_noncanonical_url():
    with pytest.raises(EvidenceValidationError, match="canonical"):
        IngestedDocument(
            "d",
            "HTTPS://ISSUER.TEST:443/a?utm_source=email",
            "a" * 64,
            Path(__file__),
            (),
        )


@pytest.mark.parametrize("invalid_id", ["", "bad id", "private\ud800"])
def test_claim_lineage_ids_fail_with_typed_redacted_error(
    migrated_store, invalid_id
):
    ingestor = EvidenceIngestor(migrated_store)

    with pytest.raises(EvidenceError) as add_error:
        ingestor.add_claim("Revenue grew.", ClaimKind.FACT, [invalid_id])
    with pytest.raises(EvidenceError) as lookup_error:
        ingestor.lineage(invalid_id)
    with pytest.raises(EvidenceError) as dependency_error:
        ingestor.supporting_claim_ids(invalid_id)

    for error in (add_error, lookup_error, dependency_error):
        assert "Revenue grew" not in str(error.value)
        assert "bad id" not in str(error.value)
        assert "private" not in str(error.value)


def test_evidence_passage_rejects_inconsistent_offsets():
    with pytest.raises(EvidenceValidationError, match="offset"):
        EvidencePassage(
            "p", "d", 0, "text", hashlib.sha256(b"text").hexdigest(), 1, 4
        )


def test_evidence_passage_rejects_mismatched_content_hash():
    with pytest.raises(EvidenceValidationError, match="content_hash"):
        EvidencePassage("p", "d", 0, "text", "a" * 64, 0, 4)
