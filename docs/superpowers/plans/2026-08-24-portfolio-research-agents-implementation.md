# Portfolio-Aware Agentic Research System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized, IBKR-aware qualitative research pipeline that maintains evidence-backed industry theses, produces institutional-style reports, emits gated research ratings, automatically falls back to Ollama, and never exceeds $5 of paid model usage per calendar month.

**Architecture:** Add a focused `news_bot/research` package around the existing newsletter. Deterministic connectors and a SQLite store create the evidence base; typed specialist agents consume that evidence through a provider-neutral model gateway; an orchestrator enforces quality, freshness, security, and cost gates before rendering HTML/PDF reports. External inference is preferred when configured, while a standard Ollama service is the zero-key fallback.

**Tech Stack:** Python 3.11, dataclasses and Pydantic 2, SQLite/WAL, requests, OpenAI Responses API, Ollama HTTP API, Jinja2, WeasyPrint, APScheduler, pytest, Docker Compose.

---

## Scope and Execution Order

The approved specification covers four dependent subsystems. Execute them in order so each phase leaves working, testable software:

1. Platform foundation and IBKR portfolio ingestion.
2. Evidence ingestion and exposure mapping.
3. Model routing, specialist agents, and publication gates.
4. Reports, scheduling, Docker runtime, and research evaluations.

Before Task 1, create an isolated worktree with `superpowers:using-git-worktrees`. The verified baseline command is:

```bash
venv/bin/python -m pytest -q
```

Expected baseline: `305 passed, 1 warning`. Running system `pytest` uses Python 3.13 without project dependencies and is not a valid baseline.

## File Map

| Path | Responsibility |
|---|---|
| `news_bot/research/config.py` | Research-specific typed configuration and secret-file loading |
| `news_bot/research/models.py` | Shared enums and immutable domain records |
| `news_bot/research/store.py` | SQLite connection, migrations, transactions, and repositories |
| `news_bot/research/migrations/001_initial.sql` | Initial research schema |
| `news_bot/research/budget.py` | Paid-model reservation and reconciliation ledger |
| `news_bot/research/ibkr_flex.py` | Flex request, polling, XML normalization, and redaction |
| `news_bot/research/entities.py` | Canonical entity resolution and exposure relationships |
| `news_bot/research/evidence.py` | Document hashing, passages, claims, and lineage |
| `news_bot/research/connectors/` | News, SEC, Form D, patent, awards, and clinical-trial connectors |
| `news_bot/research/providers/` | OpenAI, Ollama, and fallback routing |
| `news_bot/research/agents/` | Typed specialist roles and prompts |
| `news_bot/research/quality.py` | Freshness, evidence, recommendation, and publication gates |
| `news_bot/research/orchestrator.py` | Dependency-aware daily, weekly, monthly, and landscape workflows |
| `news_bot/research/reports/` | Report view models, exhibits, templates, HTML, and PDF rendering |
| `news_bot/research/cli.py` | One-shot workflow commands, dry run, replay, and regeneration |
| `news_bot/research/scheduler.py` | APScheduler entry point and non-overlapping run lock |
| `tests/research/conftest.py` | Shared UTC, fixture-loading, database, and typed factory helpers |
| `tests/research/` | Unit, contract, integration, fixture, and golden-report tests |

## Phase 1: Foundation and Portfolio Ingestion

### Task 1: Research package, domain types, and configuration

**Files:**
- Create: `news_bot/research/__init__.py`
- Create: `news_bot/research/models.py`
- Create: `news_bot/research/config.py`
- Create: `tests/research/__init__.py`
- Create: `tests/research/conftest.py`
- Create: `tests/research/test_config.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `env.example`

- [ ] **Step 1: Add failing configuration and domain tests**

```python
# tests/research/test_config.py
from datetime import datetime, timezone
from decimal import Decimal

from news_bot.research.config import ResearchConfig
from news_bot.research.models import InferenceMode, PortfolioSnapshot


def test_missing_openai_key_selects_local_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    config = ResearchConfig.from_env()
    assert config.inference_mode is InferenceMode.LOCAL_ONLY
    assert config.ollama_base_url == "http://localhost:11434"


def test_secret_file_wins_over_environment(monkeypatch, tmp_path):
    secret = tmp_path / "openai_key"
    secret.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(secret))
    assert ResearchConfig.from_env().openai_api_key == "file-key"


def test_portfolio_snapshot_is_timezone_aware():
    snapshot = PortfolioSnapshot(
        snapshot_id="snapshot-1",
        as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
        base_currency="USD",
        nav=Decimal("4500.00"),
        cash=Decimal("500.00"),
        is_stale=False,
    )
    assert snapshot.as_of.tzinfo is not None
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `venv/bin/python -m pytest tests/research/test_config.py -v`  
Expected: FAIL because `news_bot.research` does not exist.

- [ ] **Step 3: Add dependencies and focused domain types**

Add to both `requirements.txt` and `[project].dependencies` in `pyproject.toml`:

```text
pydantic>=2.9,<3
openai>=2,<3
jinja2>=3.1,<4
weasyprint>=62,<70
apscheduler>=3.10,<4
```

Create `news_bot/research/models.py` with string enums for `InferenceMode`, `AgentRole`, `ClaimKind`, `RecommendationRating`, and `ReviewVerdict`, plus frozen dataclasses for `PortfolioSnapshot`, `Position`, `SourceDocument`, `EvidenceClaim`, `ModelUsage`, and `AgentRunResult`. `AgentRole` has the exact values `research_director`, `portfolio_mapper`, `event_scout`, `emerging_company_scout`, `evidence_analyst`, `industry_strategist`, `fundamental_analyst`, `skeptical_reviewer`, and `research_editor`. Store money and quantities as `Decimal`, serialized as strings at persistence boundaries.

```python
class InferenceMode(str, Enum):
    EXTERNAL = "external"
    LOCAL_ONLY = "local_only"


class RecommendationRating(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL_REDUCE = "sell_reduce"
    NO_RATING = "no_rating"


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_id: str
    as_of: datetime
    base_currency: str
    nav: Decimal
    cash: Decimal
    is_stale: bool

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("PortfolioSnapshot.as_of must be timezone-aware")
```

- [ ] **Step 4: Implement secret-aware research configuration**

`ResearchConfig.from_env()` must read `*_FILE` before the direct environment value, default to local-only mode when `OPENAI_API_KEY` is absent, validate `MODEL_BUDGET_SOFT_USD < MODEL_BUDGET_HARD_USD`, and expose paths under `RESEARCH_DATA_DIR`.

```python
def _read_secret(name: str) -> str | None:
    file_value = os.getenv(f"{name}_FILE")
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    value = os.getenv(name)
    return value.strip() if value else None


@classmethod
def from_env(cls) -> "ResearchConfig":
    openai_key = _read_secret("OPENAI_API_KEY")
    config = cls(
        enabled=_get_bool("RESEARCH_ENABLED", False),
        data_dir=Path(os.getenv("RESEARCH_DATA_DIR", "research_data")),
        openai_api_key=openai_key,
        inference_mode=(
            InferenceMode.EXTERNAL if openai_key else InferenceMode.LOCAL_ONLY
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_RESEARCH_MODEL", "llama3.1:8b"),
        budget_soft_usd=Decimal(os.getenv("MODEL_BUDGET_SOFT_USD", "4.00")),
        budget_hard_usd=Decimal(os.getenv("MODEL_BUDGET_HARD_USD", "5.00")),
    )
    config.validate()
    return config
```

- [ ] **Step 5: Add shared test helpers with explicit factories**

`tests/research/conftest.py` initially provides `utc()`, UTF-8 `fixture()`, and JSON `fixture_json()`. Later tasks extend it only after the relevant production type exists. Test factories accept explicit overrides and otherwise use fixed 2026-08-24 UTC values; they never read the real environment, `.env`, network, portfolio, or secrets.

```python
def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict[str, object]:
    return json.loads(fixture(name))
```

- [ ] **Step 6: Document every new environment setting in `env.example`**

Include research enablement, data paths, Flex secret-file paths, OpenAI secret-file path, Ollama fallback model, model routes, price effective-until date, budget limits, staleness limits, SEC user agent, and report schedules. Use non-secret examples such as `/run/secrets/openai_api_key`.

- [ ] **Step 7: Run focused and baseline tests**

Run: `venv/bin/python -m pytest tests/research/test_config.py tests/test_config.py -v`  
Expected: PASS.  
Run: `venv/bin/python -m pytest -q`  
Expected: 308 or more tests pass.

- [ ] **Step 8: Commit the package foundation**

```bash
git add news_bot/research tests/research requirements.txt pyproject.toml env.example
git commit -m "feat: add research configuration and domain models"
```

### Task 2: SQLite schema, migrations, and repositories

**Files:**
- Create: `news_bot/research/migrations/001_initial.sql`
- Create: `news_bot/research/store.py`
- Create: `tests/research/test_store.py`
- Modify: `tests/research/conftest.py`

- [ ] **Step 1: Write failing migration and lineage tests**

```python
def test_store_migrates_empty_database(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    names = store.table_names()
    assert {"portfolio_snapshots", "positions", "source_documents", "claims"} <= names


def test_claim_requires_evidence(tmp_path):
    store = make_migrated_store(tmp_path)
    with pytest.raises(IntegrityError):
        store.insert_claim(make_claim("claim-1"), evidence_ids=[])


def test_thesis_revisions_are_append_only(tmp_path):
    store = make_migrated_store(tmp_path)
    first = store.append_thesis_revision("semiconductors", "base thesis", [])
    second = store.append_thesis_revision("semiconductors", "revised thesis", [])
    assert second.version == first.version + 1
    assert len(store.list_thesis_revisions("semiconductors")) == 2
```

- [ ] **Step 2: Verify the tests fail**

Run: `venv/bin/python -m pytest tests/research/test_store.py -v`  
Expected: FAIL because `ResearchStore` is missing.

- [ ] **Step 3: Create the complete initial schema**

The migration must create `schema_migrations`, `portfolio_snapshots`, `positions`, `entities`, `securities`, `relationships`, `source_documents`, `document_passages`, `claims`, `claim_evidence`, `industry_theses`, `scenarios`, `recommendations`, `research_tasks`, `agent_runs`, `reports`, `model_usage`, and `budget_reservations`. Add foreign keys, unique content hashes, append-only version uniqueness, UTC timestamps, and indexes on document URL, entity symbol, claim status, and task state.

```sql
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    entity_id TEXT REFERENCES entities(entity_id),
    kind TEXT NOT NULL CHECK(kind IN ('fact','guidance','estimate','inference')),
    text TEXT NOT NULL,
    as_of TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    status TEXT NOT NULL CHECK(status IN ('active','contradicted','superseded'))
);

CREATE TABLE claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE RESTRICT,
    passage_id TEXT NOT NULL REFERENCES document_passages(passage_id) ON DELETE RESTRICT,
    stance TEXT NOT NULL CHECK(stance IN ('supports','contradicts')),
    PRIMARY KEY (claim_id, passage_id, stance)
);
```

- [ ] **Step 4: Implement connection and repository boundaries**

`ResearchStore` must enable `PRAGMA foreign_keys=ON`, `journal_mode=WAL`, and `busy_timeout=5000`; provide a transaction context manager; apply numbered migrations once; and expose focused methods rather than raw SQL outside the store.

```python
@contextmanager
def transaction(self) -> Iterator[sqlite3.Connection]:
    connection = self.connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
```

Extend `tests/research/conftest.py` after `ResearchStore` exists:

```python
def make_migrated_store(tmp_path: Path) -> ResearchStore:
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    return store


@pytest.fixture
def migrated_store(tmp_path: Path) -> ResearchStore:
    return make_migrated_store(tmp_path)


def make_claim(claim_id: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        entity_id=None,
        kind=ClaimKind.FACT,
        text="Revenue grew 20%.",
        as_of=utc(2026, 8, 24),
        confidence=Decimal("0.90"),
        status="active",
    )
```

- [ ] **Step 5: Run store and baseline tests**

Run: `venv/bin/python -m pytest tests/research/test_store.py -v`  
Expected: PASS.  
Run: `venv/bin/python -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 6: Commit the research store**

```bash
git add news_bot/research/migrations news_bot/research/store.py tests/research/test_store.py
git commit -m "feat: add persistent research evidence store"
```

### Task 3: Paid-model budget ledger

**Files:**
- Create: `news_bot/research/budget.py`
- Create: `tests/research/test_budget.py`

- [ ] **Step 1: Write failing reservation tests**

```python
def test_reservation_cannot_cross_hard_limit(migrated_store):
    ledger = BudgetLedger(migrated_store, soft_limit=Decimal("4"), hard_limit=Decimal("5"))
    ledger.reserve("run-1", Decimal("4.75"), now=utc(2026, 8, 1))
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run-2", Decimal("0.26"), now=utc(2026, 8, 2))


def test_reconcile_uses_provider_reported_cost(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve("run-1", Decimal("1.00"), now=utc(2026, 8, 1))
    ledger.reconcile(reservation.id, actual_cost=Decimal("0.42"))
    assert ledger.month_total(2026, 8) == Decimal("0.42")


def test_expired_price_table_blocks_paid_call(migrated_store):
    table = PriceTable(effective_until=date(2026, 8, 23), prices={})
    with pytest.raises(ExpiredPricing):
        table.estimate("gpt-5.6-sol", 1000, 100)
```

- [ ] **Step 2: Run and verify failure**

Run: `venv/bin/python -m pytest tests/research/test_budget.py -v`  
Expected: FAIL because `BudgetLedger` is missing.

- [ ] **Step 3: Implement pessimistic reservation and reconciliation**

Use calendar-month UTC boundaries. A reservation counts against available budget until reconciled or explicitly released. Never release an unknown provider result as zero cost; leave it reserved and mark it `usage_unknown`.

```python
def estimate_cost(self, model: str, input_tokens: int, max_output_tokens: int) -> Decimal:
    self.assert_current(date.today())
    price = self.prices[model]
    input_cost = Decimal(input_tokens) * price.input_per_million / Decimal(1_000_000)
    output_cost = Decimal(max_output_tokens) * price.output_per_million / Decimal(1_000_000)
    return (input_cost + output_cost).quantize(Decimal("0.000001"), rounding=ROUND_UP)
```

- [ ] **Step 4: Test soft-limit escalation and concurrent reservations**

Add tests proving noncritical roles are rejected after $4, critical reviewer/editor roles remain eligible below $5, and two transactions cannot reserve the same remaining budget.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_budget.py tests/research/test_store.py -v`  
Expected: PASS.

```bash
git add news_bot/research/budget.py tests/research/test_budget.py
git commit -m "feat: enforce monthly model spending ceiling"
```

### Task 4: IBKR Flex ingestion and portfolio snapshots

**Files:**
- Create: `news_bot/research/ibkr_flex.py`
- Create: `tests/research/fixtures/ibkr_send_success.xml`
- Create: `tests/research/fixtures/ibkr_statement.xml`
- Create: `tests/research/test_ibkr_flex.py`

- [ ] **Step 1: Add failing request, parsing, and redaction tests**

```python
def test_sync_uses_reference_code_and_never_logs_token(requests_mock, caplog, flex_config):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))
    result = FlexClient(flex_config).sync()
    assert result.snapshot.base_currency == "USD"
    assert result.positions[0].symbol == "NVDA"
    assert flex_config.token not in caplog.text


def test_account_id_is_replaced_by_stable_local_hash(flex_config):
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt="local-test-salt")
    assert result.account_ref.startswith("acct_")
    assert "U1234567" not in result.account_ref


def test_stale_snapshot_suppresses_sizing():
    assert portfolio_is_stale(utc(2026, 8, 22), utc(2026, 8, 24), max_hours=36)
```

- [ ] **Step 2: Verify tests fail**

Run: `venv/bin/python -m pytest tests/research/test_ibkr_flex.py -v`  
Expected: FAIL because the Flex client is missing.

- [ ] **Step 3: Implement two-stage Flex retrieval with bounded polling**

The client sends `t`, `q`, and `v=3`; parses `ReferenceCode` and response URL; polls only error `1019` with capped exponential backoff; maps authentication, IP, invalid query, throttling, and incomplete-statement codes to typed errors; and sets a `User-Agent`.

```python
for attempt in range(config.max_polls):
    response = session.get(statement_url, params={"t": token, "q": reference, "v": 3}, timeout=30)
    root = ElementTree.fromstring(response.text)
    if root.tag != "FlexStatementResponse":
        return parse_statement(response.text, config.account_salt)
    code = root.findtext("ErrorCode")
    if code != "1019":
        raise FlexError.from_code(code, root.findtext("ErrorMessage") or "Unknown Flex error")
    sleeper(min(2 ** attempt, 16))
raise FlexError("Statement generation did not complete within the polling limit")
```

- [ ] **Step 4: Normalize money and positions, then persist atomically**

Parse `OpenPosition` records and the latest equity summary. Reject non-finite decimals, preserve identifiers, make all timestamps UTC, and insert the snapshot plus positions in one store transaction.

- [ ] **Step 5: Run focused and full tests**

Run: `venv/bin/python -m pytest tests/research/test_ibkr_flex.py -v`  
Expected: PASS.  
Run: `venv/bin/python -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 6: Commit Flex support**

```bash
git add news_bot/research/ibkr_flex.py tests/research/fixtures tests/research/test_ibkr_flex.py
git commit -m "feat: ingest read-only IBKR Flex portfolios"
```

## Phase 2: Evidence and Exposure Mapping

### Task 5: Document ingestion, passages, and claim lineage

**Files:**
- Create: `news_bot/research/evidence.py`
- Create: `tests/research/test_evidence.py`

- [ ] **Step 1: Write failing deduplication and passage tests**

```python
def test_canonical_content_hash_deduplicates_tracking_urls(migrated_store):
    first = make_document("https://issuer.test/release?utm_source=email", "Revenue grew 20%.")
    second = make_document("https://issuer.test/release", "Revenue grew 20%.")
    ingestor = EvidenceIngestor(migrated_store)
    assert ingestor.ingest(first).document_id == ingestor.ingest(second).document_id


def test_claim_lineage_returns_exact_supporting_passage(migrated_store):
    ingestor = EvidenceIngestor(migrated_store)
    document = ingestor.ingest(make_document("https://issuer.test/10q", "Data center revenue was $30 billion."))
    passage = document.passages[0]
    claim = ingestor.add_claim("Data center revenue was $30 billion.", ClaimKind.FACT, [passage.passage_id])
    assert ingestor.lineage(claim.claim_id)[0].text == passage.text
```

- [ ] **Step 2: Run and verify failure**

Run: `venv/bin/python -m pytest tests/research/test_evidence.py -v`  
Expected: FAIL because `EvidenceIngestor` is missing.

- [ ] **Step 3: Implement canonicalization and immutable source caching**

Remove known tracking query parameters, normalize whitespace before hashing, store raw bytes under `source_cache/{sha256_hex}`, and never overwrite content for an existing hash. Split text into deterministic passages with stable IDs and source-relative character offsets.

- [ ] **Step 4: Enforce claim evidence policy in code and database**

Facts, guidance, and estimates require at least one passage. Inferences require at least one supporting claim or passage. Contradicting passages remain linked rather than replacing earlier evidence.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_evidence.py tests/research/test_store.py -v`  
Expected: PASS.

```bash
git add news_bot/research/evidence.py tests/research/test_evidence.py
git commit -m "feat: add traceable evidence ingestion"
```

### Task 6: News, SEC, and investor-relations connectors

**Files:**
- Create: `news_bot/research/connectors/__init__.py`
- Create: `news_bot/research/connectors/base.py`
- Create: `news_bot/research/connectors/news.py`
- Create: `news_bot/research/connectors/sec.py`
- Create: `news_bot/research/connectors/investor_relations.py`
- Create: `news_bot/research/connectors/fmp.py`
- Create: `tests/research/fixtures/sec_submissions.json`
- Create: `tests/research/fixtures/sec_companyfacts.json`
- Create: `tests/research/fixtures/fmp_quote.json`
- Create: `tests/research/fixtures/fmp_income_statement.json`
- Create: `tests/research/test_connectors_core.py`

- [ ] **Step 1: Write connector contract tests**

```python
@pytest.mark.parametrize("connector", [MarketAuxResearchConnector(), RSSResearchConnector()])
def test_news_connectors_emit_normalized_documents(connector, article):
    document = connector.from_article(article)
    assert document.publisher
    assert document.published_at.tzinfo is not None
    assert document.canonical_url.startswith("https://")


def test_sec_connector_sends_identifying_user_agent(requests_mock, sec_config):
    requests_mock.get(sec_config.submissions_url("0001045810"), json=fixture_json("sec_submissions.json"))
    SECConnector(sec_config).fetch_submissions("0001045810")
    assert requests_mock.last_request.headers["User-Agent"] == "newsletter-research research@example.com"


def test_fmp_connector_emits_dated_price_and_fundamentals(fmp_connector):
    packet = fmp_connector.parse(
        quote=fixture_json("fmp_quote.json"),
        statements=fixture_json("fmp_income_statement.json"),
    )
    assert packet.price.effective_date == date(2026, 8, 21)
    assert packet.financial_period == "2026-Q2"
```

- [ ] **Step 2: Verify contract tests fail**

Run: `venv/bin/python -m pytest tests/research/test_connectors_core.py -v`  
Expected: FAIL because connectors are missing.

- [ ] **Step 3: Define the connector protocol and checkpoint contract**

```python
class ResearchConnector(Protocol):
    name: str

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        """Return normalized documents and the next durable checkpoint."""
```

Checkpoints advance only after the full batch commits. Connector errors record status, retryability, and redacted diagnostics.

- [ ] **Step 4: Adapt existing MarketAux/RSS output without duplicating clients**

Wrap `fetch_articles_by_section()` and RSS-normalized `ArticleMeta` objects. Preserve source sections as tags and route extracted article content through `EvidenceIngestor`.

- [ ] **Step 5: Implement SEC submissions, filing documents, and Company Facts**

Use `data.sec.gov`, a configured identifying user agent, conditional requests, and a connector-level rate limiter below SEC's published ceiling. Limit forms to 10-K, 10-Q, 8-K, 20-F, 6-K, and registration statements selected by the research director.

- [ ] **Step 6: Implement investor-relations and FMP adapters**

Investor-relations URLs are allowlisted per issuer and pass through the existing article extractor. FMP personal-use endpoints normalize dated prices, statements, ratios, and market metadata; a missing or disallowed endpoint degrades to SEC facts and returns an explicit unavailable field instead of fabricating a value.

- [ ] **Step 7: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_connectors_core.py tests/test_news_client.py tests/test_rss_client.py -v`  
Expected: PASS.

```bash
git add news_bot/research/connectors tests/research/fixtures/sec_* tests/research/test_connectors_core.py
git commit -m "feat: ingest news and SEC research sources"
```

### Task 7: Emerging-company public-signal connectors

**Files:**
- Create: `news_bot/research/connectors/form_d.py`
- Create: `news_bot/research/connectors/uspto.py`
- Create: `news_bot/research/connectors/awards.py`
- Create: `news_bot/research/connectors/clinical_trials.py`
- Create: `news_bot/research/connectors/manual_import.py`
- Create: `tests/research/fixtures/form_d_sample.tsv`
- Create: `tests/research/fixtures/uspto_sample.json`
- Create: `tests/research/fixtures/sbir_sample.json`
- Create: `tests/research/fixtures/usaspending_sample.json`
- Create: `tests/research/fixtures/clinical_trials_sample.json`
- Create: `tests/research/fixtures/manual_company_import.csv`
- Create: `tests/research/test_connectors_signals.py`

- [ ] **Step 1: Add one failing normalization test per source**

```python
@pytest.mark.parametrize(
    ("connector", "fixture_name", "expected_signal"),
    [
        (FormDConnector(), "form_d_sample.tsv", "funding"),
        (USPTOConnector(), "uspto_sample.json", "patent"),
        (SBIRConnector(), "sbir_sample.json", "grant"),
        (USASpendingConnector(), "usaspending_sample.json", "contract"),
        (ClinicalTrialsConnector(), "clinical_trials_sample.json", "clinical_trial"),
        (ManualImportConnector(), "manual_company_import.csv", "private_market_profile"),
    ],
)
def test_public_signal_normalization(connector, fixture_name, expected_signal):
    records = connector.parse(fixture(fixture_name))
    assert records[0].signal_type == expected_signal
    assert records[0].source_document_id
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_connectors_signals.py -v`  
Expected: FAIL because the signal connectors are missing.

- [ ] **Step 3: Implement downloadable and API-backed adapters**

Form D supports quarterly ZIP/TSV imports. SBIR supports bulk download when its API health check fails. USPTO uses authenticated ODP requests. USAspending and ClinicalTrials use their documented JSON APIs. Manual CSV imports support licensed exports without coupling the system to a proprietary API. All adapters emit `EmergingSignal` records with company name, signal type, amount or stage where available, effective date, geography, technology terms, and evidence passage ID.

- [ ] **Step 4: Add source-specific pacing and safe degradation**

Tests must prove a maintenance response marks the connector unavailable without blocking unrelated sources, and no connector retries authentication failures.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_connectors_signals.py -v`  
Expected: PASS.

```bash
git add news_bot/research/connectors tests/research/fixtures tests/research/test_connectors_signals.py
git commit -m "feat: collect emerging company public signals"
```

### Task 8: Entity resolution and portfolio exposure graph

**Files:**
- Create: `news_bot/research/entities.py`
- Create: `tests/research/fixtures/sec_company_tickers.json`
- Create: `tests/research/test_entities.py`

- [ ] **Step 1: Write failing deterministic resolution tests**

```python
def test_identifier_match_beats_name_similarity(resolver):
    entity = resolver.resolve(SecurityIdentity(symbol="NVDA", cik="0001045810", isin=None))
    assert entity.canonical_name == "NVIDIA CORP"
    assert entity.resolution_method == "cik"


def test_relationship_requires_evidence(resolver):
    with pytest.raises(MissingEvidence):
        resolver.add_relationship("company-a", "company-b", "supplier", evidence_ids=[])


def test_etf_overlap_rolls_up_underlying_exposure(mapper):
    exposure = mapper.map_positions([position("NVDA", 20), etf_position("SMH", 30, {"NVDA": 0.18})])
    assert exposure["NVDA"].direct_weight == Decimal("0.20")
    assert exposure["NVDA"].lookthrough_weight == Decimal("0.054")
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_entities.py -v`  
Expected: FAIL because the resolver is missing.

- [ ] **Step 3: Implement identifier-first resolution**

Resolve conid/CIK/ISIN/FIGI/CUSIP before normalized names. Ambiguous name matches remain unresolved and create a research task. Store aliases and resolution provenance.

- [ ] **Step 4: Implement evidence-backed directed relationships and ETF look-through**

Separate direct portfolio weight from look-through weight and relationship-derived qualitative exposure. Never convert semantic similarity into a stored relationship without cited evidence.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_entities.py tests/research/test_ibkr_flex.py -v`  
Expected: PASS.

```bash
git add news_bot/research/entities.py tests/research/fixtures/sec_company_tickers.json tests/research/test_entities.py
git commit -m "feat: map portfolio and value chain exposures"
```

## Phase 3: Providers, Agents, and Quality Gates

### Task 9: OpenAI/Ollama providers and automatic fallback

**Files:**
- Create: `news_bot/research/providers/__init__.py`
- Create: `news_bot/research/providers/base.py`
- Create: `news_bot/research/providers/openai_provider.py`
- Create: `news_bot/research/providers/ollama_provider.py`
- Create: `news_bot/research/providers/router.py`
- Create: `tests/research/test_providers.py`

- [ ] **Step 1: Write failing zero-key and outage-routing tests**

```python
def test_no_api_key_routes_every_role_to_ollama(local_config, ollama_provider):
    router = ProviderRouter(local_config, external=None, ollama=ollama_provider)
    for role in AgentRole:
        assert router.provider_for(role).name == "ollama"


def test_authentication_failure_disables_external_for_run(external_config, openai_provider, ollama_provider):
    openai_provider.generate.side_effect = ExternalAuthenticationError("invalid key")
    router = ProviderRouter(external_config, openai_provider, ollama_provider)
    response = router.generate(request(role=AgentRole.EVENT_SCOUT, allow_local_fallback=True))
    assert response.provider == "ollama"
    assert router.external_disabled_reason == "authentication"


def test_frontier_review_defers_when_fallback_is_not_allowed(router):
    router.external.generate.side_effect = ExternalUnavailable("timeout")
    with pytest.raises(TaskDeferred):
        router.generate(request(role=AgentRole.SKEPTICAL_REVIEWER, allow_local_fallback=False))
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_providers.py -v`  
Expected: FAIL because provider modules are missing.

- [ ] **Step 3: Define provider-neutral structured requests**

```python
class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return validated JSON data and provider-reported token usage."""
```

`ModelRequest` includes role, system prompt, evidence packet, Pydantic output schema, maximum output tokens, reasoning effort, fallback policy, and run ID. Providers return parsed data, raw response hash, input/output/reasoning token counts, model, and latency.

- [ ] **Step 4: Implement Ollama structured generation**

POST to `/api/generate` with `stream=false` and JSON format. Validate model availability first. Parse the response string with the requested Pydantic schema and raise a typed validation error containing no prompt contents.

- [ ] **Step 5: Implement OpenAI Responses structured generation with budget reservation**

Reserve pessimistic cost before the call, use the role-configured model and structured output schema, reconcile provider-reported usage, and hash rather than log raw prompts. Authentication errors disable external inference for the remainder of the run.

- [ ] **Step 6: Implement routing and reportable inference metadata**

No key selects Ollama immediately. Outages use per-role fallback policy. Every response records `external` or `local_only`; no silent provider changes are permitted.

- [ ] **Step 7: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_providers.py tests/research/test_budget.py -v`  
Expected: PASS.

```bash
git add news_bot/research/providers tests/research/test_providers.py
git commit -m "feat: route research models with Ollama fallback"
```

### Task 10: Agent contracts and specialist implementations

**Files:**
- Create: `news_bot/research/agents/__init__.py`
- Create: `news_bot/research/agents/contracts.py`
- Create: `news_bot/research/agents/base.py`
- Create: `news_bot/research/agents/director.py`
- Create: `news_bot/research/agents/scouts.py`
- Create: `news_bot/research/agents/analysts.py`
- Create: `news_bot/research/agents/reviewer.py`
- Create: `news_bot/research/agents/editor.py`
- Create: `news_bot/research/prompts/director.md`
- Create: `news_bot/research/prompts/event_scout.md`
- Create: `news_bot/research/prompts/emerging_scout.md`
- Create: `news_bot/research/prompts/evidence_analyst.md`
- Create: `news_bot/research/prompts/industry_strategist.md`
- Create: `news_bot/research/prompts/fundamental_analyst.md`
- Create: `news_bot/research/prompts/skeptical_reviewer.md`
- Create: `news_bot/research/prompts/research_editor.md`
- Create: `tests/research/test_agents.py`

- [ ] **Step 1: Write failing typed-contract tests**

```python
def test_evidence_analyst_rejects_uncited_fact(fake_provider, evidence_agent):
    fake_provider.result = {"claims": [{"text": "Revenue grew 20%", "kind": "fact", "evidence_ids": []}]}
    with pytest.raises(AgentContractError):
        evidence_agent.run(make_evidence_task())


def test_private_company_cannot_receive_trade_rating(fundamental_agent):
    with pytest.raises(IneligibleSecurity):
        fundamental_agent.run(make_private_company_task())


def test_reviewer_has_three_explicit_verdicts():
    assert {item.value for item in ReviewVerdict} == {"pass", "revise", "block"}
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_agents.py -v`  
Expected: FAIL because contracts are missing.

- [ ] **Step 3: Define versioned Pydantic contracts**

Create explicit inputs and outputs for each role. All analytical outputs carry `evidence_ids`, `as_of`, `confidence`, `inference_mode`, and `schema_version="1"`. Recommendation output includes thesis, horizon, valuation range, assumptions, catalysts, counter-thesis, risks, invalidation conditions, rating, and eligibility.

- [ ] **Step 4: Implement the bounded agent base**

The base loads a versioned prompt resource, builds an evidence packet from IDs, calls the router once, validates output, stores the run, and retries exactly once only for schema validation feedback.

```python
def run(self, task: AgentTask[InputT]) -> OutputT:
    request = self.build_request(task)
    try:
        response = self.router.generate(request)
        return self.output_type.model_validate(response.data)
    except ValidationError as error:
        response = self.router.generate(self.repair_request(request, error))
        return self.output_type.model_validate(response.data)
```

- [ ] **Step 5: Implement each specialist role without inter-agent chat**

Director creates tasks; scouts rank events/signals; evidence analyst creates claims; strategist updates scenarios; fundamental analyst rates eligible public securities; reviewer issues pass/revise/block; editor emits a report outline using approved claim IDs only.

- [ ] **Step 6: Add prompt-injection resistance tests**

Feed a document passage containing instructions to reveal secrets and prove it remains delimited as untrusted evidence, cannot alter the system prompt, and never appears in provider tool configuration.

- [ ] **Step 7: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_agents.py tests/research/test_providers.py -v`  
Expected: PASS.

```bash
git add news_bot/research/agents news_bot/research/prompts tests/research/test_agents.py
git commit -m "feat: add specialized qualitative research agents"
```

### Task 11: Freshness, recommendation, and publication gates

**Files:**
- Create: `news_bot/research/quality.py`
- Create: `tests/research/test_quality.py`

- [ ] **Step 1: Write failing gate tests**

```python
def test_stale_portfolio_blocks_sizing_but_not_event_research():
    context = quality_context(portfolio_age_hours=40)
    result = QualityGate().evaluate(context)
    assert result.allow_event_report is True
    assert result.allow_sizing is False


def test_material_quantitative_claim_without_lineage_blocks_publication():
    result = QualityGate().evaluate(report_with_uncited_material_number())
    assert result.verdict is ReviewVerdict.BLOCK


def test_latest_completed_price_within_four_days_is_fresh():
    assert price_is_fresh(date(2026, 8, 21), date(2026, 8, 24), exchange="NASDAQ")


def test_uncertain_filing_freshness_forces_no_rating():
    result = RecommendationGate().evaluate(recommendation_context(filing_due=None))
    assert result.rating is RecommendationRating.NO_RATING
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_quality.py -v`  
Expected: FAIL because quality gates are missing.

- [ ] **Step 3: Implement deterministic gate composition**

Implement separate gates for evidence lineage, corroboration, portfolio age, price age, periodic filing availability, security eligibility, contradictory evidence, reviewer verdict, editor claim set, and calculated-exhibit reconciliation. Combine gate results without letting a model override deterministic failures.

- [ ] **Step 4: Test local-only disclosure and no-rating behavior**

Local-only reports must name the model and mode. A weaker local result may publish only when the same deterministic gates and reviewer pass; otherwise preserve research as draft and emit no rating.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_quality.py tests/research/test_agents.py -v`  
Expected: PASS.

```bash
git add news_bot/research/quality.py tests/research/test_quality.py
git commit -m "feat: enforce research publication quality gates"
```

### Task 12: Dependency-aware research orchestration

**Files:**
- Create: `news_bot/research/orchestrator.py`
- Create: `tests/research/test_orchestrator.py`

- [ ] **Step 1: Write failing workflow and idempotency tests**

```python
def test_daily_run_does_not_recompute_all_ratings(orchestrator):
    result = orchestrator.run_daily(as_of=utc(2026, 8, 24))
    assert result.completed_stages == ["portfolio", "ingestion", "materiality", "event_update"]
    assert "full_recommendation_refresh" not in result.completed_stages


def test_same_source_and_period_is_idempotent(orchestrator):
    first = orchestrator.run_daily(as_of=utc(2026, 8, 24))
    second = orchestrator.run_daily(as_of=utc(2026, 8, 24))
    assert second.report_ids == first.report_ids
    assert second.new_agent_runs == 0


def test_blocked_review_returns_task_to_originating_role(orchestrator):
    result = orchestrator.run_weekly(reviewer_verdict="revise")
    assert result.pending_tasks[0].assigned_role is AgentRole.FUNDAMENTAL_ANALYST
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_orchestrator.py -v`  
Expected: FAIL because the orchestrator is missing.

- [ ] **Step 3: Implement explicit workflow graphs**

Daily: portfolio → ingest → resolve → materiality → optional event analysis → review → publish. Weekly: portfolio → changed theses → selected recommendations → review → publish. Monthly: public signals → emerging map → one industry refresh → review → publish. Backfill: bounded historical documents → evidence only unless explicitly authorized for analysis.

- [ ] **Step 4: Implement durable tasks, retries, and run locks**

Tasks have stable idempotency keys, dependency IDs, attempt count, state, assigned role, and defer reason. Use a SQLite lease with expiry and owner ID so scheduler and one-shot runner cannot publish concurrently.

- [ ] **Step 5: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_orchestrator.py tests/research/test_quality.py -v`  
Expected: PASS.

```bash
git add news_bot/research/orchestrator.py tests/research/test_orchestrator.py
git commit -m "feat: orchestrate durable research workflows"
```

## Phase 4: Reports, Runtime, and Evaluation

### Task 13: Institutional HTML/PDF reports and email attachments

**Files:**
- Create: `news_bot/research/reports/__init__.py`
- Create: `news_bot/research/reports/models.py`
- Create: `news_bot/research/reports/exhibits.py`
- Create: `news_bot/research/reports/renderer.py`
- Create: `news_bot/research/reports/templates/base.html`
- Create: `news_bot/research/reports/templates/event_update.html`
- Create: `news_bot/research/reports/templates/portfolio_brief.html`
- Create: `news_bot/research/reports/templates/industry_landscape.html`
- Create: `news_bot/research/reports/templates/emerging_monitor.html`
- Create: `tests/research/test_reports.py`
- Modify: `news_bot/email_client.py:349-423`
- Modify: `tests/test_email_client.py`

- [ ] **Step 1: Write failing rendering and attachment tests**

```python
def test_event_report_contains_claim_citations_and_mode(renderer, tmp_path):
    artifact = renderer.render_event_update(approved_event_report(), tmp_path)
    assert "Evidence E-001" in artifact.html
    assert "Inference mode: local_only" in artifact.html
    assert artifact.pdf_path.read_bytes().startswith(b"%PDF")


def test_exhibit_values_come_from_rows_not_model_html():
    exhibit = build_exposure_exhibit([ExposureRow("NVDA", Decimal("0.25"))])
    assert exhibit.rows[0].display_weight == "25.0%"


def test_send_email_accepts_pdf_and_audio(mock_smtp, config, tmp_path):
    pdf = write_file(tmp_path / "brief.pdf", b"%PDF-test")
    audio = write_file(tmp_path / "brief.mp3", b"ID3test")
    send_email(config, "Research", "<p>Approved</p>", attachment_paths=[pdf, audio])
    assert mock_smtp.sent_message_count == 1
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_reports.py tests/test_email_client.py -v`  
Expected: FAIL because report rendering and multi-attachment support are missing.

- [ ] **Step 3: Implement report view models and calculated exhibits**

Templates receive only escaped display models. Exhibit builders calculate weights, valuation ranges, scenario matrices, and source notes from typed rows. They reject values that do not reconcile to stored data.

- [ ] **Step 4: Implement Jinja HTML and WeasyPrint PDF rendering**

Use package-relative templates, a restricted Jinja environment with autoescape, local CSS/assets only, and atomic writes. Include report ID, as-of date, inference mode/model, data freshness, citations, methodology, and research-only disclosure.

- [ ] **Step 5: Extend email without breaking existing callers**

Change `send_email` to accept `attachment_paths: Sequence[Path] | None = None` while retaining `attachment_path` for backward compatibility during this commit. Attach PDF as `application/pdf`, MP3/WAV as audio, and reject unrecognized types with a warning.

- [ ] **Step 6: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_reports.py tests/test_email_client.py -v`  
Expected: PASS.

```bash
git add news_bot/research/reports news_bot/email_client.py tests/research/test_reports.py tests/test_email_client.py
git commit -m "feat: render and deliver institutional research reports"
```

### Task 14: CLI, scheduler, and legacy pipeline integration

**Files:**
- Create: `news_bot/research/cli.py`
- Create: `news_bot/research/scheduler.py`
- Create: `tests/research/test_cli.py`
- Modify: `news_bot/main.py:149-228`
- Modify: `scripts/run.sh`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI and lock tests**

```python
def test_cli_supports_all_approved_workflows():
    for command in ["daily", "weekly", "monthly", "backfill", "dry-run", "regenerate"]:
        exit_code = main([command, "--as-of", "2026-08-24"], services=fake_services())
        assert exit_code == 0


def test_dry_run_never_sends_email(orchestrator, email_sender):
    orchestrator.run_daily(as_of=utc(2026, 8, 24), dry_run=True)
    email_sender.assert_not_called()


def test_second_scheduler_cannot_acquire_active_lease(store):
    first = RunLease.acquire(store, "scheduler", ttl_seconds=3600)
    with pytest.raises(RunAlreadyActive):
        RunLease.acquire(store, "scheduler", ttl_seconds=3600)
    first.release()
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_cli.py -v`  
Expected: FAIL because CLI and scheduler are missing.

- [ ] **Step 3: Implement argparse CLI with explicit exit codes**

Commands initialize config/store once, validate workflow credentials, acquire a lease, invoke one orchestrator workflow, print a redacted summary, and return nonzero only for blocked required outputs or operational failure.

- [ ] **Step 4: Implement APScheduler cadence**

Read daily/weekly/monthly cron settings from configuration. Jobs call the same CLI service functions, use `max_instances=1`, `coalesce=True`, and preserve durable task state after restart.

- [ ] **Step 5: Preserve the legacy newsletter path behind configuration**

When `RESEARCH_ENABLED=false`, `python -m news_bot.main` retains current behavior. When enabled, it invokes the research daily workflow. Update `scripts/run.sh` to use `venv/bin/python` directly so it cannot select the wrong interpreter.

- [ ] **Step 6: Update operating documentation**

Document Flex query setup, secret files, zero-key Ollama mode, external API mode, $5 ceiling, workflow commands, report paths, data backups, and the research-only/no-trading boundary.

- [ ] **Step 7: Run tests and commit**

Run: `venv/bin/python -m pytest tests/research/test_cli.py tests/test_config.py -v`  
Expected: PASS.  
Run: `venv/bin/python -m pytest -q`  
Expected: all tests pass.

```bash
git add news_bot/research/cli.py news_bot/research/scheduler.py tests/research/test_cli.py news_bot/main.py scripts/run.sh README.md
git commit -m "feat: schedule portfolio research workflows"
```

### Task 15: Docker Compose runtime, secrets, health, and backups

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.dockerignore`
- Modify: `.gitignore`
- Create: `scripts/ollama-init.sh`
- Create: `scripts/backup-research-db.sh`
- Create: `tests/research/test_compose_config.py`

- [ ] **Step 1: Write failing static Compose assertions**

```python
def test_compose_has_required_services(compose_config):
    assert {"research-bot", "research-runner", "ollama", "ollama-init", "test"} <= set(compose_config["services"])


def test_research_services_run_as_non_root(compose_config):
    assert compose_config["services"]["research-bot"]["user"] != "root"


def test_persistent_volumes_cover_state_reports_and_models(compose_config):
    mounts = str(compose_config["services"]["research-bot"]["volumes"])
    assert "research-data" in mounts
    assert "research-reports" in mounts
    assert "research-backups" in mounts
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/python -m pytest tests/research/test_compose_config.py -v`  
Expected: FAIL because the current Compose file lacks the research services.

- [ ] **Step 3: Update the production image**

Install WeasyPrint runtime libraries, copy templates/migrations/prompts/scripts, create `/app/data`, `/app/reports`, `/app/cache`, `/app/backups`, `/app/logs`, and `/app/audio_output`, and run under a fixed non-root UID/GID. Add a health command that validates configuration and SQLite connectivity without making paid calls.

- [ ] **Step 4: Replace the Compose topology**

`research-bot` runs the scheduler; `research-runner` is profile `runner`; `ollama` is standard with health check; `ollama-init` waits for health and idempotently pulls `OLLAMA_RESEARCH_MODEL`; `test` builds the development target. Use named volumes for data, reports, cache, backups, logs, audio, and Ollama models. Mount `./secrets` read-only at `/run/secrets`; add `secrets/` to `.gitignore` and provide names only in `env.example`.

- [ ] **Step 5: Add safe model initialization and online SQLite backup scripts**

`ollama-init.sh` checks `ollama list` before pulling. `backup-research-db.sh` invokes the application backup command, writes a timestamped database, verifies `PRAGMA integrity_check`, and retains the most recent 14 backups without deleting outside `/app/backups`.

- [ ] **Step 6: Validate Compose and image behavior**

Run: `docker compose config --quiet`  
Expected: exit 0.  
Run: `docker compose --profile test build test`  
Expected: image builds successfully.  
Run: `docker compose --profile test run --rm test python -m pytest -q`  
Expected: all tests pass.  
Run: `docker compose up -d ollama ollama-init`  
Expected: Ollama becomes healthy and init exits 0.

- [ ] **Step 7: Commit Docker runtime**

```bash
git add Dockerfile docker-compose.yml .dockerignore .gitignore scripts tests/research/test_compose_config.py
git commit -m "feat: dockerize the portfolio research runtime"
```

### Task 16: Golden reports, claim trace, historical replay, and final verification

**Files:**
- Create: `tests/research/golden/source_packet/`
- Create: `tests/research/golden/expected_characteristics.json`
- Create: `tests/research/test_golden_reports.py`
- Create: `tests/research/test_historical_replay.py`
- Create: `tests/research/test_claim_trace.py`
- Create: `tests/research/test_end_to_end.py`
- Modify: `README.md`

- [ ] **Step 1: Build a public, redistributable golden source packet**

Use a small set of primary-source filing and issuer-release excerpts committed as fixtures. Record expected facts, required counterarguments, prohibited future facts, expected evidence IDs, and maximum allowed dollar cost in `expected_characteristics.json`. Do not commit the supplied Morgan Stanley PDF or proprietary text.

- [ ] **Step 2: Write failing golden and replay tests**

```python
def test_every_material_report_claim_resolves_to_passage(golden_run):
    for claim in golden_run.report.material_claims:
        assert golden_run.store.lineage(claim.claim_id)


def test_replay_cannot_use_future_document(replay_runner):
    result = replay_runner.run(as_of=date(2025, 10, 1))
    assert all(document.published_at.date() <= date(2025, 10, 1) for document in result.documents)


def test_synthetic_portfolio_dry_run_never_calls_trade_endpoint(http_history):
    run_synthetic_end_to_end()
    assert not any("order" in request.url.lower() for request in http_history)


def test_monthly_cost_simulation_stops_at_five_dollars(cost_simulator):
    result = cost_simulator.run_month(days=31)
    assert result.paid_cost <= Decimal("5.00")
    assert result.deferred_tasks > 0
```

- [ ] **Step 3: Verify the new tests fail for missing harnesses**

Run: `venv/bin/python -m pytest tests/research/test_golden_reports.py tests/research/test_historical_replay.py tests/research/test_claim_trace.py tests/research/test_end_to_end.py -v`  
Expected: FAIL because golden and replay harnesses are missing.

- [ ] **Step 4: Implement deterministic evaluation harnesses**

Golden tests use fixture-backed providers and score factual accuracy, citation coverage, causal links, counterarguments, rating consistency, inference disclosure, and cost. Historical replay adds `published_at <= as_of` to every document query and fails closed when a source date is unknown.

- [ ] **Step 5: Run the complete verification matrix**

Run: `venv/bin/python -m pytest -q`  
Expected: all tests pass.  
Run: `venv/bin/python -m pytest --cov=news_bot --cov-report=term-missing`  
Expected: no untested critical budget, security, Flex, quality-gate, or routing branch.  
Run: `docker compose config --quiet`  
Expected: exit 0.  
Run: `docker compose --profile test run --rm test python -m pytest -q`  
Expected: all tests pass in Docker.  
Run: `docker compose --profile runner run --rm research-runner dry-run --synthetic-portfolio --as-of 2026-08-24`  
Expected: exit 0, one HTML report and one PDF report, no email, no trade request, and paid cost at or below $5.

- [ ] **Step 6: Inspect generated artifacts manually**

Open the HTML and PDF from the synthetic dry run. Verify thesis-first structure, readable exhibits, citations, inference mode, stale-data behavior, counter-thesis, disclosures, and no exact account identifier or NAV.

- [ ] **Step 7: Update README verification evidence and commit**

Record the exact commands, passing counts, Docker image build date, synthetic report paths, and any expected warnings.

```bash
git add tests/research README.md
git commit -m "test: verify end-to-end research quality and safety"
```

## Final Completion Gate

Before claiming implementation is complete, invoke `superpowers:verification-before-completion` and rerun the Task 16 matrix from a clean worktree. Confirm `git status --short` contains no unintended files, secrets, generated reports, caches, databases, or model artifacts. Then invoke `superpowers:requesting-code-review` for a requirements and maintainability review before offering merge or pull-request options.

## Specification Coverage Check

| Approved requirement | Implemented by |
|---|---|
| Read-only IBKR Flex portfolio snapshots and redaction | Tasks 1, 4 |
| Persistent evidence, theses, scenarios, and recommendation history | Tasks 2, 5 |
| MarketAux, RSS, SEC, IR, FMP, Form D, patent, award, trial, and manual-import sources | Tasks 6, 7 |
| Direct, ETF look-through, and value-chain exposure mapping | Task 8 |
| External API preference, Ollama zero-key fallback, and provider neutrality | Tasks 1, 9, 15 |
| Specialized functional agents with typed handoffs | Task 10 |
| Freshness, citation, eligibility, skeptical-review, and no-rating gates | Task 11 |
| Daily, weekly, monthly, landscape, backfill, and replay workflows | Tasks 12, 14, 16 |
| Event, portfolio, emerging-company, and landscape HTML/PDF reports | Task 13 |
| Research-only ratings and no trading endpoint | Tasks 10, 11, 16 |
| $4 soft limit and $5 hard monthly paid-model ceiling | Tasks 3, 9, 16 |
| Dockerized runtime, standard Ollama backup, secrets, health checks, and backups | Task 15 |
| Claim trace, golden quality evaluation, cost simulation, and historical replay | Task 16 |
