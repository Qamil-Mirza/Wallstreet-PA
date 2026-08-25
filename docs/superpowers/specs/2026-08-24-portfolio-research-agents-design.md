# Portfolio-Aware Agentic Research System Design

**Date:** 2026-08-24
**Status:** Approved in design discussion; awaiting review of this written specification

## 1. Purpose

Extend the existing daily finance newsletter into a portfolio-aware qualitative research system. The system will read an Interactive Brokers portfolio through the read-only Flex Web Service, maintain cumulative industry and company research, and publish institutional-style reports whose claims are traceable to evidence.

The product has three connected jobs:

1. Find and explain news that is directly relevant to the portfolio.
2. Discover broader changes one or two layers away in an industry's value chain, including emerging private and early public companies.
3. Produce research-only buy, hold, sell, or no-rating views for investable public securities, supported by fundamentals, valuation, scenarios, contrary evidence, and portfolio constraints.

The target style is a thesis-first research note such as the locally supplied Morgan Stanley industry update: concise interpretation, causal decomposition, cross-company read-through, quantified exhibits, explicit analyst judgment, and source attribution. The system must reproduce those characteristics without copying proprietary wording or branding.

## 2. User and Investment Context

- Single-user, personal research system.
- Cash account with less than $5,000 of NAV at the time of design.
- Capital is dedicated investment capital and is not expected to fund living expenses or a major purchase in the next 12–24 months.
- Security recommendation horizon: 6–24 months.
- Industry landscape horizon: 5–10 years.
- Recommendations are limited to long-only, liquid stocks and unleveraged ETFs. Fractional-share feasibility may be considered.
- Margin, shorts, options, futures, leveraged or inverse funds, OTC/penny stocks, and automated order placement are out of scope.
- Private companies may be assessed as emerging or watch-list entities, but never receive buy/sell ratings.

## 3. Goals

- Make portfolio holdings, weights, cash, cost basis, and correlated exposures part of research planning.
- Maintain persistent evidence, theses, scenarios, recommendations, and report history instead of restarting from blank prompts.
- Specialize agents by function and use deterministic orchestration with typed handoffs.
- Generate HTML email and archival PDF reports in a consistent institutional style.
- Keep exact portfolio values and all brokerage credentials local.
- Keep paid model API usage below a hard monthly ceiling of $5.
- Dockerize the complete runtime, including optional local inference and persistent data.
- Preserve the existing newsletter's source ingestion, email, summary validation, and optional audio capabilities where they remain useful.

## 4. Non-Goals

- Placing, staging, or transmitting trades.
- Intraday trading signals or technical-analysis-driven recommendations.
- Personalized tax, legal, or accounting advice.
- Pretending sparse private-company data is complete or current.
- Generating a rating when evidence is insufficient.
- Building a multi-user SaaS platform, web dashboard, or microservice fleet.
- Purchasing expensive proprietary news or private-market data in the initial implementation.

## 5. Chosen Architecture

Use a deterministic role-based agent pipeline with a shared SQLite research store.

```text
IBKR Flex + research sources
          |
          v
normalization and evidence ingestion
          |
          v
SQLite research and thesis memory
          |
          v
specialist agent workflow
          |
          v
skeptical review and publication gates
          |
          v
HTML email + archival PDF + optional audio
```

Agents are logical roles inside one Python application. They are not independent containers or free-chatting autonomous processes. The orchestrator creates bounded tasks, validates structured outputs, records lineage, enforces dependencies and budget, and determines whether work can be published.

This approach was selected over:

- A single autonomous researcher, which would be simpler but inconsistent and difficult to audit.
- A full agent framework with dedicated vector and graph infrastructure, which would add unnecessary operational complexity for a single-user deployment.

## 6. Core Components

### 6.1 Portfolio Gateway

The portfolio gateway uses IBKR Flex Web Service Version 3 to request and retrieve a configured statement. It normalizes:

- account-scoped snapshot timestamp;
- NAV and available cash;
- positions and quantities;
- cost basis when present;
- currency and FX-to-base information;
- symbol, description, asset class, conid, ISIN, FIGI, CUSIP, and underlying identifiers when present.

Account identifiers are replaced by a local opaque identifier before persistence. The Flex token and query ID are read from secret files or environment variables and never stored in SQLite or logs.

The last successful snapshot is retained. If synchronization fails, research may continue with a stale-data banner, but portfolio-sizing guidance is suppressed.

### 6.2 Entity and Exposure Mapper

The mapper resolves positions and discovered companies into canonical entities and relationships:

- security to issuer;
- issuer to industries and technologies;
- supplier, customer, competitor, complement, and substitute relationships;
- ETF to disclosed underlying holdings where available;
- direct portfolio exposure and first- or second-order value-chain exposure.

Relationships store direction, evidence, as-of date, confidence, and provenance. The mapper must not infer a supply-chain link solely from semantic similarity.

### 6.3 Source Connectors

Initial connectors are:

- existing MarketAux and RSS news sources;
- SEC EDGAR company submissions, filings, and structured XBRL facts;
- SEC Form D structured data for private capital formation;
- company investor-relations releases and filings;
- Financial Modeling Prep free endpoints where their current license and quota permit personal use;
- USPTO Open Data Portal for patent and application signals;
- SBIR/STTR award downloads or API when operational;
- USAspending contracts, grants, and awards;
- ClinicalTrials.gov for relevant healthcare and biotechnology landscapes.

Connectors for proprietary sources such as Crunchbase remain optional. They must support manual imports as well as APIs so the core system never depends on an enterprise license.

Every connector returns a normalized source document with source type, canonical URL or local document identifier, publisher, publication and retrieval timestamps, content hash, raw-content location, and extraction status.

### 6.4 Evidence and Research Store

SQLite is the system of record. Raw documents and generated artifacts live in mounted filesystem volumes; SQLite stores metadata, normalized content references, and lineage.

The logical schema includes:

- `portfolio_snapshots` and `positions`;
- `entities`, `securities`, and `relationships`;
- `source_documents` and `document_passages`;
- `claims` and `claim_evidence`;
- `industry_theses`, `scenarios`, and thesis revisions;
- `recommendations` and recommendation revisions;
- `research_tasks` and `agent_runs`;
- `reports` and report sections;
- `model_usage` and the monthly budget ledger.

Claims are atomic and labeled as fact, company guidance, third-party estimate, or agent inference. Each claim records freshness, confidence, status, and supporting or contradicting passages. Thesis and recommendation revisions are append-only; superseded records remain available for audit and historical replay.

SQLite migrations are explicit and versioned. Concurrent writes use short transactions, WAL mode, and application-level orchestration to avoid multiple report writers.

### 6.5 Model Gateway

The model gateway exposes provider-neutral operations for structured generation. Initial production support targets OpenAI's Responses API, with an Ollama-compatible provider retained as an optional offline fallback. The interface permits later Anthropic or Gemini adapters without changing agent logic.

The initial OpenAI routing is:

- GPT-5.6 Luna for high-volume triage and straightforward extraction;
- GPT-5.6 Terra for evidence synthesis and routine analyst tasks;
- GPT-5.6 Sol for research direction, difficult scenario analysis, final recommendation judgment, skeptical review, and final synthesis.

Model names, prices, effort levels, timeouts, and per-role routes are configuration rather than hard-coded policy. Production uses pinned model configuration and records the exact provider/model for every agent run.

Ollama is not the primary research engine. It is an optional Docker profile for offline operation, private preprocessing, and budget-exhaustion fallback. A consumer chat subscription is not treated as API access.

## 7. Specialist Agents

Every agent accepts a versioned typed input, returns a versioned typed result, and cites evidence IDs rather than free-form URLs.

### 7.1 Research Director

- Converts portfolio exposures, report schedules, material events, and thesis gaps into research tasks.
- Defines the question, scope, expected output, admissible sources, and completion criteria.
- Selects which tasks justify paid frontier inference.
- Does not perform final research or publish reports.

### 7.2 Portfolio Mapper

- Calculates position and sector weights, cash, concentration, ETF overlap, and correlated exposures.
- Maps holdings to value chains and candidate adjacent industries.
- Uses exact values locally, but emits rounded weight bands for external model prompts.

### 7.3 Event Scout

- Finds portfolio-relevant news, filings, earnings, policy changes, and broader market events.
- Scores novelty, materiality, portfolio relevance, source quality, and likely thesis impact.
- Avoids sending duplicate or immaterial stories downstream.

### 7.4 Emerging-Company Scout

- Finds private and early public companies through funding, patents, grants, government contracts, product activity, hiring, trials, partnerships, and industry events.
- Maps each company to a value-chain role and evidence-backed adoption signals.
- Produces emerging/watch assessments only.

### 7.5 Evidence Analyst

- Extracts atomic claims and exact supporting passages.
- Separates reported facts, guidance, third-party estimates, and inference.
- Records corroboration, contradiction, freshness, and uncertainty.
- Cannot create an uncited factual claim.

### 7.6 Industry Strategist

- Maintains 5–10 year industry landscapes and value-chain maps.
- Describes base, upside, and downside scenarios; adoption drivers; bottlenecks; disruption paths; and signposts.
- Explains how emerging companies may change incumbent economics and which public securities may benefit or be threatened.

### 7.7 Fundamental Analyst

- Evaluates investable public companies over 6–24 months.
- Covers business quality, financial trajectory, balance-sheet risk, valuation range, catalysts, industry positioning, and portfolio fit.
- Emits buy, hold, sell/reduce, or no rating with confidence and explicit assumptions.

### 7.8 Skeptical Reviewer

- Challenges causality, stale inputs, unsupported valuation assumptions, unmodeled risks, correlated exposure, selection bias, and missing counterarguments.
- Returns `pass`, `revise`, or `block` with actionable findings.
- A blocked result cannot be published as approved research.

### 7.9 Research Editor

- Converts approved structured research into a consistent report.
- Preserves the distinction between facts, estimates, and judgments.
- May clarify or reorganize but cannot introduce new factual claims.
- Produces citation lists, exhibit notes, methodology, and disclosures.

## 8. Workflows and Cadence

### 8.1 Daily Monitoring

1. Refresh the portfolio snapshot.
2. Ingest new documents since the prior successful run.
3. Resolve entities, deduplicate documents, and update evidence.
4. Score events for materiality and thesis impact.
5. Create an event update only when a threshold is met.
6. Send the daily email with portfolio-relevant news, broader industry signals, and explicit omissions or stale-data warnings.

Daily monitoring does not recompute all company recommendations.

### 8.2 Weekly Portfolio Research Brief

- Summarizes thesis changes and relevant new evidence.
- Reviews concentration, cash, sector overlap, and investable opportunity set.
- Re-evaluates a security only when scheduled or triggered by material evidence or valuation movement.
- Includes current research ratings and the reason each changed or remained unchanged.

### 8.3 Monthly Emerging-Company Monitor

- Updates emerging-company and technology maps.
- Highlights new funding, technical, adoption, contract, clinical, and hiring signals.
- Connects private-company activity to investable public beneficiaries, threats, suppliers, and complements.

### 8.4 Industry Landscape Reports

- A foundational report is created for each portfolio-linked industry.
- One industry is initialized or deeply refreshed per month under the normal budget.
- Full landscape rewrites occur quarterly when budget and material changes justify them.
- Interim event notes update individual claims, scenarios, and signposts without rewriting the foundation.

### 8.5 Research Initialization

Historical initialization is staged across successive months to honor the $5 monthly ceiling. A larger one-time backfill is allowed only after an explicit configuration change by the user; there is no implicit overspend.

All workflows are idempotent by portfolio as-of date, source hash, task definition, and report period. A run lock prevents overlapping scheduled workflows from publishing duplicates.

## 9. Reports

### 9.1 Event Update

- Two to five pages.
- Thesis-led headline and immediate interpretation.
- Event decomposition and causal mechanisms.
- Cross-company and portfolio read-through.
- Quantified exhibits where evidence supports them.
- Thesis changes, unchanged assumptions, unresolved questions, and next signposts.

### 9.2 Weekly Portfolio Research Brief

- Portfolio summary using rounded values in external-model-authored prose.
- Directly relevant news and thesis impact.
- Broader value-chain and landscape developments.
- Buy/hold/sell/no-rating research views with change history.
- Concentration and correlation observations.

### 9.3 Industry Landscape

- Industry definition and value chain.
- Current structure, profit pools, and bottlenecks.
- Emerging technologies and companies.
- 5–10 year base, upside, and downside scenarios.
- Adoption signposts and invalidation conditions.
- Public beneficiaries, threats, and portfolio relevance.

### 9.4 Emerging-Company Monitor

- Company and technology map.
- Evidence-backed operating and adoption signals.
- Confidence and data limitations.
- Public-market translation without private-company trade ratings.

Reports render from Jinja templates to HTML and then to PDF using a reproducible renderer installed in the container. Exhibits are generated from normalized data, not model-authored HTML tables. The existing email delivery path is reused. Optional TTS may summarize approved report content but never gates publication.

## 10. Recommendation Semantics

- **Buy/Add:** the 6–24 month risk/reward and valuation range are attractive relative to alternatives, the thesis is evidence-backed, and portfolio constraints permit additional exposure.
- **Hold:** the thesis remains intact but valuation, uncertainty, or portfolio fit does not support adding or reducing.
- **Sell/Reduce:** the thesis is impaired, valuation exceeds a defensible range, a superior substitute exists, or concentration/liquidity risk dominates.
- **No Rating:** evidence, freshness, comparability, or valuation confidence is insufficient.

Every rated recommendation includes:

- thesis and time horizon;
- valuation range and assumptions;
- supporting evidence;
- catalysts and signposts;
- counter-thesis and principal risks;
- confidence and data freshness;
- what would change the view;
- portfolio feasibility without generating an order.

The default freshness policy is:

- portfolio snapshot no older than 36 hours;
- market price from the latest completed trading session and no older than four calendar days;
- latest required periodic issuer filing available as of the research date;
- valuation inputs labeled with their effective date;
- event evidence evaluated against a connector-specific materiality window, with seven days as the default for daily monitoring.

Weekends, exchange holidays, foreign issuers, and non-US filing schedules use calendar-aware overrides. If the system cannot determine whether a required filing or price should be available, it returns no rating rather than assuming the data is current.

## 11. Publication and Quality Gates

- Every material factual or quantitative claim resolves to one or more stored evidence passages.
- Consequential conclusions require a primary source or two independent corroborating sources.
- Search-result snippets are discovery aids, not admissible evidence.
- Source and data dates appear in the report.
- Unsupported claims, missing citations, and unresolved contradictions block affected sections.
- The skeptical reviewer must pass recommendations, foundational reports, and portfolio briefs.
- The editor cannot add facts after skeptical review.
- Historical replay restricts agents to evidence available as of the replay date.
- Generated charts and tables reconcile to stored normalized data.

## 12. Model Budget

Paid model spend uses a local ledger based on provider-reported token usage and configured current prices.

- Soft monthly limit: $4.00.
- Hard monthly limit: $5.00.
- Before a paid request, the gateway reserves a pessimistic estimated cost.
- After completion, the reservation is reconciled to actual usage.
- A request that could cross the hard limit is not sent.
- If provider usage is missing or current pricing is not configured, the paid request is rejected safely.
- Pricing configuration has an effective date. Expired pricing blocks paid calls until it is refreshed, preventing a stale price table from bypassing the dollar ceiling.
- At the soft limit, only critical final judgment or review tasks may use paid models.
- At the hard limit, tasks are deferred or routed to Ollama when their risk level permits.
- Deferred work remains visible in the report and task queue.

The initial planning envelope, using prices available on 2026-08-24, is:

- Luna: approximately 5 million input and 300,000 output tokens, $1.36;
- Terra: approximately 500,000 input and 50,000 output tokens, $1.60;
- Sol: approximately 300,000 input and 20,000 output tokens, $1.60;
- reserve: $0.44.

These are routing targets, not guaranteed allocations. Configuration and tests use the hard dollar ceiling as the authoritative requirement.

## 13. Failure Handling

- **IBKR unavailable:** retry with bounded backoff, then use the last successful snapshot with a visible age; suppress sizing guidance.
- **Source unavailable:** record the failure, continue independent sources, and block conclusions that depend on the missing source.
- **Rate limiting:** honor retry metadata and connector-specific pacing; do not bypass provider limits.
- **Invalid structured model output:** validate, retry once with validation feedback, then fall back or block.
- **Contradictory evidence:** retain both claims, surface the contradiction, and reduce confidence or return no rating.
- **Budget exhausted:** defer frontier work or use approved local fallback; never overspend silently.
- **PDF rendering failure:** retain HTML, record the failure, and send HTML only when the underlying report passed quality gates.
- **Email failure:** preserve the approved report and retry delivery without rerunning research.
- **Partial daily run:** may publish with explicit omissions. Weekly, monthly, foundational, and recommendation outputs require all critical gates.

## 14. Security and Privacy

- IBKR access is read-only and no trading endpoint is implemented.
- Flex, SMTP, and model credentials are Docker secrets or mounted secret files.
- Credentials, account IDs, exact NAV, and exact position values never enter model prompts, trace logs, reports, or source caches.
- External prompts may include ticker, issuer, and rounded exposure bands only.
- Logs use run IDs and redacted structured metadata.
- Downloaded documents are treated as untrusted data and cannot alter system instructions or tool permissions.
- HTML and PDF content is escaped and sanitized before rendering.
- Containers run as a non-root user with only required writable volumes.

## 15. Docker Topology

Docker Compose defines:

- `research-bot`: the Python application, scheduler, connectors, orchestration, report rendering, email, and optional audio.
- `research-runner`: an on-demand Compose profile for `daily`, `weekly`, `monthly`, `backfill`, `dry-run`, and report-regeneration commands.
- `ollama`: an optional profile with a persistent model volume; it is not required for external-API-first production operation.

Named volumes persist:

- SQLite state;
- raw source cache;
- generated reports;
- logs and redacted traces;
- audio output;
- optional Ollama models.

The image includes reproducible PDF-rendering, extraction, audio, and system dependencies. Compose includes health checks, dependency ordering, restart policy, secret mounts, and explicit service commands. Database backups use SQLite's online backup mechanism and are written to a separate mounted backup directory.

The scheduler and on-demand runner share an application lock so they cannot publish the same report concurrently.

## 16. Configuration

Configuration remains environment-driven and typed. New settings cover:

- Flex endpoint, token secret path, query ID, polling timeout, and snapshot staleness;
- enabled research connectors and connector credentials;
- report cadences and materiality thresholds;
- provider/model routing by agent role;
- reasoning effort, token limits, timeouts, and retry policy;
- $4 soft and $5 hard monthly API limits;
- storage, report, cache, backup, and template paths;
- optional Ollama profile and fallback policy;
- dry-run and historical replay dates.

Configuration validation fails before network activity when a selected workflow lacks required credentials or has unsafe budget values.

## 17. Testing and Evaluation

### 17.1 Unit Tests

- Flex request, polling, XML parsing, normalization, redaction, and staleness.
- Entity resolution and relationship provenance.
- Source hashing, deduplication, passage extraction, and claim lineage.
- Materiality and portfolio-relevance scoring.
- Agent input/output schemas and orchestration state transitions.
- Recommendation and publication gates.
- Model routing, reservation, reconciliation, soft-limit behavior, and hard-limit enforcement.
- Report calculations, exhibits, HTML escaping, and citations.

### 17.2 Contract and Integration Tests

- Stored external API fixtures prevent tests from spending money or depending on provider availability.
- SQLite migration and backup/restore tests.
- Docker fresh-start, health, persistent-volume, one-shot runner, and scheduled workflow tests.
- HTML and PDF generation tests.
- SMTP tests use a local fake server.
- End-to-end dry runs use synthetic portfolios before real Flex credentials.

### 17.3 Research Evaluations

A golden evaluation set contains source packets and expected characteristics derived from public primary materials and the supplied report style. It scores:

- factual and numerical accuracy;
- citation completeness and passage entailment;
- causal depth and portfolio read-through;
- treatment of uncertainty and contrary evidence;
- rating consistency and change justification;
- report structure and clarity;
- token and dollar cost.

Historical replay checks for future-information leakage. Provider comparisons use the same source packet and rubric before changing production routes.

## 18. Delivery Sequence

Implementation will be planned and executed in dependency order:

1. Domain models, SQLite migrations, configuration, and cost ledger.
2. IBKR Flex ingestion and portfolio snapshots.
3. Source-document ingestion and evidence lineage.
4. Provider-neutral model gateway and budget enforcement.
5. Agent contracts, deterministic orchestration, and quality gates.
6. Industry, emerging-company, portfolio, and recommendation workflows.
7. HTML/PDF reports and existing email/audio integration.
8. Docker Compose runtime, scheduler, secrets, health checks, and backups.
9. Golden evaluations, historical replay, and end-to-end dry runs.

Each step must preserve or extend the existing test suite. The implementation plan may split these into smaller commits but may not omit an approved component.

## 19. Success Criteria

The design is complete when the implemented system can:

1. Run from Docker Compose with persistent state and no host-specific Python setup.
2. Sync a read-only IBKR Flex portfolio without leaking secrets or exact account values.
3. Build and revise evidence-backed company and industry theses over time.
4. Discover and map emerging companies using accessible public signals.
5. Generate cited event, portfolio, emerging-company, and landscape reports in HTML and PDF.
6. Produce gated buy/hold/sell/no-rating research for eligible public securities without placing orders.
7. Trace every material published claim and exhibit to stored evidence.
8. Degrade safely under stale data, unavailable sources, invalid model output, and provider failure.
9. Prevent cumulative paid model spend from crossing $5 in a calendar month.
10. Pass unit, contract, Docker integration, golden-report, claim-lineage, and historical-replay tests.

## 20. Source Notes Used During Design

- IBKR Flex Web Service and Open Positions documentation.
- SEC EDGAR developer resources and Form D data sets.
- USPTO Open Data Portal, SBIR/STTR, USAspending, and ClinicalTrials.gov API documentation.
- Current OpenAI model and API pricing documentation as of 2026-08-24.
- Anthropic and Gemini model documentation used only to ensure the provider abstraction remains viable.
- The user-supplied Morgan Stanley PDF was inspected as a structural and editorial reference, not as reusable content.
