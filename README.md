# Meridian

> 🇧🇷 [Leia em Português](./README.pt-BR.md)

A reference implementation of an internal **engineering knowledge platform**: a
developer asks a question in natural language, and the system routes it,
retrieves the knowledge the developer is permitted to see, and returns a
grounded answer with citations.

it's built to demonstrate a specific set of production engineering practices
end to end - semantic routing, an LLM output contract, retrieval-augmented
generation with access control, and clean separation of concerns - on a stack
of Python, Redis Stack, and pluggable embedding/LLM providers.

> **It runs with zero setup.** The default configuration uses a deterministic
> fake embedder, an in-memory vector store, and a rule-based fake LLM, so you
> can see the whole system work with no API key, no network, and no Docker:
>
> ```bash
> uv sync
> uv run python -m meridian.interfaces.cli.main --demo
> ```

---

## What it demonstrates

| Concern | Where | What to look at |
|---|---|---|
| **Semantic router** | `application/router/` | Negative-aware scoring, three ambiguity rules, fingerprint-cached matrices |
| **Routing engine** | `application/router/routing_engine.py` | Signal → decision, kept separate from the math |
| **LLM output contract** | `application/services/query_understanding.py` | Pydantic schema + coercion that absorbs format drift (DSPy-style) |
| **RAG with access control** | `application/pipelines/rag_pipeline.py` | Retrieval-time ACL filter, mandatory citations, honest "I don't know" |
| **Fat/slim data model** | `domain/models/knowledge.py`, stores | Slim projection for search, fat document fetched on demand via JSON.GET |
| **DSPy (real) + Groq** | `application/dspy_modules/` | `dspy.Predict` routing + `dspy.Refine` self-correction with a grounding reward, on Groq; fake fallback by default |
| **Clean Architecture/SOLID** | whole tree | Dependencies point inward; concretions chosen only at the composition root |
| **Twelve-factor config** | `infrastructure/config/settings.py` | All settings from the environment |
| **Redis Stack vector store** | `infrastructure/redis/` | RediSearch KNN with a metadata ACL filter |
| **Structured query (RediSearch)** | `application/query/` | Typed filter → compiled `FT.SEARCH` over the service catalog, ACL-scoped, injection-sanitised |
| **Routing metrics** | `infrastructure/metrics/` | In-process degradation counters plus buffered Redis HASH persistence |
| **Anaphora entity stack** | `application/services/entity_stack.py` | Standalone bounded LIFO primitive, ready for a future conversation adapter |
| **Observability** | `infrastructure/observability/` | Structured event per routing decision and retrieval |

---

## The request flow

```
                    ┌──────────────────────────────────────────────┐
   "how do I        │                 AskService                   │
    configure  ───► │  (application/services/ask_service.py)       │
    auth?"          └───────────────┬──────────────────────────────┘
                                    │
                    1. SemanticRouter.route()    ── layer 1: scores + ambiguity
                                    │
                    2. RoutingEngine.decide()    ── layer 2: signal → action
                                    │
                    3. QueryUnderstanding        ── LLM behind a Pydantic contract
                       (coercion absorbs drift)
                                    │
                    4. RagPipeline.run()         ── ACL-filtered retrieval,
                                    │                 grounded generation, citations
                                    ▼
                             Answer + Citations
```

Two pipelines run at different times, and the code keeps them separate:

- **Ingestion (offline):** documents are chunked, embedded, and indexed with
  their ACL metadata. Seeded from the packaged resources under
  `src/meridian/data/catalog/`.
- **Query (online):** the flow above.

---

## The semantic router, concretely

Each intent is defined by **positive** example phrases (what it looks like) and
**negative** phrases (confusables from other intents). At build time these are
embedded into per-intent matrices and cached in the vector store under a
**SHA-256 fingerprint** of the catalog, thresholds, embedding dimension, and
provider/model identity - change any of those and the cache invalidates.

For a query vector `q`, each intent scores as:

```
score = max(M_pos @ q) − NEG_PENALTY · max(0, max(M_neg @ q))
```

The `max` over rows rewards the closest example (not the average), and the
negative term (clamped at zero, so it only ever subtracts) pushes the boundary
away from known confusables. Then three ambiguity rules run in order:

1. **Per-intent threshold** - top score below the winning intent's threshold.
2. **Absolute floor** (`AMBIG_MIN`) - top score too weak overall, unless the
   margin to the runner-up is comfortable.
3. **Margin** (`AMBIG_DELTA`) - top two too close to separate safely.

The routing engine then turns this signal into one of: route directly, ask for
disambiguation, or fall back to general QA.

> The scoring math lives in `application/router/scoring.py`, is completely pure
> (no I/O), and is unit-tested with hand-built matrices.

---

## Access control is a retrieval-time filter

The single most important security property: **a user never retrieves a chunk
outside their groups, even transiently.** The ACL check is a metadata filter
*inside* the vector search - a RediSearch tag clause combined with the KNN
clause - not a step applied after retrieval, and never delegated to the LLM.

See it in isolation:

```bash
uv run python -m meridian.interfaces.cli.main --acl-demo
```

```
ACL probe: retrieving 'security post mortem for the payments outage root cause'

  [carol groups=security             ] -> Security Post-Mortem, Credential Rotation Guide
  [alice groups=payments,platform    ] -> Payments Service Auth Guide, Transfer API Reference, Database Failover Runbook
  [dan   groups=(no groups)          ] -> (nothing visible)
```

Carol (security) sees the restricted post-mortem; Alice (payments/platform)
never does; Dan (no groups) sees nothing - the filter fails closed.

---

## Structured knowledge is a query problem, not a retrieval problem

Not all knowledge is unstructured prose. "Who owns the payments service" or
"which tier-1 services have no owner" are questions over a **service catalog** -
structured data where the right answer should be query-driven, not a top-K
semantic sample. The structured path executes a bounded catalog query and
explicitly reports when more rows exist beyond the configured result limit.

```bash
uv run python -m meridian.interfaces.cli.main --structured-demo
```

```
[alice] Q: who owns the payments service
    compiled: @visibility:{payments | platform} @domain:{payments}
      - payments-api (team payments, tier1)
      - transfer-service (team payments, tier1)

[bob]   Q: list tier1 services in the gateway domain
    compiled: @visibility:{sre | platform} @domain:{gateway} @tier:{tier1}
      - api-gateway (team platform, tier1)
```

The `ServiceQueryBuilder` (`application/query/`) classifies each filter field as
a **TAG** (exact), **TEXT** (fuzzy with tokenisation rules), or **NUMERIC**
(range) and compiles a RediSearch expression. Two properties are structural: the
visibility clause is always prepended (a caller cannot build an unscoped query),
and the result passes an injection **sanitiser** that rejects aggregation verbs,
over-length input, and control characters - failing closed to an impossible
ACL-scoped query. This is
the same lesson, in code, that once turned a failing RAG-over-tabular-data
approach into a text-to-query one: the pattern follows the shape of the data.

---

## One document, two representations: fat/slim

The same knowledge document lives in Redis Stack as two payloads, each sized for
a different stage. Searching pays for the small one; the large one is fetched
only for the few documents that survive ranking.

```bash
uv run python -m meridian.interfaces.cli.main --fatslim-demo
```

```
fat/slim probe: 'how do I configure authentication...' as alice

  Phase 1 - slim search (cheap, projections only):
    · Payments Service Authentication  [snippet: To configure authentication for the payments...]
    · Credential Rotation              [snippet: To rotate a service credential, generate a new...]
    · Database Failover Procedure      [snippet: For a database failover, first confirm the...]

  Phase 2 - fat fetch (JSON.GET) for survivors only:
    · Payments Service Authentication  owner=payments  updated=2026-02-02  chars=687
    · Credential Rotation              owner=platform  updated=2026-01-30  chars=718
```

The **slim projection** (title, snippet, source, ACL) is an indexed hash the KNN
search returns - small, fast, enough to rank and cite. The **fat document** (full
text plus rich metadata) is a RedisJSON body fetched by `JSON.GET`, and only for
the survivors that will actually enter the generation context. The `RagPipeline`
runs exactly this flow - `search_slim` → select survivors → `fetch_fat` - so the
fat payload is paid for a handful of times per query, not once per candidate.
Each stage gets the payload it needs, right-sized end to end.

---

## Before the split: a flat retrieval model

Alongside the fat/slim index, the same knowledge is also seeded into a
simpler, older model: one flat `KnowledgeChunk` record per document, full
text and citation metadata together, no separate `JSON.GET` fetch. It exists
to make the fat/slim split's cost trade-off concrete by comparison, and it
carries the identical access-control guarantee.

```bash
uv run python -m meridian.interfaces.cli.main --flat-demo
```

```
flat-chunk probe: 'security post mortem for the payments outage root cause'

  [carol groups=security                ] -> Security Post-Mortem, Credential Rotation Guide
  [alice groups=payments,platform       ] -> Payments Service Auth Guide, Database Failover Runbook, Transfer API Reference
  [dan   groups=(no groups)             ] -> (nothing visible)
```

Same probe, same ACL filter applied inside the search - `search_chunks`
instead of `search_slim` - and the same outcome: Carol sees the restricted
post-mortem, Alice never does, Dan sees nothing. The difference is cost, not
security: every candidate here carries its full text into the search index,
so ranking pays for the whole document on every candidate, not just the
survivors. That's the bill the fat/slim split above exists to avoid.

---

## Real DSPy on Groq, with a fake fallback by default

The routing and generation contracts are backed by **real DSPy modules** when you
opt in - `dspy.Predict` for routing and `dspy.Refine` for generation - running on
**Groq** via `GROQ_API_KEY`.

```bash
uv sync --extra groq
export GROQ_API_KEY=gsk_...
MERIDIAN_LLM_BACKEND=groq uv run python -m meridian.interfaces.cli.main --demo
```

The `DSPyRefineModule` is the interesting one: it generates an answer, scores it
with a **grounding reward** (does it cite a source? do its claims overlap the
context? does it avoid unsupported hedging?), and regenerates up to a retry
budget until the score clears the threshold - the self-correction pattern from a
production compliance advisor, adapted to a knowledge domain. The output still
passes the same Pydantic coercion contract as the fake path, so drift is absorbed
identically.

Crucially, **the fake provider is the default**, and the Groq backend degrades to
it gracefully when `dspy` or the key is absent. The zero-setup demo never depends
on the network - you turn Groq on deliberately, with the key in hand.

---

## Observing router health
Every routing decision is recorded by a metrics collector
(`infrastructure/metrics/`). In-process counters drive a fast degradation check
- if too large a share of decisions fall back to the generic route, the router
may be drifting and needs recompilation. An optional Redis backend persists
buffered deltas into a single HASH with `HINCRBY`; a scheduler or shutdown hook
calls `flush()` outside the request path. Its TTL expires the aggregate after 24
hours without writes. Backend failures are swallowed and never break requests.

---

## Running against real infrastructure

The whole point of the abstractions is that the same application code runs
against different backends. To use **Redis Stack** instead of the in-memory
store:

```bash
uv sync --extra redis
docker compose up -d          # Redis Stack on :6379, RedisInsight on :8001
MERIDIAN_BACKEND=redis uv run python -m meridian.interfaces.cli.main --demo
```

To use **Azure OpenAI** embeddings/LLM, set `MERIDIAN_EMBEDDING_BACKEND=azure`
and `MERIDIAN_LLM_BACKEND=azure` and provide the Azure variables (see
`.env.example`). The provider classes in `infrastructure/embeddings/` and
`infrastructure/llm/` carry the production-grade scaffolding (retry with
backoff and jitter, corporate TLS trust) with the SDK call marked as the single
documented gap - filling it in does not touch any other layer.

To use a **free, real semantic embedder** with no credentials at all, set
`MERIDIAN_EMBEDDING_BACKEND=local` and `MERIDIAN_EMBEDDING_DIM=384` after
`uv sync --extra local`. This runs `sentence-transformers/all-MiniLM-L6-v2`
locally (CPU-friendly, ~80MB, cached after first download) via
`SentenceTransformerEmbeddingProvider` - unlike the Azure skeleton, this path
is fully implemented, so it is the fastest way to see real semantic routing
and retrieval behaviour instead of the fake hashing embedder's lexical
approximation.

That substitutability is the Dependency Inversion Principle in practice: swap
happens at `interfaces/composition.py`, one `if` per component, and nothing
upstream changes.

---

## Project layout

```
src/meridian/
  domain/            # models (incl. fat/slim knowledge), interfaces, policy - pure, no I/O
  application/       # router, engine, RAG + structured pipelines, query builder, dspy modules
  infrastructure/    # embeddings, vector/catalog stores, redis, llm (fake/azure/groq), metrics
  interfaces/        # composition root, CLI
  data/catalog/      # packaged intents + knowledge + service catalog resources
tests/               # unit (pure pieces) + integration (full flows, incl. ACL, structured, fat/slim)
```

---

## Development

```bash
make install     # editable install with dev extras
make demo        # scripted end-to-end demo
make acl-demo    # access-control filter in isolation
make structured-demo  # structured query compiled to RediSearch
make fatslim-demo     # fat/slim retrieval split
make flat-demo        # flat, pre fat/slim knowledge-chunk model
make test        # test suite
make check       # lint + typecheck + test
make redis-up    # start Redis Stack
```

---

## Notes on scope

This is a learning/demo artifact. The fake providers are lexical, not semantic - they
exercise the plumbing deterministically, not the quality of a real embedder. The
router thresholds ship with a separate calibration for the fake backend
(`domain/policies`), which is itself a point worth making: **thresholds are a
property of the embedding model, so switching models means recalibrating, not
editing prompts.** The DSPy + Groq path is real (`dspy.Predict` + `dspy.Refine`
with a grounding reward) and runs once you install the `groq` extra and set
`GROQ_API_KEY`; without them the system falls back to the fake provider so the
default demo always runs. The Azure providers are scaffolded to the point where
the only missing piece is the external SDK call.

**Content sources.** The prose under `data/catalog/` is original writing for
this fictional demo, informed by publicly documented engineering practices:
retry/backoff-with-jitter, failover discipline, and SLO/error-budget framing
draw on ideas from Google's *Site Reliability Engineering* book (free at
sre.google/books, CC BY-NC-ND 4.0) and its blameless-postmortem culture;
token-bucket rate limiting draws on public CNCF/Envoy/Kubernetes
rate-limiting documentation (CC BY 4.0); the idempotency-key pattern draws on
publicly documented conventions used by major payment APIs; credential
rotation draws on OWASP secret-management guidance. No text is copied from
these sources - everything here is original paraphrase written for this
repository. The `wiki.internal/...` URLs in the fixtures are fictional
placeholders for an invented internal wiki, not links to the real sources
above.
