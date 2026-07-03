# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repository. This file
is the harness: it encodes the architecture, the conventions, and the
guardrails so that generated changes match the codebase instead of fighting it.

## What this project is

Meridian is a reference implementation of an internal engineering knowledge
platform. A developer asks a question in natural language; the system routes the
question, retrieves the knowledge they are permitted to see, and returns a
grounded, cited answer. It exists to demonstrate a specific set of engineering
practices end to end - it's a learning/demo artifact, not a product.

The pieces that matter:

- **Semantic router** (`application/router/`) - embedding-based intent routing
  with negative-aware scoring and three ambiguity rules.
- **Routing engine** - turns the router's signal into an action; kept separate
  from scoring so policy and math evolve independently.
- **DSPy-style output contract** (`application/services/query_understanding.py`)
  - a Pydantic schema plus a coercion layer that absorbs LLM format drift.
- **RAG pipeline** (`application/pipelines/rag_pipeline.py`) - retrieval with
  access control applied at query time, grounded generation, mandatory citations.
  Uses the fat/slim flow: cheap slim search, then fat fetch for survivors only.
- **Fat/slim data model** (`domain/models/knowledge.py`) - one document, two
  representations: an indexed slim projection for search, a fat body fetched by
  `JSON.GET` on demand. Search must never read the fat body.
- **DSPy modules** (`application/dspy_modules/`) - real `dspy.Predict` routing
  and `dspy.Refine` self-correction with a grounding reward, run on Grok (xAI).
  The fake provider is the default; the Grok path degrades to it when `dspy` or
  `XAI_API_KEY` is absent.
- **Structured query** (`application/query/`) - a typed filter compiled into an
  ACL-scoped, injection-sanitised RediSearch expression over the service
  catalog. The structured counterpart to RAG: the pattern follows the shape of
  the data.
- **Routing metrics** (`infrastructure/metrics/`) - in-process counters plus an
  optional Redis HASH backend, driving fallback-rate degradation detection.
- **Anaphora entity stack** (`application/services/entity_stack.py`) - a bounded
  LIFO of discussed entities, JSON-serialisable for a Redis TTL cache.
- **Vector store** - two implementations (Redis Stack and in-memory) behind one
  interface.
- **Catalog store** - two implementations (Redis Stack and in-memory) behind one
  interface, for structured service records.

## Architecture: Clean Architecture, dependencies point inward

```
interfaces/      → composition root, CLI, (API)         [outermost]
  application/   → router, engine, pipelines, services
    domain/      → models, interfaces (ports), policies  [innermost]
infrastructure/  → embeddings, vectorstore, redis, llm, observability
```

The rule that must never be violated: **source-code dependencies point
inward.** `domain` imports nothing from `application` or `infrastructure`.
`application` imports from `domain` only. `infrastructure` implements `domain`
interfaces. The only place concretions are chosen is
`interfaces/composition.py` - the composition root.

When adding a capability, add the *port* (an ABC or Protocol) in
`domain/interfaces`, implement it in `infrastructure`, and wire it in the
composition root. Do not let application code import an infrastructure module
directly.

## Conventions

- **Language:** all code, comments, docstrings, and identifiers in English.
- **Docstrings:** every module, class, and public function has one. Google style
  (see `pyproject.toml`). Docstrings explain *why*, not just *what*.
- **Type hints:** required on all function signatures (`mypy` enforces).
- **Formatting & linting:** `ruff` for both. Run `make check` before committing.
- **No I/O in the domain or in the scoring math.** Those layers must stay pure
  and unit-testable without mocks.
- **Injection over instantiation:** components receive their collaborators in
  `__init__`; they do not construct them. This keeps everything testable.

## Testing

- Unit tests cover the pure pieces (scoring, routing engine, coercion) with
  hand-built inputs - fast, no I/O.
- Integration tests wire the whole flow through the in-memory backend with fake
  providers.
- The access-control guarantee has dedicated tests
  (`tests/integration/test_ask_flow.py`): a user must never retrieve a chunk
  outside their groups, and a user with no groups retrieves nothing.
- Run `make test`. Keep the suite green; add tests with any behavioural change.

## Guardrails (do not regress these)

1. **Access control is a retrieval-time filter, never a post-filter.** Do not
   introduce a code path that fetches chunks and filters them afterwards. The
   same applies to structured queries: the visibility clause is always prepended
   by the builder, never applied after execution.
2. **Citations are mandatory on grounded answers.** Do not return a grounded
   answer without its sources.
3. **The router's scoring math stays pure.** No I/O, no logging with side
   effects, in `application/router/scoring.py`.
4. **Structured queries are always sanitised and ACL-scoped.** Every catalog
   query goes through the builder (which prepends visibility) and the sanitiser
   (which rejects injection). Never hand-build a RediSearch string.
5. **Config comes from the environment.** No hard-coded endpoints, secrets, or
   tunables outside `infrastructure/config/settings.py`.
6. **The fake providers must keep the whole system runnable with zero setup.**
   Any change must preserve `python -m meridian.interfaces.cli.main --demo`
   working with no network and no credentials. The Grok/DSPy backend is opt-in
   and must always fall back to the fake provider when unavailable.
7. **Metrics must never break the request path.** Backend failures are
   swallowed; durable writes stay off the hot path.
8. **Search reads slim, never fat.** The slim projection serves search and
   ranking; the fat body is fetched by id only for survivors entering the
   context. Do not read fat bodies during search.

## How to run

```bash
make demo       # end-to-end scripted demo, fake providers, in-memory store
make acl-demo   # access-control filter shown in isolation
make test       # the test suite
make redis-up   # start Redis Stack, then MERIDIAN_BACKEND=redis make demo
```

## Where to look first

- New to the code? Read `domain/models` then `domain/interfaces` - they define
  the whole vocabulary.
- Understanding routing? `application/router/scoring.py` (the math), then
  `semantic_router.py` (the orchestration), then `routing_engine.py` (the
  policy).
- Understanding the swap points? `interfaces/composition.py` - every
  fake-versus-real decision is one `if` there.
