# Contributing to Meridian

> 🇧🇷 [Leia em Português](./CONTRIBUTING.pt-BR.md)

Meridian is a learning/demo artifact, not a product. Contributions that sharpen its educational value
are welcome: cleaner explanations, more idiomatic code, better test coverage, and
corrections when the implementation drifts from the principles it claims to demonstrate.

## Before you start

Read `CLAUDE.md` - it encodes the architecture rules and guardrails that any
change must respect. A pull request that violates them will not be merged.

The most important constraints:

1. **Dependencies point inward.** `domain` imports nothing from `application` or
   `infrastructure`. `application` imports from `domain` only. Concretions are
   chosen exclusively in `interfaces/composition.py`.
2. **Access control is a retrieval-time filter.** Never introduce a code path that
   fetches then filters; the ACL must be inside the search.
3. **Citations are mandatory on grounded answers.**
4. **The fake providers must keep `make demo` running with zero setup.** Any
   change must leave the demo runnable with no credentials and no Docker.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned exactly
in `pyproject.toml`, resolved in `uv.lock` for a reproducible environment.

```bash
make install   # uv sync --extra dev
make demo      # verify the system works end to end
make test      # 52 tests, all must pass
```

Add a dependency with `uv add <package>` (or `uv add --optional <extra> <package>`
for an optional extra) - it updates `pyproject.toml` and `uv.lock` together. Don't
hand-edit version pins; run `uv lock` after any manual `pyproject.toml` change.

## Making changes

- Add a test for every behavioural change. Pure functions get unit tests
  (fast, no I/O). End-to-end flows get integration tests through the in-memory
  backend.
- Run `make check` (lint + typecheck + tests) before opening a PR. CI runs the
  same gate.
- Keep docstrings in Google style. Explain *why*, not just *what*.
- Type hints are required on all public function signatures; `mypy` enforces this.

## Commit style

Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
`chore:`. One logical change per commit.

## Opening a pull request

- Describe what changed and why in the PR body.
- Link to the relevant guardrail in `CLAUDE.md` if your change touches a
  critical path (access control, citations, scoring purity, config).
- Keep PRs small. A focused, reviewable diff is better than a large one.
