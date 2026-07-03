# AGENTS.md

Vendor-neutral guidance for coding agents working in this repository.

The full guidance lives in [`CLAUDE.md`](./CLAUDE.md). It applies to any agent,
not only Claude Code. In short:

- **Architecture is Clean Architecture; dependencies point inward.** `domain`
  depends on nothing; `application` depends only on `domain`; `infrastructure`
  implements `domain` interfaces; concretions are chosen only in
  `src/meridian/interfaces/composition.py`.
- **Everything in English, fully docstringed, type-hinted, `ruff`-clean.**
- **Five guardrails must not regress:** access control is a retrieval-time
  filter; citations are mandatory on grounded answers; the scoring math stays
  pure; config comes from the environment; the fake providers keep the system
  runnable with zero setup.
- **Run `make check`** (lint + typecheck + test) before proposing changes.

Read `CLAUDE.md` in full before making non-trivial edits.
