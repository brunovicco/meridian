# Security Policy

## Scope

Meridian is a reference implementation and learning artifact, not a production
service. it's not deployed publicly. Security reports are most valuable when they
concern patterns the codebase teaches. For example, if a guardrail described in
`CLAUDE.md` (access-control filtering, query sanitisation, injection prevention)
can be bypassed.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Email the maintainer directly at the address on the GitHub profile. Include:

- A description of the vulnerability and which component is affected.
- Steps to reproduce, or a minimal proof-of-concept.
- The potential impact and, if known, a suggested fix.

You will receive an acknowledgment within 48 hours and a resolution timeline
within 7 days.

## Out of scope

- Vulnerabilities requiring physical access to the machine.
- Denial-of-service attacks against the local demo process.
- Theoretical attacks with no practical exploitation path.

## Credentials

All external service credentials (Azure OpenAI, xAI/Grok) are supplied through
environment variables. Never commit a `.env` file with real keys; only
`.env.example` (with placeholder values) belongs in version control.
