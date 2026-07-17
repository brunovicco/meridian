"""Sanitisation and injection protection for RediSearch queries.

User- and LLM-derived text can reach the query layer. RediSearch has a rich
query syntax, and an unsanitised string can smuggle in aggregation or command
verbs that change what the search does - expanding result limits, reordering,
or escaping the intended scope. This module is the guard: it rejects dangerous
verbs, caps the length, and rejects control characters. Invalid input becomes
an impossible ACL-scoped query, so validation can never widen visibility.

Modelled on the production sanitiser used in the system this reference is based
on. The forbidden set covers the RediSearch/AGGREGATE surface that could alter
query structure; the length cap is a blunt buffer-overflow guard; the control
character check stops a query being split across the wire.
"""

import re

SEARCH_MAX_QUERY_LENGTH = 512
"""Maximum accepted query length."""

NO_MATCH_QUERY = "@visibility:{__none__}"
"""Impossible scoped query used whenever validation fails."""

SEARCH_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "RETURN",
    "LIMIT",
    "SORTBY",
    "GROUPBY",
    "REDUCE",
    "APPLY",
    "FILTER",
    "LOAD",
    "FT.SEARCH",
    "FT.AGGREGATE",
    "FT.CREATE",
    "FT.DROP",
    "DEL",
    "SET",
    "EVAL",
)
"""Verbs that could alter query structure or scope; their presence fails safe."""


def sanitize_search_query(raw_query: str) -> str:
    """Rigorously sanitise a query destined for RediSearch.

    Rejects command-like forbidden verbs, over-length input, non-printable
    control characters, and expressions without the mandatory visibility
    prefix. Rejected input becomes :data:`NO_MATCH_QUERY`, which matches no
    authorised records instead of widening the query to ``*``.

    :param raw_query: The raw search string proposed by the application flow.
    :returns: The cleaned query, or :data:`NO_MATCH_QUERY` when unsafe.
    """
    sanitized = (raw_query or "").strip()

    if not sanitized:
        return NO_MATCH_QUERY

    if len(sanitized) > SEARCH_MAX_QUERY_LENGTH:
        return NO_MATCH_QUERY

    # Control characters could split the command across the wire.
    if any(ord(character) < 32 for character in sanitized):
        return NO_MATCH_QUERY

    if not sanitized.startswith("@visibility:{"):
        return NO_MATCH_QUERY

    upper = sanitized.upper()
    for forbidden in SEARCH_FORBIDDEN_SUBSTRINGS:
        # Match command tokens, not substrings in legitimate values such as
        # the ACL group ``asset``.
        if re.search(rf"(?<![A-Z0-9_]){re.escape(forbidden)}(?![A-Z0-9_])", upper):
            return NO_MATCH_QUERY

    return sanitized
