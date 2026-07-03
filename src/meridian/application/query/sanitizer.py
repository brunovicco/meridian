"""Sanitisation and injection protection for RediSearch queries.

User- and LLM-derived text can reach the query layer. RediSearch has a rich
query syntax, and an unsanitised string can smuggle in aggregation or command
verbs that change what the search does - expanding result limits, reordering,
or escaping the intended scope. This module is the guard: it rejects the
dangerous verbs, caps the length, and strips control characters, failing safe
to a match-all wildcard rather than executing anything suspicious.

Modelled on the production sanitiser used in the system this reference is based
on. The forbidden set covers the RediSearch/AGGREGATE surface that could alter
query structure; the length cap is a blunt buffer-overflow guard; the control
character check stops a query being split across the wire.
"""

SEARCH_MAX_QUERY_LENGTH = 512
"""Maximum accepted query length; longer inputs fail safe to a wildcard."""

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

    Rejects SQL-like forbidden verbs, over-length input (buffer-overflow guard),
    and non-printable control characters. On any of these it returns the
    match-all wildcard ``*`` rather than the suspicious query - fail safe, not
    fail open.

    :param raw_query: The raw search string proposed by the application flow.
    :returns: The cleaned query, or ``*`` if it was classified as unsafe.
    """
    sanitized = (raw_query or "").strip()

    if not sanitized:
        return "*"

    if len(sanitized) > SEARCH_MAX_QUERY_LENGTH:
        return "*"

    # Control characters could split the command across the wire.
    if any(ord(character) < 32 for character in sanitized):
        return "*"

    upper = sanitized.upper()
    for forbidden in SEARCH_FORBIDDEN_SUBSTRINGS:
        if forbidden in upper:
            return "*"

    return sanitized
