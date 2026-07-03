"""In-memory service catalog store.

Implements :class:`CatalogStore` without Redis so the structured-query path runs
anywhere. It evaluates the specific RediSearch clause shapes the
:class:`~meridian.application.query.builder.ServiceQueryBuilder` emits - a
visibility tag clause, exact tag clauses, fuzzy text clauses, and numeric
ranges - against in-memory records.

This is deliberately a *focused* evaluator, not a general RediSearch engine: it
understands exactly the queries this system builds and nothing more. The Redis
implementation runs the identical expressions against a real ``FT.SEARCH`` index;
having both behind one interface is what lets the same application code target
either backend.
"""

import re

from meridian.domain.interfaces import CatalogStore, Tracer
from meridian.domain.models.service_catalog import ServiceRecord

_TAG_CLAUSE = re.compile(r"@(\w+):\{([^}]*)\}")
_TEXT_CLAUSE = re.compile(r"@(\w+):\(([^)]*)\)")
_NUMERIC_CLAUSE = re.compile(r"@(\w+):\[(\S+)\s+(\S+)\]")


class InMemoryCatalogStore(CatalogStore):
    """Dictionary-backed catalog with a focused RediSearch-clause evaluator."""

    def __init__(self, *, tracer: Tracer) -> None:
        """Initialise an empty catalog.

        :param tracer: Structured observability sink.
        """
        self._tracer = tracer
        self._services: list[ServiceRecord] = []

    def upsert_services(self, services: list[ServiceRecord]) -> None:
        """Append service records to the in-memory catalog."""
        self._services.extend(services)
        self._tracer.event("memory.catalog.upserted", count=len(services), total=len(self._services))

    def execute(self, compiled_query: str, *, limit: int = 25) -> list[ServiceRecord]:
        """Evaluate the compiled query against the in-memory records.

        Every clause the builder can emit is ANDed together (RediSearch default),
        except the visibility clause, whose pipe-separated groups are ORed. A
        record matches only if it satisfies all clauses.
        """
        matches = [s for s in self._services if self._matches(s, compiled_query)]
        self._tracer.event("memory.catalog.executed", query=compiled_query[:120], matched=len(matches))
        return matches[:limit]

    def _matches(self, service: ServiceRecord, query: str) -> bool:
        """Return whether a record satisfies every clause in the query."""
        # Visibility clause (OR over groups).
        vis = _TAG_CLAUSE.search(query.split(" ", 1)[0]) if query.startswith("@visibility:") else None
        if vis:
            groups = [self._unescape(g) for g in vis.group(2).split("|")]
            if not set(groups).intersection(service.visibility):
                return False

        # Remaining tag clauses (exact match), skipping the visibility one.
        for field_name, raw_value in _TAG_CLAUSE.findall(query):
            if field_name == "visibility":
                continue
            value = self._unescape(raw_value)
            actual = getattr(service, field_name, None)
            if isinstance(actual, bool):
                if str(actual).lower() != value.lower():
                    return False
            elif str(actual).lower() != value.lower():
                return False

        # Text clauses (fuzzy/prefix - approximated by case-insensitive substring).
        for field_name, patterns in _TEXT_CLAUSE.findall(query):
            actual = str(getattr(service, field_name, "")).lower()
            terms = [p.strip("%*").lower() for p in patterns.split()]
            if not all(term in actual for term in terms if term):
                return False

        # Numeric range clauses.
        for field_name, lower, upper in _NUMERIC_CLAUSE.findall(query):
            actual_num = getattr(service, field_name, None)
            if actual_num is None:
                return False
            low = float("-inf") if lower == "-inf" else float(lower)
            high = float("inf") if upper == "+inf" else float(upper)
            if not (low <= float(actual_num) <= high):
                return False

        return True

    def _unescape(self, value: str) -> str:
        """Reverse the builder's backslash and space escaping for comparison."""
        return value.replace("\\ ", " ").replace("\\", "").strip()
