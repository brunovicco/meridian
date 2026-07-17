"""The structured-query pipeline for service-catalog questions.

This is the third route's engine, and the concrete embodiment of a principle:
the right pattern depends on the shape of the data. A question like "who owns the
payments service" is not a retrieval problem - it's a query problem. Stuffing
catalog rows into an LLM context returns a sample; compiling the question into a
precise query returns the complete, correct answer.

The pipeline extracts a typed filter from the question, sanitises and compiles
it into a RediSearch expression via the query builder, executes it against the
catalog store, and formats a direct answer. There is no LLM in the hot path
here - the answer is derived from structured data, not generated.
"""

from meridian.application.query.builder import ServiceQueryBuilder
from meridian.application.query.sanitizer import NO_MATCH_QUERY, sanitize_search_query
from meridian.domain.interfaces import CatalogStore, Tracer
from meridian.domain.models import Answer, Citation, RouteType, UserContext
from meridian.domain.models.service_catalog import StructuredQueryResult
from meridian.domain.models.service_filter import ServiceFilterModel

# Small keyword tables to turn a natural-language catalog question into a typed
# filter. In production this extraction is the LLM's structured-output step
# behind a contract; here it's deterministic so the demo needs no model.
_TEAM_KEYWORDS = ("platform", "payments", "notifications", "fraud", "sre", "security")
_DOMAIN_KEYWORDS = ("payments", "notifications", "fraud", "gateway", "identity")
_CATALOG_CITATION = Citation(source="Service Catalog", source_url="meridian://service-catalog")


class StructuredQueryPipeline:
    """Answers service-catalog questions by compiling and running a query."""

    def __init__(
        self,
        *,
        builder: ServiceQueryBuilder,
        store: CatalogStore,
        tracer: Tracer,
        result_limit: int = 250,
    ) -> None:
        """Wire the pipeline to the builder and the catalog store.

        :param builder: The RediSearch query builder.
        :param store: The catalog store to execute against.
        :param tracer: Structured observability sink.
        :param result_limit: Safety bound for one catalog response.
        """
        self._builder = builder
        self._store = store
        self._tracer = tracer
        self._result_limit = result_limit

    def run(self, question: str, user: UserContext) -> Answer:
        """Answer a structured catalog question for a user.

        :param question: The natural-language catalog question.
        :param user: The asking user, whose ACL groups scope visibility.
        :returns: A direct, non-generated :class:`Answer`.
        """
        result = self.query(question, user)
        return self._to_answer(question, result)

    def query(self, question: str, user: UserContext) -> StructuredQueryResult:
        """Compile and execute the query, returning records and the expression.

        Exposed separately from :meth:`run` so the compiled query can be
        inspected - useful for demonstrating that the system queries rather than
        retrieves.
        """
        filters = self._extract_filters(question)
        compiled = self._builder.build_service_query(acl_groups=user.acl_groups, filters=filters)
        # The builder produces a structured expression; the sanitiser is a final
        # guard on the whole string before it reaches the store.
        safe = sanitize_search_query(compiled)
        fetched = [] if safe == NO_MATCH_QUERY else self._store.execute(safe, limit=self._result_limit + 1)
        truncated = len(fetched) > self._result_limit
        services = fetched[: self._result_limit]
        self._tracer.event(
            "structured.query",
            question=question[:120],
            compiled=safe[:160],
            matched=len(services),
        )
        return StructuredQueryResult(services=services, compiled_query=safe, truncated=truncated)

    def _extract_filters(self, question: str) -> ServiceFilterModel:
        """Derive a typed filter from the question via keyword heuristics."""
        lowered = question.lower()
        filters = ServiceFilterModel()

        for team in _TEAM_KEYWORDS:
            if f"{team} team" in lowered or f"team {team}" in lowered:
                filters.team = team
                break

        for domain in _DOMAIN_KEYWORDS:
            if domain in lowered and filters.team != domain:
                filters.domain = domain
                break

        if any(t in lowered for t in ("tier1", "tier-1", "tier 1")):
            filters.tier = "tier1"
        elif any(t in lowered for t in ("tier2", "tier-2", "tier 2")):
            filters.tier = "tier2"

        if "without an owner" in lowered or "no owner" in lowered or "unowned" in lowered:
            filters.has_owner = False

        return filters

    def _to_answer(self, question: str, result: StructuredQueryResult) -> Answer:
        """Format the matched records into a direct answer."""
        if not result.services:
            return Answer(
                text="No services in the catalog match that, within what you can see.",
                citations=[_CATALOG_CITATION.model_copy()],
                route_type=RouteType.STRUCTURED_QUERY,
                grounded=True,
            )

        lines = [
            f"{s.name} - team {s.team}, domain {s.domain}, {s.tier}, {s.dependencies} dependencies"
            for s in result.services
        ]
        header = (
            f"Showing the first {len(result.services)} matching service(s):"
            if result.truncated
            else f"{len(result.services)} service(s) matched:"
        )
        return Answer(
            text=header + "\n" + "\n".join(lines),
            citations=[_CATALOG_CITATION.model_copy()],
            route_type=RouteType.STRUCTURED_QUERY,
            grounded=True,
        )
