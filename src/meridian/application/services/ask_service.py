"""The ask service: the application's single entry point for a user question.

This orchestrator is the top of the application layer. It composes the routing
and generation steps into the end-to-end flow a caller cares about: take a
question and a user, return an answer. It depends only on the other application
components and on domain abstractions - never on infrastructure directly.

Flow:

1. The semantic router scores the query and returns a signal.
2. The routing engine turns that signal into a decision.
3. If the decision is to disambiguate, the service returns a clarifying answer
   rather than guessing.
4. Otherwise the query-understanding step produces retrieval-optimised terms,
   and the RAG pipeline answers, grounded and cited.

Out-of-scope routes short-circuit before any retrieval, which is the whole
point of routing first: cheap questions never pay the cost of the expensive
path.
"""

from meridian.application.pipelines.rag_pipeline import RagPipeline
from meridian.application.pipelines.structured_query_pipeline import StructuredQueryPipeline
from meridian.application.router.routing_engine import RoutingEngine
from meridian.application.router.semantic_router import SemanticRouter
from meridian.application.services.query_understanding import (
    QueryUnderstanding,
    coerce_understanding,
)
from meridian.domain.interfaces import LLMProvider, RouterMetricsPort, Tracer
from meridian.domain.models import (
    Answer,
    DecisionType,
    RouteType,
    UserContext,
)


class AskService:
    """Orchestrates routing and retrieval to answer a developer's question."""

    def __init__(
        self,
        *,
        router: SemanticRouter,
        engine: RoutingEngine,
        rag: RagPipeline,
        structured: StructuredQueryPipeline,
        llm: LLMProvider,
        tracer: Tracer,
        metrics: RouterMetricsPort,
    ) -> None:
        """Compose the service from its collaborators (all injected).

        :param router: The semantic router (layer 1).
        :param engine: The routing engine (layer 2).
        :param rag: The retrieval-augmented generation pipeline.
        :param structured: The structured service-catalog query pipeline.
        :param llm: LLM provider, used for the query-understanding step.
        :param tracer: Structured observability sink.
        :param metrics: Router metrics port for degradation detection.
        """
        self._router = router
        self._engine = engine
        self._rag = rag
        self._structured = structured
        self._llm = llm
        self._tracer = tracer
        self._metrics = metrics

    def ask(self, query: str, user: UserContext) -> Answer:
        """Answer ``query`` for ``user``, end to end.

        :param query: The developer's natural-language question.
        :param user: The asking user, whose ACL groups gate retrieval.
        :returns: A grounded, cited :class:`Answer`.
        """
        self._tracer.event("ask.start", user_id=user.user_id, query=query[:120])

        router_result = self._router.route(query)
        decision = self._engine.decide(router_result)

        # Record the routing outcome for degradation monitoring. A fallback
        # decision is the signal the router may be drifting.
        self._metrics.record_routing(
            decision.route_type.value,
            was_fallback=decision.decision == DecisionType.FALLBACK,
        )
        if self._metrics.is_degraded():
            self._tracer.event("router.degraded", **self._metrics.snapshot())

        if decision.route_type == RouteType.OUT_OF_SCOPE:
            return Answer(
                text=(
                    "I'm Meridian, the engineering knowledge assistant. Ask me about "
                    "internal docs, runbooks, services, or the codebase."
                ),
                citations=[],
                route_type=RouteType.OUT_OF_SCOPE,
                grounded=False,
            )

        if decision.decision == DecisionType.ASK_DISAMBIGUATION:
            options = ", ".join(s.intent for s in router_result.topk[:2])
            return Answer(
                text=(
                    "Your question could go a couple of ways "
                    f"({options}). Could you add a bit more detail so I search the right place?"
                ),
                citations=[],
                route_type=decision.route_type,
                grounded=False,
            )

        # Structured questions about the service catalog are a query problem,
        # not a retrieval problem: route them to the structured pipeline.
        if decision.route_type == RouteType.STRUCTURED_QUERY:
            return self._structured.run(query, user)

        understanding = self._understand(query, router_result.topk, decision.route_type)
        if understanding.needs_clarification:
            return Answer(
                text="Could you be a bit more specific? I want to make sure I search the right area.",
                citations=[],
                route_type=decision.route_type,
                grounded=False,
            )

        return self._rag.run(understanding.search_terms, user, decision.route_type)

    def _understand(
        self,
        query: str,
        candidates: list,
        decided_route: RouteType,
    ) -> QueryUnderstanding:
        """Run the query-understanding LLM step behind its output contract.

        The LLM is prompted to emit JSON matching the contract; whatever it
        returns is passed through :func:`coerce_understanding`, which repairs
        drift and guarantees a valid object. The service never handles a raw,
        unvalidated model response.
        """
        candidate_names = [c.intent for c in candidates]
        prompt = (
            "Classify the developer question and produce retrieval terms. "
            "Respond with JSON only, matching this schema: "
            '{"route_type": one of ["knowledge_qa","code_lookup","structured_query","out_of_scope"], '
            '"search_terms": string, "needs_clarification": boolean}. '
            f"Question: {query}\nRouter candidates: {candidate_names}"
        )
        raw = self._llm.complete(prompt, system="You output only valid JSON.")
        understanding = coerce_understanding(raw, fallback_query=query)
        # The semantic router and routing engine are authoritative. The LLM
        # rewrites retrieval terms but cannot silently reroute the request.
        understanding = understanding.model_copy(update={"route_type": decided_route})
        self._tracer.event(
            "ask.understanding",
            route_type=understanding.route_type.value,
            needs_clarification=understanding.needs_clarification,
        )
        return understanding
