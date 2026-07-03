"""The routing engine (layer 2 of routing).

The semantic router produces a *signal*: scores and an ambiguity flag. The
routing engine turns that signal into an *action* - a :class:`RoutingDecision`
naming which pipeline should handle the query. Keeping this policy layer
separate from the scoring layer means the two can evolve and be tested
independently: the engine can be exercised with hand-built
:class:`RouterResult` objects and never needs an embedding provider.

The decision logic is intentionally simple and explainable:

* If the router flagged ambiguity, ask for disambiguation.
* If the top score is comfortably above threshold with a clear margin, route
  directly.
* Otherwise, fall back to general knowledge QA rather than guess.
"""

from meridian.domain.interfaces import Tracer
from meridian.domain.models import (
    DecisionType,
    RouterResult,
    RouteType,
    RoutingDecision,
)
from meridian.domain.policies import RoutingPolicy


class RoutingEngine:
    """Applies routing policy to a router signal to produce a decision."""

    def __init__(self, *, policy: RoutingPolicy, tracer: Tracer) -> None:
        """Inject the policy and the tracer.

        :param policy: Thresholds and the intent-to-route mapping.
        :param tracer: Structured observability sink.
        """
        self._policy = policy
        self._tracer = tracer

    def decide(self, result: RouterResult) -> RoutingDecision:
        """Turn a :class:`RouterResult` into a :class:`RoutingDecision`.

        :param result: The raw signal from the semantic router.
        :returns: The action to take, with a human-readable reason.
        """
        decision = self._decide_inner(result)
        self._tracer.event(
            "routing_engine.decide",
            decision=decision.decision.value,
            route=decision.route_type.value,
            intent=decision.intent,
            reason=decision.reason,
        )
        return decision

    def _decide_inner(self, result: RouterResult) -> RoutingDecision:
        """Core decision logic, without tracing (kept pure for testability)."""
        best = result.best_intent
        threshold = self._policy.threshold_for(best)
        margin = self._policy.ambiguity.high_confidence_margin
        runner_up = result.runner_up
        gap = (result.best_score - runner_up.score) if runner_up else float("inf")

        # Ambiguity was already detected upstream; honour it.
        if result.ambiguous:
            return RoutingDecision(
                decision=DecisionType.ASK_DISAMBIGUATION,
                route_type=self._policy.route_for(best),
                intent=best,
                reason=f"router flagged ambiguity ({result.disambiguation_rule})",
                router_result=result,
            )

        # High confidence and a clear margin: route directly.
        if result.best_score >= threshold and gap >= margin:
            return RoutingDecision(
                decision=DecisionType.ROUTE_DIRECT,
                route_type=self._policy.route_for(best),
                intent=best,
                reason=(
                    f"high confidence: score={result.best_score:.3f} "
                    f">= threshold={threshold:.3f}, margin={gap:.3f} >= {margin:.3f}"
                ),
                router_result=result,
            )

        # Neither ambiguous nor confident: fall back to general QA.
        return RoutingDecision(
            decision=DecisionType.FALLBACK,
            route_type=RouteType.KNOWLEDGE_QA,
            intent="knowledge_qa",
            reason=(
                f"no intent cleared confidence bar (score={result.best_score:.3f}, "
                f"threshold={threshold:.3f}); falling back to knowledge QA"
            ),
            router_result=result,
        )
