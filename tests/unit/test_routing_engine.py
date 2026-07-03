"""Unit tests for the routing engine.

The engine is pure policy over a :class:`RouterResult`, so these tests build
results by hand and assert on the decision - no router, no embeddings. This is
the payoff of separating the scoring layer from the policy layer: each is
testable in isolation.
"""

from meridian.application.router.routing_engine import RoutingEngine
from meridian.domain.models import DecisionType, RouterResult, RouteType, ScoredIntent
from meridian.domain.policies import RoutingPolicy
from meridian.infrastructure.observability.tracer import NullTracer


def _engine() -> RoutingEngine:
    """Build an engine with the production policy and a silent tracer."""
    return RoutingEngine(policy=RoutingPolicy(), tracer=NullTracer())


def _result(intent: str, s1: float, s2: float, ambiguous: bool = False) -> RouterResult:
    """Construct a router result with a top-two score gap."""
    return RouterResult(
        query="q",
        best_intent=intent,
        best_score=s1,
        topk=[ScoredIntent(intent=intent, score=s1), ScoredIntent(intent="other", score=s2)],
        ambiguous=ambiguous,
        disambiguation_rule="none" if not ambiguous else "margin_too_small",
    )


def test_high_confidence_routes_direct() -> None:
    """A confident, unambiguous result routes straight to its pipeline."""
    decision = _engine().decide(_result("knowledge_qa", 0.90, 0.50))
    assert decision.decision == DecisionType.ROUTE_DIRECT
    assert decision.route_type == RouteType.KNOWLEDGE_QA


def test_ambiguous_result_asks_disambiguation() -> None:
    """An upstream ambiguity flag leads to a disambiguation request."""
    decision = _engine().decide(_result("code_lookup", 0.90, 0.89, ambiguous=True))
    assert decision.decision == DecisionType.ASK_DISAMBIGUATION


def test_below_threshold_falls_back_to_qa() -> None:
    """A weak signal falls back to general knowledge QA rather than guessing."""
    decision = _engine().decide(_result("structured_query", 0.40, 0.10))
    assert decision.decision == DecisionType.FALLBACK
    assert decision.route_type == RouteType.KNOWLEDGE_QA


def test_narrow_margin_below_high_confidence_falls_back() -> None:
    """Clearing the threshold but with a thin margin is not confident enough."""
    # score above threshold (0.72) but margin (0.02) below high_confidence (0.10)
    decision = _engine().decide(_result("knowledge_qa", 0.74, 0.72))
    assert decision.decision == DecisionType.FALLBACK
