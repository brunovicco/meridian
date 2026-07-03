"""Unit tests for the router metrics collector and the anaphora entity stack.

Cover the two remaining Redis-backed capabilities in pure form: the metrics
collector's fallback-rate and degradation logic (no Redis needed for the
in-process path) and the entity stack's dedup, ordering, and JSON round-trip.
"""

from meridian.application.services.entity_stack import DiscussedEntity, EntityStack
from meridian.infrastructure.metrics.router_metrics import RouterMetricsCollector


def test_metrics_fallback_rate() -> None:
    """The fallback rate reflects the share of fallback routings."""
    m = RouterMetricsCollector()
    for _ in range(3):
        m.record_routing("knowledge_qa", was_fallback=False)
    m.record_routing("knowledge_qa", was_fallback=True)
    assert m.fallback_rate() == 0.25


def test_metrics_degradation_needs_minimum_sample() -> None:
    """Degradation only fires once enough requests have accumulated."""
    m = RouterMetricsCollector()
    # 100% fallback but below the minimum sample size -> not yet degraded.
    for _ in range(5):
        m.record_routing("knowledge_qa", was_fallback=True)
    assert m.is_degraded() is False


def test_metrics_degradation_fires_above_threshold() -> None:
    """A sustained high fallback rate past the sample floor flags degradation."""
    m = RouterMetricsCollector()
    for _ in range(25):
        m.record_routing("knowledge_qa", was_fallback=True)
    assert m.is_degraded() is True


def test_metrics_snapshot_has_distribution() -> None:
    """The snapshot reports the route distribution and backend type."""
    m = RouterMetricsCollector()
    m.record_routing("code_lookup")
    snap = m.snapshot()
    assert snap["route_distribution"]["code_lookup"] == 1
    assert snap["backend"] == "in_process"


def test_entity_stack_push_and_order() -> None:
    """The most recent push is latest; the prior one is previous."""
    stack = EntityStack()
    stack.push(DiscussedEntity(name="payments-api"))
    stack.push(DiscussedEntity(name="gateway"))
    assert stack.latest.name == "gateway"
    assert stack.previous.name == "payments-api"


def test_entity_stack_dedup_promotes() -> None:
    """Re-pushing an entity promotes it rather than duplicating it."""
    stack = EntityStack()
    stack.push(DiscussedEntity(name="payments-api"))
    stack.push(DiscussedEntity(name="gateway"))
    stack.push(DiscussedEntity(name="payments-api"))
    names = [e.name for e in stack.entities]
    assert names == ["payments-api", "gateway"]


def test_entity_stack_bounded_to_five() -> None:
    """The stack never holds more than five entities."""
    stack = EntityStack()
    for i in range(8):
        stack.push(DiscussedEntity(name=f"svc-{i}"))
    assert len(stack.entities) == 5


def test_entity_stack_json_round_trip() -> None:
    """Serialising and deserialising preserves order and contents."""
    stack = EntityStack()
    stack.push(DiscussedEntity(name="a", route="knowledge_qa"))
    stack.push(DiscussedEntity(name="b", route="structured_query"))
    restored = EntityStack.from_json(stack.to_json())
    assert [e.name for e in restored.entities] == ["b", "a"]


def test_entity_stack_from_malformed_json_is_empty() -> None:
    """Corrupt JSON yields an empty stack rather than raising."""
    assert EntityStack.from_json("not json").is_empty
