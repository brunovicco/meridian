"""Integration tests for the structured-query path.

Exercise the full structured route: a catalog question compiled into a
RediSearch expression, executed against the in-memory catalog, and returned with
ACL scoping enforced. These complement the RAG-path tests in
``test_ask_flow.py``.
"""

from pathlib import Path

from meridian.application.pipelines.structured_query_pipeline import StructuredQueryPipeline
from meridian.application.query.builder import ServiceQueryBuilder
from meridian.domain.models import UserContext
from meridian.infrastructure.config.catalog_loader import load_service_catalog
from meridian.infrastructure.observability.tracer import NullTracer
from meridian.infrastructure.vectorstore.in_memory_catalog import InMemoryCatalogStore

_DATA = Path(__file__).resolve().parents[2] / "data" / "catalog"


def _pipeline() -> StructuredQueryPipeline:
    """Build a structured-query pipeline over the seeded catalog."""
    store = InMemoryCatalogStore(tracer=NullTracer())
    store.upsert_services(load_service_catalog(_DATA / "service_catalog.json"))
    return StructuredQueryPipeline(builder=ServiceQueryBuilder(), store=store, tracer=NullTracer())


def test_structured_query_returns_scoped_results() -> None:
    """A domain question returns only services in that domain and visibility."""
    pipeline = _pipeline()
    user = UserContext(user_id="alice", acl_groups=["payments", "platform"])
    result = pipeline.query("who owns the payments service", user)
    assert result.services
    assert all(s.domain == "payments" for s in result.services)
    assert "@domain:{payments}" in result.compiled_query


def test_structured_query_respects_visibility() -> None:
    """A user cannot see services outside their visibility groups."""
    pipeline = _pipeline()
    # A user in 'fraud' only should not see payments-domain services.
    fraud_user = UserContext(user_id="f", acl_groups=["fraud"])
    result = pipeline.query("who owns the payments service", fraud_user)
    assert result.services == []


def test_structured_query_finds_unowned_service() -> None:
    """The 'no owner' filter surfaces the deprecated unowned service."""
    pipeline = _pipeline()
    user = UserContext(user_id="bob", acl_groups=["platform", "sre"])
    result = pipeline.query("which services have no owner", user)
    names = {s.name for s in result.services}
    assert "legacy-batch-runner" in names
