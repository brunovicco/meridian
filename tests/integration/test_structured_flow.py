"""Integration tests for the structured-query path.

Exercise the full structured route: a catalog question compiled into a
RediSearch expression, executed against the in-memory catalog, and returned with
ACL scoping enforced. These complement the RAG-path tests in
``test_ask_flow.py``.
"""

from importlib.resources import files

from meridian.application.pipelines.structured_query_pipeline import StructuredQueryPipeline
from meridian.application.query.builder import ServiceQueryBuilder
from meridian.domain.models import UserContext
from meridian.domain.models.service_catalog import ServiceRecord
from meridian.infrastructure.config.catalog_loader import load_service_catalog
from meridian.infrastructure.observability.tracer import NullTracer
from meridian.infrastructure.vectorstore.in_memory_catalog import InMemoryCatalogStore

_DATA = files("meridian.data.catalog")


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


def test_multiple_acl_groups_still_enforce_visibility() -> None:
    """Spaces in a multi-group clause must not disable the ACL filter."""
    store = InMemoryCatalogStore(tracer=NullTracer())
    store.upsert_services(
        [
            ServiceRecord(
                service_id="visible",
                name="visible",
                team="payments",
                domain="payments",
                tier="tier1",
                visibility=["payments"],
            ),
            ServiceRecord(
                service_id="restricted",
                name="restricted",
                team="security",
                domain="identity",
                tier="tier1",
                visibility=["security"],
            ),
        ]
    )
    pipeline = StructuredQueryPipeline(builder=ServiceQueryBuilder(), store=store, tracer=NullTracer())

    result = pipeline.query("list services", UserContext(user_id="u", acl_groups=["payments", "platform"]))

    assert [service.service_id for service in result.services] == ["visible"]


def test_suspicious_or_long_acl_never_widens_visibility() -> None:
    """Sanitizer rejection must return no records rather than a wildcard."""
    pipeline = _pipeline()

    asset = pipeline.query("list services", UserContext(user_id="u", acl_groups=["asset"]))
    long_acl = pipeline.query(
        "list services",
        UserContext(user_id="u", acl_groups=[f"group-{index}" for index in range(80)]),
    )

    assert asset.services == []
    assert long_acl.services == []


def test_catalog_store_rejects_unscoped_queries() -> None:
    """The store is a defense-in-depth boundary for compiled queries."""
    store = InMemoryCatalogStore(tracer=NullTracer())
    store.upsert_services(load_service_catalog(_DATA / "service_catalog.json"))

    assert store.execute("*") == []


def test_no_match_marker_is_never_executed_as_a_real_acl_group() -> None:
    """The fail-closed marker cannot expose a record using the reserved value."""
    store = InMemoryCatalogStore(tracer=NullTracer())
    store.upsert_services(
        [
            ServiceRecord(
                service_id="reserved",
                name="reserved",
                team="security",
                domain="identity",
                tier="tier1",
                visibility=["__none__"],
            )
        ]
    )
    pipeline = StructuredQueryPipeline(builder=ServiceQueryBuilder(), store=store, tracer=NullTracer())

    result = pipeline.query("list services", UserContext(user_id="u", acl_groups=[]))

    assert result.services == []


def test_structured_result_reports_truncation() -> None:
    """A bounded catalog response never pretends to be complete."""
    store = InMemoryCatalogStore(tracer=NullTracer())
    store.upsert_services(
        [
            ServiceRecord(
                service_id=f"service-{index}",
                name=f"service-{index}",
                team="platform",
                domain="gateway",
                tier="tier2",
                visibility=["platform"],
            )
            for index in range(3)
        ]
    )
    pipeline = StructuredQueryPipeline(
        builder=ServiceQueryBuilder(),
        store=store,
        tracer=NullTracer(),
        result_limit=2,
    )

    result = pipeline.query("list services", UserContext(user_id="u", acl_groups=["platform"]))
    answer = pipeline.run("list services", UserContext(user_id="u", acl_groups=["platform"]))

    assert result.truncated is True
    assert len(result.services) == 2
    assert answer.text.startswith("Showing the first 2")
