"""Redis Stack service catalog store.

Runs the exact RediSearch expressions the query builder emits against a real
``FT.SEARCH`` index of service records. This is the production counterpart to
:class:`~meridian.infrastructure.vectorstore.in_memory_catalog.InMemoryCatalogStore`;
the two share an interface so the same application code targets either.

The index declares the service fields with the kinds the builder assumes: tag
fields for exact matches and the visibility scope, text fields for fuzzy
matches, and a numeric field for dependency ranges. Because the builder already
compiled a complete, ACL-scoped query, this store simply executes it.
"""

from meridian.domain.interfaces import CatalogStore, Tracer
from meridian.domain.models.service_catalog import ServiceRecord

try:  # pragma: no cover - import guarded so the module loads without redis
    import redis
    from redis.commands.search.field import NumericField, TagField, TextField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REDIS_AVAILABLE = False


class RedisCatalogStore(CatalogStore):
    """Service catalog backed by a Redis Stack RediSearch index."""

    def __init__(
        self,
        *,
        url: str,
        tracer: Tracer,
        namespace: str = "meridian",
        index_name: str | None = None,
    ) -> None:
        """Connect to Redis and ensure the catalog index exists.

        :param url: Redis connection URL.
        :param tracer: Structured observability sink.
        :param namespace: Key prefix for catalog keys.
        :param index_name: Name of the RediSearch index over services.
        :raises RuntimeError: If the ``redis`` package is not installed.
        """
        if not _REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed; use InMemoryCatalogStore for local runs.")
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._tracer = tracer
        self._namespace = namespace
        self._index_name = index_name or f"idx:{namespace}:services"
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index over service hashes if absent."""
        try:
            self._client.ft(self._index_name).info()
            return
        except Exception:  # noqa: BLE001 - redis raises generically when absent
            pass

        schema = [
            TextField("name"),
            TextField("description"),
            TagField("team"),
            TagField("domain"),
            TagField("tier"),
            TagField("lifecycle"),
            TagField("has_owner"),
            TagField("visibility", separator=","),
            NumericField("dependencies"),
        ]
        definition = IndexDefinition(prefix=[f"{self._namespace}:service:"], index_type=IndexType.HASH)
        self._client.ft(self._index_name).create_index(schema, definition=definition)
        self._tracer.event("redis.catalog.index_created", index=self._index_name)

    def upsert_services(self, services: list[ServiceRecord]) -> None:
        """Index service records as RediSearch-visible hashes."""
        pipe = self._client.pipeline()
        for service in services:
            key = f"{self._namespace}:service:{service.service_id}"
            pipe.hset(
                key,
                mapping={
                    "name": service.name,
                    "description": service.description,
                    "team": service.team,
                    "domain": service.domain,
                    "tier": service.tier,
                    "lifecycle": service.lifecycle,
                    "has_owner": "true" if service.has_owner else "false",
                    "dependencies": service.dependencies,
                    "visibility": ",".join(service.visibility),
                },
            )
        pipe.execute()
        self._tracer.event("redis.catalog.upserted", count=len(services))

    def execute(self, compiled_query: str, *, limit: int = 25) -> list[ServiceRecord]:
        """Run the compiled expression against the RediSearch index."""
        if not compiled_query.startswith("@visibility:{"):
            self._tracer.event("redis.catalog.rejected_unscoped_query")
            return []
        query = Query(compiled_query).paging(0, limit).dialect(2)
        results = self._client.ft(self._index_name).search(query)
        records = [
            ServiceRecord(
                service_id=doc.id.split(":")[-1],
                name=doc.name,
                team=doc.team,
                domain=doc.domain,
                tier=doc.tier,
                lifecycle=getattr(doc, "lifecycle", "active"),
                has_owner=getattr(doc, "has_owner", "true") == "true",
                dependencies=int(getattr(doc, "dependencies", 0)),
                description=getattr(doc, "description", ""),
                visibility=[],
            )
            for doc in results.docs
        ]
        self._tracer.event("redis.catalog.executed", matched=len(records))
        return records
