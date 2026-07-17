"""Domain interfaces (ports) for the Meridian platform.

These abstract base classes are the seam between the application core and the
outside world. They express *what* the core needs - the ability to embed text,
to store and search vectors, to call an LLM, to emit a trace - without saying
*how* those things are done. Concrete implementations live in
``meridian.infrastructure`` and are injected at composition time.

This is the Dependency Inversion Principle (the "D" in SOLID) made concrete:
high-level policy (the router, the pipelines) depends on these abstractions,
and low-level detail (Azure OpenAI, Redis) depends on them too. Neither depends
on the other directly. Swapping the fake embedding provider for the real Azure
one is a one-line change at the composition root because both satisfy
:class:`EmbeddingProvider`.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from meridian.domain.models import KnowledgeChunk, UserContext
from meridian.domain.models.knowledge import FatChunk, SlimChunk
from meridian.domain.models.service_catalog import ServiceRecord


class EmbeddingProvider(ABC):
    """Turns text into normalised embedding vectors.

    Implementations must return L2-normalised vectors so that a dot product is
    equivalent to cosine similarity - the router's scoring math relies on this
    invariant and does not re-normalise.
    """

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return its normalised vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Order of the output matches the input."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vectors this provider produces."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cache_identity(self) -> str:
        """Stable identity of the model and preprocessing used for vectors.

        Cache keys must change when two providers can emit different vectors,
        even when their dimensionality happens to be the same.
        """
        raise NotImplementedError


class VectorStore(ABC):
    """Stores route embedding matrices and knowledge chunks, and searches them.

    The store has two distinct responsibilities that happen to share a backend:
    it holds the router's per-intent positive/negative matrices (written by the
    ingestion path, read on every route call), and it holds the knowledge
    chunks that RAG retrieves. Both are modelled here because in this reference
    implementation they live in the same Redis instance; a production system
    might split them.
    """

    @abstractmethod
    def save_route_matrices(
        self,
        fingerprint: str,
        positives: dict[str, list[list[float]]],
        negatives: dict[str, list[list[float]]],
    ) -> None:
        """Persist per-intent positive and negative embedding matrices."""
        raise NotImplementedError

    @abstractmethod
    def load_route_matrices(
        self, fingerprint: str
    ) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]] | None:
        """Load matrices for a fingerprint, or ``None`` on cache miss."""
        raise NotImplementedError

    @abstractmethod
    def upsert_chunks(self, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> None:
        """Index knowledge chunks alongside their embedding vectors."""
        raise NotImplementedError

    @abstractmethod
    def search_chunks(self, query_vector: list[float], user: UserContext, top_k: int) -> list[KnowledgeChunk]:
        """Return the ``top_k`` chunks visible to ``user`` nearest to the query.

        Access control is the implementation's responsibility and must be
        applied *inside* the search (as a metadata filter), not by the caller
        afterwards. A user must never receive a chunk outside their ACL groups.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_fat_chunks(self, fats: list[FatChunk], vectors: list[list[float]]) -> None:
        """Index the slim projection of each fat chunk, storing the fat body too.

        The search index holds only the slim projection and the vector; the fat
        body is stored separately (as a JSON document) keyed by ``chunk_id`` and
        fetched on demand via :meth:`fetch_fat`.
        """
        raise NotImplementedError

    @abstractmethod
    def search_slim(self, query_vector: list[float], user: UserContext, top_k: int) -> list[SlimChunk]:
        """Return the ``top_k`` slim projections visible to ``user``.

        This is the cheap search path: it returns only the small slim
        projections, never the fat bodies. ACL filtering is applied inside the
        search, exactly as in :meth:`search_chunks`.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_fat(self, chunk_id: str) -> FatChunk | None:
        """Fetch one fat document by id (the ``JSON.GET`` path).

        Called only for chunks that survive ranking and will enter the context,
        so the fat payload is paid for a handful of times per query, not once
        per candidate.
        """
        raise NotImplementedError


class CatalogStore(ABC):
    """Stores structured service records and runs compiled queries over them.

    This is the structured-knowledge counterpart to :class:`VectorStore`. It
    holds :class:`ServiceRecord` rows and executes a RediSearch expression
    (produced by the query builder) against them. Access scoping is baked into
    the compiled query - the visibility clause is a prefix the builder always
    emits - so the store executes exactly what it's given.
    """

    @abstractmethod
    def upsert_services(self, services: list[ServiceRecord]) -> None:
        """Index a batch of service records."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, compiled_query: str, *, limit: int = 25) -> list[ServiceRecord]:
        """Run a compiled RediSearch expression and return matching records.

        :param compiled_query: The RediSearch expression from the query builder.
        :param limit: Maximum number of records to return.
        :returns: The matching service records.
        """
        raise NotImplementedError


class LLMProvider(ABC):
    """Generates a natural-language answer from a prompt.

    The interface is intentionally minimal. Everything about prompt
    construction, grounding instructions, and citation formatting lives in the
    generation pipeline; the provider only knows how to turn a prompt into text.
    """

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's completion for ``prompt``."""
        raise NotImplementedError


class Tracer(Protocol):
    """Structured observability sink.

    Modelled as a ``Protocol`` rather than an ABC so that any object exposing an
    ``event`` method satisfies it - including a no-op tracer used in tests. The
    platform's guiding principle is "observability before optimisation": every
    routing decision and retrieval is emitted here so that behaviour is
    explainable from the trace alone.
    """

    def event(self, name: str, **fields: object) -> None:
        """Record a named event with arbitrary structured fields."""
        ...


class RouterMetricsPort(Protocol):
    """Port for recording router decisions and detecting degradation.

    Modelled as a ``Protocol`` so any object exposing these three methods
    satisfies it - including the in-process :class:`RouterMetricsCollector` in
    ``infrastructure`` and a no-op stand-in for tests. The application layer
    depends on this abstraction; the concrete collector is injected at the
    composition root.
    """

    def record_routing(
        self,
        route: str,
        *,
        was_fallback: bool = False,
        anaphora_resolved: bool | None = None,
        coercion_applied: bool = False,
    ) -> None:
        """Record one routing decision and optional anaphora attempt."""
        ...

    def is_degraded(self) -> bool:
        """Whether the fallback rate has crossed the alarm threshold."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return a structured snapshot of the accumulated metrics."""
        ...
