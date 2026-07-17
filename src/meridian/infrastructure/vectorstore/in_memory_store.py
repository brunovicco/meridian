"""In-memory vector store: the no-Docker, no-network fallback.

Implements :class:`VectorStore` with plain Python data structures and numpy so
the reference runs anywhere - CI, a laptop without Docker, a unit test. It
mirrors the Redis implementation's *semantics* exactly, most importantly the
access-control rule: search intersects the user's groups with each chunk's
groups and fails closed when the user has no groups.

Having two implementations behind one interface is the point. The application
layer is written once against :class:`VectorStore`; whether it talks to Redis
Stack or to this dictionary is a composition-root decision driven by an
environment variable. That is the Liskov substitution and dependency inversion
principles working together - either store can stand in for the other without
the caller noticing.
"""

import numpy as np

from meridian.domain.interfaces import Tracer, VectorStore
from meridian.domain.models import KnowledgeChunk, UserContext
from meridian.domain.models.knowledge import FatChunk, SlimChunk


class InMemoryVectorStore(VectorStore):
    """Dictionary-backed vector store with cosine KNN and ACL filtering."""

    def __init__(self, *, tracer: Tracer) -> None:
        """Initialise empty stores.

        :param tracer: Structured observability sink.
        """
        self._tracer = tracer
        self._routes: dict[str, tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]]] = {}
        self._chunks: dict[str, tuple[KnowledgeChunk, np.ndarray]] = {}
        # Fat/slim state: slim projections are searched; fat bodies are fetched
        # by id on demand. They are kept in separate structures to mirror the
        # Redis layout (an indexed hash for slim, a JSON document for fat).
        self._slim: dict[str, tuple[SlimChunk, np.ndarray]] = {}
        self._fat_by_id: dict[str, FatChunk] = {}

    def save_route_matrices(
        self,
        fingerprint: str,
        positives: dict[str, list[list[float]]],
        negatives: dict[str, list[list[float]]],
    ) -> None:
        """Store matrices in memory under the fingerprint."""
        self._routes[fingerprint] = (positives, negatives)
        self._tracer.event("memory.routes.saved", fingerprint=fingerprint, intents=len(positives))

    def load_route_matrices(
        self, fingerprint: str
    ) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]] | None:
        """Return matrices for a fingerprint, or ``None`` on miss."""
        hit = self._routes.get(fingerprint)
        self._tracer.event("memory.routes.hit" if hit else "memory.routes.miss", fingerprint=fingerprint)
        return hit

    def upsert_chunks(self, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks and vectors by stable chunk id."""
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = (chunk, np.asarray(vector, dtype=np.float32))
        self._tracer.event("memory.chunks.upserted", count=len(chunks), total=len(self._chunks))

    def search_chunks(self, query_vector: list[float], user: UserContext, top_k: int) -> list[KnowledgeChunk]:
        """Cosine KNN over chunks visible to the user; fails closed.

        A chunk is a candidate only if its groups intersect the user's groups.
        With no user groups, nothing is visible - the same fail-closed rule the
        Redis implementation enforces.
        """
        if not user.acl_groups:
            return []

        allowed = set(user.acl_groups)
        query = np.asarray(query_vector, dtype=np.float32)
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, vector in self._chunks.values():
            if not allowed.intersection(chunk.acl_groups):
                continue
            similarity = float(np.dot(vector, query))
            enriched = chunk.model_copy(update={"score": similarity})
            scored.append((similarity, enriched))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [chunk for _, chunk in scored[:top_k]]
        self._tracer.event("memory.chunks.searched", returned=len(top), candidates=len(scored))
        return top

    def upsert_fat_chunks(self, fats: list[FatChunk], vectors: list[list[float]]) -> None:
        """Store fat bodies by id and index their slim projections for search."""
        for fat, vector in zip(fats, vectors, strict=True):
            self._fat_by_id[fat.chunk_id] = fat
            self._slim[fat.chunk_id] = (fat.to_slim(), np.asarray(vector, dtype=np.float32))
        self._tracer.event("memory.fat.upserted", count=len(fats), total=len(self._slim))

    def search_slim(self, query_vector: list[float], user: UserContext, top_k: int) -> list[SlimChunk]:
        """Cosine KNN over slim projections visible to the user; fails closed.

        Returns only the small slim projections. The fat body is never touched
        here - that is the whole point of the split.
        """
        if not user.acl_groups:
            return []

        allowed = set(user.acl_groups)
        query = np.asarray(query_vector, dtype=np.float32)
        scored: list[tuple[float, SlimChunk]] = []
        for slim, vector in self._slim.values():
            if not allowed.intersection(slim.acl_groups):
                continue
            similarity = float(np.dot(vector, query))
            scored.append((similarity, slim.model_copy(update={"score": similarity})))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [slim for _, slim in scored[:top_k]]
        self._tracer.event("memory.slim.searched", returned=len(top), candidates=len(scored))
        return top

    def fetch_fat(self, chunk_id: str) -> FatChunk | None:
        """Fetch one fat document by id (the in-memory JSON.GET analogue)."""
        fat = self._fat_by_id.get(chunk_id)
        self._tracer.event("memory.fat.fetched", chunk_id=chunk_id, hit=fat is not None)
        return fat
