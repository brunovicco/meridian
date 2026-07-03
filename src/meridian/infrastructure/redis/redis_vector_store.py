"""Redis Stack vector store implementation.

Backs both responsibilities of :class:`VectorStore` on a single Redis Stack
instance: the router's per-intent matrices (as compressed numpy blobs under a
fingerprint) and the knowledge chunks (as hashes indexed by RediSearch for KNN
vector search with metadata filtering).

Two design points worth calling out at interview:

* **Route matrices are cached, not recomputed.** They are keyed by the catalog
  fingerprint, so an unchanged catalog is loaded, never re-embedded. A changed
  catalog produces a new fingerprint and a clean rebuild - no stale-cache bug.
* **Access control is a query-time filter.** The KNN search restricts to chunks
  whose ``acl_groups`` intersect the user's groups, expressed as a RediSearch
  tag filter combined with the vector clause. The user cannot retrieve a chunk
  outside their groups even transiently, because the filter is part of the
  search, not a step after it.
"""

import io
import json

import numpy as np

from meridian.domain.interfaces import Tracer, VectorStore
from meridian.domain.models import KnowledgeChunk, UserContext
from meridian.domain.models.knowledge import FatChunk, SlimChunk

try:  # pragma: no cover - import guarded so the module loads without redis
    import redis
    from redis.commands.search.field import TagField, TextField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REDIS_AVAILABLE = False


def _ndarray_to_bytes(arr: np.ndarray) -> bytes:
    """Serialise a numpy array to compressed bytes for Redis storage."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=arr.astype(np.float32))
    return buffer.getvalue()


def _bytes_to_ndarray(data: bytes) -> np.ndarray:
    """Deserialise bytes produced by :func:`_ndarray_to_bytes`."""
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return loaded["data"]


class RedisVectorStore(VectorStore):
    """Vector store backed by Redis Stack (RediSearch + JSON/hash)."""

    def __init__(
        self,
        *,
        url: str,
        dimension: int,
        tracer: Tracer,
        namespace: str = "meridian",
        index_name: str = "idx:chunks",
    ) -> None:
        """Connect to Redis and ensure the search index exists.

        :param url: Redis connection URL, e.g. ``redis://localhost:6379``.
        :param dimension: Embedding dimensionality (must match the provider).
        :param tracer: Structured observability sink.
        :param namespace: Key prefix for all of this store's keys.
        :param index_name: Name of the RediSearch index over chunks.
        :raises RuntimeError: If the ``redis`` package is not installed.
        """
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis package not installed; use InMemoryVectorStore for local runs "
                "without Docker, or `uv sync --extra redis`."
            )
        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._dimension = dimension
        self._tracer = tracer
        self._namespace = namespace
        self._index_name = index_name
        self._slim_index_name = "idx:slim"
        self._ensure_index()
        self._ensure_slim_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index over chunk hashes if absent.

        The index declares the vector field (for KNN), a tag field over
        ``acl_groups`` (for the access-control filter), and text/metadata fields
        for citation. Creating it's idempotent - an existing index is left
        alone.
        """
        try:
            self._client.ft(self._index_name).info()
            return  # already exists
        except Exception:  # noqa: BLE001 - redis raises a generic error when absent
            pass

        schema = [
            TextField("text"),
            TextField("source"),
            TextField("source_url"),
            TagField("acl_groups", separator=","),
            VectorField(
                "embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": self._dimension, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        definition = IndexDefinition(prefix=[f"{self._namespace}:chunk:"], index_type=IndexType.HASH)
        self._client.ft(self._index_name).create_index(schema, definition=definition)
        self._tracer.event("redis.index.created", index=self._index_name, dim=self._dimension)

    def _ensure_slim_index(self) -> None:
        """Create the RediSearch index over slim projection hashes if absent.

        The slim index carries only the projection fields plus the vector - no
        full text. That is what keeps search cheap: KNN and ranking touch this
        index, and the fat JSON body is never read until :meth:`fetch_fat`.
        """
        try:
            self._client.ft(self._slim_index_name).info()
            return
        except Exception:  # noqa: BLE001
            pass

        schema = [
            TextField("title"),
            TextField("snippet"),
            TextField("source"),
            TextField("source_url"),
            TagField("acl_groups", separator=","),
            VectorField(
                "embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": self._dimension, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        definition = IndexDefinition(prefix=[f"{self._namespace}:slim:"], index_type=IndexType.HASH)
        self._client.ft(self._slim_index_name).create_index(schema, definition=definition)
        self._tracer.event("redis.slim_index.created", index=self._slim_index_name)

    # ---- route matrices -------------------------------------------------

    def _matrix_key(self, fingerprint: str, kind: str, intent: str) -> str:
        """Build the key for one intent's positive/negative matrix."""
        return f"{self._namespace}:routes:{fingerprint}:{kind}:{intent}"

    def _meta_key(self, fingerprint: str) -> str:
        """Build the key holding the intent list for a fingerprint."""
        return f"{self._namespace}:routes:{fingerprint}:meta"

    def save_route_matrices(
        self,
        fingerprint: str,
        positives: dict[str, list[list[float]]],
        negatives: dict[str, list[list[float]]],
    ) -> None:
        """Persist matrices and the intent-name index for a fingerprint."""
        pipe = self._client.pipeline()
        pipe.set(self._meta_key(fingerprint), json.dumps(sorted(positives.keys())))
        for intent, rows in positives.items():
            pipe.set(self._matrix_key(fingerprint, "pos", intent), _ndarray_to_bytes(np.asarray(rows)))
        for intent, rows in negatives.items():
            pipe.set(self._matrix_key(fingerprint, "neg", intent), _ndarray_to_bytes(np.asarray(rows)))
        pipe.execute()
        self._tracer.event("redis.routes.saved", fingerprint=fingerprint, intents=len(positives))

    def load_route_matrices(
        self, fingerprint: str
    ) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]] | None:
        """Load matrices for a fingerprint, or ``None`` on cache miss."""
        meta = self._client.get(self._meta_key(fingerprint))
        if meta is None:
            self._tracer.event("redis.routes.miss", fingerprint=fingerprint)
            return None
        intents = json.loads(meta)
        positives: dict[str, list[list[float]]] = {}
        negatives: dict[str, list[list[float]]] = {}
        for intent in intents:
            pos_raw = self._client.get(self._matrix_key(fingerprint, "pos", intent))
            neg_raw = self._client.get(self._matrix_key(fingerprint, "neg", intent))
            if pos_raw is None:
                return None
            # decode_responses=False guarantees bytes, not str, at runtime.
            assert isinstance(pos_raw, bytes)
            positives[intent] = _bytes_to_ndarray(pos_raw).tolist()
            if neg_raw:
                assert isinstance(neg_raw, bytes)
                negatives[intent] = _bytes_to_ndarray(neg_raw).tolist()
            else:
                negatives[intent] = []
        self._tracer.event("redis.routes.hit", fingerprint=fingerprint, intents=len(intents))
        return positives, negatives

    # ---- knowledge chunks ----------------------------------------------

    def upsert_chunks(self, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> None:
        """Index chunks with their vectors as RediSearch-visible hashes."""
        pipe = self._client.pipeline()
        for chunk, vector in zip(chunks, vectors, strict=True):
            key = f"{self._namespace}:chunk:{chunk.chunk_id}"
            pipe.hset(
                key,
                mapping={
                    "text": chunk.text,
                    "source": chunk.source,
                    "source_url": chunk.source_url,
                    "acl_groups": ",".join(chunk.acl_groups),
                    "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
                },
            )
        pipe.execute()
        self._tracer.event("redis.chunks.upserted", count=len(chunks))

    def search_chunks(self, query_vector: list[float], user: UserContext, top_k: int) -> list[KnowledgeChunk]:
        """KNN search restricted to chunks the user is allowed to read.

        The access-control filter is expressed as a RediSearch tag clause over
        ``acl_groups`` and combined with the KNN vector clause, so the database
        never returns a chunk outside the user's groups. If the user has no
        groups, nothing matches - fail closed, not open.
        """
        if not user.acl_groups:
            return []

        group_filter = "|".join(user.acl_groups)
        base = f"(@acl_groups:{{{group_filter}}})"
        query = (
            Query(f"{base}=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("text", "source", "source_url", "acl_groups", "score")
            .paging(0, top_k)
            .dialect(2)
        )
        params: dict[str, str | int | float | bytes] = {
            "vec": np.asarray(query_vector, dtype=np.float32).tobytes()
        }
        results = self._client.ft(self._index_name).search(query, query_params=params)

        chunks: list[KnowledgeChunk] = []
        for doc in results.docs:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=doc.id.split(":")[-1],
                    text=doc.text.decode() if isinstance(doc.text, bytes) else doc.text,
                    source=doc.source.decode() if isinstance(doc.source, bytes) else doc.source,
                    source_url=(
                        doc.source_url.decode() if isinstance(doc.source_url, bytes) else doc.source_url
                    ),
                    acl_groups=[],
                    # cosine distance -> similarity
                    score=1.0 - float(doc.score),
                )
            )
        self._tracer.event("redis.chunks.searched", returned=len(chunks))
        return chunks

    # ---- fat/slim -------------------------------------------------------

    def upsert_fat_chunks(self, fats: list[FatChunk], vectors: list[list[float]]) -> None:
        """Index each fat chunk's slim projection and store its fat JSON body.

        Two writes per chunk, mirroring the production layout:

        * a hash under ``{ns}:slim:{id}`` holding the small projection fields and
          the embedding, indexed by RediSearch for KNN search;
        * a JSON document under ``{ns}:fat:{id}`` holding the full body, fetched
          later by ``JSON.GET`` - never touched during search.
        """
        pipe = self._client.pipeline()
        for fat, vector in zip(fats, vectors, strict=True):
            slim = fat.to_slim()
            slim_key = f"{self._namespace}:slim:{fat.chunk_id}"
            pipe.hset(
                slim_key,
                mapping={
                    "title": slim.title,
                    "snippet": slim.snippet,
                    "source": slim.source,
                    "source_url": slim.source_url,
                    "acl_groups": ",".join(slim.acl_groups),
                    "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
                },
            )
            fat_key = f"{self._namespace}:fat:{fat.chunk_id}"
            pipe.json().set(fat_key, "$", fat.model_dump())
        pipe.execute()
        self._tracer.event("redis.fat.upserted", count=len(fats))

    def search_slim(self, query_vector: list[float], user: UserContext, top_k: int) -> list[SlimChunk]:
        """KNN over the slim index, returning only the small projections.

        Runs against the slim hash index (``idx:slim``), returns the projection
        fields only, and never reads the fat JSON. ACL filtering is a tag clause
        combined with the KNN clause, exactly as for full chunks.
        """
        if not user.acl_groups:
            return []

        group_filter = "|".join(user.acl_groups)
        base = f"(@acl_groups:{{{group_filter}}})"
        query = (
            Query(f"{base}=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("title", "snippet", "source", "source_url", "score")
            .paging(0, top_k)
            .dialect(2)
        )
        params: dict[str, str | int | float | bytes] = {
            "vec": np.asarray(query_vector, dtype=np.float32).tobytes()
        }
        results = self._client.ft(self._slim_index_name).search(query, query_params=params)

        slims: list[SlimChunk] = []
        for doc in results.docs:
            slims.append(
                SlimChunk(
                    chunk_id=doc.id.split(":")[-1],
                    title=_decode(doc.title),
                    source=_decode(doc.source),
                    source_url=_decode(doc.source_url),
                    snippet=_decode(getattr(doc, "snippet", "")),
                    acl_groups=[],
                    score=1.0 - float(doc.score),
                )
            )
        self._tracer.event("redis.slim.searched", returned=len(slims))
        return slims

    def fetch_fat(self, chunk_id: str) -> FatChunk | None:
        """Fetch one fat document via ``JSON.GET`` on its key."""
        fat_key = f"{self._namespace}:fat:{chunk_id}"
        try:
            raw = self._client.json().get(fat_key)
        except Exception:  # noqa: BLE001 - a missing/broken doc degrades gracefully
            raw = None
        self._tracer.event("redis.fat.fetched", chunk_id=chunk_id, hit=raw is not None)
        if not raw:
            return None
        # RedisJSON returns the object (or a single-element list at the root path).
        payload = raw[0] if isinstance(raw, list) else raw
        if not isinstance(payload, dict):
            return None
        return FatChunk(**payload)


def _decode(value: object) -> str:
    """Decode a possibly-bytes RediSearch field to a string."""
    return value.decode() if isinstance(value, bytes) else str(value)
