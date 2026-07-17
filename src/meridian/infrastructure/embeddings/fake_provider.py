"""A deterministic, dependency-free embedding provider.

This is the piece that lets the whole system run on a laptop with no network,
no API key, and no GPU - which is exactly what you want when demonstrating the
architecture live. It produces stable, repeatable vectors from text using
hashed character n-grams projected into a fixed-dimensional space, then L2
normalises them.

it's emphatically *not* a good semantic embedder - it captures lexical overlap,
not meaning. That is fine for the reference: the point is to exercise the
routing math, the ambiguity rules, the caching, and the full request path
deterministically. Because it implements :class:`EmbeddingProvider`, swapping it
for the real Azure provider is a one-line change at the composition root. That
substitutability is the Dependency Inversion Principle doing its job.
"""

import hashlib
import math

from meridian.domain.interfaces import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing embedder for local runs and tests."""

    def __init__(self, dimension: int = 256) -> None:
        """Create a provider producing vectors of the given dimension.

        :param dimension: Output vector dimensionality. Larger reduces hash
            collisions between n-grams; 256 is ample for the demo catalog.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The fixed output dimensionality."""
        return self._dimension

    @property
    def cache_identity(self) -> str:
        """Identify the hashing algorithm and vector dimension."""
        return f"fake-hashing-v1:{self._dimension}"

    def embed_one(self, text: str) -> list[float]:
        """Embed one string into a normalised vector.

        The text is lowercased and split into character trigrams; each trigram
        is hashed to a bucket and a sign, and accumulates into that dimension.
        The result is L2-normalised so dot products behave as cosine similarity.

        :param text: The string to embed.
        :returns: A normalised vector of length :pyattr:`dimension`.
        """
        vec = [0.0] * self._dimension
        cleaned = (text or "").lower().strip()
        if not cleaned:
            return vec

        tokens = cleaned.split()
        # Whole-word features plus character trigrams give the fake embedder
        # enough lexical resolution to separate the demo intents cleanly.
        features: list[str] = list(tokens)
        for token in tokens:
            padded = f"  {token} "
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))

        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, preserving order.

        :param texts: Strings to embed.
        :returns: One normalised vector per input, in the same order.
        """
        return [self.embed_one(t) for t in texts]
