"""Application configuration, read from the environment (twelve-factor).

Config comes from the environment, never from code or checked-in files. This
module centralises that reading into a single typed object so the rest of the
system receives validated settings rather than scattered ``os.getenv`` calls.

The one switch that matters most for the demo is ``backend``: ``memory`` (the
default) runs everything in-process with the fake providers, so the reference
works with zero setup; ``redis`` uses Redis Stack and expects ``REDIS_URL`` to
point at it. Flipping that switch is how you show the same application code
running against two different infrastructures.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Typed, immutable view of the environment configuration."""

    backend: str
    """Vector store backend: ``memory`` or ``redis``."""

    embedding_backend: str
    """Embedding provider: ``fake``, ``azure``, or ``local`` (sentence-transformers)."""

    llm_backend: str
    """LLM provider: ``fake`` or ``azure``."""

    redis_url: str
    """Redis connection URL, used when ``backend == 'redis'``."""

    embedding_dimension: int
    """Vector dimensionality; must match the active embedding provider."""

    top_k: int
    """Number of chunks to retrieve per query."""

    @staticmethod
    def from_env() -> "Settings":
        """Build a :class:`Settings` from the process environment.

        Every value has a safe default that makes the system run locally with no
        configuration at all. Overriding any of them is a matter of setting the
        corresponding environment variable - no code change.
        """
        return Settings(
            backend=os.getenv("MERIDIAN_BACKEND", "memory").strip().lower(),
            embedding_backend=os.getenv("MERIDIAN_EMBEDDING_BACKEND", "fake").strip().lower(),
            llm_backend=os.getenv("MERIDIAN_LLM_BACKEND", "fake").strip().lower(),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            embedding_dimension=int(os.getenv("MERIDIAN_EMBEDDING_DIM", "256")),
            top_k=int(os.getenv("MERIDIAN_TOP_K", "5")),
        )
