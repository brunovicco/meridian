"""Semantic-router lifecycle and cache identity tests."""

import pytest

from meridian.application.router.semantic_router import SemanticRouter
from meridian.domain.policies import AmbiguityConfig
from meridian.infrastructure.embeddings.fake_provider import FakeEmbeddingProvider
from meridian.infrastructure.observability.tracer import NullTracer
from meridian.infrastructure.vectorstore.in_memory_store import InMemoryVectorStore


class _AlternativeEmbeddingProvider(FakeEmbeddingProvider):
    """Provider with identical dimensions but a different vector identity."""

    @property
    def cache_identity(self) -> str:
        """Return a distinct model identity for fingerprinting."""
        return f"alternative-model:{self.dimension}"


def _router(embedder: FakeEmbeddingProvider) -> SemanticRouter:
    """Build an uninitialised router around the supplied provider."""
    tracer = NullTracer()
    return SemanticRouter(
        positive_texts={"knowledge_qa": ["help with a runbook"]},
        negative_texts={"knowledge_qa": []},
        intent_thresholds={"knowledge_qa": 0.2},
        embedder=embedder,
        store=InMemoryVectorStore(tracer=tracer),
        config=AmbiguityConfig(),
        tracer=tracer,
    )


def test_model_identity_invalidates_router_fingerprint() -> None:
    """Models with equal dimensions cannot share route matrices."""
    original = _router(FakeEmbeddingProvider(dimension=32))
    replacement = _router(_AlternativeEmbeddingProvider(dimension=32))
    assert original.fingerprint != replacement.fingerprint


def test_empty_route_catalog_fails_at_startup() -> None:
    """A malformed empty catalog raises an actionable startup error."""
    tracer = NullTracer()
    with pytest.raises(ValueError, match="at least one intent"):
        SemanticRouter(
            positive_texts={},
            negative_texts={},
            intent_thresholds={},
            embedder=FakeEmbeddingProvider(),
            store=InMemoryVectorStore(tracer=tracer),
            config=AmbiguityConfig(),
            tracer=tracer,
        )
