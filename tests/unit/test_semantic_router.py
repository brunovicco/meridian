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


def test_exact_score_tie_breaks_by_higher_threshold() -> None:
    """Among equally-scored intents, the one demanding more confidence wins.

    A higher per-intent threshold signals a more specific, better-calibrated
    match, so it should take precedence over a laxer intent with the same raw
    score - even when the laxer intent's name would sort first alphabetically.
    """
    tracer = NullTracer()
    embedder = FakeEmbeddingProvider(dimension=32)
    router = SemanticRouter(
        positive_texts={"aaa_intent": ["deploy the service"], "zzz_intent": ["deploy the service"]},
        negative_texts={"aaa_intent": [], "zzz_intent": []},
        intent_thresholds={"aaa_intent": 0.1, "zzz_intent": 0.5},
        embedder=embedder,
        store=InMemoryVectorStore(tracer=tracer),
        config=AmbiguityConfig(),
        tracer=tracer,
    )
    router.build()

    result = router.route("deploy the service")

    assert result.best_intent == "zzz_intent"
    assert result.topk[0].score == pytest.approx(result.topk[1].score)


def test_exact_score_tie_with_equal_thresholds_breaks_by_intent_name() -> None:
    """When thresholds also tie, the intent name is the final, deterministic tiebreak."""
    tracer = NullTracer()
    embedder = FakeEmbeddingProvider(dimension=32)
    router = SemanticRouter(
        positive_texts={"zzz_intent": ["deploy the service"], "aaa_intent": ["deploy the service"]},
        negative_texts={"zzz_intent": [], "aaa_intent": []},
        intent_thresholds={"zzz_intent": 0.1, "aaa_intent": 0.1},
        embedder=embedder,
        store=InMemoryVectorStore(tracer=tracer),
        config=AmbiguityConfig(),
        tracer=tracer,
    )
    router.build()

    result = router.route("deploy the service")

    assert result.best_intent == "aaa_intent"
    assert result.topk[0].score == pytest.approx(result.topk[1].score)


def test_exact_score_tie_is_flagged_ambiguous() -> None:
    """A genuine tie for first place is always ambiguous, regardless of which side wins the sort.

    The margin between top two is zero, which is always below ``ambig_delta``,
    so rule 3 (margin too small) must fire even when the tied score clears
    every other threshold comfortably.
    """
    tracer = NullTracer()
    embedder = FakeEmbeddingProvider(dimension=32)
    router = SemanticRouter(
        positive_texts={"zzz_intent": ["deploy the service"], "aaa_intent": ["deploy the service"]},
        negative_texts={"zzz_intent": [], "aaa_intent": []},
        intent_thresholds={"zzz_intent": 0.1, "aaa_intent": 0.1},
        embedder=embedder,
        store=InMemoryVectorStore(tracer=tracer),
        config=AmbiguityConfig(),
        tracer=tracer,
    )
    router.build()

    result = router.route("deploy the service")

    assert result.ambiguous is True
    assert result.disambiguation_rule == "margin_too_small"
