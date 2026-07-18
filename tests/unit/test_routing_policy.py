"""Unit tests for the per-embedding-backend routing policy calibration.

Thresholds are a property of the embedding model's score geometry (see the
module docstring in ``domain.policies``), so each backend that needs its own
calibration gets its own profile. These tests lock in which backend maps to
which profile, so a future edit cannot silently point two backends at the
same thresholds without a test noticing.
"""

from meridian.domain.policies import (
    FAKE_AMBIGUITY,
    FAKE_INTENT_THRESHOLDS,
    LOCAL_AMBIGUITY,
    LOCAL_INTENT_THRESHOLDS,
    AmbiguityConfig,
    RoutingPolicy,
)


def test_fake_backend_uses_fake_calibration() -> None:
    """The hashing embedder's low, tightly-clustered scores get their own profile."""
    policy = RoutingPolicy.for_embedding_backend("fake")
    assert policy.ambiguity == FAKE_AMBIGUITY
    assert policy.intent_thresholds == FAKE_INTENT_THRESHOLDS


def test_local_backend_uses_local_calibration() -> None:
    """The MiniLM embedder's negative-penalty-sensitive scores get their own profile."""
    policy = RoutingPolicy.for_embedding_backend("local")
    assert policy.ambiguity == LOCAL_AMBIGUITY
    assert policy.intent_thresholds == LOCAL_INTENT_THRESHOLDS


def test_azure_backend_falls_back_to_production_calibration() -> None:
    """Azure is the production embedder this reference models, so it keeps the default."""
    policy = RoutingPolicy.for_embedding_backend("azure")
    assert policy.ambiguity == AmbiguityConfig()


def test_unknown_backend_falls_back_to_production_calibration() -> None:
    """An unrecognised backend name is not silently miscalibrated - it gets the default."""
    policy = RoutingPolicy.for_embedding_backend("something-new")
    assert policy.ambiguity == AmbiguityConfig()


def test_local_calibration_is_more_lenient_than_production() -> None:
    """The local profile must actually be looser, or it would not fix anything.

    MiniLM's true-positive scores for an unambiguous top pick land around
    0.4-0.8 (see the calibration note in domain.policies), well under the
    production thresholds (0.70-0.78) - if the local profile weren't strictly
    looser, every demo query would still be flagged ambiguous.
    """
    for intent, production_threshold in RoutingPolicy().intent_thresholds.items():
        assert LOCAL_INTENT_THRESHOLDS[intent] < production_threshold
    assert LOCAL_AMBIGUITY.ambig_min < AmbiguityConfig().ambig_min
    assert LOCAL_AMBIGUITY.negative_penalty < AmbiguityConfig().negative_penalty
