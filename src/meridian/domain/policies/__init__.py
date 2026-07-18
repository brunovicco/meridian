"""Routing policy constants and the intent-to-route mapping.

This module holds pure policy: the thresholds and margins that decide when a
routing result is confident enough to act on, and the mapping from catalog
intent names to the :class:`RouteType` pipelines that serve them. It contains
no I/O and no framework code, so it can be unit-tested in isolation and read at
a glance during an interview.

The numbers here mirror the calibration used in the production system this
reference is modelled on. They are deliberately conservative: the cost of
routing to the wrong pipeline (a scope error) is treated as far more serious
than the cost of asking the user to disambiguate.
"""

from dataclasses import dataclass, field

from meridian.domain.models import RouteType


@dataclass(frozen=True)
class AmbiguityConfig:
    """Thresholds governing the three ambiguity rules.

    The rules are applied in order by the router:

    1. **Per-intent threshold** - if the top score is below the winning
       intent's own threshold, the result is ambiguous. Intents with more
       dispersed examples get lower thresholds.
    2. **Absolute minimum** (``ambig_min``) - if the top score is below this
       floor, the signal is too weak to trust, unless rule 3's margin is
       comfortable.
    3. **Margin** (``ambig_delta``) - if the gap between the top two scores is
       smaller than this, the two intents are too close to separate safely.

    These defaults match the production calibration: an absolute floor around
    0.78 and a separating margin around 0.04.
    """

    ambig_min: float = 0.78
    ambig_delta: float = 0.04
    high_confidence_margin: float = 0.10
    negative_penalty: float = 0.8
    default_intent_threshold: float = 0.75


# Per-intent similarity thresholds. Intents whose positive examples span a wider
# semantic area (like the catch-all knowledge QA) sit lower; sharply defined
# intents (like a structured service-catalog lookup) sit higher.
INTENT_THRESHOLDS: dict[str, float] = {
    "knowledge_qa": 0.72,
    "code_lookup": 0.76,
    "structured_query": 0.78,
    "greeting": 0.70,
}

# Mapping from catalog intent names to the pipeline that serves them. Keeping
# this as data rather than branching logic means adding a route is a one-line
# change, and the router never needs to know which pipeline exists.
INTENT_TO_ROUTE: dict[str, RouteType] = {
    "knowledge_qa": RouteType.KNOWLEDGE_QA,
    "code_lookup": RouteType.CODE_LOOKUP,
    "structured_query": RouteType.STRUCTURED_QUERY,
    "greeting": RouteType.OUT_OF_SCOPE,
}

# Calibration for the fake (hashing) embedder. The lexical hashing embedder
# produces a very different score distribution from a real semantic model -
# scores cluster lower and closer together - so the production thresholds would
# flag everything as ambiguous. These thresholds are recalibrated to the fake
# embedder's distribution so the local demo exercises every route and decision
# path. This is itself a lesson worth stating aloud: thresholds are a property
# of the embedding model, and switching models means recompiling the
# calibration, not editing prompts.
FAKE_AMBIGUITY = AmbiguityConfig(
    ambig_min=0.30,
    ambig_delta=0.02,
    high_confidence_margin=0.03,
    negative_penalty=0.8,
    default_intent_threshold=0.20,
)

FAKE_INTENT_THRESHOLDS: dict[str, float] = {
    "knowledge_qa": 0.20,
    "code_lookup": 0.22,
    "structured_query": 0.22,
    "greeting": 0.20,
}

# Calibration for the local, real semantic embedder (sentence-transformers'
# all-MiniLM-L6-v2). A general-purpose sentence embedder gives correct
# positive/negative separation but a different absolute geometry than the
# production model this reference is modelled on: cosine similarity between
# on-topic-but-different-intent phrases (e.g. "who owns the payments service"
# vs. "how do I configure authentication for the payments service") sits
# around 0.5-0.6 just from shared domain vocabulary, well above what the
# production ``negative_penalty`` of 0.8 was tuned to tolerate. Left at the
# production calibration, a correctly top-ranked, unambiguous match routinely
# scores 0.4-0.6 - under every production threshold - so the demo asks for
# disambiguation on almost every query even though the ranking itself is
# right and the margin to the runner-up is large.
#
# These numbers were derived empirically, not guessed: scored the catalog's
# own confusable pairs plus ~20 hand-written paraphrases per intent (not
# copy-pasted from the catalog) against MiniLM, and picked values that sit
# between the observed true-positive floor (~0.4-0.6 for a correct, unambiguous
# top pick) and the observed true-negative ceiling (~0.1-0.2 for genuinely
# out-of-scope queries like "what's the capital of France"). The lower
# negative_penalty keeps ranking correctness (already robust across penalty
# values) from being swamped by MiniLM's coarser separation. As with the fake
# calibration, this is a starting point to refine against real traffic, not a
# permanent constant - see the note on `for_embedding_backend`.
LOCAL_AMBIGUITY = AmbiguityConfig(
    ambig_min=0.35,
    ambig_delta=0.05,
    high_confidence_margin=0.15,
    negative_penalty=0.5,
    default_intent_threshold=0.35,
)

LOCAL_INTENT_THRESHOLDS: dict[str, float] = {
    "knowledge_qa": 0.32,
    "code_lookup": 0.35,
    "structured_query": 0.38,
    "greeting": 0.30,
}


@dataclass(frozen=True)
class RoutingPolicy:
    """Bundles the ambiguity config with the lookup tables.

    Passed to the routing engine at construction so that policy is injected, not
    hard-coded - which keeps the engine testable with alternative calibrations.
    """

    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    intent_thresholds: dict[str, float] = field(default_factory=lambda: dict(INTENT_THRESHOLDS))
    intent_to_route: dict[str, RouteType] = field(default_factory=lambda: dict(INTENT_TO_ROUTE))

    def threshold_for(self, intent: str) -> float:
        """Return the calibrated threshold for ``intent`` or the default."""
        return self.intent_thresholds.get(intent, self.ambiguity.default_intent_threshold)

    def route_for(self, intent: str) -> RouteType:
        """Map an intent name to its serving pipeline, defaulting to QA."""
        return self.intent_to_route.get(intent, RouteType.KNOWLEDGE_QA)

    @staticmethod
    def for_embedding_backend(backend: str) -> "RoutingPolicy":
        """Return a policy calibrated for the active embedding backend.

        Thresholds are a property of the embedding model's score geometry, not
        of the application: the fake hashing embedder and the local MiniLM
        embedder each need their own calibration because neither matches the
        production model's distribution. The Azure backend is left on the
        production calibration - it *is* the production embedder this
        reference models, so no separate profile exists (or has been
        validated) for it.

        :param backend: The embedding backend name (``fake``, ``azure``, or ``local``).
        :returns: A :class:`RoutingPolicy` calibrated for that backend.
        """
        if backend == "fake":
            return RoutingPolicy(
                ambiguity=FAKE_AMBIGUITY,
                intent_thresholds=dict(FAKE_INTENT_THRESHOLDS),
                intent_to_route=dict(INTENT_TO_ROUTE),
            )
        if backend == "local":
            return RoutingPolicy(
                ambiguity=LOCAL_AMBIGUITY,
                intent_thresholds=dict(LOCAL_INTENT_THRESHOLDS),
                intent_to_route=dict(INTENT_TO_ROUTE),
            )
        return RoutingPolicy()
