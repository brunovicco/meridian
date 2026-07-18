"""The semantic router (layer 1 of routing).

The router turns a natural-language query into a :class:`RouterResult`: a ranked
list of candidate intents with scores, plus an ambiguity verdict. It does *not*
decide what to do with that verdict - that is the routing engine's job (layer
2). This separation keeps the scoring math free of business policy.

Lifecycle:

* At construction the router is given an intent catalog (positive and negative
  example phrases per intent), an :class:`EmbeddingProvider`, a
  :class:`VectorStore`, and an :class:`AmbiguityConfig`.
* ``build`` embeds the catalog into per-intent matrices, normalises them, and
  persists them to the vector store under a content ``fingerprint``. If the
  store already holds matrices for that fingerprint they are loaded instead of
  rebuilt - so an unchanged catalog is never re-embedded.
* ``route`` embeds a query, scores every intent, ranks them, and applies the
  three ambiguity rules.

The fingerprint is a SHA-256 of the catalog, thresholds, provider identity, and
dimension. Changing a model or deployment invalidates the cache even when the
new vectors happen to have the same dimensionality.
"""

import hashlib
import json

import numpy as np

from meridian.application.router.scoring import score_intent, unit_norm
from meridian.domain.interfaces import EmbeddingProvider, Tracer, VectorStore
from meridian.domain.models import RouterResult, ScoredIntent
from meridian.domain.policies import AmbiguityConfig


class SemanticRouter:
    """Embedding-based intent router with negative-aware scoring."""

    def __init__(
        self,
        *,
        positive_texts: dict[str, list[str]],
        negative_texts: dict[str, list[str]],
        intent_thresholds: dict[str, float],
        embedder: EmbeddingProvider,
        store: VectorStore,
        config: AmbiguityConfig,
        tracer: Tracer,
    ) -> None:
        """Wire the router to its collaborators.

        :param positive_texts: Per-intent lists of phrases that exemplify it.
        :param negative_texts: Per-intent lists of confusable phrases from
            other intents, used to push the boundary away.
        :param intent_thresholds: Per-intent confidence thresholds.
        :param embedder: The embedding provider (injected).
        :param store: The vector store for matrix persistence (injected).
        :param config: Ambiguity thresholds and the negative penalty.
        :param tracer: Structured observability sink (injected).
        """
        self._positive_texts = positive_texts
        self._negative_texts = negative_texts
        self._thresholds = intent_thresholds
        self._embedder = embedder
        self._store = store
        self._config = config
        self._tracer = tracer

        self._intents: list[str] = sorted(positive_texts.keys())
        if not self._intents:
            raise ValueError("the route catalog must define at least one intent")
        empty_intents = [intent for intent in self._intents if not positive_texts[intent]]
        if empty_intents:
            raise ValueError(f"route intents require positive examples: {', '.join(empty_intents)}")
        self._pos_matrices: dict[str, np.ndarray] = {}
        self._neg_matrices: dict[str, np.ndarray] = {}
        self._fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """SHA-256 over catalog, thresholds, and embedding provider identity.

        Any change to catalog content, thresholds, dimension, or provider/model
        identity yields a new fingerprint and forces a rebuild.
        """
        payload = json.dumps(
            {
                "positives": self._positive_texts,
                "negatives": self._negative_texts,
                "thresholds": self._thresholds,
                "dimension": self._embedder.dimension,
                "embedding_provider": self._embedder.cache_identity,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build(self) -> None:
        """Load matrices from cache, or embed the catalog and persist them.

        Called once at startup. On a cache hit (matching fingerprint) the
        matrices are loaded from the store; on a miss they are embedded, saved,
        and kept in memory for the process lifetime.
        """
        cached = self._store.load_route_matrices(self._fingerprint)
        if cached is not None:
            positives, negatives = cached
            self._pos_matrices = {k: np.asarray(v, dtype=np.float32) for k, v in positives.items()}
            self._neg_matrices = {k: np.asarray(v, dtype=np.float32) for k, v in negatives.items()}
            self._tracer.event(
                "router.build.cache_hit",
                fingerprint=self._fingerprint,
                intents=len(self._pos_matrices),
            )
            return

        for intent in self._intents:
            pos_vecs = self._embedder.embed_many(self._positive_texts.get(intent, []))
            self._pos_matrices[intent] = unit_norm(np.asarray(pos_vecs, dtype=np.float32))
            neg_vecs = self._embedder.embed_many(self._negative_texts.get(intent, []))
            self._neg_matrices[intent] = (
                unit_norm(np.asarray(neg_vecs, dtype=np.float32))
                if neg_vecs
                else np.zeros((0, self._embedder.dimension), dtype=np.float32)
            )

        self._store.save_route_matrices(
            self._fingerprint,
            {k: v.tolist() for k, v in self._pos_matrices.items()},
            {k: v.tolist() for k, v in self._neg_matrices.items()},
        )
        self._tracer.event(
            "router.build.cache_miss",
            fingerprint=self._fingerprint,
            intents=len(self._pos_matrices),
        )

    def route(self, query: str, *, k: int = 3) -> RouterResult:
        """Score every intent for ``query`` and return a ranked, judged result.

        Ties (equal scores, which happen with near-duplicate catalog phrases or
        degenerate zero vectors) are broken by the intent's own confidence
        threshold, descending: the intent that demands more evidence to win
        signals a more specific, better-calibrated match, so it takes
        precedence over a laxer intent with the same raw score. Intents whose
        thresholds also tie fall back to intent name, ascending, so the order
        is always deterministic. This is an explicit sort key rather than an
        accident of dict order: without it, tie-break behaviour would depend on
        Python's sort stability plus whatever order intents happen to be built
        in, which is exactly the kind of hidden coupling that breaks silently
        on refactor. A tied ``best_intent`` still surfaces to the caller (trace
        events, the disambiguation prompt in :class:`AskService`, and per-route
        metrics), so the choice must be deterministic even though score ties
        are almost always also flagged ambiguous by rule 3 below.

        :param query: The user's natural-language message.
        :param k: How many top candidates to include in ``topk``.
        :returns: A :class:`RouterResult` with scores and an ambiguity verdict.
        """
        cleaned = " ".join((query or "").strip().split())
        q_vec = unit_norm(np.asarray(self._embedder.embed_one(cleaned), dtype=np.float32))

        scored = [
            ScoredIntent(
                intent=intent,
                score=score_intent(
                    self._pos_matrices.get(intent, np.zeros((0, 0), dtype=np.float32)),
                    self._neg_matrices.get(intent, np.zeros((0, 0), dtype=np.float32)),
                    q_vec,
                    self._config.negative_penalty,
                ),
            )
            for intent in self._intents
        ]
        scored.sort(key=lambda s: (-s.score, -self._threshold_for(s.intent), s.intent))
        topk = scored[: max(1, k)]

        ambiguous, rule = self._detect_ambiguity(topk)

        result = RouterResult(
            query=cleaned,
            best_intent=topk[0].intent,
            best_score=topk[0].score,
            topk=topk,
            ambiguous=ambiguous,
            disambiguation_rule=rule,
        )
        self._tracer.event(
            "router.route",
            query=cleaned[:120],
            best_intent=result.best_intent,
            best_score=round(result.best_score, 4),
            ambiguous=result.ambiguous,
            rule=result.disambiguation_rule,
        )
        return result

    def _threshold_for(self, intent: str) -> float:
        """Return the calibrated confidence threshold for ``intent``.

        Falls back to the config default for intents absent from the
        per-intent threshold map.
        """
        return self._thresholds.get(intent, self._config.default_intent_threshold)

    def _detect_ambiguity(self, topk: list[ScoredIntent]) -> tuple[bool, str]:
        """Apply the three ambiguity rules in order.

        See :class:`AmbiguityConfig` for the meaning of each threshold. Returns
        the ambiguity verdict and the name of the rule that produced it, which
        is recorded in the trace for auditability.
        """
        best = topk[0]
        threshold = self._threshold_for(best.intent)
        runner_up = topk[1] if len(topk) > 1 else None
        gap = (best.score - runner_up.score) if runner_up else float("inf")

        # Rule 1: below the winning intent's own threshold.
        if best.score < threshold:
            return True, "top1_below_intent_threshold"

        # Rule 2: below the absolute floor, unless the margin is comfortable.
        if best.score < self._config.ambig_min:
            margin_ok = runner_up is not None and gap >= self._config.ambig_delta and runner_up.score > 0.0
            if not margin_ok:
                return True, "top1_below_ambig_min"

        # Rule 3: the top two are too close to separate.
        if runner_up is not None and gap < self._config.ambig_delta and runner_up.score > 0.0:
            return True, "margin_too_small"

        return False, "none"

    @property
    def fingerprint(self) -> str:
        """The catalog fingerprint, exposed for diagnostics and tests."""
        return self._fingerprint
