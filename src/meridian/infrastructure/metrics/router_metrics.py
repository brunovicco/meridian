"""Routing observability metrics.

Implements the "observability before optimisation" principle for the router. The
collector keeps fast in-process counters for hot-path threshold checks (is the
fallback rate alarming?) and optionally delegates to a durable backend for
multi-worker aggregation. The Redis backend uses a single HASH with ``HINCRBY``
and a rolling TTL, which aggregates correctly across Gunicorn/uvicorn workers -
each worker's counters land in the same HASH.

The collector detects model degradation the way the production system does: if
the share of routing decisions that fell back to the generic route exceeds a
threshold, that is a signal the compiled router is drifting and needs
recompilation. Catching that early is only possible because every routing
decision is recorded here.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Fire an alarm if more than this share of routings fall back to the generic
# route - a proxy for router degradation.
FALLBACK_RATE_ALARM_THRESHOLD = 0.25

_REDIS_METRICS_KEY = "metrics:router"
_REDIS_METRICS_TTL_SECONDS = 86_400  # 24h rolling window


@runtime_checkable
class MetricsBackend(Protocol):
    """Durable metrics sink. Implementations increment counters durably."""

    def increment(self, field_name: str, amount: int = 1) -> None:
        """Increment a counter identified by ``field_name``."""
        ...

    def get_all(self) -> dict[str, int]:
        """Return every stored counter."""
        ...

    def reset(self) -> None:
        """Clear all counters."""
        ...


class RedisMetricsBackend:
    """Metrics backend over a single Redis HASH using HINCRBY.

    A single HASH key holds every counter as a field, so aggregation across
    workers is correct without coordination. The TTL is renewed on every write,
    giving a rolling 24-hour window that cleans itself up.

    Example HASH contents::

        { "route:knowledge_qa": 42, "fallback:knowledge_qa": 3, "anaphora_hits": 15 }
    """

    def __init__(self, *, redis_client: Any, key: str = _REDIS_METRICS_KEY) -> None:
        """Bind to a Redis client and the HASH key.

        :param redis_client: A connected ``redis.Redis`` instance.
        :param key: The HASH key holding the counters.
        """
        self._client = redis_client
        self._key = key

    def increment(self, field_name: str, amount: int = 1) -> None:
        """Increment a HASH field and renew the rolling TTL."""
        try:
            pipe = self._client.pipeline()
            pipe.hincrby(self._key, field_name, amount)
            pipe.expire(self._key, _REDIS_METRICS_TTL_SECONDS)
            pipe.execute()
        except Exception:  # noqa: BLE001 - metrics must never break the request path
            pass

    def get_all(self) -> dict[str, int]:
        """Return every counter as an ``{field: int}`` dict."""
        try:
            raw = self._client.hgetall(self._key)
            if not raw:
                return {}
            return {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            return {}

    def reset(self) -> None:
        """Delete the HASH key."""
        try:
            self._client.delete(self._key)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class RouterMetricsCollector:
    """In-process routing counters with an optional durable backend.

    Without a backend it behaves as a per-worker singleton of in-memory
    counters. With one configured, each record also increments the durable
    store for multi-worker aggregation. The in-process counters drive the
    fast fallback-rate alarm without any I/O on the hot path.
    """

    route_counter: Counter[str] = field(default_factory=Counter)
    fallback_counter: Counter[str] = field(default_factory=Counter)
    anaphora_hits: int = 0
    anaphora_misses: int = 0
    coercion_fallbacks: int = 0
    total_requests: int = 0

    _backend: MetricsBackend | None = field(default=None, init=False, repr=False)

    def configure_backend(self, backend: MetricsBackend) -> None:
        """Register a durable backend for multi-worker aggregation.

        Call once at startup, before the first request.

        :param backend: A :class:`MetricsBackend` implementation.
        """
        self._backend = backend

    def record_routing(
        self,
        route: str,
        *,
        was_fallback: bool = False,
        anaphora_resolved: bool = False,
        coercion_applied: bool = False,
    ) -> None:
        """Record one routing decision.

        Updates in-process counters synchronously and mirrors them to the
        durable backend when configured. Emits nothing on the hot path beyond a
        counter bump.

        :param route: The route that was chosen.
        :param was_fallback: Whether the decision was a fallback.
        :param anaphora_resolved: Whether anaphora resolution fired.
        :param coercion_applied: Whether output coercion had to repair drift.
        """
        self.total_requests += 1
        self.route_counter[route] += 1
        if was_fallback:
            self.fallback_counter[route] += 1
        if anaphora_resolved:
            self.anaphora_hits += 1
        else:
            self.anaphora_misses += 1
        if coercion_applied:
            self.coercion_fallbacks += 1

        if self._backend is not None:
            self._backend.increment(f"route:{route}")
            self._backend.increment("total_requests")
            if was_fallback:
                self._backend.increment(f"fallback:{route}")
            self._backend.increment("anaphora_hits" if anaphora_resolved else "anaphora_misses")
            if coercion_applied:
                self._backend.increment("coercion_fallbacks")

    def fallback_rate(self) -> float:
        """Return the current fallback rate from in-process counters (0.0-1.0)."""
        total = sum(self.route_counter.values())
        if total == 0:
            return 0.0
        return sum(self.fallback_counter.values()) / total

    def is_degraded(self) -> bool:
        """Whether the fallback rate has crossed the alarm threshold.

        Only meaningful once enough requests have accumulated to be
        statistically useful, hence the minimum-sample guard.
        """
        return self.total_requests >= 20 and self.fallback_rate() > FALLBACK_RATE_ALARM_THRESHOLD

    def snapshot(self) -> dict[str, Any]:
        """Return a structured snapshot of the accumulated metrics."""
        attempts = self.anaphora_hits + self.anaphora_misses
        hit_rate = (self.anaphora_hits / attempts) if attempts else 0.0
        return {
            "event": "router_metrics.snapshot",
            "total_routings": sum(self.route_counter.values()),
            "route_distribution": dict(self.route_counter),
            "fallback_rate": self.fallback_rate(),
            "fallback_by_route": dict(self.fallback_counter),
            "anaphora_hit_rate": hit_rate,
            "coercion_fallbacks": self.coercion_fallbacks,
            "backend": type(self._backend).__name__ if self._backend else "in_process",
        }

    def reset(self) -> None:
        """Clear all in-process counters."""
        self.route_counter.clear()
        self.fallback_counter.clear()
        self.anaphora_hits = 0
        self.anaphora_misses = 0
        self.coercion_fallbacks = 0
        self.total_requests = 0
