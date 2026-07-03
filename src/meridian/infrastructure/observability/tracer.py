"""Structured tracer implementations.

The platform's first principle is "observability before optimisation": every
routing decision and retrieval emits a structured event so behaviour is
explainable from the trace alone. These implementations satisfy the
:class:`~meridian.domain.interfaces.Tracer` protocol.

* :class:`StructuredTracer` writes one JSON object per event to stdout. In a
  real deployment the same events would flow to Langfuse or an OpenTelemetry
  collector; the interface is what matters, not the sink.
* :class:`NullTracer` discards everything, for tests that do not assert on
  telemetry.
"""

import json
import sys
import time
from typing import TextIO


class StructuredTracer:
    """Emit newline-delimited JSON events to a stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        """Create a tracer writing to ``stream`` (stdout by default)."""
        self._stream = stream or sys.stdout

    def event(self, name: str, **fields: object) -> None:
        """Write one structured event.

        :param name: The event name, e.g. ``router.route``.
        :param fields: Arbitrary structured fields to attach.
        """
        record = {"ts": round(time.time(), 3), "event": name, **fields}
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()


class NullTracer:
    """A tracer that discards every event (for tests)."""

    def event(self, name: str, **fields: object) -> None:
        """Do nothing."""
        return None
