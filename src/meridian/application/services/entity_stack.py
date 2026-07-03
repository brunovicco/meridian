"""Discussed-entity stack for multi-entity anaphora resolution.

Original problem (carried over from the system this reference is modelled on):
a per-user cache that stored only the *last* entity discussed cannot resolve a
follow-up like "which of those has more dependencies?" after the user asked
about two services in a row - the first entity is already gone.

Solution: a bounded LIFO stack of up to five discussed entities with
dedup-by-name, serialisable to JSON for temporary persistence in Redis under a
TTL. ``latest`` and ``previous`` give the two most recent entities, which is
exactly what a comparison ("which of the two…") needs. When the user writes a
pronoun instead of a name, the resolver injects these entities into the query
before it reaches the rest of the pipeline, so downstream never sees a bare
pronoun.

In this knowledge-platform domain the entities are services or topics the user
has been asking about, rather than financial products - but the mechanism is
identical.
"""

import json
from collections import deque
from dataclasses import dataclass, field

MAX_ENTITIES = 5
ANAPHORA_STACK_KEY_PREFIX = "anaphora:stack:"
ANAPHORA_TTL_SECONDS = 300


@dataclass
class DiscussedEntity:
    """An entity (service or topic) referenced in a conversational turn."""

    name: str
    route: str = ""
    turn_index: int = 0


@dataclass
class EntityStack:
    """A bounded LIFO stack of discussed entities with dedup by name."""

    entities: deque[DiscussedEntity] = field(default_factory=lambda: deque(maxlen=MAX_ENTITIES))

    def push(self, entity: DiscussedEntity) -> None:
        """Push an entity onto the stack, promoting a repeat to the top.

        A previously-mentioned entity is removed before the new mention is
        pushed, so mentioning the same service twice keeps it at the top rather
        than duplicating it - bounded memory, most-recent-relevant ordering.

        :param entity: The entity mentioned this turn.
        """
        deduped = [e for e in self.entities if e.name.lower() != entity.name.lower()]
        self.entities = deque(deduped, maxlen=MAX_ENTITIES)
        self.entities.appendleft(entity)

    @property
    def latest(self) -> DiscussedEntity | None:
        """The most recently discussed entity, or ``None`` if empty."""
        return self.entities[0] if self.entities else None

    @property
    def previous(self) -> DiscussedEntity | None:
        """The second-most-recent entity, for comparisons; ``None`` if absent."""
        return self.entities[1] if len(self.entities) > 1 else None

    @property
    def is_empty(self) -> bool:
        """Whether the stack currently holds no entities."""
        return len(self.entities) == 0

    def to_json(self) -> str:
        """Serialise the stack to a JSON string for Redis storage.

        :returns: A JSON array of the stacked entities, newest first.
        """
        return json.dumps(
            [{"name": e.name, "route": e.route, "turn_index": e.turn_index} for e in self.entities]
        )

    @classmethod
    def from_json(cls, raw: str) -> "EntityStack":
        """Reconstruct a stack from its JSON representation.

        Malformed input yields an empty stack rather than raising - a corrupt
        cache entry should degrade anaphora, not break the request.

        :param raw: The JSON string previously produced by :meth:`to_json`.
        :returns: The reconstructed :class:`EntityStack`.
        """
        stack = cls()
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                return stack
            # Rebuild oldest-last so the newest ends up on top after appendleft
            # semantics; here we append in stored order (already newest-first).
            for item in items:
                if isinstance(item, dict):
                    stack.entities.append(
                        DiscussedEntity(
                            name=item.get("name", ""),
                            route=item.get("route", ""),
                            turn_index=item.get("turn_index", 0),
                        )
                    )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return stack
