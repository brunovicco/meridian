"""Unit tests for the fat/slim split and the DSPy grounding reward.

The fat/slim tests assert the defining property: search returns lean slim
projections (no full text), and the fat body is fetched separately by id. The
reward tests pin the grounding scorer that drives the DSPy Refine loop; they run
without the ``dspy`` package because the reward function itself is pure Python.
"""

from meridian.domain.models.knowledge import FatChunk
from meridian.infrastructure.dspy.groq import grounding_reward
from meridian.infrastructure.observability.tracer import NullTracer
from meridian.infrastructure.vectorstore.in_memory_store import InMemoryVectorStore


class _Prediction:
    """Minimal stand-in for a dspy.Prediction carrying an answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer


def _fat(chunk_id: str, text: str, groups: list[str]) -> FatChunk:
    """Build a fat chunk for tests."""
    return FatChunk(
        chunk_id=chunk_id,
        title=f"Title {chunk_id}",
        text=text,
        source=f"Source {chunk_id}",
        source_url=f"https://wiki/{chunk_id}",
        acl_groups=groups,
    )


def test_to_slim_drops_full_text() -> None:
    """The slim projection carries a snippet, never the full text."""
    fat = _fat("a", "word " * 100, ["platform"])
    slim = fat.to_slim(snippet_chars=50)
    assert len(slim.snippet) <= 50
    assert not hasattr(slim, "text")


def test_search_slim_returns_projections_not_bodies() -> None:
    """Slim search returns slim projections; the body is not present."""
    store = InMemoryVectorStore(tracer=NullTracer())
    store.upsert_fat_chunks([_fat("a", "full body text here", ["platform"])], [[1.0, 0.0]])
    results = store.search_slim([1.0, 0.0], _user(["platform"]), 3)
    assert results
    assert results[0].chunk_id == "a"
    assert not hasattr(results[0], "text")


def test_fetch_fat_returns_full_body() -> None:
    """Fetching a fat document by id returns the full text."""
    store = InMemoryVectorStore(tracer=NullTracer())
    store.upsert_fat_chunks([_fat("a", "the complete body", ["platform"])], [[1.0, 0.0]])
    fat = store.fetch_fat("a")
    assert fat is not None
    assert fat.text == "the complete body"


def test_search_slim_respects_acl() -> None:
    """Slim search never returns a projection outside the user's groups."""
    store = InMemoryVectorStore(tracer=NullTracer())
    store.upsert_fat_chunks([_fat("secret", "restricted", ["security"])], [[1.0, 0.0]])
    assert store.search_slim([1.0, 0.0], _user(["platform"]), 3) == []
    assert store.search_slim([1.0, 0.0], _user([]), 3) == []


def test_reward_rewards_grounded_answer() -> None:
    """An answer overlapping its context scores highly."""
    context = "[Source: Runbook] The failover promotes the standby replica to primary."
    prediction = _Prediction("The failover promotes the standby replica to primary.")
    assert grounding_reward({"context": context}, prediction) >= 0.75


def test_reward_penalises_empty_answer() -> None:
    """An empty answer scores zero."""
    assert grounding_reward({"context": "anything"}, _Prediction("")) == 0.0


def test_reward_accepts_honest_decline() -> None:
    """An honest 'not found' is treated as grounded, not penalised."""
    prediction = _Prediction("I could not find that in the knowledge base.")
    score = grounding_reward({"context": ""}, prediction)
    assert score >= 0.5


def _user(groups: list[str]):
    """Build a user context for tests."""
    from meridian.domain.models import UserContext

    return UserContext(user_id="u", acl_groups=groups)
