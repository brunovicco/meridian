"""Security-boundary and configuration contract tests."""

import json

import pytest
from pydantic import ValidationError

from meridian.domain.models import Answer, Citation, RouteType
from meridian.infrastructure.config.settings import Settings
from meridian.infrastructure.llm.providers import FakeLLMProvider
from meridian.infrastructure.redis.redis_vector_store import _escape_tag_value


def _settings(
    *,
    backend: str = "memory",
    embedding_backend: str = "fake",
    llm_backend: str = "fake",
    embedding_dimension: int = 256,
    top_k: int = 5,
) -> Settings:
    """Build valid settings while allowing one tested override."""
    return Settings(
        backend=backend,
        embedding_backend=embedding_backend,
        llm_backend=llm_backend,
        redis_url="redis://localhost:6379",
        embedding_dimension=embedding_dimension,
        top_k=top_k,
    )


def test_grounded_answer_requires_a_citation() -> None:
    """The domain model prevents uncited grounded answers."""
    with pytest.raises(ValidationError, match="grounded answers require"):
        Answer(
            text="An unsupported factual answer.",
            citations=[],
            route_type=RouteType.KNOWLEDGE_QA,
            grounded=True,
        )


def test_grounded_answer_accepts_evidence() -> None:
    """A grounded answer with provenance remains valid."""
    answer = Answer(
        text="A supported answer.",
        citations=[Citation(source="Runbook", source_url="https://wiki/runbook")],
        route_type=RouteType.KNOWLEDGE_QA,
        grounded=True,
    )
    assert answer.grounded is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "redsi"),
        ("embedding_backend", "typo"),
        ("llm_backend", "typo"),
    ],
)
def test_settings_reject_unknown_backend_selectors(field: str, value: str) -> None:
    """Misspelled selectors cannot silently fall back to fake components."""
    with pytest.raises(ValueError, match="must be one of"):
        if field == "backend":
            _settings(backend=value)
        elif field == "embedding_backend":
            _settings(embedding_backend=value)
        else:
            _settings(llm_backend=value)


@pytest.mark.parametrize(("field", "value"), [("embedding_dimension", 0), ("top_k", -1)])
def test_settings_reject_non_positive_numbers(field: str, value: int) -> None:
    """Invalid dimensions and retrieval limits fail at startup."""
    with pytest.raises(ValueError, match="greater than zero"):
        if field == "embedding_dimension":
            _settings(embedding_dimension=value)
        else:
            _settings(top_k=value)


def test_redis_acl_value_escapes_query_delimiters() -> None:
    """Directory group names cannot escape a RediSearch TAG clause."""
    assert _escape_tag_value("payments}|*") == r"payments\}\|\*"


def test_fake_llm_classifies_only_the_question_text() -> None:
    """The schema's code_lookup label must not classify every prompt as code."""
    prompt = (
        "Classify and respond with JSON. "
        'Schema: {"route_type": ["knowledge_qa", "code_lookup"]}. '
        "Question: how do I configure authentication?\n"
        "Router candidates: ['knowledge_qa', 'code_lookup']"
    )
    payload = json.loads(FakeLLMProvider().complete(prompt, system="You output only valid JSON."))
    assert payload["route_type"] == "knowledge_qa"


def test_fake_llm_selects_relevant_context_instead_of_concatenating() -> None:
    """The deterministic demo answer does not mix unrelated runbooks."""
    prompt = (
        "Context from the internal knowledge base:\n\n"
        "[Source: Auth]\nConfigure payments authentication with a client credential.\n\n"
        "[Source: Database]\nPromote the standby during database failover.\n\n"
        "Question: how do I configure payments authentication?\n\n"
        "Answer the question using only the context above."
    )

    answer = FakeLLMProvider().complete(prompt)

    assert "client credential" in answer
    assert "standby" not in answer


def test_fake_llm_declines_when_context_is_only_lexically_adjacent() -> None:
    """Accessible but unrelated context cannot become a fabricated answer."""
    prompt = (
        "Context from the internal knowledge base:\n\n"
        "[Source: Auth]\nConfigure payments authentication with a client credential.\n\n"
        "Question: show the security post mortem for the payments outage\n\n"
        "Answer the question using only the context above."
    )

    assert FakeLLMProvider().complete(prompt) == "INSUFFICIENT_CONTEXT"
