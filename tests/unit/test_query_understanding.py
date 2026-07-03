"""Unit tests for the query-understanding output contract and coercion.

These tests pin down the contract's forgiveness on input and strictness on
output: malformed or drifting LLM responses are repaired into a valid object,
and unsalvageable ones fall back to a safe default rather than raising into the
request path.
"""

from meridian.application.services.query_understanding import (
    QueryUnderstanding,
    coerce_understanding,
)
from meridian.domain.models import RouteType


def test_coerce_clean_json() -> None:
    """A well-formed JSON payload validates unchanged."""
    raw = '{"route_type": "code_lookup", "search_terms": "retry decorator", "needs_clarification": false}'
    result = coerce_understanding(raw, fallback_query="original")
    assert result.route_type == RouteType.CODE_LOOKUP
    assert result.search_terms == "retry decorator"
    assert result.needs_clarification is False


def test_coerce_repairs_casing_and_string_boolean() -> None:
    """Uppercased keys and a stringified boolean are normalised."""
    raw = '{"Route_Type": "KNOWLEDGE_QA", "Search_Terms": "auth setup", "Needs_Clarification": "true"}'
    result = coerce_understanding(raw, fallback_query="original")
    assert result.route_type == RouteType.KNOWLEDGE_QA
    assert result.needs_clarification is True


def test_coerce_unknown_route_defaults_to_qa() -> None:
    """An unrecognised route label maps to the safe knowledge-QA default."""
    raw = '{"route_type": "banana", "search_terms": "something"}'
    result = coerce_understanding(raw, fallback_query="original")
    assert result.route_type == RouteType.KNOWLEDGE_QA


def test_coerce_invalid_json_falls_back() -> None:
    """Unparseable output yields the safe default using the original query."""
    result = coerce_understanding("not json at all", fallback_query="how do I deploy")
    assert result.route_type == RouteType.KNOWLEDGE_QA
    assert result.search_terms == "how do I deploy"


def test_coerce_empty_search_terms_uses_fallback() -> None:
    """Empty search terms are replaced by the original query, never left blank."""
    raw = '{"route_type": "knowledge_qa", "search_terms": ""}'
    result = coerce_understanding(raw, fallback_query="fallback query")
    assert result.search_terms == "fallback query"


def test_contract_rejects_empty_terms_directly() -> None:
    """The Pydantic model itself refuses empty search terms."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueryUnderstanding(route_type=RouteType.KNOWLEDGE_QA, search_terms="   ")
