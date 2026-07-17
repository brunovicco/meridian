"""Unit tests for the structured query builder and sanitiser.

These pin down the two structural safety properties - mandatory ACL scoping and
injection rejection - plus the field-taxonomy compilation, all without touching
Redis.
"""

from meridian.application.query.builder import ServiceQueryBuilder
from meridian.application.query.sanitizer import NO_MATCH_QUERY, sanitize_search_query
from meridian.domain.models.service_filter import ServiceFilterModel


def _builder() -> ServiceQueryBuilder:
    """Return a fresh query builder."""
    return ServiceQueryBuilder()


def test_query_always_scoped_to_acl_groups() -> None:
    """Every compiled query begins with the visibility clause."""
    q = _builder().build_service_query(acl_groups=["payments", "platform"], filters=ServiceFilterModel())
    assert q.startswith("@visibility:{payments | platform}")


def test_no_groups_produces_impossible_clause() -> None:
    """With no groups the query matches nothing - fail closed."""
    q = _builder().build_service_query(acl_groups=[], filters=ServiceFilterModel())
    assert "__none__" in q


def test_tag_field_is_exact_match() -> None:
    """A tag field compiles to an exact tag clause."""
    q = _builder().build_service_query(acl_groups=["platform"], filters=ServiceFilterModel(team="platform"))
    assert "@team:{platform}" in q


def test_boolean_tag_compiles_to_true_false() -> None:
    """A boolean tag field compiles to a true/false tag."""
    q = _builder().build_service_query(acl_groups=["platform"], filters=ServiceFilterModel(has_owner=False))
    assert "@has_owner:{false}" in q


def test_numeric_range_is_bounded() -> None:
    """Numeric bounds compile to a RediSearch range clause."""
    q = _builder().build_service_query(
        acl_groups=["platform"],
        filters=ServiceFilterModel(min_dependencies=5, max_dependencies=10),
    )
    assert "@dependencies:[5 10]" in q


def test_special_characters_are_escaped() -> None:
    """Tag values with RediSearch specials are backslash-escaped."""
    q = _builder().build_service_query(acl_groups=["team&ops"], filters=ServiceFilterModel())
    assert "team\\&ops" in q


def test_sanitizer_passes_clean_query() -> None:
    """A normal query is returned unchanged."""
    query = "@visibility:{platform} @team:{platform}"
    assert sanitize_search_query(query) == query


def test_sanitizer_does_not_match_forbidden_substrings_inside_acl_values() -> None:
    """A group such as asset must not be mistaken for the SET command."""
    query = "@visibility:{asset}"
    assert sanitize_search_query(query) == query


def test_sanitizer_rejects_unscoped_query() -> None:
    """A query without mandatory visibility fails closed."""
    assert sanitize_search_query("@team:{platform}") == NO_MATCH_QUERY


def test_sanitizer_blocks_forbidden_verb() -> None:
    """A query containing a forbidden verb becomes an impossible query."""
    assert sanitize_search_query("@visibility:{x} LIMIT 0 10000") == NO_MATCH_QUERY


def test_sanitizer_blocks_overlength() -> None:
    """An over-length query becomes an impossible query."""
    assert sanitize_search_query("@visibility:{" + "a" * 600 + "}") == NO_MATCH_QUERY


def test_sanitizer_blocks_control_characters() -> None:
    """A query with control characters becomes an impossible query."""
    assert sanitize_search_query("@visibility:{x}\x00") == NO_MATCH_QUERY
