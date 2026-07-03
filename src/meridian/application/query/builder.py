"""Structured, safe RediSearch query builder for the service catalog.

This is the counterpart to RAG for *structured* knowledge. Where RAG retrieves
loosely-relevant chunks, a structured question ("who owns the payments service",
"how many tier-1 services has the platform team") deserves a precise query whose
result is complete, not a top-K sample. Compiling the question into a RediSearch
expression is how you get that - the same lesson that, in an earlier project,
turned a failing RAG-over-tabular-data approach into a text-to-query one.

The builder classifies every field into one of three kinds and formats each
accordingly:

* **TAG** fields - exact match, e.g. ``@team:{platform}``.
* **TEXT** fields - fuzzy match with tokenisation rules, e.g. ``@name:(%%payments%%)``.
* **NUMERIC** fields - ranges, e.g. ``@dependencies:[5 +inf]``.

Two safety properties are structural, not optional:

* **Access scoping is mandatory.** Every catalog query is prefixed with a
  visibility clause over the caller's ACL groups, built here in code - the same
  discipline as row-level security in a portfolio query. A caller cannot
  construct a query without it.
* **Special characters are escaped.** RediSearch tag values routinely contain
  characters (``&``, ``(``, ``)``, ``+``, ``-``) that otherwise cause silent
  parse errors or empty results. The escaping table below covers them.
"""

import re
from typing import Any

from meridian.domain.models.service_filter import ServiceFilterModel

# Field taxonomy. Membership decides how a field is compiled. Keeping these as
# data means adding a field is a one-line change, not a new code branch.
SERVICE_TAG_FIELDS: set[str] = {"team", "domain", "tier", "lifecycle", "has_owner"}
SERVICE_TEXT_FIELDS: set[str] = {"name", "description"}
SERVICE_NUMERIC_FIELDS: set[str] = {"dependencies"}

# RediSearch tag values must escape these or the FT.SEARCH parser breaks or,
# worse, silently returns nothing. This mirrors the set hardened in production
# against real catalog values like "Alternatives & Custom" or "(recon) tax".
_REDISEARCH_TAG_SPECIAL: tuple[str, ...] = (
    "\\",
    "-",
    "_",
    ".",
    ",",
    ":",
    "/",
    "@",
    '"',
    "'",
    "|",
    "(",
    ")",
    "+",
    "&",
    "!",
    "#",
    "$",
    "%",
    "^",
    "*",
    "=",
    "~",
    "<",
    ">",
    "[",
    "]",
    ";",
)


class ServiceQueryBuilder:
    """Compiles a :class:`ServiceFilterModel` into a scoped RediSearch query."""

    def build_service_query(self, *, acl_groups: list[str], filters: ServiceFilterModel) -> str:
        """Build a catalog query scoped to the caller's visibility.

        The returned string always begins with the mandatory visibility clause
        over ``acl_groups``; the field filters follow. With no groups the query
        matches nothing - fail closed.

        :param acl_groups: The caller's ACL groups; the visibility scope.
        :param filters: The structured filter to compile.
        :returns: A RediSearch query string, always ACL-scoped.
        """
        if not acl_groups:
            # No visibility: an impossible clause so the search returns nothing.
            return "@visibility:{__none__}"

        scope = " | ".join(self._format_tag_value(group) for group in acl_groups)
        visibility_clause = f"@visibility:{{{scope}}}"

        field_clauses = self._compile_filters(filters)
        if not field_clauses:
            return visibility_clause
        return f"{visibility_clause} {field_clauses}"

    def _compile_filters(self, filters: ServiceFilterModel) -> str:
        """Compile the set filter fields into RediSearch clauses."""
        clauses: list[str] = []
        data = filters.model_dump(exclude_none=True)

        for field_name, value in data.items():
            if field_name in SERVICE_TAG_FIELDS:
                clauses.append(self._format_tag(field_name, value))
            elif field_name in SERVICE_TEXT_FIELDS:
                text_clause = self._format_text_fuzzy(field_name, str(value))
                if text_clause:
                    clauses.append(text_clause)

        # Numeric range on dependency count, assembled from the two bounds.
        min_deps = data.get("min_dependencies")
        max_deps = data.get("max_dependencies")
        if min_deps is not None or max_deps is not None:
            clauses.append(self._format_numeric_range("dependencies", min_deps, max_deps))

        return " ".join(clauses)

    def _format_tag(self, field_name: str, value: Any) -> str:
        """Format a TAG field clause with a boolean or escaped string value."""
        if isinstance(value, bool):
            return f"@{field_name}:{{{'true' if value else 'false'}}}"
        return f"@{field_name}:{{{self._format_tag_value(str(value))}}}"

    def _format_tag_value(self, value: str) -> str:
        """Escape a single tag value for safe inclusion in a tag clause.

        Braces are stripped (they delimit the tag), every RediSearch special
        character is backslash-escaped, and spaces are escaped so multi-word tag
        values match as a unit rather than as separate terms.
        """
        safe = value.replace("{", "").replace("}", "")
        for special in _REDISEARCH_TAG_SPECIAL:
            safe = safe.replace(special, f"\\{special}")
        return safe.strip().replace(" ", "\\ ")

    def _format_text_fuzzy(self, field_name: str, text: str) -> str:
        """Format a TEXT field clause with tokenised fuzzy matching.

        Tokenisation rules, matching the production builder:

        * numeric tokens of 1-2 digits match exactly;
        * numeric tokens of 3+ digits become a prefix (e.g. ``2024*``);
        * alphabetic tokens of <= 2 chars are dropped (too noisy);
        * tokens of 3 chars become a prefix (e.g. ``api*``);
        * tokens of 4+ chars use Levenshtein fuzzy matching (e.g. ``%%payments%%``).
        """
        cleaned = self._escape_text(text)
        if not cleaned:
            return ""
        terms = cleaned.split()
        if not terms:
            return ""

        patterns: list[str] = []
        for term in terms:
            if term.isdigit():
                patterns.append(term if len(term) <= 2 else f"{term}*")
                continue
            if len(term) <= 2:
                continue
            patterns.append(f"{term}*" if len(term) <= 3 else f"%%{term}%%")

        if not patterns:
            return f"@{field_name}:{terms[0]}*"
        return f"@{field_name}:({' '.join(patterns)})"

    def _format_numeric_range(self, field_name: str, minimum: int | None, maximum: int | None) -> str:
        """Format a NUMERIC field as an open or closed range clause."""
        lower = str(minimum) if minimum is not None else "-inf"
        upper = str(maximum) if maximum is not None else "+inf"
        return f"@{field_name}:[{lower} {upper}]"

    def _escape_text(self, raw: str) -> str:
        """Strip punctuation from free text as a defence against injection."""
        return re.sub(r"[^\w\s]", "", raw, flags=re.UNICODE).strip()
