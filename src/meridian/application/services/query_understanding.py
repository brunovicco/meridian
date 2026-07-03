"""DSPy-style output contract for query understanding.

In a system with an LLM in the loop, the most common source of instability is
not the model - it's the absence of a contract on what the model must produce.
This module defines that contract for the query-understanding step: a Pydantic
schema the LLM output must satisfy before any downstream pipeline runs, plus a
coercion layer that absorbs the format drift LLMs routinely exhibit (a field in
unexpected casing, a number serialised as a string, a boolean rendered as the
word "true").

The design mirrors DSPy's discipline: a *signature* declares the input/output
fields and their meaning, a Pydantic model enforces the output shape, and a
coercion function normalises rather than fails when the model wanders slightly.
When the model provider ships a new version, the fix is to recompile and
re-validate against a held-out set - not to hand-edit a prompt and hope.

The signature is expressed as a plain class with a docstring so the reference
runs without a hard dependency on the ``dspy`` package. Adopting DSPy proper is
a drop-in: the ``QueryUnderstanding`` fields map one-to-one onto a
``dspy.Signature``.
"""

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from meridian.domain.models import RouteType


class QueryUnderstandingSignature:
    """Declarative I/O contract for the query-understanding LLM step.

    ----- Signature -----
    Input:
        query (str): the developer's natural-language question.
        candidate_intents (list[str]): the router's top candidates, for context.
    Output:
        route_type (RouteType): which pipeline should serve the query.
        search_terms (str): a cleaned, expanded query optimised for retrieval.
        needs_clarification (bool): true if the question is under-specified.
    ---------------------

    Adopting DSPy proper means turning the fields above into
    ``dspy.InputField``/``dspy.OutputField`` declarations on a
    ``dspy.Signature`` subclass; the semantics are identical.
    """

    INPUT_FIELDS = ("query", "candidate_intents")
    OUTPUT_FIELDS = ("route_type", "search_terms", "needs_clarification")


class QueryUnderstanding(BaseModel):
    """The validated output contract for the query-understanding step.

    This is the Pydantic gate every LLM response must pass before a pipeline
    acts on it. Validation failures are caught by :func:`coerce_understanding`,
    which repairs common drift and, only if repair is impossible, falls back to
    a safe default rather than propagating an exception into the request path.
    """

    route_type: RouteType = Field(..., description="Which pipeline serves this query.")
    search_terms: str = Field(..., description="Retrieval-optimised query text.")
    needs_clarification: bool = Field(
        default=False, description="True when the query is too vague to answer well."
    )

    @field_validator("search_terms")
    @classmethod
    def _non_empty_terms(cls, value: str) -> str:
        """Guarantee retrieval always has something to search for."""
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("search_terms must not be empty")
        return cleaned


def _coerce_bool(value: Any) -> bool:
    """Interpret the many ways an LLM might render a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _coerce_route_type(value: Any) -> RouteType:
    """Map a loosely-formatted route label onto a canonical enum value."""
    if isinstance(value, RouteType):
        return value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return RouteType(text)
    except ValueError:
        return RouteType.KNOWLEDGE_QA


def coerce_understanding(raw: str | dict[str, Any], *, fallback_query: str) -> QueryUnderstanding:
    """Parse and repair a raw LLM response into a valid contract object.

    The function is deliberately forgiving on input format and strict on output
    shape. It accepts either a JSON string or an already-parsed dict, normalises
    the well-known drift patterns, and validates. If the payload cannot be
    salvaged it returns a safe default (route to knowledge QA using the original
    query as search terms) rather than raising - a single malformed response
    should degrade one answer, not break the request path.

    :param raw: The LLM output, as a JSON string or a dict.
    :param fallback_query: The original query, used to build a safe default.
    :returns: A validated :class:`QueryUnderstanding`.
    """
    data: dict[str, Any]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {}
    else:
        data = dict(raw)

    # Normalise casing of keys, then coerce each known field.
    lowered = {str(k).strip().lower(): v for k, v in data.items()}
    route_type = _coerce_route_type(lowered.get("route_type", RouteType.KNOWLEDGE_QA.value))
    search_terms = str(lowered.get("search_terms") or fallback_query).strip()
    needs_clarification = _coerce_bool(lowered.get("needs_clarification", False))

    try:
        return QueryUnderstanding(
            route_type=route_type,
            search_terms=search_terms,
            needs_clarification=needs_clarification,
        )
    except ValidationError:
        return QueryUnderstanding(
            route_type=RouteType.KNOWLEDGE_QA,
            search_terms=fallback_query.strip() or "unknown",
            needs_clarification=False,
        )
