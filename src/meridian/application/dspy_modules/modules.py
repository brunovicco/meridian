"""Application-layer DSPy module: routing via Predict behind the output contract.

:class:`DSPyRouterModule` is the application-layer DSPy module. It wraps
``dspy.Predict`` over :class:`KnowledgeRouteSignature` and coerces the output
through the same Pydantic contract the rest of the system uses, so the real
DSPy path and the fake path behave identically from the caller's point of view.

Infrastructure-specific DSPy code (Grok configuration, the grounded-answer
signature, the Refine self-correction module) lives in
``infrastructure/dspy/grok.py`` because it depends on external services, not on
application-layer abstractions.
"""

from typing import Any

from meridian.application.dspy_modules.signatures import _DSPY_AVAILABLE
from meridian.application.services.query_understanding import (
    QueryUnderstanding,
    coerce_understanding,
)

if _DSPY_AVAILABLE:  # pragma: no cover - requires dspy
    import dspy

    from meridian.application.dspy_modules.signatures import KnowledgeRouteSignature

    class DSPyRouterModule(dspy.Module):
        """Routing via ``dspy.Predict`` behind the Pydantic output contract."""

        def __init__(self) -> None:
            """Construct the underlying predictor."""
            super().__init__()
            self._predict = dspy.Predict(KnowledgeRouteSignature)

        def forward(self, question: str, candidate_intents: list[str]) -> QueryUnderstanding:
            """Predict the route and coerce it into the validated contract.

            The raw prediction is passed through :func:`coerce_understanding`,
            so the same drift-absorbing contract governs the real DSPy path and
            the fake path identically.

            :param question: The developer's question.
            :param candidate_intents: The router's candidate intent names.
            :returns: A validated :class:`QueryUnderstanding`.
            """
            prediction = self._predict(
                question=question,
                candidate_intents=", ".join(candidate_intents),
            )
            raw: dict[str, Any] = {
                "route_type": getattr(prediction, "route_type", "knowledge_qa"),
                "search_terms": getattr(prediction, "search_terms", question),
                "needs_clarification": getattr(prediction, "needs_clarification", "false"),
            }
            return coerce_understanding(raw, fallback_query=question)
