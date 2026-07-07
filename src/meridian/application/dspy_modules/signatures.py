"""DSPy routing signature for the knowledge-router module.

Declares the input/output contract for :class:`DSPyRouterModule`. The signature
subclasses ``dspy.Signature`` when the package is installed; when it's not, the
module remains importable so the fake path still runs without errors.

Grounding signatures and the reward function (used by the generation pipeline)
live in ``infrastructure/dspy/groq.py`` because they belong to the infrastructure
layer alongside :class:`GroqDSPyLLMProvider`.
"""

try:  # pragma: no cover - exercised only with dspy installed
    import dspy

    _DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DSPY_AVAILABLE = False

if _DSPY_AVAILABLE:  # pragma: no cover - requires dspy

    class KnowledgeRouteSignature(dspy.Signature):
        """Classify a developer question and produce retrieval terms.

        Decide which pipeline should serve the question and rewrite it into
        retrieval-optimised search terms.
        """

        question: str = dspy.InputField(desc="The developer's natural-language question.")
        candidate_intents: str = dspy.InputField(desc="Comma-separated router candidates.")
        route_type: str = dspy.OutputField(
            desc="One of: knowledge_qa, code_lookup, structured_query, out_of_scope."
        )
        search_terms: str = dspy.OutputField(desc="Retrieval-optimised query text.")
        needs_clarification: str = dspy.OutputField(desc="'true' or 'false'.")
