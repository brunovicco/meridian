"""LLM providers: a deterministic fake and an Azure OpenAI skeleton.

The fake provider lets the full request path run without credentials. it's
rule-based, not intelligent: for the query-understanding step it returns valid
contract JSON, and for grounded generation it extractively summarises the
supplied context and echoes the insufficient-context marker when the context is
empty. That is enough to demonstrate the pipeline end to end and to keep tests
deterministic.

The Azure skeleton implements the same :class:`LLMProvider` interface, so the
composition root swaps between them via configuration - the same dependency
inversion pattern as the embedding providers.
"""

import json
import os

from meridian.domain.interfaces import LLMProvider

_INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


class FakeLLMProvider(LLMProvider):
    """Deterministic, rule-based stand-in for a real LLM.

    Detects which of the two prompt shapes it's being asked for - the
    JSON-only query-understanding step or grounded generation - and returns a
    plausible, valid response for each. No network, no randomness.
    """

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return a deterministic completion appropriate to the prompt shape.

        :param prompt: The composed prompt.
        :param system: The system instruction, used to detect the JSON path.
        :returns: Valid contract JSON, or an extractive grounded answer.
        """
        if system and "valid JSON" in system:
            return self._understanding_json(prompt)
        return self._grounded_answer(prompt)

    def _understanding_json(self, prompt: str) -> str:
        """Emit contract-shaped JSON for the query-understanding step.

        Uses light keyword heuristics to pick a route so the demo shows
        different routes firing, then hands the raw question through as search
        terms. The coercion layer would repair this even if the shape drifted.
        """
        lowered = prompt.lower()
        if any(word in lowered for word in ("who owns", "which team", "service catalog", "how many")):
            route = "structured_query"
        elif any(word in lowered for word in ("function", "class", "method", "code", "implement")):
            route = "code_lookup"
        else:
            route = "knowledge_qa"

        question = prompt.split("Question:", 1)[-1].split("Router candidates:", 1)[0].strip()
        return json.dumps(
            {
                "route_type": route,
                "search_terms": question or "general question",
                "needs_clarification": False,
            }
        )

    def _grounded_answer(self, prompt: str) -> str:
        """Produce an extractive answer from the context block in the prompt.

        Pulls the context out of the prompt and returns its most relevant lines,
        or the insufficient-context marker if there is no context - exactly the
        contract the RAG pipeline expects, so grounding detection works.
        """
        if "Context from the internal knowledge base:" not in prompt:
            return _INSUFFICIENT_MARKER
        context = prompt.split("Context from the internal knowledge base:", 1)[1]
        context = context.split("Question:", 1)[0]
        lines = [ln.strip() for ln in context.splitlines() if ln.strip() and not ln.startswith("[Source:")]
        if not lines:
            return _INSUFFICIENT_MARKER
        # Extractive: return the first couple of substantive lines as the answer.
        return " ".join(lines[:2])


class AzureLLMProvider(LLMProvider):  # pragma: no cover - depends on external SDK
    """Azure OpenAI chat completion provider (skeleton).

    Reads configuration from the environment (twelve-factor) and leaves the SDK
    call as a documented gap, mirroring the embedding provider. The retry/TLS
    scaffolding would be shared with the embedding client in a real build.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
    ) -> None:
        """Read configuration, preferring explicit args over the environment."""
        self._endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
        self._api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Call the Azure OpenAI chat endpoint (to be wired in)."""
        raise NotImplementedError(
            "Wire the Azure OpenAI chat completions SDK call here. The fake "
            "provider is used for local runs; see the composition root."
        )


class GrokDSPyLLMProvider(LLMProvider):
    """LLM provider backed by real DSPy modules running on Grok (xAI).

    This is the real DSPy path. On construction it configures DSPy to use Grok
    via ``XAI_API_KEY`` and builds the self-correcting ``DSPyRefineModule``.
    Generation prompts (which carry a context block) are answered through the
    Refine loop - generate, score for grounding, regenerate up to the budget.

    Grok is an xAI model, not an Anthropic one; it's wired here because it was
    chosen as the real backend. If ``dspy`` is not installed or ``XAI_API_KEY``
    is absent, construction raises so the composition root can fall back to the
    fake provider - the default demo therefore never depends on this path.
    """

    def __init__(self) -> None:
        """Configure Grok and build the DSPy modules, or fail for fallback."""
        from meridian.infrastructure.dspy.grok import (
            configure_grok_lm,
            dspy_available,
        )

        if not dspy_available():
            raise RuntimeError("dspy is not installed; cannot use the Grok/DSPy backend.")
        if not configure_grok_lm():
            raise RuntimeError("XAI_API_KEY not set; cannot configure Grok for DSPy.")

        # Imported here (not at module top) because the class is only defined
        # when dspy is installed, which the guard above has confirmed.
        from meridian.infrastructure.dspy.grok import DSPyRefineModule

        self._refine = DSPyRefineModule()

    def complete(self, prompt: str, *, system: str | None = None) -> str:  # pragma: no cover - network
        """Answer via the DSPy Refine loop when a context block is present.

        The RAG pipeline's generation prompt embeds the retrieved context under
        a known header; when present, the prompt is split into context and
        question and answered through the self-correcting Refine module. Other
        prompts (the JSON routing step) are answered with a direct DSPy predict.
        """
        marker = "Context from the internal knowledge base:"
        if marker in prompt:
            context = prompt.split(marker, 1)[1].split("Question:", 1)[0].strip()
            question = prompt.split("Question:", 1)[1].split("Answer the question", 1)[0].strip()
            return self._refine(context=context, question=question)

        import dspy

        predictor = dspy.Predict("prompt -> completion")
        return str(predictor(prompt=prompt).completion)
