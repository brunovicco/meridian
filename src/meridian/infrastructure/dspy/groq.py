"""Groq DSPy infrastructure: signatures, reward, and the Refine module.

Contains everything needed to run grounded generation through DSPy on Groq:

* DSPy signatures declaring the grounded-answer input/output contract.
* The grounding reward function that drives ``dspy.Refine`` self-correction.
* ``configure_groq_lm`` for wiring DSPy to the Groq endpoint.
* ``DSPyRefineModule``: the self-correcting generation module used by
  :class:`GroqDSPyLLMProvider`.

All routing-specific DSPy code (``DSPyRouterModule``, ``KnowledgeRouteSignature``)
lives in ``application/dspy_modules`` because it wraps the application-layer
output contract (:class:`QueryUnderstanding`). This file holds only the pieces
that depend on DSPy and the environment but not on application-layer abstractions.
"""

import os
import re
from typing import Any

try:  # pragma: no cover - exercised only with dspy installed
    import dspy

    _DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DSPY_AVAILABLE = False


class DSPyUnavailable(RuntimeError):
    """Raised when a real DSPy module is used without the ``dspy`` package."""


def dspy_available() -> bool:
    """Whether the ``dspy`` package is importable in this environment."""
    return _DSPY_AVAILABLE


def configure_groq_lm(
    *,
    model: str,
    api_key: str | None = None,
    api_base: str,
) -> bool:
    """Configure DSPy to use Groq as its language model.

    Groq is reached through DSPy's ``dspy.LM`` abstraction, which speaks the
    OpenAI-compatible protocol the Groq API exposes. Configuration is read from
    the environment (twelve-factor): ``GROQ_API_KEY`` for the key, with the model
    and base URL overridable.

    The call is a no-op returning ``False`` if ``dspy`` is missing or no key is
    present, so the caller can fall back gracefully.

    :param model: Groq model name from application settings.
    :param api_key: Groq API key; defaults to ``$GROQ_API_KEY``.
    :param api_base: API base URL from application settings.
    :returns: ``True`` if DSPy was configured with Groq, ``False`` otherwise.
    """
    if not _DSPY_AVAILABLE:
        return False

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return False

    lm = dspy.LM(model, api_key=key, api_base=api_base)  # pragma: no cover - network
    # dspy.Refine's own internal feedback step (OfferFeedback) JSON-encodes its
    # float/list arguments before passing them to a signature that types them
    # as float/list[str], so its type checker flags a mismatch against dspy's
    # own output on every refinement retry. This is internal to dspy.Refine,
    # not a signal about our signatures, so the warning is disabled here
    # rather than suppressed at the logging layer.
    dspy.configure(lm=lm, warn_on_type_mismatch=False)  # pragma: no cover - network
    return True


# Grounding heuristics used by the reward function.
_SOURCE_MENTION = re.compile(r"\bsource\b|\[source:", re.IGNORECASE)
_HEDGE_WITHOUT_CONTEXT = re.compile(
    r"\b(i think|probably|might be|i believe|as far as i know|i guess)\b", re.IGNORECASE
)


def grounding_reward(arguments: dict[str, Any], prediction: Any) -> float:
    """Score how well a generated answer is grounded in its context.

    Acts as the reward function for the ``dspy.Refine`` loop. Four checks, each
    worth a quarter, mirroring the production compliance reward's shape:

    * **Non-empty** - the answer actually says something.
    * **Grounded in context** - the answer's substantive words overlap the
      supplied context (it's not inventing content wholesale).
    * **Cites or declines** - it references a source, or it honestly declines.
    * **No unsupported hedging** - it does not hedge vaguely as a substitute for
      grounding.

    :param arguments: The original inputs, including the ``context``.
    :param prediction: The DSPy prediction carrying the generated ``answer``.
    :returns: A score in ``[0.0, 1.0]``; 1.0 clears the Refine threshold.
    """
    answer = str(getattr(prediction, "answer", "") or "").strip()
    context = str(arguments.get("context", "") or "")

    if not answer:
        return 0.0

    declined = "not found" in answer.lower() or "could not find" in answer.lower()

    non_empty = bool(answer)
    cites_or_declines = bool(_SOURCE_MENTION.search(answer)) or declined or "http" in answer
    no_unsupported_hedge = not bool(_HEDGE_WITHOUT_CONTEXT.search(answer))

    context_words = {w.lower() for w in re.findall(r"\w{5,}", context)}
    answer_words = [w.lower() for w in re.findall(r"\w{5,}", answer)]
    if not answer_words:
        grounded = declined
    else:
        overlap = sum(1 for w in answer_words if w in context_words) / len(answer_words)
        grounded = declined or overlap >= 0.3

    score = non_empty + grounded + cites_or_declines + no_unsupported_hedge
    return score / 4.0


if _DSPY_AVAILABLE:  # pragma: no cover - requires dspy

    class GroundedAnswerSignature(dspy.Signature):
        """Answer a question strictly from the provided context, with citation.

        Ground every claim in the context. If the context is insufficient, say
        so plainly rather than speculating.
        """

        context: str = dspy.InputField(
            desc="Retrieved knowledge context, each block labelled with its source."
        )
        question: str = dspy.InputField(desc="The developer's question.")
        answer: str = dspy.OutputField(
            desc="A grounded answer citing the source(s), or an honest 'not found'."
        )

    class _AnswerPredictor(dspy.Module):
        """Base predictor wrapped by ``dspy.Refine`` for regeneration."""

        def __init__(self) -> None:
            """Construct the grounded-answer predictor."""
            super().__init__()
            self._predict = dspy.Predict(GroundedAnswerSignature)

        def forward(self, context: str, question: str) -> Any:
            """Generate one grounded answer for the context and question."""
            return self._predict(context=context, question=question)

    class DSPyRefineModule(dspy.Module):
        """Grounded answering with ``dspy.Refine`` self-correction.

        Generates an answer, scores it with :func:`grounding_reward`, and
        regenerates up to ``n`` times until the score reaches ``threshold``.
        This is the knowledge-domain analogue of the production advisor's
        compliance-driven Refine loop.
        """

        def __init__(self, *, n: int = 3, threshold: float = 1.0) -> None:
            """Wrap the predictor in a Refine loop.

            :param n: Maximum generation attempts.
            :param threshold: Reward score that ends the loop early.
            """
            super().__init__()
            self._refine = dspy.Refine(
                module=_AnswerPredictor(),
                N=n,
                reward_fn=grounding_reward,
                threshold=threshold,
            )

        def forward(self, context: str, question: str) -> str:
            """Return the best-scoring grounded answer for the inputs."""
            result = self._refine(context=context, question=question)
            return str(getattr(result, "answer", "") or "")
