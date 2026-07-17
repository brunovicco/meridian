"""Domain models for the Meridian knowledge platform.

This module defines the core data structures that flow through the system.
They belong to the innermost layer of the Clean Architecture: they depend on
nothing outside the standard library and Pydantic, and every other layer is
free to depend on them.

The models here are deliberately behaviour-light. They describe *what* the
system reasons about (intents, routing decisions, retrieved knowledge) without
prescribing *how* any of it's computed. The "how" lives in the application and
infrastructure layers behind interfaces defined in ``domain.interfaces``.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class RouteType(str, Enum):
    """The four top-level actions the router can decide on.

    The knowledge platform serves developers who ask questions in natural
    language. Not every question should be answered the same way, so the router
    classifies each incoming message into one of these buckets before any
    retrieval or generation happens. Deciding this up front saves latency and
    cost (a greeting never needs a vector search) and improves accuracy (a
    structured-data question is better served by SQL than by RAG).
    """

    KNOWLEDGE_QA = "knowledge_qa"
    """Free-text question answerable from unstructured docs via RAG."""

    CODE_LOOKUP = "code_lookup"
    """Question about the codebase; retrieval favours source over prose."""

    STRUCTURED_QUERY = "structured_query"
    """Question best answered from structured data (service catalog, ownership)."""

    OUT_OF_SCOPE = "out_of_scope"
    """Greeting, chit-chat, or anything outside the platform's remit."""


class DecisionType(str, Enum):
    """The routing engine's verdict after weighing the router's raw signal.

    The semantic router (layer 1) produces a *signal*: scores per intent and an
    ambiguity flag. The routing engine (layer 2) turns that signal into an
    *action*, taking confidence and ambiguity into account. Keeping the two
    separate means the scoring math never has to know about business policy, and
    the policy never has to know about embeddings.
    """

    ROUTE_DIRECT = "route_direct"
    """High confidence, unambiguous: route straight to the target pipeline."""

    ASK_DISAMBIGUATION = "ask_disambiguation"
    """Low confidence or ambiguous: ask the user (or a small LLM) to clarify."""

    FALLBACK = "fallback"
    """No intent cleared its threshold: fall back to general knowledge QA."""


class ScoredIntent(BaseModel):
    """A single intent paired with its similarity score.

    Scores are produced by the semantic router as
    ``max(positive_similarity) - NEG_PENALTY * max(0, negative_similarity)``
    and are therefore not bounded to ``[0, 1]``; a strong negative match can
    push a score below zero. Consumers should treat scores as comparable within
    a single routing call, not as absolute probabilities.
    """

    intent: str = Field(..., description="Intent name, matching a catalog route.")
    score: float = Field(..., description="Similarity score; higher is better.")


class RouterResult(BaseModel):
    """The raw output of the semantic router (layer 1).

    This is the contract between the router and the routing engine. It carries
    everything the engine needs to make a decision without having to recompute
    anything: the winning intent, the full ranked list, whether the result was
    judged ambiguous, and which ambiguity rule (if any) fired. The
    ``disambiguation_rule`` field exists for observability - every routing
    decision should be explainable after the fact from the trace alone.
    """

    ok: bool = Field(default=True, description="False if routing failed and this is a fallback.")
    query: str = Field(..., description="The (preprocessed) user query that was routed.")
    best_intent: str = Field(..., description="Top-ranked intent name.")
    best_score: float = Field(..., description="Score of the top-ranked intent.")
    topk: list[ScoredIntent] = Field(default_factory=list, description="Full ranked candidate list.")
    ambiguous: bool = Field(default=False, description="True if the ambiguity rules flagged this result.")
    disambiguation_rule: str = Field(default="none", description="Name of the ambiguity rule that fired.")

    @property
    def runner_up(self) -> ScoredIntent | None:
        """The second-place intent, or ``None`` if only one candidate exists."""
        return self.topk[1] if len(self.topk) > 1 else None


class RoutingDecision(BaseModel):
    """The routing engine's final verdict (layer 2).

    Where :class:`RouterResult` is "what the embeddings say", this is "what we
    are going to do about it". It names the decision type, the resolved route,
    and a human-readable reason that is logged for auditability.
    """

    decision: DecisionType = Field(..., description="The action to take.")
    route_type: RouteType = Field(..., description="Which pipeline handles this query.")
    intent: str = Field(..., description="The resolved intent name.")
    reason: str = Field(default="", description="Human-readable justification for the trace.")
    router_result: RouterResult = Field(..., description="The raw signal this decision was based on.")


class KnowledgeChunk(BaseModel):
    """A retrievable unit of knowledge with its provenance.

    Every chunk carries the metadata needed both to answer the user (the text
    and its source URL for citation) and to enforce access control (the
    ``acl_groups`` it belongs to). Access control is applied as a metadata
    filter at retrieval time, never after the fact - a user must not be able to
    retrieve a chunk they cannot see, even transiently.
    """

    chunk_id: str = Field(..., description="Stable unique identifier for the chunk.")
    text: str = Field(..., description="The chunk's textual content.")
    source: str = Field(..., description="Human-readable source name, e.g. 'Payments Runbook'.")
    source_url: str = Field(..., description="Deep link back to the origin, used for citation.")
    acl_groups: list[str] = Field(default_factory=list, description="Groups permitted to read this chunk.")
    score: float = Field(default=0.0, description="Retrieval similarity score, set at query time.")


class Citation(BaseModel):
    """A single source reference attached to a generated answer."""

    source: str = Field(..., description="Human-readable source name.")
    source_url: str = Field(..., description="Deep link to the source.")


class Answer(BaseModel):
    """The final synthesised response returned to the caller.

    An answer without citations is, in this platform, a bug rather than a
    feature: a wrong answer stated confidently is worse than no answer, because
    a developer will trust it and break production. The generation step is
    instructed to ground every claim in the retrieved context and to say so
    plainly when the context does not support an answer.
    """

    text: str = Field(..., description="The synthesised natural-language answer.")
    citations: list[Citation] = Field(default_factory=list, description="Sources backing the answer.")
    route_type: RouteType = Field(..., description="Which pipeline produced this answer.")
    grounded: bool = Field(default=True, description="False when the model reported insufficient context.")

    @model_validator(mode="after")
    def _grounded_answers_have_sources(self) -> "Answer":
        """Reject grounded answers without evidence.

        Keeping this invariant in the domain model prevents a new interface or
        pipeline from accidentally bypassing the citation guardrail.
        """
        if self.grounded and not self.citations:
            raise ValueError("grounded answers require at least one citation")
        return self


class UserContext(BaseModel):
    """Who is asking, and what they are allowed to see.

    The platform is multi-tenant in the access-control sense: not every
    developer may read every repository or runbook. The user's ``acl_groups``
    are intersected with each chunk's ``acl_groups`` at retrieval time.
    """

    user_id: str = Field(..., description="Opaque stable identifier for the user.")
    acl_groups: list[str] = Field(default_factory=list, description="Groups the user belongs to.")
