"""The RAG pipeline: retrieval, context assembly, and grounded generation.

This is the online query path. Given a routed query and the asking user, it
retrieves the knowledge chunks the user is permitted to see, assembles them into
a token-budgeted context, asks the LLM to answer *only* from that context, and
attaches citations.

Two design commitments shape the code:

* **Access control is a retrieval-time filter, not a post-filter.** The
  vector store intersects the user's ACL groups with each chunk's groups inside
  the search. The pipeline never sees a chunk the user may not read.
* **Citations are mandatory.** The generation prompt instructs the model to
  ground every claim in the supplied context and to say plainly when the
  context is insufficient. The answer carries the sources back to the caller so
  the developer can verify - a wrong answer stated confidently is worse than an
  honest "I don't know".
"""

from meridian.domain.interfaces import (
    EmbeddingProvider,
    LLMProvider,
    Tracer,
    VectorStore,
)
from meridian.domain.models import Answer, Citation, RouteType, UserContext
from meridian.domain.models.knowledge import SlimChunk

_SYSTEM_PROMPT = (
    "You are Meridian, an internal engineering knowledge assistant. "
    "Answer strictly from the provided context. Every factual claim must be "
    "supported by the context. If the context does not contain the answer, say "
    "so plainly and do not speculate. Be concise and precise."
)

_INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


class RagPipeline:
    """Retrieval-augmented generation with ACL-filtered retrieval and citations."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        tracer: Tracer,
        top_k: int = 5,
        max_context_chars: int = 6000,
    ) -> None:
        """Wire the pipeline to its collaborators and tune its budgets.

        :param embedder: Embedding provider for the query vector.
        :param store: Vector store to retrieve chunks from.
        :param llm: LLM provider for generation.
        :param tracer: Structured observability sink.
        :param top_k: How many chunks to retrieve before context assembly.
        :param max_context_chars: Character budget for the assembled context.
        """
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._tracer = tracer
        self._top_k = top_k
        self._max_context_chars = max_context_chars

    def run(self, search_terms: str, user: UserContext, route_type: RouteType) -> Answer:
        """Execute retrieval and grounded generation using the fat/slim flow.

        The flow is the whole point of the fat/slim split:

        1. **Search slim** - a cheap KNN over the small slim projections returns
           the top-k candidates. No full text is read here.
        2. **Rank and select survivors** - the candidates that fit the context
           budget are chosen from the slim projections alone.
        3. **Fetch fat on demand** - only for the survivors, the full document is
           fetched (the ``JSON.GET`` path). The fat payload is paid for a handful
           of times, never once per candidate.

        Citations come from the slim projection (source, url); the context body
        comes from the fat document.

        :param search_terms: The retrieval-optimised query text.
        :param user: The asking user, whose ACL groups gate retrieval.
        :param route_type: The route that led here (recorded on the answer).
        :returns: A grounded :class:`Answer` with citations.
        """
        query_vector = self._embedder.embed_one(search_terms)
        slims = self._store.search_slim(query_vector, user, self._top_k)

        self._tracer.event(
            "rag.slim_search",
            terms=search_terms[:120],
            candidates=len(slims),
            user_groups=",".join(user.acl_groups),
        )

        if not slims:
            return Answer(
                text="I could not find anything about that in the knowledge base you have access to.",
                citations=[],
                route_type=route_type,
                grounded=False,
            )

        # Select survivors from the slim projections alone, using the snippet
        # length as a cheap proxy for the eventual fat size against the budget.
        survivors = self._select_survivors(slims)

        # Fetch fat only for survivors - the JSON.GET path, paid a handful of times.
        context_parts: list[str] = []
        citations: list[Citation] = []
        fat_fetched = 0
        for slim in survivors:
            fat = self._store.fetch_fat(slim.chunk_id)
            if fat is None:
                continue
            fat_fetched += 1
            context_parts.append(f"[Source: {fat.source}]\n{fat.text}\n")
            citations.append(Citation(source=fat.source, source_url=fat.source_url))

        self._tracer.event(
            "rag.fat_fetch",
            survivors=len(survivors),
            fat_fetched=fat_fetched,
        )

        context = "\n".join(context_parts)
        prompt = self._build_prompt(search_terms, context)
        raw = self._llm.complete(prompt, system=_SYSTEM_PROMPT).strip()

        grounded = _INSUFFICIENT_MARKER not in raw and bool(context)
        final_citations = citations if grounded else []
        text = (
            "I could not find enough in the knowledge base to answer that confidently."
            if not grounded
            else raw
        )

        self._tracer.event(
            "rag.generation",
            grounded=grounded,
            citations=len(final_citations),
            context_chars=len(context),
        )
        return Answer(text=text, citations=final_citations, route_type=route_type, grounded=grounded)

    def _select_survivors(self, slims: list[SlimChunk]) -> list[SlimChunk]:
        """Choose which ranked slim projections will enter the context.

        Uses the slim snippet length as a cheap proxy for eventual fat size,
        stopping once the estimated budget is spent. This decides *which*
        documents to pay the fat fetch for, using only slim data.
        """
        survivors: list = []
        estimated = 0
        for slim in slims:
            # Estimate the fat contribution; snippets are ~a tenth of full text.
            estimate = max(len(slim.snippet) * 10, 400)
            if estimated + estimate > self._max_context_chars and survivors:
                break
            survivors.append(slim)
            estimated += estimate
        return survivors

    def _build_prompt(self, query: str, context: str) -> str:
        """Compose the grounded-generation prompt.

        The instruction to emit ``INSUFFICIENT_CONTEXT`` gives the model an
        explicit, machine-detectable way to decline - which the pipeline turns
        into an honest "I don't know" rather than a hallucinated answer.
        """
        return (
            f"Context from the internal knowledge base:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question using only the context above. If the context "
            f"does not contain the answer, reply with exactly "
            f"'{_INSUFFICIENT_MARKER}' and nothing else."
        )
