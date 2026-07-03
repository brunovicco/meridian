"""Fat/slim knowledge representations.

A single knowledge document has two representations in Redis Stack, each sized
for a different stage of the pipeline:

* **Slim projection** - the handful of fields needed to *find* and *rank* a
  document and to *cite* it: id, title, source, a short snippet, and the ACL
  groups. This is what the search index holds and what a KNN query returns. It
  is small, indexed, and cheap to read in bulk.

* **Fat document** - the full text plus rich metadata (owner, last-updated,
  tags), fetched by ``JSON.GET`` on the document's key. it's only ever
  retrieved for the handful of documents that survive ranking and will actually
  enter the generation context.

The point is to never pay for the fat payload during search. Listing and ranking
run entirely on slim projections; the fat document is fetched on demand, for the
few documents that need it. Each stage gets exactly the payload it needs -
right-sized end to end.

This mirrors the production pattern: an indexed slim projection for ``FT.SEARCH``
and a ``JSON.GET`` against RedisJSON to enrich the survivors into fat documents
just before they are used.
"""

from pydantic import BaseModel, Field


class SlimChunk(BaseModel):
    """The lightweight projection used for search, ranking, and citation.

    Deliberately small. It carries a short ``snippet`` for preview and ranking
    context, not the full text - the full text lives in the fat document and is
    fetched only when the chunk is selected for generation.
    """

    chunk_id: str = Field(..., description="Stable unique identifier, and the fat-document key suffix.")
    title: str = Field(..., description="Short human-readable title.")
    source: str = Field(..., description="Human-readable source name, for citation.")
    source_url: str = Field(..., description="Deep link to the origin, for citation.")
    snippet: str = Field(default="", description="Short preview used for ranking and display.")
    acl_groups: list[str] = Field(default_factory=list, description="Groups permitted to read this chunk.")
    score: float = Field(default=0.0, description="Retrieval similarity score, set at query time.")


class FatChunk(BaseModel):
    """The full document, fetched on demand via ``JSON.GET``.

    Holds everything the generation step could need: the complete text and the
    rich metadata that would bloat the search index if carried there. Retrieved
    only for chunks that survive ranking.
    """

    chunk_id: str = Field(..., description="Stable unique identifier, matching the slim projection.")
    title: str = Field(..., description="Short human-readable title.")
    text: str = Field(..., description="The full document text.")
    source: str = Field(..., description="Human-readable source name.")
    source_url: str = Field(..., description="Deep link to the origin.")
    owner: str = Field(default="", description="Team or person that owns the document.")
    last_updated: str = Field(default="", description="ISO date the document was last updated.")
    tags: list[str] = Field(default_factory=list, description="Free-form metadata tags.")
    acl_groups: list[str] = Field(default_factory=list, description="Groups permitted to read this chunk.")

    def to_slim(self, *, snippet_chars: int = 160) -> SlimChunk:
        """Derive the slim projection from this fat document.

        The snippet is the head of the full text, truncated to a small budget -
        enough to rank and preview, never the whole payload.

        :param snippet_chars: Maximum snippet length in characters.
        :returns: The :class:`SlimChunk` projection of this document.
        """
        snippet = self.text[:snippet_chars].rsplit(" ", 1)[0] if len(self.text) > snippet_chars else self.text
        return SlimChunk(
            chunk_id=self.chunk_id,
            title=self.title,
            source=self.source,
            source_url=self.source_url,
            snippet=snippet,
            acl_groups=list(self.acl_groups),
        )
