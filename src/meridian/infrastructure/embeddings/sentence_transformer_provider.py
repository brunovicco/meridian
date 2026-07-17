"""A free, real semantic embedding provider running entirely on the local machine.

Unlike :class:`~meridian.infrastructure.embeddings.azure_provider.AzureEmbeddingProvider`,
this is not a documented skeleton: `sentence-transformers` models run locally,
so there is no credential, no endpoint, and no billed API call to wire in. It
gives the reference a genuinely semantic embedder that still satisfies
guardrail 6 (a backend that can be exercised with no network and no
credentials, once the model weights are cached on disk).

The trade-off against Azure is operational, not architectural: inference runs
on the caller's CPU/GPU instead of a managed endpoint, and the model choice
fixes the vector dimensionality (384 for the default MiniLM model) rather than
letting it be configured per deployment.
"""

from typing import Any

from meridian.domain.interfaces import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real semantic embeddings via a local `sentence-transformers` model.

    The default model, ``all-MiniLM-L6-v2``, is small (~80MB), CPU-friendly,
    and produces 384-dimensional vectors with solid retrieval quality for a
    reference system. ``MERIDIAN_EMBEDDING_DIM`` must be set to match whatever
    model is configured, since the vector store's index dimension is read from
    settings, not inferred from the provider.
    """

    def __init__(self, *, model_name: str) -> None:
        """Load the configured local model.

        :param model_name: A `sentence-transformers` model id or local path.
        :raises RuntimeError: If the `sentence-transformers` package is not
            installed; the composition root catches this and falls back to the
            fake provider so the demo still runs with zero setup.
        """
        self._model_name = model_name
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """Import and instantiate the model, downloading weights on first use.

        The import is local to this method (not the module top) so importing
        this module never requires the ``sentence-transformers`` package;
        only actually constructing the provider does.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed; install the 'local' "
                "extra (pip install '.[local]') or switch "
                "MERIDIAN_EMBEDDING_BACKEND back to 'fake'."
            ) from error
        return SentenceTransformer(self._model_name)

    @property
    def dimension(self) -> int:
        """The model's native output dimensionality."""
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def cache_identity(self) -> str:
        """Identify the concrete sentence-transformers model and dimension."""
        return f"sentence-transformers:{self._model_name}:{self.dimension}"

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string via the batch path."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, L2-normalising so dot products are cosine similarity.

        :param texts: Strings to embed.
        :returns: One normalised vector per input, order preserved.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
