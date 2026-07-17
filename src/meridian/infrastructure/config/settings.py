"""Application configuration, read from the environment (twelve-factor).

Config comes from the environment, never from code or checked-in files. This
module centralises that reading into a single typed object so the rest of the
system receives validated settings rather than scattered ``os.getenv`` calls.

The one switch that matters most for the demo is ``backend``: ``memory`` (the
default) runs everything in-process with the fake providers, so the reference
works with zero setup; ``redis`` uses Redis Stack and expects ``REDIS_URL`` to
point at it. Flipping that switch is how you show the same application code
running against two different infrastructures.
"""

import os
from dataclasses import dataclass

_ALLOWED_BACKENDS = {"memory", "redis"}
_ALLOWED_EMBEDDING_BACKENDS = {"fake", "azure", "local"}
_ALLOWED_LLM_BACKENDS = {"fake", "azure", "groq", "dspy"}


@dataclass(frozen=True)
class Settings:
    """Typed, immutable view of the environment configuration."""

    backend: str
    """Vector store backend: ``memory`` or ``redis``."""

    embedding_backend: str
    """Embedding provider: ``fake``, ``azure``, or ``local`` (sentence-transformers)."""

    llm_backend: str
    """LLM provider: ``fake``, ``azure``, ``groq``, or ``dspy``."""

    redis_url: str
    """Redis connection URL, used when ``backend == 'redis'``."""

    embedding_dimension: int
    """Vector dimensionality; must match the active embedding provider."""

    top_k: int
    """Number of chunks to retrieve per query."""

    max_context_chars: int = 6000
    """Maximum retrieved context characters sent to generation."""

    catalog_result_limit: int = 250
    """Safety bound for one structured catalog response."""

    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Local embedding model id or path."""

    groq_model: str = "groq/llama-3.3-70b-versatile"
    """DSPy language model identifier for Groq."""

    groq_api_base: str = "https://api.groq.com/openai/v1"
    """Groq OpenAI-compatible API base URL."""

    dspy_router_artifact: str = ""
    """Optional compiled DSPy routing program path."""

    azure_openai_endpoint: str = ""
    """Azure OpenAI resource endpoint."""

    azure_embedding_deployment: str = ""
    """Azure embedding deployment name."""

    azure_chat_deployment: str = ""
    """Azure chat deployment name."""

    azure_api_version: str = "2024-02-01"
    """Azure OpenAI API version."""

    corporate_ca_bundle: str = ""
    """Optional corporate TLS CA bundle path."""

    def __post_init__(self) -> None:
        """Validate configuration at the process boundary and fail closed."""
        self._validate_choice("MERIDIAN_BACKEND", self.backend, _ALLOWED_BACKENDS)
        self._validate_choice(
            "MERIDIAN_EMBEDDING_BACKEND",
            self.embedding_backend,
            _ALLOWED_EMBEDDING_BACKENDS,
        )
        self._validate_choice("MERIDIAN_LLM_BACKEND", self.llm_backend, _ALLOWED_LLM_BACKENDS)
        if self.embedding_dimension <= 0:
            raise ValueError("MERIDIAN_EMBEDDING_DIM must be greater than zero")
        if self.top_k <= 0:
            raise ValueError("MERIDIAN_TOP_K must be greater than zero")
        if self.max_context_chars <= 0:
            raise ValueError("MERIDIAN_MAX_CONTEXT_CHARS must be greater than zero")
        if self.catalog_result_limit <= 0:
            raise ValueError("MERIDIAN_CATALOG_RESULT_LIMIT must be greater than zero")
        if self.backend == "redis" and not self.redis_url.strip():
            raise ValueError("REDIS_URL is required when MERIDIAN_BACKEND=redis")
        if self.embedding_backend == "local" and not self.sentence_transformer_model.strip():
            raise ValueError("MERIDIAN_ST_MODEL is required for the local embedding backend")
        if self.llm_backend in {"groq", "dspy"}:
            if not self.groq_model.strip():
                raise ValueError("GROQ_MODEL is required for the Groq/DSPy backend")
            if not self.groq_api_base.strip():
                raise ValueError("GROQ_API_BASE is required for the Groq/DSPy backend")

    @staticmethod
    def from_env() -> "Settings":
        """Build a :class:`Settings` from the process environment.

        Every value has a safe default that makes the system run locally with no
        configuration at all. Overriding any of them is a matter of setting the
        corresponding environment variable - no code change.
        """
        return Settings(
            backend=os.getenv("MERIDIAN_BACKEND", "memory").strip().lower(),
            embedding_backend=os.getenv("MERIDIAN_EMBEDDING_BACKEND", "fake").strip().lower(),
            llm_backend=os.getenv("MERIDIAN_LLM_BACKEND", "fake").strip().lower(),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            embedding_dimension=Settings._positive_int("MERIDIAN_EMBEDDING_DIM", "256"),
            top_k=Settings._positive_int("MERIDIAN_TOP_K", "5"),
            max_context_chars=Settings._positive_int("MERIDIAN_MAX_CONTEXT_CHARS", "6000"),
            catalog_result_limit=Settings._positive_int("MERIDIAN_CATALOG_RESULT_LIMIT", "250"),
            sentence_transformer_model=os.getenv(
                "MERIDIAN_ST_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ).strip(),
            groq_model=os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile").strip(),
            groq_api_base=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").strip(),
            dspy_router_artifact=os.getenv("MERIDIAN_DSPY_ROUTER_ARTIFACT", "").strip(),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip(),
            azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip(),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip(),
            corporate_ca_bundle=os.getenv("CORPORATE_CA_BUNDLE", "").strip(),
        )

    @staticmethod
    def _positive_int(name: str, default: str) -> int:
        """Read a positive integer setting with an actionable error."""
        raw = os.getenv(name, default)
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from error
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return value

    @staticmethod
    def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
        """Reject misspelled backend selectors instead of silently degrading."""
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"{name} must be one of: {choices}; got {value!r}")
