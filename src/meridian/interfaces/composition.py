"""The composition root.

This is the one place in the system that knows about concrete implementations.
Every other module depends only on abstractions; here, and only here, we read
the configuration and choose which concretions to instantiate - fake or Azure
embeddings, in-memory or Redis storage, fake or Azure LLM - and wire them
together into an :class:`AskService`.

Concentrating construction here is deliberate. It keeps the dependency graph
acyclic and visible, makes the swap points obvious (each is a single ``if``),
and means the application and domain layers never import infrastructure. This is
the practical payoff of the Dependency Inversion Principle: the direction of
source-code dependency points inward, toward the abstractions, while control
flow is wired outward from this root.
"""

from meridian.application.pipelines.rag_pipeline import RagPipeline
from meridian.application.pipelines.structured_query_pipeline import StructuredQueryPipeline
from meridian.application.query.builder import ServiceQueryBuilder
from meridian.application.router.routing_engine import RoutingEngine
from meridian.application.router.semantic_router import SemanticRouter
from meridian.application.services.ask_service import AskService
from meridian.domain.interfaces import (
    CatalogStore,
    EmbeddingProvider,
    LLMProvider,
    Tracer,
    VectorStore,
)
from meridian.domain.policies import RoutingPolicy
from meridian.infrastructure.config.settings import Settings
from meridian.infrastructure.embeddings.fake_provider import FakeEmbeddingProvider
from meridian.infrastructure.llm.providers import FakeLLMProvider
from meridian.infrastructure.metrics.router_metrics import RouterMetricsCollector
from meridian.infrastructure.observability.tracer import StructuredTracer
from meridian.infrastructure.vectorstore.in_memory_catalog import InMemoryCatalogStore
from meridian.infrastructure.vectorstore.in_memory_store import InMemoryVectorStore


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Choose the embedding provider from configuration.

    :param settings: Application settings.
    :returns: A concrete :class:`EmbeddingProvider`.
    """
    if settings.embedding_backend == "azure":
        from meridian.infrastructure.embeddings.azure_provider import AzureEmbeddingProvider

        return AzureEmbeddingProvider(dimension=settings.embedding_dimension)
    if settings.embedding_backend == "local":
        try:
            from meridian.infrastructure.embeddings.sentence_transformer_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            return SentenceTransformerEmbeddingProvider()
        except RuntimeError:
            # sentence-transformers missing: fall back so the demo still runs.
            return FakeEmbeddingProvider(dimension=settings.embedding_dimension)
    return FakeEmbeddingProvider(dimension=settings.embedding_dimension)


def build_store(settings: Settings, tracer: Tracer) -> VectorStore:
    """Choose the vector store from configuration.

    :param settings: Application settings.
    :param tracer: Structured observability sink.
    :returns: A concrete :class:`VectorStore`.
    """
    if settings.backend == "redis":
        from meridian.infrastructure.redis.redis_vector_store import RedisVectorStore

        return RedisVectorStore(
            url=settings.redis_url,
            dimension=settings.embedding_dimension,
            tracer=tracer,
        )
    return InMemoryVectorStore(tracer=tracer)


def build_catalog_store(settings: Settings, tracer: Tracer) -> CatalogStore:
    """Choose the service catalog store from configuration.

    :param settings: Application settings.
    :param tracer: Structured observability sink.
    :returns: A concrete :class:`CatalogStore`.
    """
    if settings.backend == "redis":
        from meridian.infrastructure.redis.redis_catalog_store import RedisCatalogStore

        return RedisCatalogStore(url=settings.redis_url, tracer=tracer)
    return InMemoryCatalogStore(tracer=tracer)


def build_llm(settings: Settings) -> LLMProvider:
    """Choose the LLM provider from configuration.

    The ``grok`` backend runs the real DSPy modules on Grok (xAI). Because it
    requires the ``dspy`` package and ``XAI_API_KEY``, it degrades gracefully: if
    either is missing, the fake provider is used so the demo still runs. This is
    the one place that decision is made.

    :param settings: Application settings.
    :returns: A concrete :class:`LLMProvider`.
    """
    if settings.llm_backend == "azure":
        from meridian.infrastructure.llm.providers import AzureLLMProvider

        return AzureLLMProvider()
    if settings.llm_backend in ("grok", "dspy"):
        try:
            from meridian.infrastructure.llm.providers import GrokDSPyLLMProvider

            return GrokDSPyLLMProvider()
        except RuntimeError:
            # dspy missing or XAI_API_KEY absent: fall back so the demo still runs.
            return FakeLLMProvider()
    return FakeLLMProvider()


def build_ask_service(
    *,
    settings: Settings,
    positive_texts: dict[str, list[str]],
    negative_texts: dict[str, list[str]],
) -> tuple[AskService, VectorStore, EmbeddingProvider, CatalogStore]:
    """Assemble the full application graph from configuration and a catalog.

    Returns the service plus the stores and embedder, because the caller (the
    CLI or API bootstrap) also needs them to seed the knowledge base and the
    service catalog before serving.

    :param settings: Application settings.
    :param positive_texts: Per-intent positive example phrases.
    :param negative_texts: Per-intent negative example phrases.
    :returns: The wired service, the vector store, the embedder, the catalog store.
    """
    tracer: Tracer = StructuredTracer()
    embedder = build_embedder(settings)
    store = build_store(settings, tracer)
    catalog = build_catalog_store(settings, tracer)
    llm = build_llm(settings)
    policy = RoutingPolicy.for_embedding_backend(settings.embedding_backend)
    metrics = RouterMetricsCollector()

    router = SemanticRouter(
        positive_texts=positive_texts,
        negative_texts=negative_texts,
        intent_thresholds=policy.intent_thresholds,
        embedder=embedder,
        store=store,
        config=policy.ambiguity,
        tracer=tracer,
    )
    router.build()

    engine = RoutingEngine(policy=policy, tracer=tracer)
    rag = RagPipeline(embedder=embedder, store=store, llm=llm, tracer=tracer, top_k=settings.top_k)
    structured = StructuredQueryPipeline(builder=ServiceQueryBuilder(), store=catalog, tracer=tracer)
    service = AskService(
        router=router,
        engine=engine,
        rag=rag,
        structured=structured,
        llm=llm,
        tracer=tracer,
        metrics=metrics,
    )
    return service, store, embedder, catalog
