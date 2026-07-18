"""Integration tests for the full ask flow over the in-memory backend.

These wire the real router, engine, and pipeline together through the
composition root with the fake providers, then assert on end-to-end behaviour:
routing, grounded answers with citations, and - most importantly - the
access-control guarantee that a user never retrieves a chunk outside their
groups.
"""

import subprocess
import sys
from importlib.resources import files

from meridian.application.services.ask_service import AskService
from meridian.domain.interfaces import CatalogStore, EmbeddingProvider, VectorStore
from meridian.domain.models import RouteType, UserContext
from meridian.infrastructure.config.catalog_loader import (
    load_catalog,
    load_fat_knowledge_base,
    load_knowledge_base,
    load_service_catalog,
)
from meridian.infrastructure.config.settings import Settings
from meridian.interfaces.composition import build_ask_service

_DATA = files("meridian.data.catalog")


def _build() -> tuple[AskService, VectorStore, EmbeddingProvider, CatalogStore]:
    """Compose the service with fake providers and seed both stores."""
    settings = Settings(
        backend="memory",
        embedding_backend="fake",
        llm_backend="fake",
        redis_url="",
        embedding_dimension=256,
        top_k=3,
    )
    positives, negatives = load_catalog(_DATA / "routes_catalog.json")
    service, store, embedder, catalog = build_ask_service(
        settings=settings, positive_texts=positives, negative_texts=negatives
    )
    fats = load_fat_knowledge_base(_DATA / "knowledge_base_fat.json")
    store.upsert_fat_chunks(fats, embedder.embed_many([f.text for f in fats]))

    flat_chunks = load_knowledge_base(_DATA / "knowledge_base.json")
    store.upsert_chunks(flat_chunks, embedder.embed_many([c.text for c in flat_chunks]))

    catalog.upsert_services(load_service_catalog(_DATA / "service_catalog.json"))
    return service, store, embedder, catalog


def test_greeting_is_out_of_scope() -> None:
    """A greeting short-circuits before any retrieval."""
    service, _, _, _ = _build()
    answer = service.ask("hello there", UserContext(user_id="u", acl_groups=["platform"]))
    assert answer.route_type == RouteType.OUT_OF_SCOPE


def test_knowledge_question_is_grounded_and_cited() -> None:
    """A knowledge question returns a grounded answer with at least one citation."""
    service, _, _, _ = _build()
    user = UserContext(user_id="alice", acl_groups=["payments", "platform"])
    answer = service.ask("how do I configure authentication for the payments service", user)
    assert answer.grounded is True
    assert len(answer.citations) >= 1


def test_acl_filter_blocks_unauthorised_source() -> None:
    """A user without the security group never retrieves the restricted doc."""
    _, store, embedder, _ = _build()
    vector = embedder.embed_one("security post mortem payments outage root cause")

    security_user = UserContext(user_id="carol", acl_groups=["security"])
    payments_user = UserContext(user_id="alice", acl_groups=["payments", "platform"])

    security_sources = {c.source for c in store.search_slim(vector, security_user, 5)}
    payments_sources = {c.source for c in store.search_slim(vector, payments_user, 5)}

    assert "Security Post-Mortem" in security_sources
    assert "Security Post-Mortem" not in payments_sources


def test_flat_chunk_acl_filter_blocks_unauthorised_source() -> None:
    """The flat, pre fat/slim ``KnowledgeChunk`` path enforces the same ACL guarantee."""
    _, store, embedder, _ = _build()
    vector = embedder.embed_one("security post mortem payments outage root cause")

    security_user = UserContext(user_id="carol", acl_groups=["security"])
    payments_user = UserContext(user_id="alice", acl_groups=["payments", "platform"])

    security_sources = {c.source for c in store.search_chunks(vector, security_user, 5)}
    payments_sources = {c.source for c in store.search_chunks(vector, payments_user, 5)}

    assert "Security Post-Mortem" in security_sources
    assert "Security Post-Mortem" not in payments_sources


def test_acl_no_groups_fails_closed() -> None:
    """A user with no groups retrieves nothing at all."""
    _, store, embedder, _ = _build()
    vector = embedder.embed_one("anything")
    no_groups = UserContext(user_id="dan", acl_groups=[])
    assert store.search_slim(vector, no_groups, 5) == []


def test_restricted_post_mortem_is_protected_end_to_end() -> None:
    """The complete ask flow cites the restricted document only for security."""
    service, _, _, _ = _build()
    question = "show me the security post mortem for the payments outage"

    security_answer = service.ask(question, UserContext(user_id="carol", acl_groups=["security"]))
    payments_answer = service.ask(
        question,
        UserContext(user_id="alice", acl_groups=["payments", "platform"]),
    )

    assert "Security Post-Mortem" in {citation.source for citation in security_answer.citations}
    assert payments_answer.grounded is False
    assert payments_answer.citations == []


def test_cli_rejects_unknown_demo_user() -> None:
    """A misspelled user cannot inherit Alice's privileged demo groups."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "meridian.interfaces.cli.main",
            "--user",
            "mallory",
            "--ask",
            "payments authentication",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_seed_catalogs_are_available_as_package_resources() -> None:
    """The installed CLI can load every JSON resource from the package."""
    catalog = files("meridian.data.catalog")
    expected = {
        "knowledge_base.json",
        "knowledge_base_fat.json",
        "routes_catalog.json",
        "service_catalog.json",
    }
    assert expected.issubset({resource.name for resource in catalog.iterdir()})
