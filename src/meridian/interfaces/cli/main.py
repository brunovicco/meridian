"""Command-line interface for Meridian.

The runnable face of the reference. It composes the application via the
composition root, seeds the knowledge base, and then either answers a single
question passed on the command line or runs a short scripted demo that shows
each route type firing and access control in action.

Run it with::

    python -m meridian.interfaces.cli.main --demo
    python -m meridian.interfaces.cli.main --user alice --ask "how do I rotate a credential"

With the default configuration this needs nothing installed beyond the package
itself: fake embeddings, in-memory store, fake LLM. Point ``MERIDIAN_BACKEND``
at ``redis`` to run the same code against Redis Stack.
"""

import argparse
from importlib.resources import files

from meridian.application.services.ask_service import AskService
from meridian.domain.interfaces import CatalogStore, EmbeddingProvider, VectorStore
from meridian.domain.models import Answer, UserContext
from meridian.infrastructure.config.catalog_loader import (
    load_catalog,
    load_fat_knowledge_base,
    load_service_catalog,
)
from meridian.infrastructure.config.settings import Settings
from meridian.interfaces.composition import build_ask_service

_DATA_DIR = files("meridian.data.catalog")

ServiceBundle = tuple[AskService, VectorStore, EmbeddingProvider, CatalogStore]

# Demo users with different access profiles, to show ACL-filtered retrieval.
_USERS = {
    "alice": UserContext(user_id="alice", acl_groups=["payments", "platform"]),
    "bob": UserContext(user_id="bob", acl_groups=["sre", "platform"]),
    "carol": UserContext(user_id="carol", acl_groups=["security"]),
    # no groups: sees nothing
    "dan": UserContext(user_id="dan", acl_groups=[]),
}


def _seed(service_bundle: ServiceBundle) -> None:
    """Embed and index the fat knowledge base and the service catalog.

    The fat documents are embedded on their full text and indexed via
    ``upsert_fat_chunks``, which stores the slim projection for search and the
    fat body for on-demand fetch - the fat/slim split.
    """
    _, store, embedder, catalog = service_bundle
    fats = load_fat_knowledge_base(_DATA_DIR / "knowledge_base_fat.json")
    vectors = embedder.embed_many([f.text for f in fats])
    store.upsert_fat_chunks(fats, vectors)
    catalog.upsert_services(load_service_catalog(_DATA_DIR / "service_catalog.json"))


def _print_answer(user_id: str, question: str, answer: Answer) -> None:
    """Pretty-print an answer with its route and citations."""
    print(f"\n[{user_id}] Q: {question}")
    print(f"    route: {answer.route_type.value}  grounded: {answer.grounded}")
    print(f"    A: {answer.text}")
    for citation in answer.citations:
        print(f"       ↳ {citation.source} ({citation.source_url})")


def run_demo(service: AskService, users: dict[str, UserContext]) -> None:
    """Run a scripted set of questions that exercise every route and ACL.

    The last two questions make access control visible: Carol can read the
    restricted security post-mortem; Alice cannot retrieve it and therefore
    receives an honest insufficient-context answer.
    """
    script = [
        ("alice", "how do I configure authentication for the payments service"),
        ("bob", "what are the steps for a database failover"),
        ("alice", "who owns the payments service"),
        ("alice", "hello there"),
        ("carol", "show me the security post mortem for the payments outage"),
        ("alice", "show me the security post mortem for the payments outage"),
    ]
    for user_id, question in script:
        answer = service.ask(question, users[user_id])
        _print_answer(user_id, question, answer)


def run_acl_demo(bundle: ServiceBundle, users: dict[str, UserContext]) -> None:
    """Show the retrieval-time access-control filter in isolation.

    Routing and generation add noise; this probe goes straight to the store's
    ``search_slim`` to make the security property unmistakable. The same query
    vector is searched for two users with different groups. Carol (security) can
    see the restricted post-mortem; Alice (payments/platform) cannot; Dan (no
    groups) sees nothing at all. The filter is applied inside the search, so a
    user never even transiently receives a chunk outside their groups.
    """
    _, store, embedder, _ = bundle
    probe = "security post mortem for the payments outage root cause"
    vector = embedder.embed_one(probe)
    print(f"\nACL probe: retrieving '{probe}'\n")
    for user_id in ("carol", "alice", "dan"):
        user = users[user_id]
        slims = store.search_slim(vector, user, top_k=3)
        sources = [s.source for s in slims] or ["(nothing visible)"]
        groups = ",".join(user.acl_groups) or "(no groups)"
        print(f"  [{user_id:5} groups={groups:24}] -> {', '.join(sources)}")


def run_structured_demo(service: AskService, users: dict[str, UserContext]) -> None:
    """Show the structured-query path compiling questions into RediSearch.

    Each question is compiled into a RediSearch expression and executed against
    the service catalog - a query, not a retrieval. The compiled expression is
    printed to make the point that structured knowledge deserves a precise query
    whose result is complete, not a top-K sample.
    """
    print("\nStructured catalog queries (compiled to RediSearch, ACL-scoped):\n")
    questions = [
        ("alice", "who owns the payments service"),
        ("bob", "list tier1 services in the gateway domain"),
        ("alice", "which services have no owner"),
    ]
    for user_id, question in questions:
        result = service._structured.query(question, users[user_id])  # noqa: SLF001 - demo introspection
        print(f"  [{user_id}] Q: {question}")
        print(f"      compiled: {result.compiled_query}")
        for svc in result.services:
            print(f"        - {svc.name} (team {svc.team}, {svc.tier})")
        if not result.services:
            print("        (no matches within visibility)")
        print()


def run_fatslim_demo(bundle: ServiceBundle, users: dict[str, UserContext]) -> None:
    """Show the fat/slim split: cheap slim search, then fat fetch for survivors.

    Slim search returns lean projections (title + snippet, no full text). Only
    the survivors that will enter the context are enriched via a fat fetch - the
    JSON.GET path. This makes visible that the full body is paid for a handful of
    times, not once per candidate.
    """
    _, store, embedder, _ = bundle
    probe = "how do I configure authentication for the payments service"
    user = users["alice"]
    vector = embedder.embed_one(probe)
    print(f"\nfat/slim probe: '{probe}' as alice\n")

    slims = store.search_slim(vector, user, top_k=3)
    print("  Phase 1 - slim search (cheap, projections only):")
    for slim in slims:
        print(f"    · {slim.title}  [snippet: {slim.snippet[:48]}...]")

    print("\n  Phase 2 - fat fetch (JSON.GET) for survivors only:")
    for slim in slims[:2]:
        fat = store.fetch_fat(slim.chunk_id)
        if fat is not None:
            print(f"    · {fat.title}  owner={fat.owner}  updated={fat.last_updated}  chars={len(fat.text)}")
    print()


def main() -> None:
    """Parse arguments, compose the service, seed data, and answer."""
    parser = argparse.ArgumentParser(description="Meridian knowledge assistant (reference).")
    parser.add_argument("--ask", type=str, help="A single question to answer.")
    parser.add_argument(
        "--user",
        type=str,
        choices=sorted(_USERS),
        default="alice",
        help="Which demo user is asking.",
    )
    parser.add_argument("--demo", action="store_true", help="Run the scripted demo.")
    parser.add_argument("--acl-demo", action="store_true", help="Show the ACL retrieval filter.")
    parser.add_argument("--structured-demo", action="store_true", help="Show the structured query path.")
    parser.add_argument("--fatslim-demo", action="store_true", help="Show the fat/slim retrieval split.")
    args = parser.parse_args()

    settings = Settings.from_env()
    positives, negatives = load_catalog(_DATA_DIR / "routes_catalog.json")
    bundle = build_ask_service(settings=settings, positive_texts=positives, negative_texts=negatives)
    service = bundle[0]
    _seed(bundle)

    if args.acl_demo:
        run_acl_demo(bundle, _USERS)
        return

    if args.structured_demo:
        run_structured_demo(service, _USERS)
        return

    if args.fatslim_demo:
        run_fatslim_demo(bundle, _USERS)
        return

    if args.demo or not args.ask:
        run_demo(service, _USERS)
        return

    user = _USERS[args.user]
    answer = service.ask(args.ask, user)
    _print_answer(user.user_id, args.ask, answer)


if __name__ == "__main__":
    main()
