"""Loaders for the intent catalog and the seed knowledge base.

These helpers read the JSON files under ``data/catalog`` into the shapes the
router and the vector store expect. Keeping the catalog as versioned data -
rather than hard-coded in Python - means adding an intent or a document is an
edit to a JSON file, and the router's fingerprint mechanism notices the change
automatically and rebuilds.
"""

import json
from pathlib import Path

from meridian.domain.models import KnowledgeChunk
from meridian.domain.models.knowledge import FatChunk
from meridian.domain.models.service_catalog import ServiceRecord


def load_catalog(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load per-intent positive and negative example phrases.

    :param path: Path to ``routes_catalog.json``.
    :returns: A ``(positives, negatives)`` pair of intent-to-phrases dicts.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    intents = data.get("intents", {})
    positives = {name: spec.get("positive", []) for name, spec in intents.items()}
    negatives = {name: spec.get("negative", []) for name, spec in intents.items()}
    return positives, negatives


def load_knowledge_base(path: Path) -> list[KnowledgeChunk]:
    """Load the seed knowledge chunks.

    :param path: Path to ``knowledge_base.json``.
    :returns: A list of :class:`KnowledgeChunk`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeChunk(**raw) for raw in data.get("chunks", [])]


def load_service_catalog(path: Path) -> list[ServiceRecord]:
    """Load the seed service catalog records.

    :param path: Path to ``service_catalog.json``.
    :returns: A list of :class:`ServiceRecord`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ServiceRecord(**raw) for raw in data.get("services", [])]


def load_fat_knowledge_base(path: Path) -> list[FatChunk]:
    """Load the seed fat knowledge documents.

    :param path: Path to ``knowledge_base_fat.json``.
    :returns: A list of :class:`FatChunk`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [FatChunk(**raw) for raw in data.get("chunks", [])]
