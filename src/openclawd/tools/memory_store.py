"""MCP tool: store a memory."""

import hashlib
import json
import time

from ..db import MEMORY_SCHEMA, get_or_create_table
from ..embeddings import embed_one
from .. import config

VALID_CATEGORIES = {
    "general", "preference", "decision", "learning",
    "architecture", "debugging", "tool_usage",
    # v0.2 — memory-lancedb-pro taxonomy (see references/source-algorithms.md)
    "profile", "preferences", "entities", "events", "cases", "patterns",
}

VALID_TIERS = {"core", "working", "peripheral"}
VALID_TEMPORAL = {"static", "dynamic"}


def _derive_scope(project: str, scope: str) -> str:
    """Compute a scope string. Explicit scope wins; else derive from project."""
    if scope:
        return scope
    if project:
        return f"project:{project}"
    return "global"


def memory_store(
    content: str,
    category: str = "general",
    project: str = "",
    tags: list[str] | None = None,
    importance: int = 5,
    tier: str = "working",
    temporal_type: str = "static",
    abstract: str = "",
    overview: str = "",
    confidence: float = 1.0,
    scope: str = "",
    source_tag: str = "manual",
) -> str:
    """Store a memory for later recall. Deduplicates by content hash.

    Args:
        content: The full memory text (L2).
        category: Memory category (see VALID_CATEGORIES).
        project: Project scope label (empty = cross-project).
        tags: Optional list of tags.
        importance: 1-10 importance rating (default 5).
        tier: core|working|peripheral — affects decay floor and β.
        temporal_type: static|dynamic — dynamic decays 3x faster.
        abstract: One-line index (L0). Derived from content if empty.
        overview: Markdown summary (L1). Optional.
        confidence: 0..1 confidence in this memory (default 1.0).
        scope: Explicit scope string; if empty, derived from project.
        source_tag: Origin of this memory (manual, auto_extract, migration).
    """
    if tags is None:
        tags = []

    if category not in VALID_CATEGORIES:
        return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
    if tier not in VALID_TIERS:
        return f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(VALID_TIERS))}"
    if temporal_type not in VALID_TEMPORAL:
        return f"Invalid temporal_type '{temporal_type}'. Must be: {', '.join(sorted(VALID_TEMPORAL))}"

    importance = max(1, min(10, importance))
    confidence = max(0.0, min(1.0, float(confidence)))

    if not abstract:
        abstract = content[:120].replace("\n", " ").strip()

    resolved_scope = _derive_scope(project, scope)

    # Deterministic ID from content
    mem_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    now = time.time()

    vector = embed_one(content)

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)

    # Check if exists — if so, update (delete + re-add)
    try:
        existing = table.search().where(f"id = '{mem_id}'").limit(1).to_arrow()
        if len(existing) > 0:
            table.delete(f"id = '{mem_id}'")
    except Exception:
        pass

    table.add([{
        "id": mem_id,
        "content": content,
        "vector": vector,
        "category": category,
        "project": project,
        "tags": json.dumps(tags),
        "importance": importance,
        "created_at": now,
        "updated_at": now,
        "source": source_tag,
        "abstract": abstract,
        "overview": overview,
        "tier": tier,
        "temporal_type": temporal_type,
        "confidence": confidence,
        "access_count": 0,
        "last_accessed_at": now,
        "scope": resolved_scope,
    }])

    return f"Memory stored (id={mem_id}, category={category}, tier={tier}, scope={resolved_scope})"
