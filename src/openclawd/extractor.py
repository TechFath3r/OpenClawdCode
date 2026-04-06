"""Memory extraction engine — auto-capture memories from conversation text.

Takes a conversation summary, calls the configured LLM to extract structured
memories, deduplicates within the batch and against existing stored memories,
then stores survivors.

Reference: references/source-algorithms.md § Extraction prompts, § Admission control / dedup
Source:    CortexReach/memory-lancedb-pro src/extraction-prompts.ts, src/smart-extractor.ts
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass

from . import config
from .db import MEMORY_SCHEMA, get_or_create_table
from .embeddings import embed_one, embed_batch
from .llm_client import llm_json

logger = logging.getLogger("openclawd")

# --- Constants from memory-lancedb-pro ---

MAX_MEMORIES_PER_EXTRACTION = 5
BATCH_DEDUP_THRESHOLD = 0.85     # cosine similarity on L0 abstracts
STORE_DEDUP_THRESHOLD = 0.70     # vector search against existing memories
MAX_SIMILAR_FOR_PROMPT = 3       # top-N existing memories sent to dedup LLM

VALID_CATEGORIES = {
    "profile", "preferences", "entities", "events", "cases", "patterns",
}

# --- Prompts ---

EXTRACTION_SYSTEM = (
    "You are a memory extraction engine. Given a conversation summary, extract "
    "the most important facts, preferences, decisions, entities, and patterns "
    "worth remembering across sessions. Output valid JSON only."
)

EXTRACTION_USER_TEMPLATE = """\
Extract up to {max_memories} memories from this conversation summary. For each memory:
- Classify into exactly one category: profile, preferences, entities, events, cases, patterns
- Write a one-line abstract (L0 index)
- Write a structured markdown overview (L1 summary)
- Write the full narrative content (L2 detail)

Skip:
- Trivial or easily re-derivable information
- System metadata, tool output boilerplate, or recall queries
- Anything that is only relevant to the current session and has no future value

Output language should match the dominant language in the conversation.

Output schema (JSON):
{{"memories": [{{"category": "...", "abstract": "...", "overview": "...", "content": "..."}}]}}

Conversation summary:
{conversation}"""

DEDUP_SYSTEM = (
    "You are a memory deduplication engine. Given a NEW candidate memory and "
    "EXISTING memories that are similar, decide what to do. Output valid JSON only."
)

DEDUP_USER_TEMPLATE = """\
NEW candidate memory:
Category: {category}
Abstract: {abstract}
Content: {content}

EXISTING similar memories:
{existing_memories}

Decide: should the new memory be stored?

Decisions:
- "create" — genuinely new information, store it
- "skip" — already captured by an existing memory, discard
- "merge" — overlaps with an existing memory, merge them (provide match_index)
- "supersede" — replaces an outdated existing memory (provide match_index)

Output schema (JSON):
{{"decision": "create|skip|merge|supersede", "match_index": null_or_1based_int, "reason": "brief explanation"}}"""


# --- Data ---

@dataclass
class ExtractedMemory:
    category: str
    abstract: str
    overview: str
    content: str
    vector: list[float] | None = None  # L0 abstract embedding, set during dedup


# --- Core logic ---

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_memories(conversation: str) -> list[ExtractedMemory]:
    """Call LLM to extract candidate memories from conversation text.

    Returns up to MAX_MEMORIES_PER_EXTRACTION candidates.
    """
    max_mem = config.EXTRACTOR_MAX_MEMORIES
    prompt = EXTRACTION_USER_TEMPLATE.format(
        max_memories=max_mem,
        conversation=conversation[:10000],  # cap input size
    )

    try:
        result = llm_json(EXTRACTION_SYSTEM, prompt)
    except Exception as e:
        logger.warning("Memory extraction LLM call failed: %s", e)
        return []

    memories_raw = result.get("memories", []) if isinstance(result, dict) else []
    candidates = []
    for m in memories_raw[:max_mem]:
        cat = m.get("category", "").lower().strip()
        if cat not in VALID_CATEGORIES:
            cat = "patterns"  # fallback
        candidates.append(ExtractedMemory(
            category=cat,
            abstract=m.get("abstract", "")[:200],
            overview=m.get("overview", ""),
            content=m.get("content", ""),
        ))

    return candidates


def batch_dedup(candidates: list[ExtractedMemory]) -> list[ExtractedMemory]:
    """Remove near-duplicates within the extraction batch.

    Pairwise cosine on L0 abstract embeddings, threshold 0.85.
    Later candidates are dropped in favor of earlier ones.
    """
    if len(candidates) <= 1:
        return candidates

    # Embed all abstracts
    abstracts = [c.abstract for c in candidates]
    vectors = embed_batch(abstracts)
    for c, v in zip(candidates, vectors):
        c.vector = v

    survivors = []
    for i, cand in enumerate(candidates):
        is_dup = False
        for j in range(i):
            if candidates[j].vector and cand.vector:
                sim = _cosine_similarity(candidates[j].vector, cand.vector)
                if sim >= BATCH_DEDUP_THRESHOLD:
                    logger.debug("Batch dedup: dropping #%d (sim=%.3f with #%d)", i, sim, j)
                    is_dup = True
                    break
        if not is_dup:
            survivors.append(cand)

    return survivors


def store_dedup_and_save(
    candidates: list[ExtractedMemory],
    project: str = "",
    scope: str = "",
) -> list[str]:
    """For each candidate, check against existing memories and store survivors.

    Pre-store dedup:
    1. Vector search existing memories at threshold 0.70
    2. If similar exist, send top-3 to LLM dedup prompt
    3. Based on decision: create, skip, merge, or supersede

    Returns list of result strings for each candidate.
    """
    from .tools.memory_store import memory_store

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    results = []

    for cand in candidates:
        # Embed content (full L2) for storage vector
        content_vector = embed_one(cand.content)

        # Check for similar existing memories
        similar = []
        try:
            if table.count_rows() > 0:
                search_results = (
                    table.search(content_vector, query_type="vector")
                    .metric("cosine")
                    .limit(MAX_SIMILAR_FOR_PROMPT)
                    .to_arrow()
                )
                for i in range(len(search_results)):
                    dist = search_results.column("_distance")[i].as_py()
                    sim = 1.0 - dist
                    if sim >= STORE_DEDUP_THRESHOLD:
                        similar.append({
                            "id": search_results.column("id")[i].as_py(),
                            "content": search_results.column("content")[i].as_py(),
                            "category": search_results.column("category")[i].as_py(),
                            "similarity": sim,
                        })
        except Exception as e:
            logger.debug("Pre-store dedup search failed: %s", e)

        # If no similar exist, just store
        if not similar:
            result = memory_store(
                content=cand.content,
                category=cand.category,
                project=project,
                scope=scope,
                abstract=cand.abstract,
                overview=cand.overview,
                confidence=0.8,  # LLM-extracted = slightly less confident than user-stored
                tier="working",
                source_tag="auto_extract",
            )
            results.append(f"created: {cand.abstract[:60]}")
            continue

        # Ask LLM for dedup decision
        existing_text = "\n".join(
            f"{i+1}. [{s['category']}] (sim:{s['similarity']:.2f}) {s['content'][:200]}"
            for i, s in enumerate(similar)
        )
        try:
            decision = llm_json(DEDUP_SYSTEM, DEDUP_USER_TEMPLATE.format(
                category=cand.category,
                abstract=cand.abstract,
                content=cand.content[:500],
                existing_memories=existing_text,
            ))
        except Exception as e:
            logger.warning("Dedup LLM call failed, storing anyway: %s", e)
            decision = {"decision": "create"}

        action = decision.get("decision", "create")
        match_idx = decision.get("match_index")

        if action == "skip":
            results.append(f"skipped (duplicate): {cand.abstract[:60]}")

        elif action == "supersede" and match_idx and 0 < match_idx <= len(similar):
            # Delete the old memory and store the new one
            old_id = similar[match_idx - 1]["id"]
            try:
                table.delete(f"id = '{old_id}'")
            except Exception:
                pass
            memory_store(
                content=cand.content,
                category=cand.category,
                project=project,
                scope=scope,
                abstract=cand.abstract,
                overview=cand.overview,
                confidence=0.8,
                tier="working",
            )
            results.append(f"superseded #{match_idx}: {cand.abstract[:60]}")

        elif action == "merge" and match_idx and 0 < match_idx <= len(similar):
            # Merge: append new content to existing
            old = similar[match_idx - 1]
            merged_content = f"{old['content']}\n\n[Updated] {cand.content}"
            try:
                table.delete(f"id = '{old['id']}'")
            except Exception:
                pass
            memory_store(
                content=merged_content,
                category=cand.category,
                project=project,
                scope=scope,
                abstract=cand.abstract,
                overview=cand.overview,
                confidence=0.8,
                tier="working",
            )
            results.append(f"merged with #{match_idx}: {cand.abstract[:60]}")

        else:
            # create (default)
            memory_store(
                content=cand.content,
                category=cand.category,
                project=project,
                scope=scope,
                abstract=cand.abstract,
                overview=cand.overview,
                confidence=0.8,
                tier="working",
            )
            results.append(f"created: {cand.abstract[:60]}")

    return results


def auto_extract_and_store(
    conversation: str,
    project: str = "",
    scope: str = "",
) -> str:
    """Full auto-capture pipeline: extract → batch dedup → store dedup → save.

    Returns a human-readable summary of what was stored/skipped.
    """
    candidates = extract_memories(conversation)
    if not candidates:
        return "No memories extracted."

    logger.info("Extracted %d candidate memories", len(candidates))

    # Batch dedup
    deduped = batch_dedup(candidates)
    dropped = len(candidates) - len(deduped)
    if dropped:
        logger.info("Batch dedup dropped %d duplicates", dropped)

    # Store with pre-store dedup
    results = store_dedup_and_save(deduped, project=project, scope=scope)

    summary_lines = [f"Auto-extracted {len(candidates)} candidates, {dropped} batch-deduped:"]
    summary_lines.extend(f"  - {r}" for r in results)
    return "\n".join(summary_lines)
