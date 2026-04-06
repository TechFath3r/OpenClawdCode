#!/usr/bin/env python3
"""UserPromptSubmit hook — inject relevant memories before each user prompt.

Reads the user message from stdin, runs hybrid retrieval + decay scoring,
and outputs the top-K memories as an addToPrompt message so Claude sees
them as context alongside the user's prompt.

Token budget: ~2000 tokens max (~8000 chars), configurable via
OPENCLAWD_INJECT_BUDGET env var.
"""

import json
import os
import sys
import time


def main():
    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return

    user_message = hook_input.get("userMessage", "")
    if not user_message or len(user_message.strip()) < 3:
        return

    # Lazy imports — only pay the cost if we actually have a message
    try:
        from openclawd import config
        from openclawd.db import MEMORY_SCHEMA, get_or_create_table
        from openclawd.embeddings import embed_one
        from openclawd.retriever import hybrid_recall, _ensure_fts_index
    except ImportError:
        # Package not installed — skip silently
        return

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    try:
        if table.count_rows() == 0:
            return
    except Exception:
        return

    _ensure_fts_index(table)

    # Embed the user message
    try:
        vector = embed_one(user_message[:500])  # cap to avoid huge embed calls
    except Exception:
        return  # Ollama down — don't block the user

    # Derive scope from cwd
    cwd = os.getcwd()
    cwd_name = os.path.basename(cwd)
    scope_filter = f"scope = 'project:{cwd_name}' OR scope = 'global'"

    now = time.time()
    results = hybrid_recall(
        table=table,
        query_text=user_message[:500],
        query_vector=vector,
        limit=5,
        where=scope_filter,
        apply_decay=True,
        now=now,
    )

    if not results:
        return

    # Format as compact markdown, respecting token budget
    budget_chars = int(os.environ.get("OPENCLAWD_INJECT_BUDGET", "8000"))
    lines = ["[OpenClawdCode memory context]", ""]
    total = 0

    for mem in results:
        entry = (
            f"- [{mem.category}|{mem.tier}] (score:{mem.score:.2f}) "
            f"{mem.content[:300]}"
        )
        if total + len(entry) > budget_chars:
            break
        lines.append(entry)
        total += len(entry)

    if len(lines) <= 2:
        return  # only header, no actual memories fit

    context = "\n".join(lines)
    print(json.dumps({"addToPrompt": context}))


if __name__ == "__main__":
    main()
