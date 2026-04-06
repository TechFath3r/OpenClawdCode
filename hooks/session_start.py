#!/usr/bin/env python3
"""SessionStart hook — preload project-scoped memories at session open.

Fetches top memories for the current project (derived from cwd) and
injects them as a system message so Claude has context from previous
sessions immediately.
"""

import json
import os
import sys
import time


def main():
    try:
        from openclawd import config
        from openclawd.db import MEMORY_SCHEMA, get_or_create_table
        from openclawd.embeddings import embed_one
        from openclawd.retriever import hybrid_recall, _ensure_fts_index
    except ImportError:
        return

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    try:
        if table.count_rows() == 0:
            return
    except Exception:
        return

    _ensure_fts_index(table)

    cwd = os.getcwd()
    cwd_name = os.path.basename(cwd)

    # Use cwd name as the query — lightweight, surfaces project-relevant memories
    try:
        vector = embed_one(f"project: {cwd_name}")
    except Exception:
        return

    scope_filter = f"scope = 'project:{cwd_name}' OR scope = 'global'"

    results = hybrid_recall(
        table=table,
        query_text=cwd_name,
        query_vector=vector,
        limit=5,
        where=scope_filter,
        apply_decay=True,
        now=time.time(),
    )

    if not results:
        return

    lines = [
        f"[OpenClawdCode] Loaded {len(results)} memories for project '{cwd_name}':",
        "",
    ]
    for mem in results:
        lines.append(
            f"- [{mem.category}|{mem.tier}] {mem.content[:200]}"
        )

    print(json.dumps({"systemMessage": "\n".join(lines)}))


if __name__ == "__main__":
    main()
