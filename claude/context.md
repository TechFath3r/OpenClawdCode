# Architecture & Context

## Canonical Reference

**[`references/source-algorithms.md`](../references/source-algorithms.md)** contains the extracted algorithms, formulas, prompts, SQL schemas, and exact parameter defaults from memory-lancedb-pro and lossless-claw, with file:line citations. Treat it as the spec for what we're porting. Update it if the upstream repos change and we want to track.

## The Vision

OpenClawdCode is a **spiritual port** of two OpenClaw plugins to Claude Code:

1. **`memory-lancedb-pro`** (CortexReach) — auto-captured long-term memory, hybrid retrieval (vector + BM25), cross-encoder rerank, Weibull decay, per-scope isolation, before-prompt context injection
2. **`lossless-claw`** (Martian Engineering) — LCM (Lossless Context Management): every message persisted in SQLite, summarized into a DAG, agent-facing tools (`lcm_grep`, `lcm_describe`, `lcm_expand`) to drill into compacted history

It is **not a fork, not a wrapper, not a direct reimplementation** — it's the same ideas, rebuilt on Claude Code's extension points.

## Claude Code's Extension Points (what we can hook into)

| Surface | What it does | What we use it for |
|---|---|---|
| **MCP server** | Custom tools Claude can call | `store_memory`, `recall_memory`, `search_vault`, future `lcm_*` tools |
| **`UserPromptSubmit` hook** | Inject text before Claude sees the user's message | **Auto context injection** — this is the key hook |
| **`PostToolUse` hook** | Fire after every tool call | Log turns to the SQLite message archive |
| **`Stop` hook** | Fire when Claude finishes responding | Trigger memory extraction, summarization |
| **`PostCompact` hook** | Fire after Claude Code's native compaction | Flag that summaries should be re-injected |
| **`SessionStart` hook** | Fire when a session begins | Preload project-scoped memories |
| **Slash commands / skills** | User-facing commands | Management CLI, diagnostics |

## Structural Limits (what we cannot do)

- **Cannot replace Claude Code's native compaction.** We run alongside it. Our guarantee is "nothing is ever lost from *your* archive" — not "compaction never happens."
- **Cannot rewrite message history.** Only append context via hooks.
- **Cannot intercept model calls.** We're outside the request path.

## Storage Layers

| Store | Purpose | Schema location |
|---|---|---|
| **LanceDB** (`memories` table) | Long-term memories — vector + metadata + importance + timestamps | `src/openclawd/db.py::MEMORY_SCHEMA` |
| **LanceDB** (`obsidian_vault` table) | Indexed Obsidian notes for semantic search | `src/openclawd/db.py::VAULT_SCHEMA` |
| **SQLite** (planned v1.1) | Raw message log + DAG of summaries | TBD — `src/openclawd/lcm/` |
| **Filesystem** (Obsidian vault) | Session logs, human-readable markdown | `{vault}/Claude/sessions/` |

## Key Dependencies

- **LanceDB** — vector store, supports native FTS for BM25 hybrid search
- **Ollama** — local embeddings (`nomic-embed-text`, 768 dim by default) and optionally the memory extractor model
- **MCP Python SDK** (`mcp[cli]`) — FastMCP decorator API
- **httpx** — Ollama HTTP client
- **python-dotenv** — load `~/.config/openclawd/.env`

## Phased Roadmap

### v1.0 — memory-lancedb-pro parity *(current focus)*
- [x] Basic `store_memory` / `recall_memory` / `log_session`
- [x] Obsidian vault indexing + search
- [ ] Auto-capture via `Stop` hook → LLM extractor (Ollama or Haiku)
- [ ] Hybrid retrieval (vector + BM25) via LanceDB FTS
- [ ] Cross-encoder rerank (bge-reranker-v2-m3 via Ollama)
- [ ] Weibull decay scoring
- [ ] Context injection via `UserPromptSubmit` hook
- [ ] Multi-scope isolation (per-project/per-cwd/per-scope)
- [ ] Migration tool from `memory-lancedb-pro`

### v1.1 — lossless-claw parity
- [ ] SQLite message archive via `PostToolUse` hook
- [ ] Background DAG summarizer (cron or `Stop` hook)
- [ ] `lcm_grep` / `lcm_describe` / `lcm_expand` MCP tools
- [ ] `PostCompact` hook surfaces post-compaction summaries

### v2.0 — community polish
- [ ] Full diagnostics CLI (`openclawd doctor`)
- [ ] Export / import / backup
- [ ] Packaged distribution (PyPI, maybe Homebrew)

## Non-Goals

- **Not rebuilding OpenClaw itself.** No channels, no multi-platform gateway, no skill system. If you want those, use OpenClaw.
- **Not a cloud service.** Local-only. No telemetry, no sync service.
- **Not an MCP framework.** We use `mcp` (official SDK) directly — no abstraction layer.
