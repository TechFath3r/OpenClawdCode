# OpenClawdCode — Build Plan

Phase-ordered roadmap. Pick anything unclaimed, open a PR.

Legend: `[ ]` unclaimed · `[~]` in progress · `[x]` done · `[?]` needs design decision

---

## Phase 0 — Scaffolding ✅

- [x] Rename `BetterClaud` → `OpenClawdCode`
- [x] Update package name, env vars, config paths, MCP server name
- [x] Rewrite README with community-facing vision
- [x] Scaffold `CLAUDE.md` + `claude/` subdocs
- [x] Create `tasks/todo.md` + `tasks/lessons.md`
- [x] Update GitHub repo description + topics

## Phase 1 — `memory-lancedb-pro` parity

Goal: match the capabilities of CortexReach's memory plugin inside Claude Code.

### Retrieval quality
- [ ] **Hybrid retrieval** — add BM25 via LanceDB FTS, fuse with vector scores (RRF or weighted)
- [ ] **Cross-encoder rerank** — bge-reranker-v2-m3 via Ollama, top-N rerank after hybrid fetch
- [ ] **Weibull decay** — query-time scoring: `score * exp(-(age/τ)^k)` tunable per-category

### Auto-capture
- [ ] **`Stop` hook → memory extractor** — LLM call (Haiku 4.5 or local model) extracts candidate memories with category + importance
- [ ] **Dedup on store** — similarity check against existing memories before insert (current dedup is hash-exact only)
- [ ] **6-category classification** — profiles, preferences, entities, events, cases, patterns (match memory-lancedb-pro taxonomy)

### Context injection
- [ ] **`UserPromptSubmit` hook** — ranked memories + recent summaries injected before each user turn
- [ ] **Token budget** — cap injection at N tokens, rank by relevance × decay × importance
- [ ] **`SessionStart` hook** — preload top-K project-scoped memories based on cwd

### Isolation
- [ ] **Multi-scope** — `agent_id`, `user_id`, `project_id` columns; filter by current cwd / git root / explicit scope
- [ ] **Scope-aware recall** — default to current project, with override flag

### Migration & CLI
- [ ] **Migration script from memory-lancedb-pro** — read their schema, map to ours
- [ ] **`openclawd` CLI** — `doctor`, `list`, `delete`, `export`, `import`, `stats`
- [ ] **`openclawd doctor`** — check Ollama, check LanceDB, check hook wiring, check embedding dim

### Quality
- [ ] **Embed dim validation** — detect on first embed, assert against schema, fail loud if mismatch
- [ ] **Fix hook paths** — currently hardcoded to cwd, should point at installed venv
- [ ] **Integration test** — end-to-end: start server, call tools over stdio, verify responses

## Phase 2 — `lossless-claw` parity (LCM-lite)

Goal: nothing from the conversation is ever truly lost.

- [ ] **SQLite message archive** — schema for turns, summaries, DAG edges
- [ ] **`PostToolUse` hook** — append each turn to SQLite
- [ ] **Chunk summarizer** — background worker summarizes oldest N turns into DAG nodes
- [ ] **`lcm_grep` MCP tool** — keyword search over raw message archive
- [ ] **`lcm_describe` MCP tool** — return the summary for a given node
- [ ] **`lcm_expand` MCP tool** — return raw messages for a summary node
- [ ] **`PostCompact` hook** — flag that post-compaction context should include relevant summaries

## Phase 3 — Community polish

- [ ] **PyPI package**
- [ ] **Homebrew formula** (maybe)
- [ ] **Export/import** for moving memories between machines
- [ ] **Backup/restore** with timestamped snapshots
- [ ] **Docs site** (probably just GitHub Pages off `/docs`)
- [ ] **Example context profiles** (dev, repair, sysadmin, casual — mirror Dan's existing set)

## Open design questions

- [?] **Which extractor model?** Haiku 4.5 (paid, smart, cheap) vs local Ollama (free, dumber). Probably config-driven so users choose.
- [?] **LanceDB FTS vs SQLite FTS5 sidecar?** LanceDB native is simpler; SQLite FTS5 is more mature. Benchmark before deciding.
- [?] **Decay curve parameters** — per-category or global? memory-lancedb-pro uses per-category. Start global, add per-category in v1.1?
- [?] **Should we support remote Ollama?** `OPENCLAWD_OLLAMA_URL` already allows it. Document it.
