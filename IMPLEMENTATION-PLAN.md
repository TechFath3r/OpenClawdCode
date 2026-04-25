# OpenClawdCode — Implementation Plan

**Last updated:** 2026-04-25
**Maintainer:** [@TechFath3r](https://github.com/TechFath3r) (Dan)
**Status:** v0.2.0 shipped, Phase 1 ~80% complete, ready for deployment + dogfooding

> **If you (Dan or a Claude Code session) are picking this project back up, start here.** This doc captures current state, strategic context, and immediate next actions. For long-term roadmap detail see [`tasks/todo.md`](tasks/todo.md). For project rules see [`CLAUDE.md`](CLAUDE.md).

---

## TL;DR

OpenClawdCode bridges OpenClaw's plugin ecosystem to Claude Code's `claude-cli` backend. v0.2.0 is functional for personal use today. The project was paused mid-2026; resuming because:

1. **Concrete personal use case has emerged** — Dan is using OpenClaw on the `claude-cli` backend (Anthropic Max plan economics) and needs the memory + auto-injection capabilities that OpenClaw's stock plugins can't reach across the subprocess boundary.
2. **Repositioning is sharper now** — original "OpenClaw refugees" framing was for a pre-CLI-restoration world. The actual gap that exists today is "OpenClaw with `claude-cli` backend has no plugin-tool access." This is a real, ongoing problem for any OpenClaw user on the Max plan.

---

## Strategic context

### What changed since v0.2.0 was paused

1. **Anthropic restored CLI access to OpenClaw** (April 2026). The original "OpenClaw refugees who lost their plugins" framing is moot — OpenClaw is alive again. The remaining gap is narrower: **`claude-cli` backend ≠ plugin-tool surface.**
2. **OpenClaw 2026.4.x has absorbed and improved on the original plugins.** Specifically:
   - `memory-lancedb` (stock) ≈ `memory-lancedb-pro` with refinements
   - `active-memory` (stock) ≈ `lossless-claw`'s injection but bounded sub-agent semantics
   - **REM / DREAMS** — reflection-based memory promotion. **NEW capability**, not in either original plugin.
3. **OpenClawdCode v0.2.0 was built against the original plugins**, not the upstream stock implementations. So we're a port of plugins that are now themselves outdated.

### What this means for the project

- **v0.2.0 is functional and worth deploying.** Phase 1 hit "memory-lancedb-pro parity" which is roughly equivalent to stock OpenClaw memory-lancedb minus REM/DREAMS.
- **The right source-of-truth going forward is upstream OpenClaw stock**, not the original plugins. Realigning to upstream is incremental patches, not a rewrite.
- **Token efficiency matters more in the new framing.** Dan's plans for Claudia are multi-domain (personal trainer, repair business support, YouTube content, general life). Hierarchical summarization (the LCM / Phase 2 work) becomes important sooner than a single-domain use would suggest.

---

## Current state

### What's shipped (in `main`, v0.2.0)

| Capability | Reference |
|---|---|
| Schema v0.2 (three-tier memory + decay fields + scope) | `src/openclawd/db.py` |
| Hybrid retrieval (vector + BM25 weighted-sum) | `src/openclawd/retriever.py` |
| Composite decay (Weibull recency + frequency + intrinsic) | `src/openclawd/decay.py` |
| Cross-encoder rerank (Ollama, opt-in) | `src/openclawd/reranker.py` |
| Auto-capture (LLM extractor + 6 categories + dedup) | `src/openclawd/extractor.py` |
| `UserPromptSubmit` hook (auto-inject memories before each prompt) | `hooks/user_prompt_submit.py` |
| `SessionStart` hook (preload scoped memories) | `hooks/session_start.py` |
| MCP tools: `store_memory`, `recall_memory`, `extract_memories`, `log_session`, `search_vault`, `search_knowledge`, `load_context` | `src/openclawd/tools/` |
| `openclawd doctor` + `openclawd stats` CLI | `src/openclawd/cli.py` |
| Setup script (Ollama install + venv + MCP register + hook wire-up) | `setup.sh` |
| Test suite (10 test files, ~65 tests) | `tests/` |

### What's NOT shipped

- **Phase 1 polish remaining** (low-priority, see `tasks/todo.md`):
  - `PostCompact` hook (re-inject summaries after Claude Code compacts)
  - Admission control (AMAC-v1 noise filter, opt-in)
  - Migration script from `memory-lancedb-pro`
  - `setup.sh` hardcoded `SCRIPT_DIR` for hook paths (known bug)
  - Specific tests (decay formulas, fusion edges, e2e integration)
  - `openclawd list/delete/export/import` CLI commands
- **Phase 1.5 — Realign with upstream OpenClaw stock** (NEW — see roadmap below)
- **Phase 2 — LCM (lossless-claw port)** — entire phase is paper design, no code yet

---

## Immediate next action: deploy v0.2.0 on the claudia LXC

The personal trainer use case is the first dogfood target. Deploy current `main` to the claudia LXC (Proxmox CT 107, `192.168.1.31`).

### Steps

1. **On the LXC** (`ssh claudia`):
   ```bash
   # Clone fresh from GitHub (NOT via Syncthing — code/ is .stignore'd)
   cd ~
   git clone https://github.com/TechFath3r/OpenClawdCode.git
   cd OpenClawdCode

   # Run setup
   ./setup.sh
   ```

2. **Configure env** at `~/.config/openclawd/.env`:
   ```bash
   OPENCLAWD_OLLAMA_URL=http://localhost:11434
   OPENCLAWD_EMBED_MODEL=nomic-embed-text  # already pulled, see ~/.openclaw setup
   OPENCLAWD_LANCEDB_PATH=~/.local/share/openclawd/lancedb
   OPENCLAWD_VAULT_PATH=/mnt/obsidian
   ```

3. **Hand-fix the `setup.sh` hook-path bug** if it bites — paths in `~/.claude/settings.json` may need to point at the installed venv (`~/.local/share/openclawd/venv/bin/...`) rather than `SCRIPT_DIR`.

4. **Verify**:
   ```bash
   openclawd doctor
   ```
   Should report: Ollama reachable, embed dim matches, LanceDB opens, hooks in settings.json, MCP registered.

5. **Test end-to-end** — send Claudia a message in `#general` ("remember my birthday is X"), then `/new`, then ask "what's my birthday?" Memory should get extracted and auto-injected.

### Per-machine isolation guarantees

- **`~/.claude/settings.json` is per-machine** — installing on the LXC does NOT affect the Mac's Claude Code (used for dev work like RepairKeeper). Verified: setup.sh writes to `$HOME/.claude/settings.json`, never to `<vault>/.claude/settings.json` (which would Syncthing-leak).
- LanceDB store, venv, config all live in LXC's `$HOME` and never leave it.
- See [`README.md` honest trade-offs](README.md#honest-trade-offs) for what's actually in scope.

---

## Roadmap (revised priority order)

### Phase 1 polish (concurrent with deployment)

Continue items in [`tasks/todo.md`](tasks/todo.md). Top priorities driven by dogfooding:

- [ ] Fix `setup.sh` hook-path bug (will surface immediately on LXC deploy)
- [ ] `PostCompact` hook
- [ ] Admission control if early dogfooding shows memory noise

### Phase 1.5 — Realign with upstream OpenClaw stock *(NEW)*

The original plugins this project ported are themselves now superseded by improved stock implementations in OpenClaw 2026.4.x. Catch up:

- [ ] **Audit** — read upstream `dist/extensions/memory-lancedb/`, `dist/extensions/active-memory/`, and the REM/DREAMS subsystem; capture deltas vs current implementation in `references/upstream-deltas.md`
- [ ] **Port retrieval algorithm refinements** — fusion weights, BM25 floor logic, candidate pool sizing if upstream improved them
- [ ] **Port `active-memory`'s bounded sub-agent pattern** as an alternative to (or alongside) the current `UserPromptSubmit` hook. Bounded cost may be preferable to unbounded retrieval injection.
- [ ] **Port the REM / DREAMS system** — periodic reflection-based promotion to long-term memory. NEW capability not in original plugins. High value for multi-domain Claudia.
- [ ] Update `references/source-algorithms.md` to cite upstream as primary going forward
- [ ] Bump version to `0.3.0` once aligned

**Source location for upstream**: [`openclaw/openclaw`](https://github.com/openclaw/openclaw) on GitHub (MIT-licensed). Bundled extension code on any OpenClaw install lives at `/usr/lib/node_modules/openclaw/dist/extensions/`.

### Phase 2 — LCM (lossless-claw parity / port)

Now elevated from "later, maybe" to **next priority after Phase 1.5**. Reasons:

- Multi-domain Claudia (trainer + repair + YouTube + life management) will accumulate context across many domains; flat memory storage scales worse than hierarchical
- Token efficiency through summarization matters under any backend (claude-cli has rate limits too)
- LCM provides a bounded "lossless under the hood, summarized in context" guarantee that complements memory-lancedb's selective recall

Existing plan in [`tasks/todo.md` Phase 2](tasks/todo.md) — port from upstream OpenClaw + `lossless-claw` source as updated by Phase 1.5 audit.

### Phase 3 — Community polish

Defer until Phases 1.5 + 2 are closed and dogfooding has proved the shape. Items remain in `tasks/todo.md` Phase 3.

---

## Open questions / decisions made

- **Q:** Should we deploy v0.2.0 now or restart from upstream? **A:** Deploy now (2026-04-25). Phase 1 is functional; realignment is incremental, not rewrite.
- **Q:** Phase 2 priority? **A:** Elevated to "after Phase 1.5" given multi-domain Claudia plans and token-efficiency value.
- **Q:** Repository naming — keep "OpenClawdCode" or rename to reflect new bridge framing? **A:** Keep — name is established, switch costs > benefit.
- **Q:** Should the maintainer keep using own personal LanceDB on the LXC or maintain a separate dev DB? **A:** Personal use IS dogfooding. One DB. Real data informs decisions better than synthetic.
- **Open:** Does upstream OpenClaw `memory-lancedb` have a stable API surface we can match exactly, or do we need to design our own MCP-tool surface? Phase 1.5 audit will answer.
- **Open:** Should `OPENCLAWD_VAULT_PATH=/mnt/obsidian` be at vault root or scoped to `Claudia/` for the maintainer's deployment? Default to vault root; let scope filtering handle isolation.

---

## Related projects (different layers, not competitors)

Two adjacent OpenClaw-Claude bridge projects exist as of 2026-04-25. **Both operate at the inference-routing layer, not the memory layer** — no overlap with OpenClawdCode's scope. Listed for awareness; potential future cross-pollination.

| Project | License | Maturity | What it solves | Overlap with OpenClawdCode |
|---|---|---|---|---|
| [elvatis/openclaw-cli-bridge](https://github.com/elvatis/openclaw-cli-bridge-elvatis) | Apache-2.0 | v3.10.5, 2⭐ | OpenClaw plugin: route requests through Codex/Gemini/Claude CLI subprocesses | None — inference routing, no memory layer |
| [shinglokto/openclaw-claude-bridge](https://github.com/shinglokto/openclaw-claude-bridge) | MIT | v1.2.3, 144⭐, 22 forks | HTTP proxy translating OpenAI format ↔ Claude CLI text protocol | None — protocol translation, no memory layer |

Neither addresses memory persistence, vector recall, auto-injection, or context management. OpenClawdCode is unique in its layer.

**Note for the README:** consider adding a "Related projects" section that disambiguates OpenClawdCode from these — saves user confusion as the OpenClaw-Claude ecosystem grows.

---

## References

- **Repo:** https://github.com/TechFath3r/OpenClawdCode
- **README:** [`README.md`](README.md) — public-facing project intro, framing, capabilities, trade-offs
- **Project rules:** [`CLAUDE.md`](CLAUDE.md) — always-loaded essentials when working on this project
- **Long-term roadmap:** [`tasks/todo.md`](tasks/todo.md) — phase-ordered build plan, item-level tracking
- **Lessons learned:** [`tasks/lessons.md`](tasks/lessons.md)
- **Source algorithms:** [`references/source-algorithms.md`](references/source-algorithms.md) — 671-line spec extracted from original plugins (NOTE: due for upstream realignment in Phase 1.5)
- **Architectural context:** [`claude/context.md`](claude/context.md), [`claude/conventions.md`](claude/conventions.md), [`claude/workflow.md`](claude/workflow.md)
- **Upstream OpenClaw:** https://github.com/openclaw/openclaw

---

_When you finish a chunk of work that meaningfully changes the picture above (deploy, Phase 1.5 audit complete, etc.), update this doc. It's the canonical "where are we" snapshot._
