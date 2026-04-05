# Research: memory-lancedb-pro + lossless-claw algorithm extraction

## memory-lancedb-pro

### 1. Weibull decay

Recency is Weibull stretched-exponential decay; total composite is a weighted
sum of recency + frequency + intrinsic.

```ts
// src/decay-engine.ts:147-163
function recency(memory: DecayableMemory, now: number): number {
  const lastActive =
    memory.accessCount > 0 ? memory.lastAccessedAt : memory.createdAt;
  const daysSince = Math.max(0, (now - lastActive) / MS_PER_DAY);
  // Dynamic memories decay 3x faster (1/3 half-life)
  const baseHL = memory.temporalType === "dynamic" ? halfLife / 3 : halfLife;
  const effectiveHL = baseHL * Math.exp(mu * memory.importance);
  const lambda = Math.LN2 / effectiveHL;
  const beta = getTierBeta(memory.tier);
  return Math.exp(-lambda * Math.pow(daysSince, beta));
}
```

Composite (src/decay-engine.ts:192-205):
```ts
composite = rw * recency + fw * frequency + iw * intrinsic
```

Frequency (src/decay-engine.ts:170-183):
```ts
base = 1 - exp(-accessCount / 5)
recentnessBonus = exp(-avgGapDays / 30)
frequency = base * (0.5 + 0.5 * recentnessBonus)
```

Intrinsic = `importance * confidence` (src/decay-engine.ts:188-190).

**Defaults** (src/decay-engine.ts:48-62, DEFAULT_DECAY_CONFIG):
- `recencyHalfLifeDays` τ = 30 (days)
- `importanceModulation` μ = 1.5 (effective half-life = τ * exp(μ * importance))
- Weibull β: `core=0.8`, `working=1.0`, `peripheral=1.3`
- Weights: `recencyWeight=0.4`, `frequencyWeight=0.3`, `intrinsicWeight=0.3`
- `staleThreshold=0.3`, `searchBoostMin=0.3`
- Tier decay floors: `core=0.9`, `working=0.7`, `peripheral=0.5`
- Dynamic (`temporalType="dynamic"`) memories use `halfLife/3`

**Age units:** milliseconds subtracted, then divided by `MS_PER_DAY = 86_400_000` (src/decay-engine.ts:17) → days.

**Per-category:** NO — parameters are per-tier (`core`/`working`/`peripheral`), not per memory-category. Tier is a separate axis (see `MemoryTier` in src/memory-categories.ts:42).

**Combined with vector similarity** via `applySearchBoost` (src/decay-engine.ts:216-223):
```ts
const tierFloor = Math.max(getTierFloor(tier), composite);
const multiplier = boostMin + (1 - boostMin) * tierFloor;
r.score *= min(1, max(boostMin, multiplier));
```
So the search score is multiplied by `boostMin + (1-boostMin)*max(tierFloor, composite)`, clamped to `[boostMin, 1]`.

Source: `src/decay-engine.ts:17-232`

---

### 2. Hybrid retrieval fusion

Despite the file header saying "RRF", the implemented fusion is a **weighted sum** of raw vector and BM25 scores with a BM25 exact-match floor.

```ts
// src/retriever.ts:1109-1186
const weightedFusion = (vectorScore * this.config.vectorWeight)
                     + (bm25Score * this.config.bm25Weight);
const fusedScore = vectorResult
  ? clamp01(
      Math.max(
        weightedFusion,
        bm25Score >= 0.75 ? bm25Score * 0.92 : 0,
      ),
      0.1,
    )
  : clamp01(bm25Result!.score, 0.1);
```
- BM25-only "ghost" hits are dropped if the id is not in the store (src/retriever.ts:1138-1145).
- BM25 hit floor `>= 0.75` preserves exact keyword matches (API keys, ticket numbers) with weight `0.92`.

**Defaults** (src/retriever.ts:179-199, DEFAULT_RETRIEVAL_CONFIG):
- `mode: "hybrid"`, `vectorWeight=0.7`, `bm25Weight=0.3`
- `minScore=0.3`, `candidatePoolSize=20`
- `recencyHalfLifeDays=14`, `recencyWeight=0.1` (secondary recency boost, additive, separate from decay-engine)
- `lengthNormAnchor=500`, `hardMinScore=0.35`
- `timeDecayHalfLifeDays=60`, `reinforcementFactor=0.5`, `maxHalfLifeMultiplier=3`
- `tagPrefixes: ["proj","env","team","scope"]`
- `queryExpansion: true` (only applied when `source` is `"manual"` or `"cli"`, see src/retriever.ts:1100-1107)

**Top-K at each stage** (src/retriever.ts:909 + pipeline):
- Vector/BM25 fetch: `candidatePoolSize = max(20, limit*2)` for each branch (src/retriever.ts:909).
- Fusion output: union of both result sets, sorted by fused score.
- Rerank window: `filtered.slice(0, limit * 2)` (src/retriever.ts:978-980).
- Final slice: `deduplicated.slice(0, limit)` after MMR diversity (src/retriever.ts:891).

Post-fusion pipeline order (src/retriever.ts:983-1015 and `postProcessResults`):
`minScore → rerankInput window → rerank → recency_boost → importance_weight → length_norm → time_decay → hard_min_score → noise_filter → mmr_diversity → limit`.

Source: `src/retriever.ts:1109-1186` (fusion), `src/retriever.ts:898-1019` (pipeline).

---

### 3. Cross-encoder reranker

```ts
// src/retriever.ts:185-192 (defaults)
rerank: "cross-encoder",
rerankModel: "jina-reranker-v3",
rerankEndpoint: "https://api.jina.ai/v1/rerank",
rerankTimeoutMs: 5000,
```
Providers supported: `"jina"` (default), `"siliconflow"`, `"voyage"`, `"pinecone"`, `"dashscope"`, `"tei"` (src/retriever.ts:57-63).

**Candidates reranked:** `filtered.slice(0, limit * 2)` — twice the final limit (src/retriever.ts:978-980). The full slice is sent to the rerank API (`topN = results.length` via `buildRerankRequest`, src/retriever.ts:1213-1220).

**Score blend** (src/retriever.ts:1252-1270):
```ts
blendedScore = clamp01WithFloor(item.score * 0.6 + original.score * 0.4, floor)
```
60% cross-encoder + 40% fused score, with per-item "preservation floor" from `getRerankPreservationFloor`.

**Unreturned candidates** are kept but multiplied by 0.8 (penalized) rather than dropped (src/retriever.ts:1273-1281).

**Threshold:** There is no rerank-specific score threshold; the cut happens at `hardMinScore = 0.35` applied after all post-processing stages (src/retriever.ts:77-79 default; applied downstream via `postProcessResults`). Cross-encoder also falls back to cosine similarity if the API fails (src/retriever.ts:1296-1299).

Source: `src/retriever.ts:1192-1310`.

---

### 4. Memory categories

```ts
// src/memory-categories.ts:8-15
export const MEMORY_CATEGORIES = [
  "profile",
  "preferences",
  "entities",
  "events",
  "cases",
  "patterns",
] as const;
```

Category classifications (src/memory-categories.ts:20-39):
- `ALWAYS_MERGE_CATEGORIES = {"profile"}` — skip dedup entirely, always merge.
- `MERGE_SUPPORTED_CATEGORIES = {"preferences","entities","patterns"}`.
- `TEMPORAL_VERSIONED_CATEGORIES = {"preferences","entities"}` — facts replaced over time.
- `APPEND_ONLY_CATEGORIES = {"events","cases"}` — CREATE or SKIP only.

**Decay/importance defaults per category:** NO — decay config is per-tier only. However, admission-control **type priors** are per-category (src/admission-control.ts:114-121):
```ts
DEFAULT_TYPE_PRIORS = {
  profile: 0.95, preferences: 0.9, entities: 0.75,
  events: 0.45, cases: 0.8, patterns: 0.85,
}
```
Conservative / high-recall presets have slightly different priors (src/admission-control.ts:166-173, 195-202).

Three-level memory structure per candidate (src/memory-categories.ts:45-50):
- `abstract` (L0): one-sentence index
- `overview` (L1): structured markdown summary
- `content` (L2): full narrative

Dedup decisions (src/memory-categories.ts:53-60):
`create | merge | skip | support | contextualize | contradict | supersede`

Source: `src/memory-categories.ts:8-86`.

---

### 5. Extraction prompts

**Extraction prompt** (src/extraction-prompts.ts:9-132, `buildExtractionPrompt`): quoted in full below (condensed).

Key instructions from the prompt (verbatim excerpts):
- "Maximum 5 memories per extraction"
- "Preferences should be aggregated by topic"
- "Output language should match the dominant language in the conversation"
- Skip list explicitly excludes: system metadata, "[Subagent Context]", recall queries like "Do you remember X?", tool output/logs/boilerplate.

**Output schema** (src/extraction-prompts.ts:114-124):
```json
{
  "memories": [
    {
      "category": "profile|preferences|entities|events|cases|patterns",
      "abstract": "One-line index",
      "overview": "Structured Markdown summary",
      "content": "Full narrative"
    }
  ]
}
```

**Dedup prompt** (src/extraction-prompts.ts:134-176, `buildDedupPrompt`) returns:
```json
{"decision":"skip|create|merge|supersede|support|contextualize|contradict",
 "match_index":1,
 "reason":"...",
 "context_label":"evening"}
```
Required `context_label` vocabulary: `general, morning, evening, night, weekday, weekend, work, leisure, summer, winter, travel`.

**Merge prompt** (src/extraction-prompts.ts:178-217, `buildMergePrompt`) returns `{abstract, overview, content}`.

**LLM used:** Not hardcoded — `LlmClient` interface, configurable. No default model name in `smart-extractor.ts`. Cap: `MAX_MEMORIES_PER_EXTRACTION = 5` (src/smart-extractor.ts:165). Dedup vector pre-filter `SIMILARITY_THRESHOLD = 0.7`, `MAX_SIMILAR_FOR_PROMPT = 3` (src/smart-extractor.ts:163-164).

Source: `src/extraction-prompts.ts:1-217`.

---

### 6. Scoping / isolation

Scope hierarchy / patterns (src/scopes.ts:62-69):
```ts
const SCOPE_PATTERNS = {
  GLOBAL:     "global",
  AGENT:      (agentId)   => `agent:${agentId}`,
  CUSTOM:     (name)      => `custom:${name}`,
  REFLECTION: (agentId)   => `reflection:agent:${agentId}`,
  PROJECT:    (projectId) => `project:${projectId}`,
  USER:       (userId)    => `user:${userId}`,
};
```

Default config (src/scopes.ts:48-56):
```ts
DEFAULT_SCOPE_CONFIG = {
  default: "global",
  definitions: { global: { description: "Shared knowledge across all agents" } },
  agentAccess: {},
};
```

**Scope derivation from environment:** From the OpenClaw session key (src/scopes.ts:91-103):
```ts
parseAgentIdFromSessionKey(sessionKey)
// "agent:main:discord:channel:123" -> "main"
// "agent:main"                      -> "main"
```
Bypass ids: `"system"`, `"undefined"` (src/scopes.ts:71).

**Applied as filter** (src/scopes.ts:188-230, `getAccessibleScopes` / `getScopeFilter`):
- Explicit ACL → `[...explicit, "reflection:agent:${agentId}"]`.
- Default for an agent → `["global", "agent:${agentId}", "reflection:agent:${agentId}"]`.
- Bypass (system/undefined agentId) → `getScopeFilter` returns `undefined` (no store filtering).
- Empty `[]` return is explicit "deny all".
- Default write scope for an agent is `agent:${agentId}` if accessible, else `global` (src/scopes.ts:232-251).

Source: `src/scopes.ts:48-230`.

---

### 7. Admission control / dedup

**Batch-internal cosine dedup** (src/batch-dedup.ts:82-135):
- Pairwise O(n²) cosine similarity on candidate L0 abstract vectors.
- Default `threshold = 0.85`; later candidate marked duplicate of earlier.
- Called with `n <= MAX_MEMORIES_PER_EXTRACTION = 5`.

**Post-extraction dedup pre-filter** (src/smart-extractor.ts):
- `SIMILARITY_THRESHOLD = 0.7` — vector search against existing memories.
- `MAX_SIMILAR_FOR_PROMPT = 3` — top-3 similar existing memories sent to LLM dedup prompt.
- Source: src/smart-extractor.ts:163-165.

**Admission Memory Admission Control (AMAC-v1)** — gate BEFORE dedup (src/admission-control.ts):
Score = weighted sum of 5 features:
```
score = weights.utility*utility + weights.confidence*confidence
      + weights.novelty*novelty + weights.recency*recency
      + weights.typePrior*typePrior
```
- `novelty = clamp01(1 - maxSimilarity)` where `maxSimilarity` is max cosine over the candidate pool (src/admission-control.ts:543-572). `matchedIds` are those with similarity ≥ 0.55.
- `confidence` uses ROUGE-like F1 LCS between candidate content and conversation spans (src/admission-control.ts:394-540).
- Final decision: `score >= admitThreshold → pass_to_dedup` (hint "update_or_merge" if `matchedIds.length>0` else "add"); `score < rejectThreshold → reject`.

**Presets** (src/admission-control.ts:132-207):

| preset | reject | admit | weights (u/c/n/r/t) | noveltyPool | recencyHL |
|---|---|---|---|---|---|
| balanced (default) | 0.45 | 0.60 | 0.1/0.1/0.1/0.1/0.6 | 8 | 14 |
| conservative | 0.52 | 0.68 | 0.16/0.16/0.18/0.08/0.42 | 10 | 10 |
| high-recall | 0.34 | 0.52 | 0.08/0.1/0.08/0.14/0.60 | 6 | 21 |

All presets are `enabled: false` by default; admission control is opt-in.

Source: `src/admission-control.ts:106-340, 543-572`; `src/batch-dedup.ts:82-135`; `src/smart-extractor.ts:163-165`.

---

## lossless-claw

### 1. SQLite schema

Core tables (src/db/migration.ts:423-557):

```sql
CREATE TABLE conversations (
  conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id       TEXT NOT NULL,
  session_key      TEXT,
  active           INTEGER NOT NULL DEFAULT 1,
  archived_at      TEXT,
  title            TEXT,
  bootstrapped_at  TEXT,
  created_at       TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE messages (
  message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  seq             INTEGER NOT NULL,
  role            TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
  content         TEXT NOT NULL,
  token_count     INTEGER NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (conversation_id, seq)
);

CREATE TABLE summaries (
  summary_id                  TEXT PRIMARY KEY,         -- "sum_xxx"
  conversation_id             INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  kind                        TEXT NOT NULL CHECK (kind IN ('leaf','condensed')),
  depth                       INTEGER NOT NULL DEFAULT 0,
  content                     TEXT NOT NULL,
  token_count                 INTEGER NOT NULL,
  earliest_at                 TEXT,
  latest_at                   TEXT,
  descendant_count            INTEGER NOT NULL DEFAULT 0,
  descendant_token_count      INTEGER NOT NULL DEFAULT 0,
  source_message_token_count  INTEGER NOT NULL DEFAULT 0,
  created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
  file_ids                    TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE message_parts (
  part_id       TEXT PRIMARY KEY,
  message_id    INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  session_id    TEXT NOT NULL,
  part_type     TEXT NOT NULL CHECK (part_type IN (
                  'text','reasoning','tool','patch','file',
                  'subtask','compaction','step_start','step_finish',
                  'snapshot','agent','retry')),
  ordinal       INTEGER NOT NULL,
  text_content  TEXT,
  is_ignored    INTEGER,
  is_synthetic  INTEGER,
  tool_call_id  TEXT, tool_name TEXT, tool_status TEXT,
  tool_input    TEXT, tool_output TEXT, tool_error TEXT, tool_title TEXT,
  patch_hash    TEXT, patch_files TEXT,
  file_mime     TEXT, file_name TEXT, file_url TEXT,
  subtask_prompt TEXT, subtask_desc TEXT, subtask_agent TEXT,
  step_reason   TEXT, step_cost REAL, step_tokens_in INTEGER, step_tokens_out INTEGER,
  snapshot_hash TEXT
  -- plus more
);

-- DAG edges:
CREATE TABLE summary_messages (             -- leaf summary -> source messages
  summary_id  TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
  message_id  INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE RESTRICT,
  ordinal     INTEGER NOT NULL,
  PRIMARY KEY (summary_id, message_id)
);

CREATE TABLE summary_parents (              -- condensed summary -> source summaries
  summary_id         TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
  parent_summary_id  TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE RESTRICT,
  ordinal            INTEGER NOT NULL,
  PRIMARY KEY (summary_id, parent_summary_id)
);

CREATE TABLE context_items (                -- what assembler returns, ordered
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  ordinal         INTEGER NOT NULL,
  item_type       TEXT NOT NULL CHECK (item_type IN ('message','summary')),
  message_id      INTEGER REFERENCES messages(message_id) ON DELETE RESTRICT,
  summary_id      TEXT REFERENCES summaries(summary_id) ON DELETE RESTRICT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (conversation_id, ordinal),
  CHECK ((item_type='message' AND message_id IS NOT NULL AND summary_id IS NULL)
      OR (item_type='summary' AND summary_id IS NOT NULL AND message_id IS NULL))
);

CREATE TABLE large_files (
  file_id             TEXT PRIMARY KEY,
  conversation_id     INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  file_name TEXT, mime_type TEXT, byte_size INTEGER,
  storage_uri         TEXT NOT NULL,
  exploration_summary TEXT,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE conversation_bootstrap_state (
  conversation_id            INTEGER PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  session_file_path          TEXT NOT NULL,
  last_seen_size             INTEGER NOT NULL,
  last_seen_mtime_ms         INTEGER NOT NULL,
  last_processed_offset      INTEGER NOT NULL,
  last_processed_entry_hash  TEXT,
  updated_at                 TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes (src/db/migration.ts:550-557)
CREATE INDEX messages_conv_seq_idx       ON messages (conversation_id, seq);
CREATE INDEX summaries_conv_created_idx  ON summaries (conversation_id, created_at);
CREATE INDEX message_parts_message_idx   ON message_parts (message_id);
CREATE INDEX message_parts_type_idx      ON message_parts (part_type);
CREATE INDEX context_items_conv_idx      ON context_items (conversation_id, ordinal);
CREATE INDEX large_files_conv_idx        ON large_files (conversation_id, created_at);
CREATE UNIQUE INDEX conversations_active_session_key_idx
  ON conversations (session_key) WHERE session_key IS NOT NULL AND active = 1;

-- FTS5 virtual tables (src/db/migration.ts:623-687)
CREATE VIRTUAL TABLE messages_fts     USING fts5(content, tokenize='porter unicode61');
CREATE VIRTUAL TABLE summaries_fts    USING fts5(summary_id UNINDEXED, content, tokenize='porter unicode61');
CREATE VIRTUAL TABLE summaries_fts_cjk USING fts5(summary_id UNINDEXED, content, tokenize='trigram');
```

**DAG edges:**
- `summary_messages` links a **leaf** summary to its source `messages` (ordered by `ordinal`).
- `summary_parents` links a **condensed** summary to its parent summaries that were rolled up into it.
- `context_items` is a flat per-conversation ordered list of either a message or a summary (mutually exclusive via CHECK constraint) — this is what assembler reads.

Source: `src/db/migration.ts:423-557, 623-687`.

---

### 2. Summarization loop

**Trigger / chunking** (src/compaction.ts:347-398):
- Full-sweep trigger: `currentTokens > floor(contextThreshold * tokenBudget)` (default `contextThreshold=0.75`).
- Leaf (soft) trigger: `rawTokensOutsideTail >= leafChunkTokens` (default `leafChunkTokens=20000`) — token-based, not turn-count.
- "currentTokens" = max of stored context tokens and observed live token count.

**Targets** (src/db/config.ts & src/summarize.ts:48-49):
- `leafTargetTokens = 2400` (per leaf summary)
- `condensedTargetTokens = 2000`
- Compression floor (src/summarize.ts:750-751): `max(192, min(leafTarget, floor(inputTokens * 0.35)))` — minimum 35% compression.

**Hierarchy / re-summarization:** YES. Summaries have `depth: INTEGER` (0 = leaf, 1+ = condensed). A condensed summary's parent-edges live in `summary_parents`. The prompt switches on depth (src/summarize.ts:931-946):
- `depth <= 1` → `buildD1Prompt` (condense leaf summaries into a session-level node)
- `depth == 2` → `buildD2Prompt` (condense session summaries into phase-level node)
- `depth >= 3` → `buildD3PlusPrompt` (high-level durable memory)

Fanout / roll-up triggers (src/db/config.ts):
- `leafMinFanout = 8` (need ≥8 items to create a leaf)
- `condensedMinFanout = 4`
- `condensedMinFanoutHard = 2`
- `incrementalMaxDepth = 1`
- `newSessionRetainDepth = 2`

**System prompt** (src/summarize.ts:50-51):
```
"You are a context-compaction summarization engine. Follow user instructions exactly and return plain text summary content only."
```

**Leaf prompt** (src/summarize.ts:760-807, `buildLeafSummaryPrompt`):
```
You summarize a SEGMENT of an OpenClaw conversation for future model turns.
Treat this as incremental memory compaction input, not a full-conversation summary.

[normal | aggressive policy]
Operator instructions: <custom>|(none)

Output requirements:
- Plain text only.
- No preamble, headings, or markdown formatting.
- Keep it concise while preserving required details.
- Track file operations (created, modified, deleted, renamed) with file paths and current status.
- If no file operations appear, include exactly: "Files: none".
- End with exactly: "Expand for details about: <comma-separated list of what was dropped or compressed>".
- Target length: about <targetTokens> tokens or less.

<previous_context>
  <previousSummary or (none)>
</previous_context>

<conversation_segment>
  <text>
</conversation_segment>
```

Two policies:
- Normal: "Preserve key decisions, rationale, constraints, and active tasks. Keep essential technical details needed to continue work safely. Remove obvious repetition and conversational filler."
- Aggressive: "Keep only durable facts and current task state. Remove examples, repetition, and low-value narrative details. Preserve explicit TODOs, blockers, decisions, and constraints."

**Condensed D1 prompt** (src/summarize.ts:809-857, `buildD1Prompt`): "You are compacting leaf-level conversation summaries into a single condensed memory node." — preserves decisions+rationale, superseded decisions, completed tasks/outcomes, in-progress state, blockers, specific references (names/paths/URLs). Drops unchanged context, dead-ends, transients, tool-internal mechanics. "Include a timeline with timestamps (hour or half-hour)."

**D2 prompt** (src/summarize.ts:859-893): "condensing multiple session-level summaries into a higher-level memory node." — "A future model should understand trajectory, not per-session minutiae." Timeline uses "dates and approximate time of day".

**D3+ prompt** (src/summarize.ts:895-929): "creating a high-level memory node from multiple phase-level summaries." — "may persist for the rest of the conversation. Keep only durable context." Timeline uses "dates (or date ranges)".

All three condensed prompts end with the same `"Expand for details about: ..."` footer requirement.

Summary timeout: `summaryTimeoutMs = 60000` (src/db/config.ts:224-226).
Overage cap: `summaryMaxOverageFactor = 3` (reject if summary > 3x targetTokens).

Source: `src/summarize.ts:50-51, 750-946`; `src/compaction.ts:340-398`.

---

### 3. Context assembly

Algorithm (src/assembler.ts:899-1056, `ContextAssembler.assemble`):

1. Fetch `context_items` ordered by `ordinal` for the conversation (src/assembler.ts:904).
2. Resolve each into an `AgentMessage` (fetching underlying message or summary record).
3. Split into `evictable` (older) + protected `freshTail` (last `freshTailCount` items, default 8, configurable; plugin default `freshTailCount=64` via config).
4. Also protect evictable tool_result messages paired with tool_calls in the fresh tail (src/assembler.ts:939-946).
5. Always include fresh tail (even if it alone exceeds budget).
6. Fill remaining budget (`tokenBudget - tailTokens`) from evictable items:
   - If everything fits → include all.
   - Else if `prompt` provided and searchable → **BM25-lite relevance scoring**: `scoreRelevance(item.text, prompt)` on each evictable, sort desc, greedily add highest-scoring items that fit. Restore chronological order by `ordinal`.
   - Else (chronological eviction, default) → walk evictable from newest to oldest, keep items that fit, stop at first overflow (drop all older).
7. Append fresh tail after prefix.
8. Normalize assistant-string content to content-block arrays (Anthropic format).
9. Drop assistant messages with empty content.
10. Run `sanitizeToolUseResultPairing` to ensure tool_use/tool_result pairing.

**Token estimate:** `ceil(text.length / 4)` (src/assembler.ts:50-52, "same as VoltCode's Token.estimate").

**System prompt addition** (src/assembler.ts:63-117, `buildSystemPromptAddition`): dynamic guidance emitted only when summaries are present in assembled context. Includes stronger "expand before asserting specifics" guidance when `maxDepth >= 2` or `condensedCount >= 2`.

**Relevance ranking:** Token-based BM25-lite (`scoreRelevance` + `tokenizeText` in src/assembler.ts:~810-878). Only used when assembler is called with a `prompt`. Otherwise chronological newest-first.

Source: `src/assembler.ts:899-1056` (assemble), `src/assembler.ts:63-117` (prompt addition).

---

### 4. Recall tools

All three tools are Agent SDK tools with TypeBox schemas.

#### `lcm_grep` (src/tools/lcm-grep-tool.ts:83-213)

```ts
name: "lcm_grep"
description: "Search compacted conversation history using regex or full-text search.
  Searches across messages and/or summaries stored by LCM. Use this to find specific
  content that may have been compacted away from active context. Returns matching
  snippets with their summary/message IDs for follow-up with lcm_expand or lcm_describe."
parameters: {
  pattern: string,                                 // regex | text query
  mode?: "regex" | "full_text",                    // default: "regex"
  scope?: "messages" | "summaries" | "both",       // default: "both"
  conversationId?: number,                         // default: current session
  allConversations?: boolean,                      // explicit cross-conversation
  since?: string (ISO),                            // inclusive
  before?: string,                                 // exclusive
  limit?: number (1..200)                          // default: 50
}
```
Searches: **both** raw messages AND summaries by default. Returns markdown with `[msg#<id>]` and `[sum_xxx]` citations plus snippet (truncated to 200 chars), capped at `MAX_RESULT_CHARS = 40_000` (~10k tokens).

Retrieval side: `retrieval.grep({query,mode,scope,conversationId,limit,since,before})` runs messages_fts / summaries_fts (porter+unicode61) for `full_text`, or LIKE-based regex; falls back to `summaries_fts_cjk` (trigram) for CJK. Source: src/retrieval.ts:63-256.

#### `lcm_describe` (src/tools/lcm-describe-tool.ts:58-237)

```ts
name: "lcm_describe"
description: "Look up metadata and content for an LCM item by ID.
  Use this to inspect summaries (sum_xxx) or stored files (file_xxx) from
  compacted conversation history. Returns summary content, lineage, token
  counts, and file exploration results."
parameters: {
  id: string,                             // "sum_xxx" | "file_xxx"
  conversationId?: number,
  allConversations?: boolean,
  tokenCap?: number                       // subtree budget-fit annotation
}
```
Returns summary content + lineage + token counts, or large-file metadata + exploration summary.

#### `lcm_expand` (src/tools/lcm-expand-tool.ts:123-448)

```ts
name: "lcm_expand"
description: "Expand compacted conversation summaries from LCM (Lossless Context
  Management). Traverses the summary DAG to retrieve children and source messages.
  Use this to drill into previously-compacted context when you need detail that
  was summarised away. Provide either summaryIds (direct expansion) or query
  (grep-first, then expand top matches). Returns a compact text payload with
  cited IDs for follow-up."
parameters: {
  summaryIds?: string[],                 // required unless `query`
  query?: string,                        // grep-first mode
  maxDepth?: number (min 1),             // default: 3
  tokenCap?: number,                     // entire-result cap
  includeMessages?: boolean,             // default: false
  conversationId?: number,
  allConversations?: boolean
}
```

**Returns** (via `retrieval.expand` → `expandRecursive`, src/retrieval.ts:263-359):
- For a **condensed** summary: walks `summary_parents` edges, returns `children: [{summaryId, kind, content, tokenCount}]`, recursing if `depth > 1`.
- For a **leaf** summary with `includeMessages=true`: fetches `summary_messages` edges → returns `messages: [{messageId, role, content, tokenCount}]`.
- Response shape: `{children[], messages[], estimatedTokens, truncated}`.

**Auth / recursion guards** (src/tools/lcm-expand-tool.ts:158-199):
- `lcm_expand` is only available in **sub-agent sessions** (`deps.isSubagentSessionKey(sessionKey)` check, else error: "lcm_expand is only available in sub-agent sessions. Use lcm_expand_query...").
- Delegated sessions must have a propagated grant: `resolveDelegatedExpansionGrantId(sessionKey)` → `runtimeAuthManager.getGrant(id)`. Missing grant → error.
- Orchestrator wrapped with `wrapWithAuth` when a grant exists.
- Separate `lcm-expansion-recursion-guard.ts` prevents recursion loops.
- `tokenCap` enforced during recursion; sets `truncated=true` and short-circuits.

A separate `lcm_expand_query` tool (src/tools/lcm-expand-query-tool.ts) is the primary-session entry point, which delegates to `lcm_expand` in a sub-agent.

Source: `src/tools/lcm-grep-tool.ts:83-213`, `src/tools/lcm-describe-tool.ts:58-237`, `src/tools/lcm-expand-tool.ts:24-448`, `src/retrieval.ts:134-359`.

---

### 5. Config defaults

From `src/db/config.ts:129-247` (hardcoded defaults, overridden by env vars `LCM_*` then plugin config):

| Key | Default |
|---|---|
| `enabled` | `true` |
| `databasePath` | `~/.openclaw/lcm.db` |
| `ignoreSessionPatterns` | `[]` |
| `statelessSessionPatterns` | `[]` |
| `skipStatelessSessions` | `true` |
| `contextThreshold` | `0.75` (of model window) |
| `freshTailCount` | `64` |
| `newSessionRetainDepth` | `2` |
| `leafMinFanout` | `8` |
| `condensedMinFanout` | `4` |
| `condensedMinFanoutHard` | `2` |
| `incrementalMaxDepth` | `1` |
| `leafChunkTokens` | `20000` |
| `bootstrapMaxTokens` | `max(6000, floor(leafChunkTokens * 0.3))` |
| `leafTargetTokens` | `2400` |
| `condensedTargetTokens` | `2000` |
| `maxExpandTokens` | `4000` |
| `largeFileTokenThreshold` | `25000` |
| `summaryProvider` / `summaryModel` | `""` (caller-provided) |
| `largeFileSummaryProvider` / `largeFileSummaryModel` | `""` |
| `expansionProvider` / `expansionModel` | `""` |
| `delegationTimeoutMs` | `120000` (2 min) |
| `summaryTimeoutMs` | `60000` (1 min) |
| `timezone` | `process.env.TZ` or system default |
| `pruneHeartbeatOk` | `false` |
| `maxAssemblyTokenBudget` | `undefined` (no ceiling) |
| `summaryMaxOverageFactor` | `3` |
| `customInstructions` | `""` |
| `circuitBreakerThreshold` | `5` (consecutive auth failures) |
| `circuitBreakerCooldownMs` | `1_800_000` (30 min) |

Env-var precedence (highest first): `LCM_*` env → plugin config → hardcoded default.

Source: `src/db/config.ts:129-247`.

---

## Open questions / gaps

- **Is the "RRF" label in retriever.ts accurate?** The header comment (src/retriever.ts:3) says "RRF fusion" and the trace stage is called `rrf_fusion`, but `fuseResults` implements **weighted sum with BM25 floor**, not reciprocal rank fusion. Implement as weighted sum (that is what actually runs).
- **Recency boost formula** (the secondary additive one at `recencyHalfLifeDays=14`, `recencyWeight=0.1`) is in `applyRecencyBoost` (src/retriever.ts:~880), not fully quoted above — distinct from the decay-engine composite. If needed, grep `applyRecencyBoost` / `applyLengthNormalization` / `applyImportanceWeight` in retriever.ts.
- **`getRerankPreservationFloor`** logic (per-item floor) is not quoted; look in retriever.ts around the rerank call.
- **Leaf chunk selection algorithm** (`selectOldestLeafChunk`) is referenced in compaction.ts but not read here — relevant if you need the exact rule that picks which raw messages get rolled into a leaf.
- **MMR diversity** (`applyMMRDiversity` in retriever.ts) uses MMR but the lambda/diversity parameter was not quoted.
- The decay-engine `recencyHalfLifeDays=30` (composite) and retriever `recencyHalfLifeDays=14` (boost) are **two different things**; the retriever's is an additive boost, the decay engine's is a multiplicative composite. Keep them separate in the port.
- memory-lancedb-pro `LlmClient` is a generic interface — there is no "default LLM" hardcoded in the package; callers wire one in.
- lossless-claw `bootstrap_state` / `large_files` / `message_parts` tables contain additional columns (`tool_call_id`, `patch_hash`, etc.) that were truncated in the quoted schema — full list is at src/db/migration.ts:462-499.
