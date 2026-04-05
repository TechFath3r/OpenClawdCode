# Code Conventions

## Python

- **Python 3.10+** — use modern syntax (`str | None`, `list[str]`, match statements where natural)
- **Type hints** on public functions and MCP tools. Internal helpers optional.
- **Docstrings** — every MCP tool must have a clear docstring; it becomes the tool description Claude sees. Be explicit about arg types and constraints.
- **No comments stating the obvious.** Comments explain *why*, not *what*.
- **Imports** — relative imports within the package (`from ..db import ...`), absolute for external.
- **f-strings** over `.format()` / `%`.
- **pathlib.Path** over `os.path` where it reads cleaner. `os.path` is fine when we're already using it in a module.

## MCP Tools

- **One tool per file** in `src/openclawd/tools/`
- **Return strings**, not dicts — Claude reads the string output directly
- **Fail gracefully** — return a human-readable error string instead of raising, so Claude can react
- **Keep tool names stable** — they're in users' registered MCP configs. Renaming breaks installs.

## Config

- All user-facing config via env vars with `OPENCLAWD_` prefix
- Loaded once from `~/.config/openclawd/.env` in `config.py`
- **Never hardcode paths** — always go through `config.py`
- Defaults must work with zero config (LanceDB + Ollama local)

## Hooks

- Hooks live in `hooks/` and are plain Python scripts (shebang + `json.dumps` output to stdout)
- Keep hook logic **minimal and fast** — they run every turn. Offload real work to MCP tools or background workers.
- Never block on network calls inside a hook. Use a timeout and fail silent.

## Testing

- `pytest` with fixtures per the existing patterns in `tests/`
- Mock Ollama embeddings in tests (don't hit the real Ollama service)
- Use `tmp_path` for LanceDB instances per test
- Remember to `db.get_db.cache_clear()` + `importlib.reload(config, db)` in fixtures so env overrides stick

## Naming

- Package: `openclawd` (short, matches CLI entry points)
- Product name in docs: `OpenClawdCode`
- MCP server name (registered with Claude Code): `openclawd-memory`
- CLI entry points: `openclawd-server`, `openclawd-index`
