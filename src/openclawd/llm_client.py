"""Thin LLM abstraction for extraction and dedup prompts.

Supports two backends:
- "haiku": Anthropic API (claude-haiku-4-5-20251001), needs ANTHROPIC_API_KEY
- "ollama": local Ollama chat endpoint

Config: OPENCLAWD_EXTRACTOR = auto|haiku|ollama
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from . import config

logger = logging.getLogger("openclawd")


def _resolve_backend() -> str:
    """Resolve 'auto' to a concrete backend."""
    ext = config.EXTRACTOR
    if ext == "auto":
        return "haiku" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
    return ext


def _haiku_call(system: str, user: str) -> str:
    """Call Anthropic Haiku 4.5 via the SDK."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "OPENCLAWD_EXTRACTOR=haiku requires the anthropic package. "
            "Install with: pip install anthropic"
        )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _ollama_call(system: str, user: str) -> str:
    """Call local Ollama chat endpoint."""
    resp = httpx.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.EXTRACTOR_OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def llm_call(system: str, user: str) -> str:
    """Call the configured LLM backend. Returns response text (expected JSON)."""
    backend = _resolve_backend()
    if backend == "haiku":
        return _haiku_call(system, user)
    elif backend == "ollama":
        return _ollama_call(system, user)
    else:
        raise ValueError(f"Unknown extractor backend: {backend!r}")


def llm_json(system: str, user: str) -> dict | list:
    """Call LLM and parse response as JSON. Raises on parse failure."""
    raw = llm_call(system, user)
    # Strip markdown fences if present (common with Ollama models)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)
