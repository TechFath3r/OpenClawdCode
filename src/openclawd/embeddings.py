"""Ollama embedding client."""

import httpx

from . import config

_dim_validated = False


def _url() -> str:
    return f"{config.OLLAMA_URL}/api/embed"


def _validate_dim(actual: int) -> None:
    """Ensure the model's embedding dim matches config.EMBED_DIM. Fail loud on mismatch.

    The LanceDB schema uses a fixed-size vector column, so changing embedding
    model without updating OPENCLAWD_EMBED_DIM silently corrupts the database.
    """
    global _dim_validated
    if _dim_validated:
        return
    if actual != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: model {config.EMBED_MODEL!r} returned "
            f"{actual} dims, but OPENCLAWD_EMBED_DIM={config.EMBED_DIM}. "
            f"Fix by either (a) setting OPENCLAWD_EMBED_DIM={actual} in your "
            f".env and recreating the LanceDB tables, or (b) switching to a "
            f"{config.EMBED_DIM}-dim model like nomic-embed-text."
        )
    _dim_validated = True


def embed_one(text: str) -> list[float]:
    """Get embedding for a single text."""
    resp = httpx.post(
        _url(),
        json={"model": config.EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    vector = resp.json()["embeddings"][0]
    _validate_dim(len(vector))
    return vector


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Get embeddings for multiple texts in batches."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = httpx.post(
            _url(),
            json={"model": config.EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]
        if vectors:
            _validate_dim(len(vectors[0]))
        all_embeddings.extend(vectors)
    return all_embeddings
