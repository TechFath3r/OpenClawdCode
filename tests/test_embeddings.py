"""Tests for embed-dim validation."""

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def reset_dim_validation():
    """Each test starts with _dim_validated = False and config reloaded to defaults.

    test_config.py mutates config.EMBED_DIM via importlib.reload — make sure
    those changes don't leak into this module.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCLAWD_")}
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib
        from openclawd import config, embeddings
        importlib.reload(config)
        importlib.reload(embeddings)
        embeddings._dim_validated = False
        yield
        embeddings._dim_validated = False


def _fake_post(vectors):
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"embeddings": vectors}
    return FakeResponse()


def test_embed_dim_match_passes():
    from openclawd import embeddings
    with mock.patch("openclawd.embeddings.httpx.post", return_value=_fake_post([[0.0] * 768])):
        vec = embeddings.embed_one("hello")
        assert len(vec) == 768


def test_embed_dim_mismatch_raises(tmp_path):
    # Force config.EMBED_DIM to 768 (the default) then return a 384-dim vector.
    from openclawd import embeddings
    with mock.patch("openclawd.embeddings.httpx.post", return_value=_fake_post([[0.0] * 384])):
        with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
            embeddings.embed_one("hello")


def test_embed_dim_validated_only_once():
    """After the first successful validation, subsequent calls skip the check."""
    from openclawd import embeddings
    calls = {"n": 0}
    def counting_post(*args, **kwargs):
        calls["n"] += 1
        return _fake_post([[0.0] * 768])
    with mock.patch("openclawd.embeddings.httpx.post", side_effect=counting_post):
        embeddings.embed_one("a")
        embeddings.embed_one("b")
        embeddings.embed_one("c")
    assert calls["n"] == 3
    assert embeddings._dim_validated is True


def test_batch_also_validates():
    from openclawd import embeddings
    with mock.patch("openclawd.embeddings.httpx.post", return_value=_fake_post([[0.0] * 384, [0.0] * 384])):
        with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
            embeddings.embed_batch(["a", "b"])
