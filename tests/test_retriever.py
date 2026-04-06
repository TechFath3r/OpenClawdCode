"""Tests for hybrid retrieval fusion logic and end-to-end recall with new retriever."""

import os
import time
from unittest import mock

import pytest


# --- Unit tests for fusion math (no DB needed) ---

class TestFusion:
    def test_weighted_sum(self):
        from openclawd.retriever import _fuse, VECTOR_WEIGHT, BM25_WEIGHT

        vec_hits = {"m1": {"id": "m1", "vector_score": 0.9, "bm25_score": 0.0}}
        fts_hits = {"m1": {"id": "m1", "vector_score": 0.0, "bm25_score": 0.6}}
        merged = _fuse(vec_hits, fts_hits)

        expected = VECTOR_WEIGHT * 0.9 + BM25_WEIGHT * 0.6  # 0.63 + 0.18 = 0.81
        assert merged["m1"]["fused_score"] == pytest.approx(expected, rel=1e-4)

    def test_bm25_exact_match_floor(self):
        """BM25 ≥ 0.75 preserves the result even with low vector score."""
        from openclawd.retriever import _fuse, BM25_FLOOR_FACTOR

        vec_hits = {}  # not in vector results at all
        fts_hits = {"m1": {"id": "m1", "vector_score": 0.0, "bm25_score": 0.85}}
        merged = _fuse(vec_hits, fts_hits)

        # max(0.7*0 + 0.3*0.85, 0.85*0.92) = max(0.255, 0.782) = 0.782
        assert merged["m1"]["fused_score"] == pytest.approx(0.85 * BM25_FLOOR_FACTOR, rel=1e-4)

    def test_bm25_below_floor_no_boost(self):
        """BM25 < 0.75 does NOT get the floor treatment."""
        from openclawd.retriever import _fuse

        vec_hits = {"m1": {"id": "m1", "vector_score": 0.5, "bm25_score": 0.0}}
        fts_hits = {"m1": {"id": "m1", "vector_score": 0.0, "bm25_score": 0.6}}
        merged = _fuse(vec_hits, fts_hits)

        # max(0.7*0.5 + 0.3*0.6, 0) = max(0.35+0.18, 0) = 0.53
        assert merged["m1"]["fused_score"] == pytest.approx(0.53, rel=1e-3)

    def test_union_of_both_sources(self):
        """Fusion includes IDs from vector-only and FTS-only hits."""
        from openclawd.retriever import _fuse

        vec_hits = {"m1": {"id": "m1", "vector_score": 0.8, "bm25_score": 0.0}}
        fts_hits = {"m2": {"id": "m2", "vector_score": 0.0, "bm25_score": 0.9}}
        merged = _fuse(vec_hits, fts_hits)

        assert "m1" in merged
        assert "m2" in merged

    def test_clamp_minimum(self):
        """Very low scores get clamped to SCORE_CLAMP_MIN."""
        from openclawd.retriever import _fuse, SCORE_CLAMP_MIN

        vec_hits = {"m1": {"id": "m1", "vector_score": 0.01, "bm25_score": 0.0}}
        fts_hits = {}
        merged = _fuse(vec_hits, fts_hits)
        assert merged["m1"]["fused_score"] == pytest.approx(SCORE_CLAMP_MIN, rel=1e-4)


# --- Integration tests (with real LanceDB but mocked embeddings) ---

@pytest.fixture
def temp_db(tmp_path):
    env = os.environ.copy()
    env["OPENCLAWD_LANCEDB_PATH"] = str(tmp_path / "lancedb")
    with mock.patch.dict(os.environ, env):
        import importlib
        from openclawd import config, db
        importlib.reload(config)
        db.get_db.cache_clear()
        importlib.reload(db)
        yield tmp_path


@pytest.fixture
def mock_embed():
    vec = [0.1] * 768
    with mock.patch("openclawd.tools.memory_store.embed_one", return_value=vec), \
         mock.patch("openclawd.tools.memory_recall.embed_one", return_value=vec):
        yield vec


def test_recall_uses_hybrid_retriever(temp_db, mock_embed):
    """Recall returns results with score/vec/bm25 indicators."""
    from openclawd.tools.memory_store import memory_store
    from openclawd.tools.memory_recall import memory_recall

    memory_store("Python prefers spaces for indentation", category="preference")

    result = memory_recall("Python spaces")
    assert "spaces" in result
    assert "score:" in result
    assert "vec:" in result
    assert "bm25:" in result


def test_recall_respects_tier_filter(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    from openclawd.tools.memory_recall import memory_recall

    memory_store("core fact", tier="core")
    memory_store("peripheral noise", tier="peripheral")

    result = memory_recall("fact", tier="core")
    assert "core fact" in result
    assert "peripheral noise" not in result


def test_recall_returns_empty_on_no_memories(temp_db, mock_embed):
    from openclawd.tools.memory_recall import memory_recall
    result = memory_recall("anything")
    assert "No memories stored" in result
