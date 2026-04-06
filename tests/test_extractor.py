"""Tests for the auto-capture extraction + dedup pipeline."""

import json
import math
import os
from unittest import mock

import pytest


@pytest.fixture
def temp_db(tmp_path):
    env = os.environ.copy()
    env["OPENCLAWD_LANCEDB_PATH"] = str(tmp_path / "lancedb")
    env["OPENCLAWD_EXTRACTOR"] = "ollama"  # force ollama to avoid needing API key
    with mock.patch.dict(os.environ, env):
        import importlib
        from openclawd import config, db
        importlib.reload(config)
        db.get_db.cache_clear()
        importlib.reload(db)
        yield tmp_path


@pytest.fixture
def mock_embed():
    """Mock embeddings: return deterministic vectors based on text hash."""
    def _embed(text):
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Spread 32 bytes into 768 floats (repeating)
        return [(b / 255.0) for b in h * 24]

    with mock.patch("openclawd.tools.memory_store.embed_one", side_effect=_embed), \
         mock.patch("openclawd.extractor.embed_one", side_effect=_embed), \
         mock.patch("openclawd.extractor.embed_batch", side_effect=lambda texts: [_embed(t) for t in texts]), \
         mock.patch("openclawd.tools.memory_recall.embed_one", side_effect=_embed):
        yield _embed


# --- Cosine similarity unit test ---

class TestCosine:
    def test_identical_vectors(self):
        from openclawd.extractor import _cosine_similarity
        v = [0.1, 0.2, 0.3]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, rel=1e-6)

    def test_orthogonal_vectors(self):
        from openclawd.extractor import _cosine_similarity
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors(self):
        from openclawd.extractor import _cosine_similarity
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, rel=1e-6)


# --- Extraction parsing ---

class TestExtraction:
    def test_parses_valid_extraction(self, temp_db, mock_embed):
        from openclawd.extractor import extract_memories

        fake_response = {
            "memories": [
                {"category": "preferences", "abstract": "Uses dark mode",
                 "overview": "User prefers dark UI", "content": "Dan uses dark mode everywhere"},
                {"category": "entities", "abstract": "Main machine is Mac Mini M4",
                 "overview": "Dev setup", "content": "Primary dev machine is Mac Mini M4 Pro"},
            ]
        }
        with mock.patch("openclawd.extractor.llm_json", return_value=fake_response):
            results = extract_memories("some conversation")

        assert len(results) == 2
        assert results[0].category == "preferences"
        assert results[0].abstract == "Uses dark mode"
        assert results[1].category == "entities"

    def test_caps_at_max_memories(self, temp_db, mock_embed):
        from openclawd.extractor import extract_memories, MAX_MEMORIES_PER_EXTRACTION

        fake_response = {
            "memories": [
                {"category": "events", "abstract": f"event {i}",
                 "overview": "", "content": f"event content {i}"}
                for i in range(10)
            ]
        }
        with mock.patch("openclawd.extractor.llm_json", return_value=fake_response):
            results = extract_memories("conversation")

        assert len(results) <= MAX_MEMORIES_PER_EXTRACTION

    def test_invalid_category_falls_back(self, temp_db, mock_embed):
        from openclawd.extractor import extract_memories

        fake_response = {
            "memories": [{"category": "BOGUS", "abstract": "x", "overview": "", "content": "y"}]
        }
        with mock.patch("openclawd.extractor.llm_json", return_value=fake_response):
            results = extract_memories("conversation")

        assert results[0].category == "patterns"  # fallback

    def test_llm_failure_returns_empty(self, temp_db, mock_embed):
        from openclawd.extractor import extract_memories

        with mock.patch("openclawd.extractor.llm_json", side_effect=RuntimeError("LLM down")):
            results = extract_memories("conversation")

        assert results == []


# --- Batch dedup ---

class TestBatchDedup:
    def test_identical_abstracts_deduped(self, temp_db, mock_embed):
        from openclawd.extractor import batch_dedup, ExtractedMemory

        cands = [
            ExtractedMemory("preferences", "User likes tabs", "", "tabs content"),
            ExtractedMemory("preferences", "User likes tabs", "", "tabs content 2"),
        ]
        # Same abstract → same embedding → cosine = 1.0 → second dropped
        result = batch_dedup(cands)
        assert len(result) == 1

    def test_different_abstracts_kept(self, temp_db, mock_embed):
        from openclawd.extractor import batch_dedup, ExtractedMemory

        cands = [
            ExtractedMemory("preferences", "Likes dark mode", "", "dark mode"),
            ExtractedMemory("entities", "Mac Mini M4 Pro", "", "dev machine"),
        ]
        result = batch_dedup(cands)
        assert len(result) == 2

    def test_single_candidate_passes(self, temp_db, mock_embed):
        from openclawd.extractor import batch_dedup, ExtractedMemory

        cands = [ExtractedMemory("events", "deployed v2", "", "we deployed")]
        result = batch_dedup(cands)
        assert len(result) == 1


# --- Store dedup + save ---

class TestStoreDedupAndSave:
    def test_stores_when_no_existing(self, temp_db, mock_embed):
        from openclawd.extractor import store_dedup_and_save, ExtractedMemory
        from openclawd.db import get_or_create_table, MEMORY_SCHEMA
        from openclawd import config

        cands = [ExtractedMemory("preferences", "Likes Python", "overview", "Dan likes Python")]
        results = store_dedup_and_save(cands, project="test")

        assert len(results) == 1
        assert "created" in results[0]

        table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
        assert table.count_rows() >= 1

    def test_skip_decision_respected(self, temp_db, mock_embed):
        """When LLM says skip, memory is not stored."""
        from openclawd.extractor import store_dedup_and_save, ExtractedMemory
        from openclawd.tools.memory_store import memory_store

        # Pre-store a memory with same content so cosine = 1.0 (hash-based mock)
        memory_store("Dan prefers tabs over spaces", category="preferences")

        # Try to store same content — dedup should find it, LLM says skip
        cands = [ExtractedMemory("preferences", "Indentation", "", "Dan prefers tabs over spaces")]
        with mock.patch("openclawd.extractor.llm_json", return_value={"decision": "skip"}):
            results = store_dedup_and_save(cands)

        assert "skipped" in results[0]

    def test_supersede_deletes_old(self, temp_db, mock_embed):
        from openclawd.extractor import store_dedup_and_save, ExtractedMemory
        from openclawd.tools.memory_store import memory_store

        # Store with same content so dedup finds it
        memory_store("Mac Mini M4 Pro is the dev machine", category="entities")

        cands = [ExtractedMemory("entities", "Updated fact", "", "Mac Mini M4 Pro is the dev machine")]
        with mock.patch("openclawd.extractor.llm_json",
                       return_value={"decision": "supersede", "match_index": 1}):
            results = store_dedup_and_save(cands)

        assert "superseded" in results[0]


# --- Full pipeline ---

class TestFullPipeline:
    def test_auto_extract_and_store(self, temp_db, mock_embed):
        from openclawd.extractor import auto_extract_and_store

        fake_response = {
            "memories": [
                {"category": "preferences", "abstract": "Prefers Python",
                 "overview": "Lang pref", "content": "Dan prefers Python for scripting"},
                {"category": "entities", "abstract": "Mac Mini M4",
                 "overview": "Hardware", "content": "Primary dev machine is Mac Mini M4 Pro"},
            ]
        }
        # Mock both extraction and dedup LLM calls
        with mock.patch("openclawd.extractor.llm_json", return_value=fake_response):
            result = auto_extract_and_store("Session about Python dev on Mac Mini")

        assert "Auto-extracted 2" in result
        assert "created" in result
