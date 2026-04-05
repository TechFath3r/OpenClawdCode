"""Tests for schema v0.2 fields and the old→new table migration."""

import os
from unittest import mock

import pyarrow as pa
import pytest


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


def test_new_schema_has_all_v02_fields(temp_db, mock_embed):
    """Fresh table should have every v0.2 column populated after store."""
    from openclawd.tools.memory_store import memory_store
    from openclawd.db import get_or_create_table, MEMORY_SCHEMA
    from openclawd import config

    memory_store("hello world", category="preference", project="garage", importance=8,
                 tier="core", temporal_type="static", confidence=0.9)

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    row = table.to_arrow().to_pylist()[0]

    assert row["tier"] == "core"
    assert row["temporal_type"] == "static"
    assert row["scope"] == "project:garage"
    assert row["confidence"] == pytest.approx(0.9, rel=1e-5)
    assert row["access_count"] == 0
    assert row["last_accessed_at"] > 0
    assert row["abstract"] == "hello world"      # derived from first 120 chars
    assert row["overview"] == ""


def test_scope_defaults_to_global_when_no_project(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    from openclawd.db import get_or_create_table, MEMORY_SCHEMA
    from openclawd import config

    memory_store("no project here")

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    row = table.to_arrow().to_pylist()[0]
    assert row["scope"] == "global"


def test_explicit_scope_overrides_project(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    from openclawd.db import get_or_create_table, MEMORY_SCHEMA
    from openclawd import config

    memory_store("test", project="garage", scope="agent:openclawd")

    table = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    row = table.to_arrow().to_pylist()[0]
    assert row["scope"] == "agent:openclawd"


def test_invalid_tier_rejected(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    result = memory_store("test", tier="nonsense")
    assert "Invalid tier" in result


def test_tier_filter_on_recall(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    from openclawd.tools.memory_recall import memory_recall

    memory_store("durable fact", tier="core")
    memory_store("noisy fact", tier="peripheral")

    result = memory_recall("fact", tier="core")
    assert "durable fact" in result
    assert "noisy fact" not in result


def test_scope_filter_on_recall(temp_db, mock_embed):
    from openclawd.tools.memory_store import memory_store
    from openclawd.tools.memory_recall import memory_recall

    memory_store("project thing", project="garage")
    memory_store("global thing")

    result = memory_recall("thing", scope="global")
    assert "global thing" in result
    assert "project thing" not in result


def test_migration_from_old_schema(temp_db, mock_embed):
    """Simulate a pre-v0.2 table: create it manually, then let
    get_or_create_table migrate it by adding the new columns."""
    from openclawd.db import get_db, get_or_create_table, MEMORY_SCHEMA
    from openclawd import config

    old_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("category", pa.string()),
        pa.field("project", pa.string()),
        pa.field("tags", pa.string()),
        pa.field("importance", pa.int32()),
        pa.field("created_at", pa.float64()),
        pa.field("updated_at", pa.float64()),
        pa.field("source", pa.string()),
    ])
    db = get_db()
    table = db.create_table(config.MEMORY_TABLE, schema=old_schema)
    table.add([
        {"id": "m1", "content": "garage memory", "vector": [0.1] * 768,
         "category": "general", "project": "garage", "tags": "[]",
         "importance": 5, "created_at": 100.0, "updated_at": 100.0, "source": "manual"},
        {"id": "m2", "content": "global memory", "vector": [0.2] * 768,
         "category": "general", "project": "", "tags": "[]",
         "importance": 3, "created_at": 200.0, "updated_at": 200.0, "source": "manual"},
    ])
    assert "tier" not in table.schema.names    # sanity: pre-migration

    migrated = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    assert set(MEMORY_SCHEMA.names).issubset(set(migrated.schema.names))

    rows = {r["id"]: r for r in migrated.to_arrow().to_pylist()}
    assert rows["m1"]["scope"] == "project:garage"
    assert rows["m2"]["scope"] == "global"
    assert rows["m1"]["tier"] == "working"
    assert rows["m1"]["temporal_type"] == "static"
    assert rows["m1"]["confidence"] == pytest.approx(1.0, rel=1e-5)
    assert rows["m1"]["access_count"] == 0
    assert rows["m1"]["last_accessed_at"] == 100.0   # copied from created_at
    assert rows["m2"]["last_accessed_at"] == 200.0


def test_migration_is_idempotent(temp_db, mock_embed):
    """Running the migration twice should not fail or duplicate columns."""
    from openclawd.tools.memory_store import memory_store
    from openclawd.db import get_or_create_table, MEMORY_SCHEMA
    from openclawd import config

    memory_store("first")
    table1 = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    cols1 = set(table1.schema.names)
    table2 = get_or_create_table(config.MEMORY_TABLE, MEMORY_SCHEMA)
    cols2 = set(table2.schema.names)
    assert cols1 == cols2
