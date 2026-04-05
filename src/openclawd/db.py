"""LanceDB connection, table schemas, and schema migrations."""

import logging
import os
from functools import lru_cache

import lancedb
import pyarrow as pa

from . import config

logger = logging.getLogger("openclawd")

# --- Schemas ---

MEMORY_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),              # L2: full narrative
    pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
    pa.field("category", pa.string()),
    pa.field("project", pa.string()),
    pa.field("tags", pa.string()),                  # JSON-encoded list
    pa.field("importance", pa.int32()),             # 1..10
    pa.field("created_at", pa.float64()),
    pa.field("updated_at", pa.float64()),
    pa.field("source", pa.string()),
    # --- v0.2: three-tier memory structure + decay/scope ---
    pa.field("abstract", pa.string()),              # L0: one-line index
    pa.field("overview", pa.string()),              # L1: markdown summary
    pa.field("tier", pa.string()),                  # core|working|peripheral
    pa.field("temporal_type", pa.string()),         # static|dynamic
    pa.field("confidence", pa.float32()),           # 0..1
    pa.field("access_count", pa.int32()),
    pa.field("last_accessed_at", pa.float64()),
    pa.field("scope", pa.string()),                 # global|project:X|agent:X|...
])

VAULT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
    pa.field("filepath", pa.string()),
    pa.field("heading", pa.string()),
    pa.field("modified", pa.float64()),
])


# --- Migration definitions ---

# SQL expressions backfilling each new column from existing row data.
# Used when migrating a pre-v0.2 table to the current schema.
_MEMORY_MIGRATION_DEFAULTS: dict[str, str] = {
    "abstract": "''",
    "overview": "''",
    "tier": "'working'",
    "temporal_type": "'static'",
    "confidence": "cast(1.0 as float)",
    "access_count": "cast(0 as int)",
    "last_accessed_at": "created_at",
    "scope": "'project:' || project",   # post-fixed to 'global' where project = ''
}


# --- Connection ---

@lru_cache(maxsize=1)
def get_db() -> lancedb.DBConnection:
    """Get or create the LanceDB connection."""
    os.makedirs(config.LANCEDB_PATH, exist_ok=True)
    return lancedb.connect(config.LANCEDB_PATH)


def _migrate_table_if_needed(
    table: lancedb.table.Table,
    expected_schema: pa.Schema,
    defaults: dict[str, str],
) -> lancedb.table.Table:
    """Add any columns in expected_schema that are missing from table.

    Uses LanceDB add_columns with SQL default expressions. Forward-compat:
    unknown columns already on the table are left alone. Raises if a missing
    column has no migration default defined.
    """
    existing = set(table.schema.names)
    expected = set(expected_schema.names)
    missing = expected - existing
    if not missing:
        return table

    undefined = missing - set(defaults)
    if undefined:
        raise RuntimeError(
            f"Cannot migrate table {table.name!r}: no default defined "
            f"for columns {sorted(undefined)}. Update _MEMORY_MIGRATION_DEFAULTS."
        )

    to_add = {col: defaults[col] for col in missing}
    logger.info("Migrating table %s: adding columns %s", table.name, sorted(missing))
    table.add_columns(to_add)

    # Post-fix scope: empty project → 'global' (SQL CASE isn't supported by Lance)
    if "scope" in missing:
        table.update(where="project = ''", values={"scope": "global"})

    return table


def get_or_create_table(name: str, schema: pa.Schema) -> lancedb.table.Table:
    """Open a table, creating it if missing, migrating it if the schema drifted.

    Only the memory table carries a defined migration path; other tables with
    schema drift will raise if new columns cannot be backfilled.
    """
    db = get_db()
    if name in db.list_tables().tables:
        table = db.open_table(name)
        if name == config.MEMORY_TABLE:
            return _migrate_table_if_needed(table, schema, _MEMORY_MIGRATION_DEFAULTS)
        return table
    return db.create_table(name, schema=schema)
