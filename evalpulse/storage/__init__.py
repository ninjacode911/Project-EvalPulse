"""EvalPulse storage backends."""

from __future__ import annotations

from evalpulse.storage.base import StorageBackend
from evalpulse.storage.sqlite_store import SQLiteStore

_store: StorageBackend | None = None


def get_storage(config=None) -> StorageBackend:
    """Get the global storage backend singleton.

    Uses SQLite by default. Pass a config object to customize.
    """
    global _store
    if _store is None:
        if config is not None and config.storage_backend == "postgres":
            try:
                from evalpulse.storage.postgres_store import PostgresStore
            except ImportError:
                raise ImportError(
                    "PostgreSQL storage requires 'psycopg2-binary'. "
                    "Install with: pip install evalpulse[postgres]"
                ) from None
            _store = PostgresStore(config.database_url)
        else:
            db_path = config.sqlite_path if config else "evalpulse.db"
            _store = SQLiteStore(db_path)
    return _store


def reset_storage() -> None:
    """Reset the storage singleton (useful for testing)."""
    global _store
    if _store is not None:
        _store.close()
    _store = None


__all__ = ["StorageBackend", "SQLiteStore", "get_storage", "reset_storage"]
