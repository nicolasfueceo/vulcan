import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any
import duckdb

class DataLoader:
    """
    Concurrency-safe DataLoader for DuckDB databases.
    Handles DB directory, connection pooling, and context management.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "data/goodreads_curated.duckdb"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = "data/goodreads_curated.duckdb"):
        if self._initialized:
            return
        self.db_path = db_path
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def set_db_path(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Any:
        conn = duckdb.connect(self.db_path, read_only=False)
        try:
            yield conn
        finally:
            conn.close()

    def get_db_path(self) -> str:
        return self.db_path

    def extract_and_cache_schema(self, cache_path: Optional[str] = None) -> dict:
        """
        Extracts the schema (tables and columns) from the DuckDB database,
        caches it as a JSON file, and returns the schema as a dict.
        """
        import json
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(self.db_path), "schema_cache.json")
        schema = {}
        with duckdb.connect(self.db_path, read_only=True) as conn:
            tables = conn.execute("SHOW TABLES").fetchall()
            for (table_name,) in tables:
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                schema[table_name] = [
                    {"column": col[0], "type": col[1], "null": col[2], "default": col[3]}
                    for col in columns
                ]
        with open(cache_path, "w") as f:
            json.dump(schema, f, indent=2)
        return schema
