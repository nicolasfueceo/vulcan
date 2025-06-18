import os
import sqlite3
import pytest
from agentic.langgraph.data.data_loader import DataLoader

def test_data_loader_connection():
    db_path = "data/test_vulcan.db"
    loader = DataLoader(db_path=db_path)
    with loader.get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES ('foo')")
        result = conn.execute("SELECT value FROM test").fetchone()
        assert result[0] == 'foo'
    os.remove(db_path)
