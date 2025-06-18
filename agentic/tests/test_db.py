import pytest
from agentic.core.db import AgenticDB
import os

def test_db_set_and_get(tmp_path):
    db_path = tmp_path / "test.db"
    db = AgenticDB(str(db_path))
    db.set("foo", "bar")
    assert db.get("foo") == "bar"
    db.set("foo", "baz")
    assert db.get("foo") == "baz"

    db.close()
    db2 = AgenticDB(str(db_path))
    assert db2.get("foo") == "baz"
    db2.close()

    os.remove(db_path)

# Add more tests for error handling, missing keys, etc.
