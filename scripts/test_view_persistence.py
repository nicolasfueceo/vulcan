# scripts/test_view_persistence.py
"""
Minimal test: Ensure that a view created by one agent/tool is accessible in a fresh connection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Ensure src/ is importable

from src.utils.tools import create_analysis_view
from src.config.settings import DB_PATH
import duckdb

TEST_VIEW_NAME = "test_view_persistence"
TEST_SQL = "SELECT 1 AS value UNION ALL SELECT 2 AS value"


def test_view_persistence():
    # Step 1: Create the view using the tool
    result = create_analysis_view(TEST_VIEW_NAME, TEST_SQL, rationale="Test view for persistence.")
    print(f"View creation result: {result}")
    assert "Successfully created view" in result, f"View creation failed: {result}"

    # Step 2: In a new connection, attempt to query the view
    with duckdb.connect(database=str(DB_PATH), read_only=True) as conn:
        try:
            df = conn.execute(f'SELECT * FROM {TEST_VIEW_NAME}').fetchdf()
            print(df)
            assert not df.empty, "Query returned no rows, but view should exist."
            assert set(df["value"]) == {1, 2}, f"Unexpected view contents: {df}"  # Should be 1 and 2
        except Exception as e:
            raise AssertionError(f"Failed to query view in new connection: {e}")

    print("Test passed: View is accessible in a new connection.")

if __name__ == "__main__":
    test_view_persistence()
