"""
Integration test for the full VULCAN pipeline: discovery, strategy, feature realization, and optimization.
This test runs the orchestrator main function and asserts that the pipeline completes and produces an optimization report.
"""
"""
Integration test for the full VULCAN pipeline: discovery, strategy, feature realization, and optimization.
- Ensures DB views are created before pipeline run
- Checks DB schema is passed to prompts
- Checks for view accessibility issues
"""
import sys
import traceback
import subprocess
import duckdb

from src.orchestrator import main
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string

DB_PATH = "data/goodreads_curated.duckdb"
SQL_PATH = "scripts/create_interactions_view.sql"


def run_setup_views():
    # Run the setup_views script to ensure views exist
    result = subprocess.run([sys.executable, "scripts/setup_views.py"], capture_output=True, text=True)
    print("[TEST] setup_views.py output:", result.stdout)
    assert result.returncode == 0, "setup_views.py failed to run."


def test_db_schema_passed():
    # Check that DB schema is available and non-empty
    schema = get_db_schema_string()
    print("[TEST] DB schema:", schema[:300])
    assert "TABLE:" in schema or "VIEW:" in schema, "DB schema string does not contain expected content."


def test_view_access():
    # Directly connect to DuckDB and check if the interactions view exists and is queryable
    with duckdb.connect(DB_PATH) as conn:
        # Use information_schema.views for robust view discovery
        views = conn.execute("SELECT table_name FROM information_schema.views WHERE table_schema = 'main'").fetchall()
        print("[TEST] Views in DB:", views)
        found = any("interactions" in v[0] for v in views)
        assert found, "Interactions view not found in DB."
        # Try a simple query
        try:
            res = conn.execute("SELECT * FROM interactions LIMIT 1").fetchall()
            print("[TEST] Sample from interactions view:", res)
        except Exception as e:
            print("[TEST] Error querying interactions view:", e)
            assert False, f"Error querying interactions view: {e}"


def test_full_pipeline():
    try:
        # Run orchestrator main (with fast mode for speed)
        report = main(epochs=1, fast_mode_frac=0.2)
        assert 'Optimization' in report or 'optimization' in report or 'best_score' in report, "No optimization report found in output."
        print("[TEST] Full pipeline completed successfully. Optimization report present.")
    except Exception as e:
        print(f"[TEST] Full pipeline failed: {e}")
        print(traceback.format_exc())
        assert False, f"Pipeline failed: {e}"

if __name__ == "__main__":
    run_setup_views()
    test_db_schema_passed()
    test_view_access()
    test_full_pipeline()

