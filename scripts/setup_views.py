# scripts/setup_views.py
"""
Ensures DuckDB views for VULCAN compatibility are created.
Run at the start of orchestrator.py to guarantee the 'interactions' view exists with 'item_id' alias.
"""
import duckdb
import os

DB_PATH = os.environ.get("VULCAN_DB_PATH", "data/goodreads_curated.duckdb")
SQL_PATH = "scripts/create_interactions_view.sql"

def setup_views(db_path=DB_PATH, sql_path=SQL_PATH):
    with duckdb.connect(db_path) as conn:
        with open(sql_path, "r") as f:
            sql = f.read()
        conn.execute(sql)
        print(f"[setup_views] Executed {sql_path} on {db_path}")

if __name__ == "__main__":
    setup_views()
