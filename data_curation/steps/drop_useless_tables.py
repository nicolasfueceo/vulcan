#!/usr/bin/env python3
"""
drop_unused_columns.py
----------------------

Remove redundant / always-NULL fields from a local DuckDB file.

Usage
-----
$ python drop_unused_columns.py data/goodreads_curated.duckdb
"""

import sys
from pathlib import Path
import duckdb


def column_exists(con, table: str, column: str) -> bool:
    """Return True if the column is still present in `table`."""
    return (
        con.execute(
            """
            SELECT COUNT(*) = 1
            FROM duckdb_information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table, column],
        ).fetchone()[0]
        == 1
    )


def maybe_drop(con, table: str, column: str) -> None:
    """Drop a column only if it exists (makes the script re-runnable)."""
    if column_exists(con, table, column):
        print(f"Dropping {table}.{column} …")
        con.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}";')
    else:
        print(f"{table}.{column} already absent – skipping.")


def main(db_path: str) -> None:
    db_file = Path(db_path)
    if not db_file.exists():
        sys.exit(f"❌  Database file not found: {db_path}")

    con = duckdb.connect(db_path)

    # 1️⃣  user_stats.day  (all NULL)
    maybe_drop(con, "user_stats", "day")

    # 2️⃣  curated_reviews.*  (three always-NULL timestamp columns)
    for col in ("date_updated", "read_at", "started_at"):
        maybe_drop(con, "curated_reviews", col)

    con.close()
    print("✅  All done!")


if __name__ == "__main__":
    db_path = "data/goodreads_curated.duckdb"
    main(db_path)
