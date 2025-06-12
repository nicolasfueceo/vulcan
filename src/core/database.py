import logging
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "data/goodreads_curated.duckdb"

logger = logging.getLogger(__name__)


def check_db_schema() -> bool:
    """
    Checks if the database has the required tables and they are not empty.
    """
    db_file = Path(DB_PATH)
    if not db_file.exists() or db_file.stat().st_size == 0:
        return False
    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            required_tables = {"books", "reviews", "users"}

            if not required_tables.issubset(tables):
                return False

            for table in required_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count == 0:
                    return False
        return True
    except duckdb.Error as e:
        logger.warning(f"Database schema check failed, will attempt to rebuild: {e}")
        return False


def ingest_json_to_duckdb():
    """
    Ingests data from gzipped JSON files into DuckDB, creating the schema.
    """
    books_json_path = "data/books.json.gz"
    reviews_json_path = "data/reviews.json.gz"

    logger.info(f"Starting ingestion from {books_json_path} and {reviews_json_path}")

    with duckdb.connect(database=DB_PATH, read_only=False) as conn:
        logger.info("Creating 'books' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE books AS 
            SELECT * 
            FROM read_json_auto('{books_json_path}', format='newline_delimited');
        """)
        logger.info("'books' table created.")

        logger.info("Creating 'reviews' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE reviews AS 
            SELECT *
            FROM read_json_auto('{reviews_json_path}', format='newline_delimited');
        """)
        logger.info("'reviews' table created.")

        logger.info("Creating 'users' table from distinct reviewers...")
        conn.execute("""
            CREATE OR REPLACE TABLE users AS
            SELECT DISTINCT user_id FROM reviews;
        """)
        logger.info("'users' table created.")

    logger.info("Data ingestion from JSON files to DuckDB complete.")


def fetch_df(query: str) -> pd.DataFrame:
    """
    Connects to the database, executes a query, and returns a DataFrame.
    """
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        return conn.execute(query).fetchdf()


def get_db_schema_string() -> str:
    """
    Introspects the database using SUMMARIZE and returns a detailed schema string
    with summary statistics. Connects in-process to avoid file locking issues.
    """
    schema_parts = []
    try:
        # Connect in-process to an in-memory database to avoid file locks
        with duckdb.connect() as conn:
            # Attach the main database file in READ_ONLY mode, giving it an alias 'db'
            conn.execute(f"ATTACH '{DB_PATH}' AS db (READ_ONLY);")

            # Query the information_schema to find tables in the attached database's 'main' schema
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_catalog = 'db';"
            ).fetchdf()

            if tables_df.empty:
                # Fallback to a simpler SHOW TABLES if the schema query fails
                try:
                    tables_df = conn.execute("SHOW TABLES FROM db;").fetchdf()
                except Exception:
                    return "ERROR: No tables found in the attached database. Could not list tables via information_schema or SHOW TABLES."

            if tables_df.empty:
                return "ERROR: No tables found in the attached database."

            for _, row in tables_df.iterrows():
                table_name = row["table_name"] if "table_name" in row else row["name"]

                # We must use the 'db' alias to refer to tables in the attached database
                qualified_table_name = f'db."{table_name}"'

                row_count_result = conn.execute(
                    f"SELECT COUNT(*) FROM {qualified_table_name};"
                ).fetchone()
                row_count = row_count_result[0] if row_count_result else 0
                schema_parts.append(f"TABLE: {table_name} ({row_count:,} rows)")

                # Use the SUMMARIZE command to get schema and statistics
                summary_df = conn.execute(
                    f"SUMMARIZE {qualified_table_name};"
                ).fetchdf()

                for _, summary_row in summary_df.iterrows():
                    col_name = summary_row["column_name"]
                    col_type = summary_row["column_type"]
                    null_pct = summary_row["null_percentage"]

                    stats = [f"NULLs: {null_pct}%"]

                    # Add type-specific stats for a richer summary
                    if "VARCHAR" in col_type.upper():
                        unique_count = summary_row.get("approx_unique")
                        if unique_count is not None:
                            stats.append(f"~{int(unique_count)} unique values")
                    elif any(
                        t in col_type.upper()
                        for t in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
                    ):
                        min_val = summary_row.get("min")
                        max_val = summary_row.get("max")
                        if min_val is not None and max_val is not None:
                            stats.append(f"range: [{min_val}, {max_val}]")

                    schema_parts.append(
                        f"  - {col_name} ({col_type}) [{', '.join(stats)}]"
                    )
                schema_parts.append("")

        return "\n".join(schema_parts)
    except Exception as e:
        logger.error(f"Failed to get database schema using SUMMARIZE method: {e}")
        logger.exception(e)
        return "ERROR: Could not retrieve database schema."
