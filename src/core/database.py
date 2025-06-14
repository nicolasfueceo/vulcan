import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import DB_PATH

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
    db_path = str(DB_PATH)  # Ensure it's a string for DuckDB

    try:
        logger.debug(f"Generating database schema from: {db_path}")

        # Connect in-process to an in-memory database to avoid file locks
        with duckdb.connect() as conn:
            # Attach the main database file in READ_ONLY mode, giving it an alias 'db'
            conn.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")

            # Query the information_schema to find tables in the attached database's 'main' schema
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_catalog = 'db';"
            ).fetchdf()

            if tables_df.empty:
                # Fallback to a simpler SHOW TABLES if the schema query fails
                try:
                    tables_df = conn.execute("SHOW TABLES FROM db;").fetchdf()
                    logger.debug("Used SHOW TABLES fallback method")
                except Exception:
                    logger.error(
                        "Failed to list tables via both information_schema and SHOW TABLES"
                    )
                    return "ERROR: No tables found in the attached database. Could not list tables via information_schema or SHOW TABLES."

            if tables_df.empty:
                logger.warning("No tables found in the database")
                return "ERROR: No tables found in the attached database."

            logger.debug(f"Found {len(tables_df)} tables in database")

            for _, row in tables_df.iterrows():
                table_name = row["table_name"] if "table_name" in row else row["name"]

                # We must use the 'db' alias to refer to tables in the attached database
                qualified_table_name = f'db."{table_name}"'

                try:
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

                except Exception as table_error:
                    logger.warning(
                        f"Failed to analyze table {table_name}: {table_error}"
                    )
                    schema_parts.append(f"TABLE: {table_name} (analysis failed)")
                    schema_parts.append("")

        result = "\n".join(schema_parts)
        logger.debug(f"Generated schema string with {len(result)} characters")
        return result

    except Exception as e:
        logger.error(f"Failed to get database schema using SUMMARIZE method: {e}")
        logger.exception(e)
        return (
            f"ERROR: Could not retrieve database schema from {db_path}. Error: {str(e)}"
        )


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Returns a read-write connection to the main DuckDB database."""
    return duckdb.connect(database=str(DB_PATH), read_only=False)


class DatabaseConnection:
    def __init__(
        self, connection_string: Optional[str] = None, engine: Optional[Engine] = None
    ):
        """Initialize database connection

        Args:
            connection_string: SQLAlchemy connection string
            engine: Existing SQLAlchemy engine (for testing)
        """
        if engine:
            self.engine = engine
        else:
            connection_string = connection_string or os.getenv(
                "DATABASE_URL",
                "sqlite:///data/vulcan.db",  # Default to SQLite
            )
            self.engine = create_engine(connection_string)

        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results

        Args:
            query: SQL query string

        Returns:
            Dictionary with query results
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return {"data": df.to_dict(orient="records")}
                return {"affected_rows": result.rowcount}
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def create_view(self, view_name: str, query: str, version: Optional[int] = None):
        """Create or replace a view

        Args:
            view_name: Name of the view
            query: SQL query defining the view
            version: Optional version number to append to view name
        """
        if version:
            view_name = f"{view_name}_v{version}"

        create_view_sql = f"CREATE OR REPLACE VIEW {view_name} AS {query}"

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_view_sql))
                conn.commit()
            logger.info(f"View {view_name} created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create view {view_name}: {str(e)}")
            raise

    def drop_view(self, view_name: str):
        """Drop a view if it exists

        Args:
            view_name: Name of the view to drop
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
                conn.commit()
            logger.info(f"View {view_name} dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop view {view_name}: {str(e)}")
            raise

    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")
