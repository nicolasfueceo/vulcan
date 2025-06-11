import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.utils.database import DB_PATH

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
