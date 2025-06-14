"""
Script to extract and cache metadata from the database for use in prompt templates.
"""

import json
from loguru import logger
from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd

from src.config.log_config import setup_logging
from src.utils.run_utils import init_run


def extract_table_samples(n_samples: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Extracts stratified samples from each table.

    Args:
        n_samples: Number of samples to extract per table

    Returns:
        Dictionary mapping table names to sample DataFrames
    """

    samples = {}

    with duckdb.connect("data/goodreads_curated.duckdb") as conn:
        # Get samples from each table
        tables = [
            "curated_books",
            "curated_reviews",
            "book_authors",
            "book_series",
            "book_shelves",
            "book_similars",
            "users",
            "user_stats_daily",
        ]

        for table in tables:
            logger.info("Sampling {}...", table)

            # For reviews, sample by user to get complete user histories
            if table == "curated_reviews":
                # Get random users
                user_sample = conn.execute(
                    """
                    SELECT DISTINCT user_id 
                    FROM curated_reviews 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """,
                    [n_samples],
                ).fetchdf()

                # Get all reviews for these users
                samples[table] = conn.execute("""
                    SELECT * 
                    FROM curated_reviews 
                    WHERE user_id IN (
                        SELECT user_id FROM user_sample
                    )
                """).fetchdf()

            # For books, sample by author to get complete author catalogs
            elif table == "curated_books":
                # Get random authors
                author_sample = conn.execute(
                    """
                    SELECT DISTINCT author 
                    FROM curated_books 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """,
                    [n_samples],
                ).fetchdf()

                # Get all books by these authors
                samples[table] = conn.execute("""
                    SELECT * 
                    FROM curated_books 
                    WHERE author IN (
                        SELECT author FROM author_sample
                    )
                """).fetchdf()

            # For other tables, just get random samples
            else:
                samples[table] = conn.execute(
                    f"""
                    SELECT * 
                    FROM {table} 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """,
                    [n_samples],
                ).fetchdf()

    return samples


def extract_column_stats() -> Dict[str, Dict[str, Any]]:
    """
    Extracts column-level statistics from each table.

    Returns:
        Dictionary mapping table names to column statistics
    """

    stats = {}

    with duckdb.connect("data/goodreads_curated.duckdb") as conn:
        tables = [
            "curated_books",
            "curated_reviews",
            "book_authors",
            "book_series",
            "book_shelves",
            "book_similars",
            "users",
            "user_stats_daily",
        ]

        for table in tables:
            logger.info("Analyzing {}...", table)
            table_stats = {}

            # Get column info
            columns = conn.execute(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = ?
            """,
                [table],
            ).fetchdf()

            for _, row in columns.iterrows():
                col_name = row["column_name"]
                col_type = row["data_type"]

                # Get null percentage
                null_pct = conn.execute(f"""
                    SELECT 
                        COUNT(*) FILTER (WHERE {col_name} IS NULL)::FLOAT / COUNT(*) as null_pct
                    FROM {table}
                """).fetchone()[0]

                # Get unique count for text columns
                unique_count = None
                if "VARCHAR" in col_type.upper():
                    unique_count = conn.execute(f"""
                        SELECT COUNT(DISTINCT {col_name}) 
                        FROM {table}
                    """).fetchone()[0]

                # Get min/max for numeric columns
                min_val = None
                max_val = None
                if any(
                    t in col_type.upper()
                    for t in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
                ):
                    min_max = conn.execute(f"""
                        SELECT MIN({col_name}), MAX({col_name})
                        FROM {table}
                    """).fetchone()
                    min_val = min_max[0]
                    max_val = min_max[1]

                table_stats[col_name] = {
                    "type": col_type,
                    "null_percentage": null_pct,
                    "unique_count": unique_count,
                    "min_value": min_val,
                    "max_value": max_val,
                }

            stats[table] = table_stats

    return stats


def cache_metadata():
    """
    Extracts and caches metadata from the database.
    """


    # Create config directory if it doesn't exist
    Path("config").mkdir(exist_ok=True)

    # Extract and save samples
    logger.info("Extracting table samples...")
    samples = extract_table_samples()
    samples_dict = {name: df.to_dict(orient="records") for name, df in samples.items()}

    with open("config/samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_dict, f, indent=2)
    logger.info("Saved samples to config/samples.json")

    # Extract and save column stats
    logger.info("Extracting column statistics...")
    stats = extract_column_stats()

    with open("config/summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved summary statistics to config/summary_stats.json")


def main():
    # Initialize run context
    init_run()

    # Setup logging
    setup_logging()

    # Run metadata caching
    cache_metadata()


if __name__ == "__main__":
    main()
