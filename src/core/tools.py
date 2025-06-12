"""
This module provides a collection of custom tools for the VULCAN agents to use
within their Python execution environment. These functions are designed to be
injected into the context available to the `execute_python` tool.
"""

import duckdb
from loguru import logger

from src.config.settings import DB_PATH


def get_table_sample(table_name: str, n_samples: int = 5) -> str:
    """
    Retrieves a random sample of rows from a specified table in the database.

    Args:
        table_name (str): The name of the table to sample from.
        n_samples (int): The number of rows to retrieve. Defaults to 5.

    Returns:
        str: A string representation of the sampled data in a markdown-friendly format,
             or an error message if the table cannot be accessed.
    """
    if not isinstance(table_name, str) or not table_name.isidentifier():
        return f"ERROR: Invalid table name '{table_name}'. Table names must be valid Python identifiers."

    if not isinstance(n_samples, int) or n_samples <= 0:
        return f"ERROR: Invalid number of samples '{n_samples}'. Must be a positive integer."

    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            # Use DuckDB's built-in TABLESAMPLE for efficient random sampling
            query = f'SELECT * FROM "{table_name}" USING SAMPLE {n_samples} ROWS;'
            sample_df = conn.execute(query).fetchdf()

            if sample_df.empty:
                return f"No data returned for table '{table_name}'. It may be empty."

            return (
                f"Sample of '{table_name}' ({n_samples} rows):\n"
                f"{sample_df.to_markdown(index=False)}"
            )
    except duckdb.CatalogException:
        return f"ERROR: Table '{table_name}' not found in the database."
    except Exception as e:
        logger.error(f"Failed to sample table '{table_name}': {e}")
        return (
            f"ERROR: An unexpected error occurred while sampling table '{table_name}'."
        )
