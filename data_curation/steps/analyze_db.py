#!/usr/bin/env python3
"""
Script to analyze the Goodreads database structure and content.
This will help us understand the schema and data before implementing the DataAnalysisAgent.
"""

import json
import os
from typing import Any, Dict, List

import duckdb
import numpy as np
from loguru import logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Create a connection to the DuckDB database."""
    db_path = os.path.join("data", "goodreads.duckdb")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
    return duckdb.connect(db_path)


def get_schema_info(conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Get detailed schema information for all tables."""
    schema_info = {}

    # Get list of tables
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()

    for (table_name,) in tables:
        # Get column information
        columns = conn.execute(f"""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()

        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        schema_info[table_name] = {
            "columns": [
                {"name": col[0], "type": col[1], "nullable": col[2] == "YES"}
                for col in columns
            ],
            "row_count": row_count,
        }

    return schema_info


def get_table_stats(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, Any]:
    """Get statistical information for a specific table."""
    stats = {}

    # Get column information
    columns = conn.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
    """).fetchall()

    for col_name, col_type in columns:
        if col_type in ("INTEGER", "BIGINT", "DOUBLE", "FLOAT"):
            # Numeric column stats
            result = conn.execute(f"""
                SELECT 
                    COUNT(*) as count,
                    AVG({col_name}) as mean,
                    STDDEV_POP({col_name}) as std,
                    MIN({col_name}) as min,
                    MAX({col_name}) as max
                FROM {table_name}
            """).fetchone()

            stats[col_name] = {
                "count": int(result[0]) if result[0] is not None else None,
                "mean": float(result[1]) if result[1] is not None else None,
                "std": float(result[2]) if result[2] is not None else None,
                "min": float(result[3]) if result[3] is not None else None,
                "max": float(result[4]) if result[4] is not None else None,
            }
        elif col_type in ("VARCHAR", "TEXT", "CHAR"):
            # Categorical column stats
            unique_count = conn.execute(
                f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}"
            ).fetchone()[0]

            sample_values = conn.execute(f"""
                SELECT DISTINCT {col_name} 
                FROM {table_name} 
                WHERE {col_name} IS NOT NULL 
                LIMIT 5
            """).fetchall()

            stats[col_name] = {
                "unique_values": int(unique_count),
                "sample_values": [
                    str(val[0]) for val in sample_values if val[0] is not None
                ],
            }

    return stats


def get_sample_data(
    conn: duckdb.DuckDBPyConnection, table_name: str, n_rows: int = 5
) -> List[Dict[str, Any]]:
    """Get a random sample of rows from a table."""
    df = conn.execute(
        f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {n_rows}"
    ).fetchdf()

    # Convert numpy types to Python native types
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, np.integer):
                record[col] = int(val)
            elif isinstance(val, np.floating):
                record[col] = float(val)
            elif isinstance(val, np.ndarray):
                record[col] = val.tolist()
            else:
                record[col] = val
        records.append(record)

    return records


def main():
    """Main function to analyze the database."""
    logger.info("Starting database analysis...")

    try:
        conn = get_db_connection()

        # Get schema information
        logger.info("Getting schema information...")
        schema_info = get_schema_info(conn)

        # Save schema info
        os.makedirs("config", exist_ok=True)
        with open("config/schema.json", "w") as f:
            json.dump(schema_info, f, indent=2, cls=NumpyEncoder)
        logger.info("Schema information saved to config/schema.json")

        # Get and save stats for each table
        global_stats = {}
        samples = {}

        for table_name in schema_info.keys():
            logger.info(f"Analyzing table: {table_name}")

            # Get table statistics
            stats = get_table_stats(conn, table_name)
            global_stats[table_name] = stats

            # Get sample data
            sample = get_sample_data(conn, table_name)
            samples[table_name] = sample

        # Save global stats
        with open("config/global_stats.json", "w") as f:
            json.dump(global_stats, f, indent=2, cls=NumpyEncoder)
        logger.info("Global statistics saved to config/global_stats.json")

        # Save samples
        with open("config/samples.json", "w") as f:
            json.dump(samples, f, indent=2, cls=NumpyEncoder)
        logger.info("Sample data saved to config/samples.json")

        # Print summary
        logger.info("\nDatabase Analysis Summary:")
        for table_name, info in schema_info.items():
            logger.info(f"\nTable: {table_name}")
            logger.info(f"Row count: {info['row_count']}")
            logger.info("Columns:")
            for col in info["columns"]:
                logger.info(f"  - {col['name']} ({col['type']})")

    except Exception as e:
        logger.error(f"Error during database analysis: {str(e)}")
        raise
    finally:
        if "conn" in locals():
            conn.close()
        logger.info("Database analysis complete.")


if __name__ == "__main__":
    main()
