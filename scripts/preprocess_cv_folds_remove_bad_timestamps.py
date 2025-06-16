import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger

# Settings
DB_PATH = "data/goodreads_curated.duckdb"
SPLITS_DIR = Path("data/processed/cv_splits")
SUMMARY_PATH = SPLITS_DIR / "cv_summary.json"
FOLDS_PATH = SPLITS_DIR / "cv_folds.json"

# Which columns to check for timestamps
TIMESTAMP_COLUMNS = ["timestamp", "created_at", "review_date", "date"]

# DuckDB's pandas timestamp limit
PANDAS_TS_MIN = pd.Timestamp.min
PANDAS_TS_MAX = pd.Timestamp.max


def find_timestamp_columns(table_name, conn):
    """Return list of timestamp/datetime columns in a DuckDB table."""
    info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
    candidates = []
    for _, row in info.iterrows():
        col_type = row['type'].lower()
        if "timestamp" in col_type or "date" in col_type or row['name'] in TIMESTAMP_COLUMNS:
            candidates.append(row['name'])
    return candidates


def remove_out_of_bounds(df, timestamp_cols):
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in timestamp_cols:
        if col in df.columns:
            vals = pd.to_datetime(df[col], errors='coerce')
            mask &= (vals >= PANDAS_TS_MIN) & (vals <= PANDAS_TS_MAX)
    cleaned = df[mask].copy()
    n_removed = n_before - len(cleaned)
    return cleaned, n_removed


def main():
    conn = duckdb.connect(DB_PATH, read_only=False)
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    logger.info(f"Tables in DB: {tables}")

    total_removed = 0
    for table in tables:
        timestamp_cols = find_timestamp_columns(table, conn)
        if not timestamp_cols:
            continue
        logger.info(f"Checking table {table} for bad timestamps in columns: {timestamp_cols}")
        df = conn.execute(f"SELECT * FROM {table}").fetchdf()
        cleaned, n_removed = remove_out_of_bounds(df, timestamp_cols)
        if n_removed > 0:
            logger.warning(f"Removed {n_removed} rows from {table} due to out-of-bounds timestamps.")
            # Overwrite table with cleaned data
            conn.execute(f"DELETE FROM {table}")
            conn.register("cleaned_df", cleaned)
            conn.execute(f"INSERT INTO {table} SELECT * FROM cleaned_df")
        total_removed += n_removed
    logger.success(f"Total rows removed across all tables: {total_removed}")
    conn.close()

if __name__ == "__main__":
    main()
