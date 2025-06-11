import os

import duckdb
import pandas as pd

# Configure pandas for better display
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

RAW_DB_PATH = "data/goodreads_raw.duckdb"

print(f"--- Inspecting Raw Date Formats in {RAW_DB_PATH} ---")

if not os.path.exists(RAW_DB_PATH):
    print(f"❌ Error: Raw database file not found at '{RAW_DB_PATH}'")
    exit()

try:
    # Connect in read-only mode to avoid locking issues
    with duckdb.connect(database=RAW_DB_PATH, read_only=True) as con:
        # First, attach the database with an alias to query it correctly
        con.execute(f"ATTACH '{RAW_DB_PATH}' AS raw_db;")

        print("\n[+] Sampling date columns from raw_db.books...")
        books_dates_df = con.sql("""
            SELECT
                publication_year,
                publication_month,
                publication_day
            FROM raw_db.books
            WHERE publication_year IS NOT NULL AND publication_year != ''
            LIMIT 10;
        """).df()
        print(books_dates_df)

        print("\n[+] Sampling date columns from raw_db.reviews...")
        reviews_dates_df = con.sql("""
            SELECT
                date_added,
                date_updated,
                read_at,
                started_at
            FROM raw_db.reviews
            WHERE date_added IS NOT NULL AND date_added != ''
            LIMIT 10;
        """).df()
        print(reviews_dates_df)

except Exception as e:
    print(f"\n❌ An error occurred: {e}")

print("\n--- Inspection Complete ---")
