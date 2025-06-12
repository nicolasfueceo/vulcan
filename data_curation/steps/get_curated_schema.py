#!/usr/bin/env python3
"""
Connects to the curated DuckDB database and prints its schema.
"""

import duckdb
import pandas as pd

# Configure pandas for full display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DB_PATH = 'data/goodreads_curated.duckdb'

print(f"--- Schema for {DB_PATH} ---")

try:
    with duckdb.connect(database=DB_PATH, read_only=True) as con:
        # Get all tables and views
        tables = con.sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;").df()
        
        for table_name in tables['table_name']:
            print(f"\n[+] Table: {table_name}")
            schema_df = con.sql(f"DESCRIBE {table_name};").df()
            print(schema_df)
            
except Exception as e:
    print(f"\n❌ An error occurred: {e}")

print("\n--- Schema extraction complete ---") 