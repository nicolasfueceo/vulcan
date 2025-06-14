# /Users/nicolasdhnr/Documents/VULCAN/debug_user_counts.py
import duckdb
import pandas as pd
from pathlib import Path

print("Analyzing user rating distribution in goodreads_curated.duckdb...")

db_path = Path("data/goodreads_curated.duckdb")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit()

conn = None
try:
    conn = duckdb.connect(str(db_path))

    print("\n--- Top 10 users by rating count ---")
    top_users = conn.execute("""
        SELECT user_id, COUNT(*) as n_ratings
        FROM interactions
        GROUP BY user_id
        ORDER BY n_ratings DESC
        LIMIT 10;
    """).fetchdf()
    print(top_users)

    print("\n--- Distribution of ratings per user ---")
    distribution = conn.execute("""
        WITH user_counts AS (
            SELECT user_id, COUNT(*) as n_ratings
            FROM interactions
            GROUP BY user_id
        )
        SELECT n_ratings, COUNT(user_id) as num_users
        FROM user_counts
        GROUP BY n_ratings
        ORDER BY n_ratings;
    """).fetchdf()
    print(distribution)

    print("\n--- Summary Statistics ---")
    n_users_ge_5 = conn.execute("""
        WITH user_counts AS (
            SELECT user_id, COUNT(*) as n_ratings
            FROM interactions
            GROUP BY user_id
        )
        SELECT COUNT(user_id)
        FROM user_counts
        WHERE n_ratings >= 5;
    """).fetchone()[0]
    print(f"Number of users with >= 5 ratings: {n_users_ge_5}")

    total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM interactions").fetchone()[0]
    print(f"Total unique users: {total_users}")

    total_interactions = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    print(f"Total interactions: {total_interactions}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if conn:
        conn.close()
        print("\nAnalysis complete. Connection closed.")
