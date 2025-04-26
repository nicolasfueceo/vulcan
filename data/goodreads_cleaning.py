#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goodreads Database Column Cleaning Script
"""

import logging
import os
import sqlite3
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("database_cleaning.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
WORKDIR = os.getcwd()

def connect_to_db(db_path):
    """Connect to SQLite database with error handling."""
    try:
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            sys.exit(1)

        conn = sqlite3.connect(db_path)
        logger.info(f"Successfully connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)


def check_if_column_exists(conn, table, column):
    """Check if a specific column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def clean_database_basic(db_path=f"{WORKDIR}/goodreads.db"):
    """
    Clean database using a simple approach with explicit SQL statements.
    This approach is more robust against syntax errors.
    """
    logger.info(f"Attempting to clean database: {db_path}")
    conn = connect_to_db(db_path)

    try:
        # Check if tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found tables: {tables}")

        if "books" not in tables or "interactions" not in tables:
            logger.error("Required tables not found in database")
            conn.close()
            return

        # Check if the popular_shelves column exists in books
        has_popular_shelves = check_if_column_exists(conn, "books", "popular_shelves")
        if not has_popular_shelves:
            logger.warning("popular_shelves column not found in books table - skipping")

        # Get columns for books
        cursor.execute("PRAGMA table_info(books)")
        books_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Books table has {len(books_columns)} columns")

        # Get columns for interactions
        cursor.execute("PRAGMA table_info(interactions)")
        interactions_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Interactions table has {len(interactions_columns)} columns")

        # Find duplicate columns
        duplicate_columns = [
            col for col in interactions_columns if col in books_columns
        ]
        logger.info(
            f"Found {len(duplicate_columns)} duplicate columns: {duplicate_columns}"
        )

        # Filter out columns to keep
        books_columns_clean = [col for col in books_columns if col != "popular_shelves"]
        interactions_columns_clean = [
            col for col in interactions_columns if col not in duplicate_columns
        ]

        # Begin transaction
        conn.execute("BEGIN TRANSACTION")

        # Step 1: Create temporary tables
        logger.info("Step 1: Creating temporary tables...")

        # Create temporary books table
        if has_popular_shelves:
            # We need to exclude popular_shelves column
            column_list = ", ".join(f'"{col}"' for col in books_columns_clean)
            cursor.execute(f"CREATE TABLE books_new AS SELECT {column_list} FROM books")
            logger.info("Created books_new table without popular_shelves column")
        else:
            # Just copy the whole table
            cursor.execute("CREATE TABLE books_new AS SELECT * FROM books")
            logger.info("Created books_new table (no changes needed)")

        # Create temporary interactions table
        column_list = ", ".join(f'"{col}"' for col in interactions_columns_clean)
        cursor.execute(
            f"CREATE TABLE interactions_new AS SELECT {column_list} FROM interactions"
        )
        logger.info("Created interactions_new table without duplicate columns")

        # Step 2: Replace original tables
        logger.info("Step 2: Replacing original tables...")
        cursor.execute("DROP TABLE books")
        cursor.execute("ALTER TABLE books_new RENAME TO books")

        cursor.execute("DROP TABLE interactions")
        cursor.execute("ALTER TABLE interactions_new RENAME TO interactions")

        # Step 3: Recreate indexes
        logger.info("Step 3: Recreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_book_id ON books(book_id)")

        if "book_id" in interactions_columns_clean:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_book_id ON interactions(book_id)"
            )
        if "user_id" in interactions_columns_clean:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id)"
            )

        # Commit changes
        conn.commit()
        logger.info("Successfully cleaned database tables.")

    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info("Database cleaning complete!")


if __name__ == "__main__":
    try:
        db_path = f"{WORKDIR}/goodreads.db"  # Default path

        # Allow command-line override of database path
        if len(sys.argv) > 1:
            db_path = sys.argv[1]

        clean_database_basic(db_path)
        print("Database cleaning completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
