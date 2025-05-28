"""Utility for merging multiple SQLite databases into one."""

import os
import sqlite3
from typing import List

import pandas as pd


class DatabaseMerger:
    """Utility for merging SQLite databases."""

    def __init__(self, source_dbs: List[str], target_db: str):
        """
        Initialize the database merger.

        Args:
            source_dbs: List of paths to source SQLite database files.
            target_db: Path where the merged database will be saved.
        """
        self.source_dbs = source_dbs
        self.target_db = target_db

    def merge_databases(self, preserve_source: bool = True) -> bool:
        """
        Merge multiple SQLite databases into a single database.

        Args:
            preserve_source: If True, keeps source databases intact. If False, removes them after merging.

        Returns:
            True if the merge was successful, False otherwise.
        """
        print(f"Merging {len(self.source_dbs)} databases into {self.target_db}")

        # Check if source databases exist
        for db in self.source_dbs:
            if not os.path.exists(db):
                print(f"Source database {db} does not exist. Aborting merge.")
                return False

        # Create target directory if it doesn't exist
        target_dir = os.path.dirname(self.target_db)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Create or connect to target database
        try:
            target_conn = sqlite3.connect(self.target_db)
            target_cursor = target_conn.cursor()
            print(f"Created/connected to target database: {self.target_db}")
        except sqlite3.Error as e:
            print(f"Error connecting to target database {self.target_db}: {e}")
            return False

        # Track tables that have been created in the target
        created_tables = set()

        # Process each source database
        for source_db in self.source_dbs:
            try:
                print(f"Processing source database: {source_db}")

                # Connect to source database
                source_conn = sqlite3.connect(source_db)
                source_cursor = source_conn.cursor()

                # Get list of tables in source database (excluding system tables)
                source_cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                tables = [row[0] for row in source_cursor.fetchall()]

                print(f"Found {len(tables)} tables in {source_db}")

                # Process each table
                for table in tables:
                    print(f"Processing table: {table}")

                    # Get table schema from source
                    source_cursor.execute(f"PRAGMA table_info({table})")
                    columns_info = source_cursor.fetchall()

                    # Create table in target if it doesn't exist
                    if table not in created_tables:
                        # Build CREATE TABLE statement
                        column_defs = []
                        for col in columns_info:
                            column_defs.append(f"{col[1]} {col[2]}")

                        create_query = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(column_defs)})"
                        target_cursor.execute(create_query)
                        created_tables.add(table)
                        print(f"Created table {table} in target database")

                    # Read data from source table
                    source_cursor.execute(f"SELECT * FROM {table}")
                    rows = source_cursor.fetchall()

                    if not rows:
                        print(f"No data in table {table}, skipping")
                        continue

                    # Get column names
                    column_names = [col[1] for col in columns_info]
                    placeholders = ",".join(["?" for _ in column_names])

                    # Insert data into target table
                    insert_query = f"INSERT INTO {table} ({','.join(column_names)}) VALUES ({placeholders})"
                    target_cursor.executemany(insert_query, rows)
                    print(f"Inserted {len(rows)} rows into {table}")

                # Close source connection
                source_conn.commit()
                source_conn.close()

                # Delete source database if requested
                if not preserve_source:
                    print(f"Removing source database: {source_db}")
                    os.remove(source_db)

            except sqlite3.Error as e:
                print(f"Error processing database {source_db}: {e}")
                target_conn.close()
                return False

        # Commit changes to target database
        target_conn.commit()
        target_conn.close()
        print(f"Successfully merged databases into {self.target_db}")
        return True

    def create_split_index(
        self,
        split_name: str,
        db_origin: str,
        id_column: str = "user_id",
        splits_dir: str = "data/splits",
    ) -> bool:
        """
        Create CSV index files containing IDs from each source database.

        Args:
            split_name: Name of the split to create
            db_origin: Path to the source database this split came from
            id_column: Column containing IDs to extract
            splits_dir: Directory to save split files to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create splits directory if it doesn't exist
            if not os.path.exists(splits_dir):
                os.makedirs(splits_dir)

            # Connect to the source database
            conn = sqlite3.connect(db_origin)

            # Get all unique IDs from the specified column in the reviews table
            query = f"SELECT DISTINCT {id_column} FROM reviews"
            df = pd.read_sql_query(query, conn)

            # Save to CSV
            output_path = os.path.join(splits_dir, f"{split_name}.csv")
            df.to_csv(output_path, index=False)

            print(
                f"Created split index for {split_name} with {len(df)} IDs, saved to {output_path}"
            )
            conn.close()
            return True

        except Exception as e:
            print(f"Error creating split index for {split_name}: {e}")
            return False

    def create_all_split_indices(
        self, id_column: str = "user_id", splits_dir: str = "data/splits"
    ) -> bool:
        """
        Create split indices for all source databases.

        Args:
            id_column: Column containing IDs to extract
            splits_dir: Directory to save split files to

        Returns:
            True if all splits were created successfully, False otherwise
        """
        # Map database files to split names
        db_to_split = {}
        for db in self.source_dbs:
            # Extract split name from filename (e.g., train.db -> train)
            split_name = os.path.splitext(os.path.basename(db))[0]
            db_to_split[db] = split_name

        # Create split indices
        success = True
        for db, split_name in db_to_split.items():
            if not self.create_split_index(split_name, db, id_column, splits_dir):
                success = False

        return success
