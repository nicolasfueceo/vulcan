"""Data access layer for VULCAN project."""

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseDAL(ABC):
    """Abstract Base Class for Data Access Layers."""

    @abstractmethod
    def connect(self):
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection to the data source."""
        pass

    @abstractmethod
    def get_data_for_evaluation(self, split_name: str) -> pd.DataFrame:
        """Fetches the pre-defined training or validation data splits.

        Args:
            split_name: Name of the split (e.g., 'train', 'validate').

        Returns:
            A pandas DataFrame containing the data for the specified split.
        """
        pass

    @abstractmethod
    def get_horizontal_data(
        self,
        entity_id: Any,
        entity_type: str,
        columns: List[str],
        current_instance_id: Optional[Any] = None,
        id_column: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetches additional data for a specific entity (horizontal access).

        Args:
            entity_id: The ID of the entity (e.g., user_id).
            entity_type: The type of the entity (e.g., 'user', 'item').
            columns: List of column names to fetch.
            current_instance_id: (Optional) ID of the current instance being processed, to potentially exclude it.
            id_column: (Optional) The name of the column holding the instance ID (e.g., 'review_id').

        Returns:
            A pandas DataFrame containing the horizontal data, or None if not found/applicable.
        """
        pass

    @abstractmethod
    def get_vertical_data(
        self,
        context_id: Any,
        entity_type: str,
        columns: List[str],
        current_instance_id: Optional[Any] = None,
        id_column: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetches data for other entities within a specific context (vertical access).

        Args:
            context_id: The ID of the context (e.g., book_id).
            entity_type: The type of the entity within the context (e.g., 'review', 'user').
            columns: List of column names to fetch.
            current_instance_id: (Optional) ID of the current instance being processed, to potentially exclude it.
            id_column: (Optional) The name of the column holding the instance ID (e.g., 'review_id').

        Returns:
            A pandas DataFrame containing the vertical data, or None if not found/applicable.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, List[str]]:
        """Returns the database schema (tables and their columns).

        Returns:
            A dictionary mapping table names to lists of column names.
        """
        pass


class SqlDAL(BaseDAL):
    """Data Access Layer implementation for SQL databases (using SQLite)."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the SqlDAL.

        Args:
            config: Configuration dictionary containing database paths and settings.
        """
        self.config = config
        data_config = config.get("data_source", {})

        # Database paths
        self.db_path = data_config.get("db_path", "data/goodreads.db")

        # Table names
        self.reviews_table = data_config.get("reviews_table", "reviews")
        self.books_table = data_config.get("books_table", "books")
        self.users_table = data_config.get("users_table", "users")

        # Split configuration
        self.split_config = data_config.get("splits", {})
        self.splits_dir = self.split_config.get("directory", "data/splits")

        self.conn = None
        self.split_ids = self._load_split_ids()
        self._schema = None

    def _load_split_ids(self) -> Dict[str, List[Any]]:
        """Loads entity IDs for each data split from specified files."""
        split_ids = {}
        split_files = self.split_config.get("files", {})

        for split_name, file_path in split_files.items():
            full_path = os.path.join(self.splits_dir, file_path)
            if not os.path.exists(full_path):
                print(
                    f"Warning: Split ID file not found: {full_path}. Split '{split_name}' will be empty."
                )
                split_ids[split_name] = []
                continue

            try:
                ids_df = pd.read_csv(full_path)
                id_column = self.split_config.get("id_column", "user_id")
                if id_column not in ids_df.columns:
                    raise ValueError(
                        f"Split file {full_path} must contain a '{id_column}' column."
                    )
                split_ids[split_name] = ids_df[id_column].tolist()
                print(
                    f"Loaded {len(split_ids[split_name])} IDs for split '{split_name}'"
                )
            except Exception as e:
                print(f"Warning: Failed to load split IDs from {full_path}: {e}")
                split_ids[split_name] = []

        return split_ids

    def connect(self):
        """Establish connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database {self.db_path}: {e}")
            self.conn = None
            raise

    def disconnect(self):
        """Close connection to the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")

    def _check_connection(self):
        if not self.conn:
            raise ConnectionError(
                "Database connection is not established. Call connect() first."
            )

    def get_schema(self) -> Dict[str, List[str]]:
        """Returns the database schema (tables and their columns)."""
        if self._schema is not None:
            return self._schema

        self._check_connection()

        schema = {}
        cursor = self.conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Get columns for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in cursor.fetchall()]
            schema[table] = columns

        self._schema = schema
        return schema

    def get_data_for_evaluation(self, split_name: str) -> pd.DataFrame:
        """Fetches data for the specified split using pre-loaded IDs."""
        self._check_connection()

        if split_name not in self.split_ids:
            raise ValueError(f"Split name '{split_name}' not found in configuration.")

        ids = self.split_ids[split_name]
        if not ids:
            print(
                f"Warning: No IDs found for split '{split_name}'. Returning empty DataFrame."
            )
            return pd.DataFrame()

        # Get the entity type and ID column from config
        entity_type = self.split_config.get("entity_type", "user")
        id_col_name = self.split_config.get("id_column", "user_id")

        table_name = self.reviews_table
        placeholders = ",".join("?" * len(ids))
        query = f"SELECT * FROM {table_name} WHERE {id_col_name} IN ({placeholders})"

        try:
            df = pd.read_sql_query(query, self.conn, params=ids)
            return df
        except Exception as e:
            print(f"Error executing query for split '{split_name}': {e}")
            return pd.DataFrame()

    def get_horizontal_data(
        self,
        entity_id: Any,
        entity_type: str,
        columns: List[str],
        current_instance_id: Optional[Any] = None,
        id_column: Optional[str] = "review_id",
    ) -> Optional[pd.DataFrame]:
        """Fetches other reviews by the same user."""
        self._check_connection()

        if entity_type != "user":
            print(
                f"Warning: Horizontal data access currently only supports entity_type='user'. Got '{entity_type}'."
            )
            return None

        # Assuming the user ID column is 'user_id'
        user_id_col = "user_id"
        cols_str = ", ".join(columns)
        if not cols_str:
            print("Warning: No columns requested for horizontal data access.")
            return pd.DataFrame(columns=columns)

        query = f"SELECT {cols_str} FROM {self.reviews_table} WHERE {user_id_col} = ?"
        params = [entity_id]

        if current_instance_id is not None and id_column is not None:
            query += f" AND {id_column} != ?"
            params.append(current_instance_id)

        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            print(f"Error fetching horizontal data for user {entity_id}: {e}")
            return None

    def get_vertical_data(
        self,
        context_id: Any,
        entity_type: str,
        columns: List[str],
        current_instance_id: Optional[Any] = None,
        id_column: Optional[str] = "review_id",
    ) -> Optional[pd.DataFrame]:
        """Fetches other reviews for the same book."""
        self._check_connection()

        if entity_type != "book":
            print(
                f"Warning: Vertical data access currently only supports entity_type='book' context. Got '{entity_type}'."
            )
            return None

        # Assuming the book ID column is 'book_id'
        book_id_col = "book_id"
        cols_str = ", ".join(columns)
        if not cols_str:
            print("Warning: No columns requested for vertical data access.")
            return pd.DataFrame(columns=columns)

        query = f"SELECT {cols_str} FROM {self.reviews_table} WHERE {book_id_col} = ?"
        params = [context_id]

        if current_instance_id is not None and id_column is not None:
            query += f" AND {id_column} != ?"
            params.append(current_instance_id)

        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            print(f"Error fetching vertical data for book {context_id}: {e}")
            return None


def get_dal(config: Dict[str, Any]) -> BaseDAL:
    """Factory function to create a Data Access Layer instance based on config."""
    dal_type = config.get("data_source", {}).get("type", "sql")

    if dal_type == "sql":
        return SqlDAL(config)
    else:
        raise ValueError(f"Unsupported DAL type: {dal_type}")
