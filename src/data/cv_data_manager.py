"""
Cross-validation data manager for VULCAN.

Handles loading and managing cross-validation splits efficiently.
"""

import concurrent.futures
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

# Type aliases
UserID = Union[str, int]
UserIDList = List[UserID]
DataFrameDict = Dict[str, pd.DataFrame]


class ConnectionPool:
    """Thread-safe connection pool for managing DuckDB connections."""

    def __init__(self, db_path: str, max_connections: int = 10, **connection_kwargs):
        """Initialize the connection pool.

        Args:
            db_path: Path to the DuckDB database file
            max_connections: Maximum number of connections in the pool
            **connection_kwargs: Additional connection parameters
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_kwargs = connection_kwargs
        self._pool: queue.Queue[duckdb.DuckDBPyConnection] = queue.Queue(
            maxsize=max_connections
        )
        self._in_use: Set[duckdb.DuckDBPyConnection] = set()
        self._lock = threading.Lock()

        # Initialize connections
        for _ in range(max_connections):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new database connection with optimized settings."""
        try:
            conn = duckdb.connect(self.db_path, **self.connection_kwargs)

            # Apply performance optimizations
            config = {
                "threads": 1,
                "enable_progress_bar": False,
                "enable_object_cache": True,
                "preserve_insertion_order": False,
                "default_null_order": "nulls_first",
                "enable_external_access": False,
            }

            for param, value in config.items():
                try:
                    conn.execute(f"SET {param} = {repr(value)}")
                except Exception as e:
                    logger.warning(f"Could not set {param}={value}: {e}")

            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    def get_connection(self, timeout: float = 10.0) -> duckdb.DuckDBPyConnection:
        """Get a connection from the pool with a timeout.

        Args:
            timeout: Maximum time to wait for a connection (seconds)

        Returns:
            An active database connection

        Raises:
            queue.Empty: If no connection is available within the timeout
        """
        try:
            conn = self._pool.get(timeout=timeout)
            with self._lock:
                self._in_use.add(conn)
            return conn
        except queue.Empty:
            raise RuntimeError(
                f"No database connections available after {timeout} seconds. "
                f"Consider increasing max_connections (current: {self.max_connections})."
            )

    def return_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool.

        Args:
            conn: The connection to return
        """
        if conn is None:
            return

        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                try:
                    # Rollback any open transaction before returning to the pool
                    try:
                        conn.rollback()
                    except duckdb.Error as e:
                        # It's okay if there's no transaction to roll back.
                        if 'no transaction is active' not in str(e):
                            logger.warning(f"Error during rollback: {e}")
                    self._pool.put_nowait(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except duckdb.Error as close_err:
                        logger.warning(f"Error closing connection: {close_err}")
                    # Replace the bad connection with a new one
                    try:
                        new_conn = self._create_connection()
                        self._pool.put_nowait(new_conn)
                    except duckdb.Error as e:
                        logger.error(f"Failed to create replacement connection: {e}")

    def close_all(self) -> None:
        """Close all connections in the pool."""
        # Close all available connections
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

        # Also close any connections that were in use
        for conn in list(self._in_use):
            try:
                conn.close()
            except duckdb.Error as e:
                logger.warning(f"Error closing database connection: {e}")

        self._in_use.clear()


class CVDataManager:
    """Manages cross-validation data splits for the VULCAN pipeline."""

    # Class-level connection pool
    _connection_pool: Optional[ConnectionPool] = None
    _pool_lock = threading.Lock()
    _instance_count: int = 0

    def __init__(
        self,
        db_path: Union[str, Path],
        splits_dir: Union[str, Path],
        random_state: int = 42,
        cache_size_mb: int = 1024,
        max_connections: int = 10,
        read_only: bool = False,
    ):
        """Initialize the CV data manager with caching and connection pooling.

        Args:
            db_path: Path to the DuckDB database file
            splits_dir: Directory containing the cross-validation splits
            random_state: Random seed for reproducibility
            cache_size_mb: Size of DuckDB's memory cache in MB
            max_connections: Maximum number of database connections in the pool
            read_only: Whether the database should be opened in read-only mode
        """
        self.db_path = Path(db_path)
        self.splits_dir = Path(splits_dir)
        self.random_state = random_state
        self._cached_folds: Optional[List[Dict]] = None
        self._cv_folds = None
        self._cache_size_mb = cache_size_mb
        self.read_only = read_only

        # Cache for loaded data
        self._data_cache: Dict[str, Any] = {}

        # Initialize the connection pool if it doesn't exist
        with CVDataManager._pool_lock:
            CVDataManager._instance_count += 1
            if CVDataManager._connection_pool is None:
                self._initialize_connection_pool(max_connections=max_connections)

    def _initialize_connection_pool(self, max_connections: int) -> None:
        """Initialize the connection pool with the specified number of connections."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {self.db_path}. "
                "Please ensure the data is downloaded and processed."
            )

        try:
            # Determine access mode
            self.read_only = not os.access(self.db_path.parent, os.W_OK)
            connection_kwargs = {
                "read_only": self.read_only,
                "config": {"memory_limit": f"{self._cache_size_mb}MB"},
            }

            db_path_str = str(self.db_path)

            logger.info(
                f"Initializing connection pool for db='{db_path_str}' with "
                f"{max_connections} connections (read_only={self.read_only})"
            )

            CVDataManager._connection_pool = ConnectionPool(
                db_path=db_path_str,
                max_connections=max_connections,
                **connection_kwargs,
            )

            # Create indexes if the database is writeable
            if not self.read_only:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """CREATE INDEX IF NOT EXISTS idx_interactions_user_id 
                               ON interactions(user_id)"""
                        )
                        cur.execute(
                            """CREATE INDEX IF NOT EXISTS idx_interactions_item_id 
                               ON interactions(item_id)"""
                        )
                    logger.info("Successfully created indexes on user_id and item_id.")
                except Exception as e:
                    logger.warning(f"Could not create indexes: {e}")
                finally:
                    self._return_connection(conn)

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a database connection from the pool."""
        if CVDataManager._connection_pool is None:
            raise RuntimeError("Connection pool not initialized")
        return CVDataManager._connection_pool.get_connection()

    def _return_connection(self, conn: Optional[duckdb.DuckDBPyConnection]) -> None:
        """Return a connection to the pool."""
        if conn is not None and CVDataManager._connection_pool is not None:
            CVDataManager._connection_pool.return_connection(conn)

    @property
    def db_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a connection from the connection pool.

        Note: The caller is responsible for returning the connection to the pool
        using _return_connection() when done.

        Returns:
            An active DuckDB connection from the pool

        Raises:
            RuntimeError: If the connection pool is not initialized
        """
        return self._get_connection()

    def _clear_previous_fold_data(self) -> None:
        """Clear any cached fold data from memory."""
        self._data_cache.clear()

        # Run garbage collection to free up memory
        import gc

        gc.collect()

    def close(self) -> None:
        """Decrement the instance counter and clean up resources."""
        with CVDataManager._pool_lock:
            if CVDataManager._instance_count > 0:
                CVDataManager._instance_count -= 1

                # Close the connection pool if this is the last instance
                if (
                    CVDataManager._instance_count <= 0
                    and CVDataManager._connection_pool is not None
                ):
                    try:
                        # Clear any cached data
                        self._clear_previous_fold_data()

                        # Close all connections in the pool
                        CVDataManager._connection_pool.close_all()
                        CVDataManager._connection_pool = None
                        logger.info(
                            "Closed all database connections and cleared cached data"
                        )
                    except Exception as e:
                        logger.warning(f"Error during cleanup: {e}")

    def __del__(self) -> None:
        """Ensure proper cleanup when the object is destroyed.""" 
        try:
            self.close()
        except Exception:
            # Suppress errors during garbage collection
            pass

    @classmethod
    def close_global_connection_pool(cls) -> None:
        """Close the global connection pool if it exists."""
        with cls._pool_lock:
            if cls._connection_pool:
                logger.info("Closing global connection pool.")
                cls._connection_pool.close_all()
                cls._connection_pool = None
                logger.debug("Global connection pool closed.")



    def load_cv_folds(self) -> List[Dict[str, List[str]]]:
        """Load the cross-validation folds.

        Returns:
            List of dictionaries with 'train', 'validation', and 'test' keys
        """
        if self._cv_folds is not None:
            return self._cv_folds

        folds_file = self.splits_dir / "cv_folds.json"
        if not folds_file.exists():
            logger.error(f"CV folds file not found at {folds_file}")
            raise FileNotFoundError(f"CV folds file not found at {folds_file}")

        try:
            with open(folds_file, "r", encoding="utf-8") as f:
                self._cv_folds = json.load(f)
            return self._cv_folds
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing CV folds file: {e}")
            raise

    def get_fold_data(
        self,
        fold_idx: int,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        batch_size: int = 500,
        show_progress: bool = True,
        max_workers: int = 4,
        split_type: str = "train_val",
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        # Clear any previous data first
        self._clear_previous_fold_data()

        # Get the fold data
        folds = self.load_cv_folds()
        if fold_idx >= len(folds):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(folds) - 1})")

        fold = folds[fold_idx]

        # Get user lists for each split
        train_users = fold["train"]
        val_users = fold["validation"]
        test_users = fold.get("test", [])

        # --- Stratified sampling by user activity ---
        def stratified_sample_users(users: List[str], frac: float, user_activity: Dict[str, int]) -> List[str]:
            """
            Stratified sampling of users by activity level.
            Args:
                users: List of user IDs to sample from.
                frac: Fraction to sample.
                user_activity: Dict mapping user_id to number of interactions.
            Returns:
                List of sampled users preserving activity distribution.
            """
            if frac is None or frac >= 1.0 or not users:
                return users
            activity_counts = np.array([user_activity.get(u, 0) for u in users])
            if len(set(activity_counts)) <= 1:
                rng = np.random.default_rng(random_state)
                sample_size = max(1, int(len(users) * frac))
                return rng.choice(users, size=sample_size, replace=False).tolist()
            bins = np.quantile(activity_counts, np.linspace(0, 1, 6))
            bins[0] = min(activity_counts) - 1
            sampled_users = []
            rng = np.random.default_rng(random_state)
            for i in range(5):
                in_bin = [u for u, c in zip(users, activity_counts) if bins[i] < c <= bins[i+1]]
                n_bin = max(1, int(len(in_bin) * frac)) if in_bin else 0
                if in_bin and n_bin > 0:
                    sampled_users.extend(rng.choice(in_bin, size=n_bin, replace=False).tolist())
            return sampled_users if sampled_users else users

        # Only compute user_activity if needed
        user_activity = {}
        if sample_frac is not None and sample_frac < 1.0:
            conn = self._get_connection()
            try:
                query = "SELECT user_id, COUNT(*) as n FROM interactions GROUP BY user_id"
                df_activity = conn.execute(query).fetchdf()
                user_activity = dict(zip(df_activity['user_id'], df_activity['n']))
            finally:
                self._return_connection(conn)

            train_users = stratified_sample_users(train_users, sample_frac, user_activity)
            val_users = stratified_sample_users(val_users, sample_frac, user_activity)
            test_users = stratified_sample_users(test_users, sample_frac, user_activity)
        # Get column list for query
        if columns:
            column_list = ", ".join([f"r.{c}" for c in columns])
        else:
            column_list = "r.*"

        def process_chunk(
            chunk: List[str], chunk_idx: int, purpose: str
        ) -> Optional[pd.DataFrame]:
            """Process a single chunk of user data."""
            if not chunk:
                return None

            temp_table = f"temp_users_{abs(hash(str(chunk[:5]))) % 10000}_{chunk_idx}"
            conn = None

            try:
                # Get a connection from the pool
                conn = self._get_connection()

                with conn.cursor() as cur:
                    # Create and populate temp table
                    cur.execute(
                        f"""
                        CREATE TEMP TABLE {temp_table} AS 
                        SELECT UNNEST(?) AS user_id
                    """,
                        [chunk],
                    )

                    # Execute main query
                    query = f"""
                        SELECT {column_list}
                        FROM interactions r
                        JOIN {temp_table} t ON r.user_id = t.user_id
                    """

                    df = cur.execute(query).fetchdf()

                    # Add purpose column for filtering later
                    if not df.empty:
                        df["_purpose"] = purpose

                    return df

            except Exception as e:
                logger.error(f"Error processing {purpose} chunk {chunk_idx}: {e}")
                return None

            finally:
                # Clean up temp table and return connection to pool
                if conn is not None:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    except Exception as e:
                        logger.warning(f"Error dropping temp table {temp_table}: {e}")
                    self._return_connection(conn)

        def process_user_list(users: List[str], purpose: str) -> pd.DataFrame:
            """Process a list of users in batches."""
            if not users:
                return pd.DataFrame()

            # Split into batches
            batches = [
                users[i : i + batch_size] for i in range(0, len(users), batch_size)
            ]

            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_chunk, batch, i, purpose)
                    for i, batch in enumerate(batches)
                ]

                # Collect results
                results = []
                for future in (
                    tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Loading {purpose} data",
                        disable=not show_progress,
                    )
                    if show_progress
                    else concurrent.futures.as_completed(futures)
                ):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        # Process each split
        train_df = (
            process_user_list(train_users, "train") if train_users else pd.DataFrame()
        )
        val_df = (
            process_user_list(val_users, "validation") if val_users else pd.DataFrame()
        )
        test_df = (
            process_user_list(test_users, "test") if test_users else pd.DataFrame()
        )

        logger.critical(f"TRAIN DF COLUMNS before return: {train_df.columns}")
        logger.critical(f"TRAIN DF HEAD before return:\n{train_df.head()}")

        # Return based on split_type
        if split_type == "train_val":
            return train_df, val_df
        elif split_type == "train_test":
            return train_df, test_df
        elif split_type == "val_test":
            return val_df, test_df
        elif split_type == "all":
            return train_df, val_df, test_df
        elif split_type == "full_train":
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            return train_val_df, test_df
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

    def iter_folds(
        self,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        split_type: str = "train_val",
    ) -> Generator[
        Union[
            Tuple[pd.DataFrame, pd.DataFrame],
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ],
        None,
        None,
    ]:
        """Iterate over all CV folds, loading data for each.

        Args:
            columns: List of columns to load (None for all).
            sample_frac: Fraction of users to sample.
            random_state: Seed for reproducibility.
            split_type: Type of data split to return.

        Yields:
            Data for each fold according to the specified split_type.
        """
        n_folds = self.get_fold_summary().get("n_folds", 0)
        if n_folds == 0:
            logger.warning("No CV folds found. Returning empty iterator.")
            return

        for i in range(n_folds):
            yield self.get_fold_data(
                fold_idx=i,
                columns=columns,
                sample_frac=sample_frac,
                random_state=random_state,
                split_type=split_type,
            )

    def get_all_folds_data(
        self,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
        split_type: str = "train_val",
    ) -> List[
        Union[
            Tuple[pd.DataFrame, pd.DataFrame],
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ]
    ]:
        """Get data for all CV folds.

        Args:
            columns: List of columns to load (None for all)
            sample_frac: If provided, sample this fraction of users
            random_state: Random seed for reproducibility
            split_type: The type of data split to retrieve.

        Returns:
            A list containing the data for all folds.
        """
        return list(
            self.iter_folds(
                columns=columns,
                sample_frac=sample_frac,
                random_state=random_state,
                split_type=split_type,
            )
        )

    def get_fold_summary(self) -> Dict[str, Any]:
        """Get a summary of the CV folds.

        Returns:
            Dictionary with fold statistics including:
            - n_folds: Number of folds
            - n_users: Total number of unique users
            - n_items: Total number of unique items
            - n_interactions: Total number of interactions
            - folds: List of fold statistics
        """
        summary_file = self.splits_dir / "cv_summary.json"



        if not summary_file.exists():
            return {
                "status": "error",
                "message": "CV summary file not found. Please generate CV splits first.",
                "n_folds": 0,
                "n_users": 0,
                "n_items": 0,
                "n_interactions": 0,
                "folds": [],
            }

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Ensure required fields exist
            if "folds" not in summary:
                summary["folds"] = []
            if "n_folds" not in summary:
                summary["n_folds"] = len(summary["folds"])

            return summary

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing CV summary file: {e}")
            return {
                "status": "error",

                "message": f"Invalid CV summary file: {e}",
                "n_folds": 0,
                "folds": [],
            }
