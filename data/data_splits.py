"""
Data splitting module for VULCAN project.

This module handles creating the data splits for the two-phase recommender system:
1. Phase 1: MCTS-Driven LLM Feature Engineering & Clustering
2. Phase 2: Conversational Cold-Start Assignment

The data splits follow this structure:
- Outer Hold-Out (20% users): Final "warm-user" test in Phase 2
- Design Set (80% users): Phase 1 feature engineering & clustering
  - K-Fold Outer CV (K=5) on Design Set:
    - TrainFE (K-1 folds): Feature search
    - ValClusters (1 fold): Cluster validation
  - Inside TrainFE: M-fold inner split (M=3) for feature evaluation:
    - FeatTrain: Train models for feature
    - FeatVal: Validate feature (RMSE)
"""

import logging
import os
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths to database files
TRAIN_DB = "data/train.db"
TEST_DB = "data/test.db"
VALIDATION_DB = "data/validation.db"


class DataSplitter:
    """Class for handling all data splitting operations for the recommender system."""

    def __init__(
        self,
        database_path: str = TRAIN_DB,
        test_size: float = 0.2,
        outer_folds: int = 5,
        inner_folds: int = 3,
        random_state: int = 42,
    ):
        """
        Initialize the data splitter.

        Args:
            database_path: Path to the SQLite database
            test_size: Proportion of users to hold out for final testing
            outer_folds: Number of folds for outer cross-validation
            inner_folds: Number of folds for inner cross-validation
            random_state: Random seed for reproducibility
        """
        self.db_path = database_path
        self.test_size = test_size
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state

        # Initialize empty sets
        self.all_user_ids = None
        self.outer_test_users = None
        self.design_users = None
        self.outer_cv_folds = None
        self.inner_cv_folds = None

    def connect_to_db(self) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def get_all_users(self) -> List[str]:
        """Get all unique user IDs from the database."""
        if self.all_user_ids is not None:
            return self.all_user_ids

        conn = self.connect_to_db()
        query = "SELECT DISTINCT user_id FROM reviews"
        df = pd.read_sql_query(query, conn)
        conn.close()

        self.all_user_ids = df["user_id"].tolist()
        logger.info(f"Found {len(self.all_user_ids)} unique users in the database")
        return self.all_user_ids

    def create_outer_split(self) -> Tuple[List[str], List[str]]:
        """
        Create the initial outer split between test users and design users.

        Returns:
            Tuple of (design_users, outer_test_users)
        """
        all_users = self.get_all_users()

        # Create the outer test/design split
        design_users, outer_test_users = train_test_split(
            all_users, test_size=self.test_size, random_state=self.random_state
        )

        self.design_users = design_users
        self.outer_test_users = outer_test_users

        logger.info(
            f"Created outer split: {len(design_users)} design users, {len(outer_test_users)} test users"
        )
        return design_users, outer_test_users

    def create_outer_cv_folds(self) -> List[Tuple[List[str], List[str]]]:
        """
        Create K-fold cross-validation splits for the design users.

        Returns:
            List of (TrainFE, ValClusters) user ID tuples
        """
        if self.design_users is None:
            self.create_outer_split()

        # Create K-fold splits on the design users
        kf = KFold(
            n_splits=self.outer_folds, shuffle=True, random_state=self.random_state
        )

        # Convert to numpy array for splitting
        design_users_array = np.array(self.design_users)

        # Generate the outer CV folds
        outer_cv_folds = []
        for train_idx, val_idx in kf.split(design_users_array):
            train_users = design_users_array[train_idx].tolist()  # TrainFE users
            val_users = design_users_array[val_idx].tolist()  # ValClusters users
            outer_cv_folds.append((train_users, val_users))

        self.outer_cv_folds = outer_cv_folds

        # Log fold sizes
        fold_info = "\n".join(
            [
                f"Fold {i + 1}: {len(train)} TrainFE users, {len(val)} ValClusters users"
                for i, (train, val) in enumerate(outer_cv_folds)
            ]
        )
        logger.info(f"Created {self.outer_folds} outer CV folds:\n{fold_info}")

        return outer_cv_folds

    def create_inner_cv_folds(
        self, fold_idx: int = 0
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Create M-fold inner cross-validation splits for feature evaluation.

        Args:
            fold_idx: Index of the outer fold to use for inner CV

        Returns:
            List of (FeatTrain, FeatVal) user ID tuples
        """
        if self.outer_cv_folds is None:
            self.create_outer_cv_folds()

        # Get the TrainFE users from the specified outer fold
        train_fe_users = self.outer_cv_folds[fold_idx][0]

        # Create M-fold splits on the TrainFE users
        kf = KFold(
            n_splits=self.inner_folds, shuffle=True, random_state=self.random_state
        )

        # Convert to numpy array for splitting
        train_fe_array = np.array(train_fe_users)

        # Generate the inner CV folds
        inner_cv_folds = []
        for inner_train_idx, inner_val_idx in kf.split(train_fe_array):
            feat_train_users = train_fe_array[
                inner_train_idx
            ].tolist()  # FeatTrain users
            feat_val_users = train_fe_array[inner_val_idx].tolist()  # FeatVal users
            inner_cv_folds.append((feat_train_users, feat_val_users))

        self.inner_cv_folds = inner_cv_folds

        # Log fold sizes
        fold_info = "\n".join(
            [
                f"Inner Fold {i + 1}: {len(train)} FeatTrain users, {len(val)} FeatVal users"
                for i, (train, val) in enumerate(inner_cv_folds)
            ]
        )
        logger.info(
            f"Created {self.inner_folds} inner CV folds for outer fold {fold_idx}:\n{fold_info}"
        )

        return inner_cv_folds

    def save_user_splits(self, output_dir: str = "data/splits"):
        """
        Save all user splits to CSV files.

        Args:
            output_dir: Directory to save the split files
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Make sure all splits have been created
        if self.outer_test_users is None or self.design_users is None:
            self.create_outer_split()

        if self.outer_cv_folds is None:
            self.create_outer_cv_folds()

        # Save the outer split
        pd.DataFrame({"user_id": self.outer_test_users}).to_csv(
            os.path.join(output_dir, "outer_test_users.csv"), index=False
        )
        pd.DataFrame({"user_id": self.design_users}).to_csv(
            os.path.join(output_dir, "design_users.csv"), index=False
        )

        # Save each outer CV fold
        for i, (train_users, val_users) in enumerate(self.outer_cv_folds):
            pd.DataFrame({"user_id": train_users}).to_csv(
                os.path.join(output_dir, f"outer_fold_{i + 1}_train_fe_users.csv"),
                index=False,
            )
            pd.DataFrame({"user_id": val_users}).to_csv(
                os.path.join(output_dir, f"outer_fold_{i + 1}_val_clusters_users.csv"),
                index=False,
            )

            # Create and save inner CV folds for this outer fold
            inner_folds = self.create_inner_cv_folds(fold_idx=i)
            for j, (feat_train_users, feat_val_users) in enumerate(inner_folds):
                pd.DataFrame({"user_id": feat_train_users}).to_csv(
                    os.path.join(
                        output_dir,
                        f"outer_fold_{i + 1}_inner_fold_{j + 1}_feat_train_users.csv",
                    ),
                    index=False,
                )
                pd.DataFrame({"user_id": feat_val_users}).to_csv(
                    os.path.join(
                        output_dir,
                        f"outer_fold_{i + 1}_inner_fold_{j + 1}_feat_val_users.csv",
                    ),
                    index=False,
                )

        logger.info(f"Saved all user splits to {output_dir}")

    def create_split_queries(self, output_dir: str = "data/queries"):
        """
        Generate SQL queries to extract data for each split.

        Args:
            output_dir: Directory to save the query files
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Make sure all splits have been created
        if self.outer_test_users is None or self.design_users is None:
            self.create_outer_split()

        if self.outer_cv_folds is None:
            self.create_outer_cv_folds()

        # Create and save queries for the outer split
        outer_test_query = self._create_user_filter_query(self.outer_test_users)
        design_query = self._create_user_filter_query(self.design_users)

        with open(os.path.join(output_dir, "outer_test_query.sql"), "w") as f:
            f.write(outer_test_query)

        with open(os.path.join(output_dir, "design_query.sql"), "w") as f:
            f.write(design_query)

        # Create and save queries for each outer CV fold
        for i, (train_users, val_users) in enumerate(self.outer_cv_folds):
            train_query = self._create_user_filter_query(train_users)
            val_query = self._create_user_filter_query(val_users)

            with open(
                os.path.join(output_dir, f"outer_fold_{i + 1}_train_fe_query.sql"), "w"
            ) as f:
                f.write(train_query)

            with open(
                os.path.join(output_dir, f"outer_fold_{i + 1}_val_clusters_query.sql"),
                "w",
            ) as f:
                f.write(val_query)

            # Create and save queries for inner CV folds
            inner_folds = self.create_inner_cv_folds(fold_idx=i)
            for j, (feat_train_users, feat_val_users) in enumerate(inner_folds):
                feat_train_query = self._create_user_filter_query(feat_train_users)
                feat_val_query = self._create_user_filter_query(feat_val_users)

                with open(
                    os.path.join(
                        output_dir,
                        f"outer_fold_{i + 1}_inner_fold_{j + 1}_feat_train_query.sql",
                    ),
                    "w",
                ) as f:
                    f.write(feat_train_query)

                with open(
                    os.path.join(
                        output_dir,
                        f"outer_fold_{i + 1}_inner_fold_{j + 1}_feat_val_query.sql",
                    ),
                    "w",
                ) as f:
                    f.write(feat_val_query)

        logger.info(f"Saved all split queries to {output_dir}")

    def _create_user_filter_query(self, user_ids: List[str]) -> str:
        """
        Create a SQL query to filter reviews by user IDs.

        Args:
            user_ids: List of user IDs to include

        Returns:
            SQL query string
        """
        # Format list for SQL IN clause - limit to chunks to avoid query size limits
        chunk_size = 1000
        user_chunks = [
            user_ids[i : i + chunk_size] for i in range(0, len(user_ids), chunk_size)
        ]

        queries = []
        for chunk in user_chunks:
            user_str = "', '".join(chunk)
            query = f"SELECT * FROM reviews WHERE user_id IN ('{user_str}')"
            queries.append(query)

        # Combine with UNION
        return " UNION ".join(queries)


def create_all_splits(
    database_path: str = TRAIN_DB,
    test_size: float = 0.2,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
) -> DataSplitter:
    """
    Create all data splits and save them to files.

    Args:
        database_path: Path to the SQLite database
        test_size: Proportion of users to hold out for final testing
        outer_folds: Number of folds for outer cross-validation
        inner_folds: Number of folds for inner cross-validation
        random_state: Random seed for reproducibility

    Returns:
        DataSplitter instance with all splits created
    """
    splitter = DataSplitter(
        database_path=database_path,
        test_size=test_size,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_state=random_state,
    )

    # Create all splits
    splitter.create_outer_split()
    splitter.create_outer_cv_folds()

    # Save splits to files
    splitter.save_user_splits()
    splitter.create_split_queries()

    return splitter


if __name__ == "__main__":
    # Create all splits with default parameters
    splitter = create_all_splits()

    # Print summary
    print("\nData Split Summary:")
    print(f"Total users: {len(splitter.all_user_ids)}")
    print(f"Outer test users (20%): {len(splitter.outer_test_users)}")
    print(f"Design users (80%): {len(splitter.design_users)}")

    # For the first outer fold
    train_fe_users = splitter.outer_cv_folds[0][0]
    val_clusters_users = splitter.outer_cv_folds[0][1]
    print("\nExample outer fold:")
    print(f"  TrainFE users (~64%): {len(train_fe_users)}")
    print(f"  ValClusters users (~16%): {len(val_clusters_users)}")

    # For the first inner fold of the first outer fold
    inner_folds = splitter.create_inner_cv_folds(0)
    feat_train_users = inner_folds[0][0]
    feat_val_users = inner_folds[0][1]
    print("\nExample inner fold:")
    print(f"  FeatTrain users (~43%): {len(feat_train_users)}")
    print(f"  FeatVal users (~21%): {len(feat_val_users)}")
