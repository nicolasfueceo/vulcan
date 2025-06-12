#!/usr/bin/env python3
"""
Generate train/test splits for the VULCAN pipeline.

This script implements a stratified split strategy for the Goodreads dataset:
1. Cold-start holdout (10% of users)
2. Active sample (~20-25% of remaining users)
3. K-fold stratified CV (default K=5) on the active sample

The splits are stratified by user activity level (quantiles of n_ratings).
"""

import json
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold


def setup_logging() -> None:
    """Configure logging for the script."""
    logger.add(
        "logs/generate_splits.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Create a connection to the DuckDB database."""
    db_path = Path("data/goodreads_curated.duckdb")
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")
    return duckdb.connect(str(db_path))


def get_user_stats(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Get user statistics from the database.

    Returns:
        DataFrame with columns: user_id, n_ratings, mean_rating, var_rating
    """
    query = """
    SELECT 
        user_id,
        COUNT(*) as n_ratings,
        AVG(rating) as mean_rating,
        STDDEV_SAMP(rating) as var_rating
    FROM curated_reviews
    GROUP BY user_id
    """
    return conn.execute(query).fetchdf()


def create_activity_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Create activity level bins for stratification.

    Args:
        df: DataFrame with user statistics
        n_bins: Number of bins to create

    Returns:
        DataFrame with added 'bin' column
    """
    df = df.copy()
    df["bin"] = pd.qcut(
        df["n_ratings"],
        q=n_bins,
        duplicates="drop",
        labels=False,
    )
    return df


def sample_users(
    df: pd.DataFrame,
    sample_frac: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample users stratified by activity level.

    Args:
        df: DataFrame with user statistics and bins
        sample_frac: Fraction of users to sample
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sampled users
    """
    return df.groupby("bin").sample(
        frac=sample_frac,
        random_state=random_state,
    )


def create_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[List[str]]:
    """
    Create stratified cross-validation folds.

    Args:
        df: DataFrame with user statistics and bins
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        List of lists containing user_ids for each fold
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    for _, val_idx in skf.split(df, df.bin):
        folds.append(list(df.iloc[val_idx].user_id))

    return folds


def save_splits(
    cold_start_users: List[str],
    active_users: List[str],
    cv_folds: List[List[str]],
    output_dir: Path,
) -> None:
    """
    Save the generated splits to disk.

    Args:
        cold_start_users: List of user_ids for cold-start holdout
        active_users: List of user_ids for active sample
        cv_folds: List of lists containing user_ids for each CV fold
        output_dir: Directory to save the splits
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cold-start users
    with open(output_dir / "cold_start_users.json", "w") as f:
        json.dump(cold_start_users, f)

    # Save active users
    with open(output_dir / "sample_users.json", "w") as f:
        json.dump(active_users, f)

    # Save CV folds
    with open(output_dir / "cv_folds.json", "w") as f:
        json.dump(cv_folds, f)


def generate_splits(
    cold_start_frac: float = 0.10,
    active_sample_frac: float = 0.25,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Generate the train/test splits for the VULCAN pipeline.

    Args:
        cold_start_frac: Fraction of users to hold out for cold-start testing
        active_sample_frac: Fraction of remaining users to use for active sample
        n_cv_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cold_start_users, active_users, cv_folds)
    """
    logger.info("Starting split generation...")

    try:
        # Connect to database
        conn = get_db_connection()
        logger.info("Connected to database")

        # Get user statistics
        df_users = get_user_stats(conn)
        logger.info(f"Retrieved statistics for {len(df_users)} users")

        # Create activity bins
        df_users = create_activity_bins(df_users)
        logger.info("Created activity level bins")

        # Sample cold-start users
        df_cold = sample_users(df_users, cold_start_frac, random_state)
        cold_start_users = list(df_cold.user_id)
        logger.info(f"Sampled {len(cold_start_users)} cold-start users")

        # Sample active users from remaining pool
        remaining = df_users[~df_users.user_id.isin(cold_start_users)]
        df_active = sample_users(remaining, active_sample_frac, random_state)
        active_users = list(df_active.user_id)
        logger.info(f"Sampled {len(active_users)} active users")

        # Create CV folds
        cv_folds = create_cv_folds(df_active, n_cv_folds, random_state)
        logger.info(f"Created {len(cv_folds)} CV folds")

        # Save splits
        save_splits(
            cold_start_users,
            active_users,
            cv_folds,
            Path("data/splits"),
        )
        logger.info("Saved splits to disk")

        return cold_start_users, active_users, cv_folds

    except Exception as e:
        logger.error(f"Error generating splits: {str(e)}")
        raise
    finally:
        if "conn" in locals():
            conn.close()
            logger.info("Closed database connection")


def main():
    """Main function to generate splits."""
    setup_logging()
    logger.info("Starting split generation script")

    try:
        cold_start_users, active_users, cv_folds = generate_splits()

        # Print summary statistics
        logger.info("\nSplit Generation Summary:")
        logger.info(f"Total cold-start users: {len(cold_start_users)}")
        logger.info(f"Total active users: {len(active_users)}")
        logger.info(f"Number of CV folds: {len(cv_folds)}")
        logger.info(f"Users per fold: {[len(fold) for fold in cv_folds]}")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
