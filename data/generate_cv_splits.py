#!/usr/bin/env python3
"""
Generate cross-validation splits for the VULCAN pipeline.

This script implements stratified k-fold cross-validation for the Goodreads dataset.
The splits are stratified by user activity level (quantiles of n_ratings).
"""

import json
from pathlib import Path
from typing import List, Dict

import duckdb
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split


def setup_logging() -> None:
    """Configure logging for the script."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "generate_cv_splits.log",
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


def get_user_stats(conn: duckdb.DuckDBPyConnection, min_ratings: int = 5) -> pd.DataFrame:
    """
    Retrieves user IDs and their rating counts from the database.
    """
    logger.info(f"Querying for users with at least {min_ratings} ratings...")
    query = f"""
    SELECT
        user_id,
        count(rating) as n_ratings
    FROM interactions
    GROUP BY
        user_id
    HAVING
        n_ratings >= {min_ratings}
    """
    user_stats = conn.execute(query).fetchdf()
    logger.info(f"Found {len(user_stats)} users meeting the criteria.")
    return user_stats


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
    # Use quantile-based binning for stratification
    df["bin"] = pd.qcut(
        df["n_ratings"],
        q=n_bins,
        duplicates="drop",
        labels=False,
    )
    return df


def create_train_val_test_split(
    df: pd.DataFrame, 
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, List[str]]:
    """
    Create stratified train/validation/test splits.
    
    Args:
        df: DataFrame with user statistics and bins
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' keys
    """
    # Validate the split sizes
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total:.2f}")
    
    # Convert user_ids to strings for consistency
    user_ids = df["user_id"].astype(str).values
    bins = df["bin"].values
    
    # First split: separate out test set
    train_val_idx, test_idx = train_test_split(
        np.arange(len(user_ids)),
        test_size=test_size,
        random_state=random_state,
        stratify=bins
    )
    
    # Calculate the proportion of validation in the remaining data
    remaining_size = 1.0 - test_size
    val_ratio = val_size / remaining_size
    
    # Second split: split remaining into train and validation
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=random_state,
        stratify=bins[train_val_idx] if bins is not None else None
    )
    
    return {
        "train": user_ids[train_idx].tolist(),
        "validation": user_ids[val_idx].tolist(),
        "test": user_ids[test_idx].tolist()
    }


def create_cv_folds(
    df: pd.DataFrame, 
    n_splits: int = 5, 
    random_state: int = 42
) -> List[Dict[str, List[str]]]:
    """
    Create stratified cross-validation folds with train/validation/test splits.
    
    For each fold, the data is split into 60% train, 20% validation, and 20% test,
    with stratification by user activity level.
    
    Args:
        df: DataFrame with user statistics and bins
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of dictionaries with 'train', 'validation', and 'test' keys
    """
    # Create a list to store all folds
    folds = []
    
    # For each fold, create a different random split
    for i in range(n_splits):
        # Use a different random state for each fold
        fold_seed = random_state + i if random_state is not None else None
        
        # Create the split for this fold
        fold = create_train_val_test_split(
            df,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=fold_seed
        )
        
        # Add metadata about the fold
        fold["fold"] = i
        fold["random_seed"] = fold_seed
        
        folds.append(fold)
    
    return folds


def save_folds(folds: List[Dict[str, List[str]]], output_dir: Path) -> None:
    """
    Save the generated CV folds to disk.

    Args:
        folds: List of fold dictionaries
        output_dir: Directory to save the folds
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full folds
    with open(output_dir / "cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    
    # Also save a summary of the fold sizes
    summary = []
    for fold in folds:
        fold_summary = {
            "fold": fold.get("fold", 0),
            "random_seed": fold.get("random_seed", 0),
            "train_users": len(fold["train"]),
            "val_users": len(fold["validation"]),
            "test_users": len(fold["test"]),
            "train_ratio": len(fold["train"]) / (len(fold["train"]) + len(fold["validation"]) + len(fold["test"])),
            "val_ratio": len(fold["validation"]) / (len(fold["train"]) + len(fold["validation"]) + len(fold["test"])),
            "test_ratio": len(fold["test"]) / (len(fold["train"]) + len(fold["validation"]) + len(fold["test"]))
        }
        summary.append(fold_summary)
    
    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"folds": summary}, f, indent=2)
    
    logger.info(f"Saved {len(folds)} CV folds to {output_dir}")
    
    # Log the summary
    for s in summary:
        logger.info(
            f"Fold {s['fold']}: "
            f"train={s['train_ratio']:.1%} ({s['train_users']} users), "
            f"val={s['val_ratio']:.1%} ({s['val_users']} users), "
            f"test={s['test_ratio']:.1%} ({s['test_users']} users)"
        )


def generate_cv_splits(
    n_splits: int = 5,
    n_activity_bins: int = 10,
    random_state: int = 42,
    output_dir: str = "data/splits"
) -> List[Dict[str, List[str]]]:
    """
    Generate cross-validation splits for the dataset.

    Args:
        n_splits: Number of CV folds
        n_activity_bins: Number of bins for activity-based stratification
        random_state: Random seed for reproducibility
        output_dir: Directory to save the splits

    Returns:
        List of fold dictionaries
    """
    logger.info(f"Generating {n_splits}-fold CV splits...")
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    
    # Get user statistics
    conn = get_db_connection()
    user_stats = get_user_stats(conn)
    conn.close()

    logger.info(f"Found {len(user_stats)} users with sufficient ratings")

    # Adjust n_bins if there are too few users for stratification.
    # Each bin for stratification should have at least 2 samples for train/test split.
    if len(user_stats) < n_activity_bins * 2:
        new_n_bins = max(1, len(user_stats) // 2)
        if new_n_bins < n_activity_bins:
            logger.warning(
                f"Number of users ({len(user_stats)}) is too low for {n_activity_bins} "
                f"bins. Adjusting to {new_n_bins} bin(s) to allow for stratification."
            )
            n_activity_bins = new_n_bins

    # Create activity-based bins for stratification
    user_stats = create_activity_bins(user_stats, n_bins=n_activity_bins)
    
    # Generate CV folds
    folds = create_cv_folds(user_stats, n_splits=n_splits, random_state=random_state)
    
    # Save the folds
    output_path = Path(output_dir)
    save_folds(folds, output_path)
    
    return folds


def main():
    """Main function to generate CV splits."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CV splits for VULCAN")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-bins", type=int, default=10, 
                       help="Number of activity level bins for stratification")
    parser.add_argument("--random-state", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                       help="Directory to save the splits")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Generate and save the splits
    generate_cv_splits(
        n_splits=args.n_splits,
        n_activity_bins=args.n_bins,
        random_state=args.random_state,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
