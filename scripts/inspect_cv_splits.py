"""
Inspect CV splits and data loading for VULCAN.

This script loads and displays the CV splits and sample data to help debug data loading issues.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tabulate import tabulate

from src.data.cv_data_manager import CVDataManager

# Configure logging
logger.add("logs/inspect_cv_splits.log", rotation="1 MB")

def inspect_cv_splits(data_dir: Path):
    """Inspect CV splits and display information about the data."""
    logger.info("Starting CV splits inspection")
    
    # Initialize data manager
    try:
        logger.info(f"Loading CV splits from {data_dir}")
        cv_data = CVDataManager(data_dir)
        
        # Get fold summary
        summary = cv_data.get_fold_summary()
        logger.info("CV Fold Summary:")
        print("\n" + "="*50)
        print("CV FOLD SUMMARY")
        print("="*50)
        print(json.dumps(summary, indent=2))
        
        # Load all folds
        cv_folds = cv_data.load_cv_folds()
        logger.info(f"Loaded {len(cv_folds)} CV folds")
        
        # Inspect each fold
        for fold_idx, fold_data in enumerate(cv_folds):
            print(f"\n{'-'*50}")
            print(f"FOLD {fold_idx + 1} DETAILS")
            print(f"{'='*50}")
            
            # Print user counts
            print(f"\nTrain users: {len(fold_data['train']):,}")
            print(f"Val users: {len(fold_data['validation']):,}")
            
            # Load sample data for this fold
            try:
                logger.info(f"Loading sample data for fold {fold_idx}")
                # Get a sample of data for this fold (returns tuple of train_df, val_df)
                # Only load a small sample for inspection
                sample_frac = 0.001  # Sample 0.1% of users for faster inspection
                train_df, val_df = cv_data.get_fold_data(
                    fold_idx=fold_idx,
                    sample_frac=sample_frac,
                    random_state=42
                )
                
                def analyze_dataframe(df: pd.DataFrame, name: str) -> None:
                    """Helper function to analyze a single dataframe."""
                    if df is None or df.empty:
                        logger.warning(f"{name} DataFrame is empty")
                        return
                        
                    print(f"\n{name} DataFrame Info:")
                    print(f"Shape: {df.shape}")
                    print("\nColumn dtypes:")
                    print(df.dtypes)
                    
                    # Check for missing values
                    print("\nMissing values:")
                    print(df.isnull().sum())
                    
                    # Show sample of the data
                    print(f"\n{name} Sample data:")
                    print(tabulate(df.head(5), headers='keys', tablefmt='psql'))
                    
                    # Check rating distribution if column exists
                    if 'rating' in df.columns:
                        print(f"\n{name} Rating distribution:")
                        print(df['rating'].value_counts().sort_index())
                        
                        # Check for NaN ratings
                        nan_ratings = df['rating'].isna().sum()
                        if nan_ratings > 0:
                            print(f"WARNING: Found {nan_ratings} rows with NaN ratings in {name}")
                        
                        # Check rating range
                        print(f"\n{name} Rating statistics:")
                        print(df['rating'].describe())
                    
                    # Check user-item interactions
                    if all(col in df.columns for col in ['user_id', 'item_id']):
                        print(f"\n{name} User-Item interaction matrix:")
                        n_users = df['user_id'].nunique()
                        n_items = df['item_id'].nunique()
                        n_interactions = len(df)
                        density = n_interactions / (n_users * n_items) if (n_users * n_items) > 0 else 0
                        print(f"Users: {n_users}, Items: {n_items}, "
                              f"Interactions: {n_interactions}, Density: {density:.6f}")
                        
                        # Check for duplicate user-item pairs
                        duplicates = df.duplicated(subset=['user_id', 'item_id']).sum()
                        if duplicates > 0:
                            print(f"WARNING: Found {duplicates} duplicate user-item pairs in {name}")
                
                # Analyze both train and validation data
                analyze_dataframe(train_df, "TRAIN")
                analyze_dataframe(val_df, "VALIDATION")
                    
            except Exception as e:
                logger.error(f"Error loading data for fold {fold_idx}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to inspect CV splits: {str(e)}")
        raise

if __name__ == "__main__":
    data_dir = Path("data")
    inspect_cv_splits(data_dir)
