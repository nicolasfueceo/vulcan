#!/usr/bin/env python3
"""
Simple script to create data splits for VULCAN.
"""

import os
import sys

sys.path.append("data")

import yaml

from data.data_splits import DataSplitter


def main():
    # Load config
    with open("data/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Creating data splits for VULCAN...")

    # Create splitter
    splitter = DataSplitter(
        database_path="data/goodreads.db",
        test_size=0.2,
        outer_folds=5,
        inner_folds=3,
        random_state=42,
    )

    # Create splits
    print("Creating outer split...")
    design_users, outer_test_users = splitter.create_outer_split()

    print("Creating CV folds...")
    outer_cv_folds = splitter.create_outer_cv_folds()

    # Create inner folds for each outer fold
    for i in range(5):
        print(f"Creating inner folds for outer fold {i + 1}...")
        splitter.create_inner_cv_folds(fold_idx=i)

    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    splitter.save_user_splits("data/splits")

    print("Data splits created successfully!")
    print(f"Total users: {len(splitter.all_user_ids)}")
    print(f"Design users: {len(design_users)}")
    print(f"Test users: {len(outer_test_users)}")


if __name__ == "__main__":
    main()
