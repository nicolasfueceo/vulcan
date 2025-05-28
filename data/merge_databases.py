#!/usr/bin/env python3
"""
Merge test, train, and validation databases into a single goodreads.db database.
Also creates split indices to identify which records came from which original database.
"""

import argparse
import os
import sys

import yaml

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.autonomous_fe_env.data.data_merger import DatabaseMerger


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Merge test, train, and validation databases into a single goodreads.db database."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--preserve",
        action="store_true",
        help="Preserve source databases after merging.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/goodreads.db",
        help="Path to the target database.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_paths = config.get("data_paths", {})

    # Get paths to source databases
    train_db = data_paths.get("train_db", "data/train.db")
    test_db = data_paths.get("test_db", "data/test.db")
    validation_db = data_paths.get("validation_db", "data/validation.db")

    # Check that all source databases exist
    source_dbs = [train_db, test_db, validation_db]
    missing_dbs = [db for db in source_dbs if not os.path.exists(db)]

    if missing_dbs:
        print(
            f"Error: The following source databases do not exist: {', '.join(missing_dbs)}"
        )
        return 1

    # Create merger
    merger = DatabaseMerger(source_dbs, args.target)

    # Merge databases
    print(f"Merging databases: {', '.join(source_dbs)} -> {args.target}")
    success = merger.merge_databases(preserve_source=args.preserve)

    if not success:
        print("Error: Failed to merge databases.")
        return 1

    # Create split indices
    print("Creating split indices...")
    splits_dir = data_paths.get("splits_dir", "data/splits")
    success = merger.create_all_split_indices(splits_dir=splits_dir)

    if not success:
        print("Warning: Failed to create some split indices.")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
