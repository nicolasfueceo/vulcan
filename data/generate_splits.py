#!/usr/bin/env python
"""
Script to generate all data splits for the VULCAN recommender system.

This script reads the configuration from config.yaml and creates all necessary
data splits for the two-phase recommender system.

Usage:
    python generate_splits.py [--config CONFIG_PATH]
"""

import argparse
import logging
import os
import time

import yaml

from .data_splits import DataSplitter


def load_config(config_path="data/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    """Set up logging based on configuration."""
    log_level = getattr(logging, config["logging"]["level"])
    log_dir = config["data_paths"]["logs_dir"]

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    log_dir, f"data_splits_{time.strftime('%Y%m%d_%H%M%S')}.log"
                )
            ),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def create_directory_structure(config):
    """Create all required directories from the configuration."""
    for path_name, path in config["data_paths"].items():
        if path_name.endswith("_dir"):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")


def main(config_path):
    """Main function to generate all data splits."""
    global logger

    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {config_path}")

    # Create required directories
    create_directory_structure(config)

    # Get data splitting parameters
    split_config = config["data_splits"]
    database_path = config["data_paths"]["train_db"]
    splits_dir = config["data_paths"]["splits_dir"]
    queries_dir = config["data_paths"]["queries_dir"]

    logger.info("Starting data split generation")

    # Create data splitter
    splitter = DataSplitter(
        database_path=database_path,
        test_size=split_config["test_size"],
        outer_folds=split_config["outer_folds"],
        inner_folds=split_config["inner_folds"],
        random_state=split_config["random_state"],
    )

    # Create all splits
    logger.info("Creating outer split (test/design)")
    design_users, outer_test_users = splitter.create_outer_split()

    logger.info(f"Creating {split_config['outer_folds']} outer CV folds")
    outer_cv_folds = splitter.create_outer_cv_folds()

    # For each outer fold, create inner folds
    for i in range(split_config["outer_folds"]):
        logger.info(
            f"Creating {split_config['inner_folds']} inner CV folds for outer fold {i + 1}"
        )
        splitter.create_inner_cv_folds(fold_idx=i)

    # Save all splits to files
    logger.info(f"Saving user splits to {splits_dir}")
    splitter.save_user_splits(output_dir=splits_dir)

    logger.info(f"Generating SQL queries and saving to {queries_dir}")
    splitter.create_split_queries(output_dir=queries_dir)

    # Print summary
    logger.info("\nData Split Summary:")
    logger.info(f"Total users: {len(splitter.all_user_ids)}")
    logger.info(
        f"Outer test users ({split_config['test_size'] * 100:.0f}%): {len(splitter.outer_test_users)}"
    )
    logger.info(
        f"Design users ({(1 - split_config['test_size']) * 100:.0f}%): {len(splitter.design_users)}"
    )

    # Example fold sizes
    train_fe_users = splitter.outer_cv_folds[0][0]
    val_clusters_users = splitter.outer_cv_folds[0][1]
    logger.info("\nExample outer fold:")
    logger.info(
        f"  TrainFE users (~{len(train_fe_users) / len(splitter.all_user_ids) * 100:.1f}%): {len(train_fe_users)}"
    )
    logger.info(
        f"  ValClusters users (~{len(val_clusters_users) / len(splitter.all_user_ids) * 100:.1f}%): {len(val_clusters_users)}"
    )

    inner_folds = splitter.inner_cv_folds
    feat_train_users = inner_folds[0][0]
    feat_val_users = inner_folds[0][1]
    logger.info("\nExample inner fold:")
    logger.info(
        f"  FeatTrain users (~{len(feat_train_users) / len(splitter.all_user_ids) * 100:.1f}%): {len(feat_train_users)}"
    )
    logger.info(
        f"  FeatVal users (~{len(feat_val_users) / len(splitter.all_user_ids) * 100:.1f}%): {len(feat_val_users)}"
    )

    logger.info("\nData splitting completed successfully!")

    return splitter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data splits for VULCAN recommender system"
    )
    parser.add_argument(
        "--config", default="data/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
