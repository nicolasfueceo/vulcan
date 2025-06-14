import json
from pathlib import Path

from loguru import logger

from src.baselines.deepfm_baseline import run_deepfm_baseline
from src.baselines.featuretools_baseline import run_featuretools_baseline
from src.baselines.svd_baseline import run_svd_baseline
from src.data.cv_data_manager import CVDataManager


def main():
    """
    Main function to run all baseline models and save their results.

    This script orchestrates the following steps:
    1. Initializes the CVDataManager to load the dataset.
    2. Retrieves the data for the first cross-validation fold.
    3. Runs three baseline models in sequence:
        - Featuretools for automated feature engineering.
        - SVD for classic collaborative filtering.
        - DeepFM for a deep learning-based recommendation.
    4. Aggregates the performance metrics (e.g., RMSE, MAE, MSE, NDCG@10) from each baseline.
    5. Saves the aggregated results to a JSON file in the 'reports' directory.
    """
    logger.info("Starting the execution of all baseline models...")

    # 1. Initialize DataManager and load data
    logger.info("Initializing CVDataManager...")
    db_path = "data/goodreads_curated.duckdb"
    splits_dir = "data/splits"
    data_manager = CVDataManager(db_path=db_path, splits_dir=splits_dir)

    logger.info("Loading users and books metadata...")
    conn = data_manager.db_connection
    try:
        users_df = conn.execute("SELECT * FROM users").fetchdf()
        books_df = conn.execute("SELECT * FROM book_series").fetchdf()
    finally:
        data_manager._return_connection(conn)
    logger.success("Metadata loaded.")

    logger.info("Loading train/test data for fold 0...")
    # Use 'full_train' to get combined training and validation data against the test set.
    train_df, test_df = data_manager.get_fold_data(fold_idx=0, split_type="full_train")

    # Dictionary to store results from all baselines
    all_results = {}

    # 2. Run Featuretools baseline
    logger.info("--- Running Featuretools Baseline ---")
    try:
        featuretools_results = run_featuretools_baseline(train_df, books_df, users_df)
        all_results["featuretools"] = {
            "status": "success",
            "feature_matrix_shape": list(featuretools_results.shape),
        }
        logger.success("Featuretools baseline completed.")
    except Exception as e:
        logger.error(f"Featuretools baseline failed: {e}")
        all_results["featuretools"] = {"status": "failure", "error": str(e)}

    # 3. Run SVD baseline (full dataset)
    logger.info("--- Running SVD Baseline ---")
    try:
        svd_results = run_svd_baseline(train_df, test_df)
        all_results["svd"] = {"status": "success", "metrics": svd_results}
        logger.success(f"SVD baseline completed. Metrics: {svd_results}")
    except Exception as e:
        logger.error(f"SVD baseline failed: {e}")
        all_results["svd"] = {"status": "failure", "error": str(e)}

    # 5. Run Popularity baseline (to be implemented)
    logger.info("--- Running Popularity Baseline ---")
    try:
        from src.baselines.popularity_baseline import run_popularity_baseline
        popularity_results = run_popularity_baseline(train_df, test_df)
        all_results["popularity"] = {"status": "success", "metrics": popularity_results}
        logger.success(f"Popularity baseline completed. Metrics: {popularity_results}")
    except Exception as e:
        logger.error(f"Popularity baseline failed: {e}")
        all_results["popularity"] = {"status": "failure", "error": str(e)}

    # 4. Run DeepFM baseline
    logger.info("--- Running DeepFM Baseline ---")
    try:
        deepfm_results = run_deepfm_baseline(train_df, test_df)
        all_results["deepfm"] = {"status": "success", "metrics": deepfm_results}
        logger.success(f"DeepFM baseline completed. Metrics: {deepfm_results}")
    except Exception as e:
        logger.error(f"DeepFM baseline failed: {e}")
        all_results["deepfm"] = {"status": "failure", "error": str(e)}

    # 5. Save results to a JSON file
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        results_path = reports_dir / "baseline_results.json"

        logger.info(f"Saving aggregated baseline results to {results_path}")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)

        logger.success(f"Results successfully saved to {results_path}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save results to file: {e}")
        logger.error(f"Current Working Directory: {Path.cwd()}")

    logger.success("All baseline models have been executed.")


if __name__ == "__main__":
    main()
