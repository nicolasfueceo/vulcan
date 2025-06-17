import json
from pathlib import Path

from loguru import logger

from src.baselines.recommender.deepfm_baseline import run_deepfm_baseline
from src.baselines.feature_engineer.featuretools_baseline import run_featuretools_baseline
from src.baselines.recommender.svd_baseline import run_svd_baseline
from torch.utils.tensorboard import SummaryWriter


from src.data.cv_data_manager import CVDataManager


def main(sampling_fraction=0.1, k_list=[5]):
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

    logger.info("Loading CV fold summary...")
    fold_summary = data_manager.get_fold_summary()
    n_folds = fold_summary.get("n_folds", 0)
    if n_folds == 0:
        logger.error("No CV folds found. Exiting.")
        return

    writer = SummaryWriter("reports/tensorboard_baselines")
    per_fold_results = {"featuretools_lightfm": [], "svd": [], "popularity": [], "deepfm": []}
    errors = {"featuretools_lightfm": [], "svd": [], "popularity": [], "deepfm": []}

    from src.baselines.recommender.popularity_baseline import run_popularity_baseline

    for fold_idx in range(n_folds):
        logger.info(f"--- Processing Fold {fold_idx+1}/{n_folds} ---")
        train_df, test_df = data_manager.get_fold_data(fold_idx=fold_idx, split_type="full_train")

        # Featuretools Baseline
        try:
            metrics = run_featuretools_baseline(train_df, books_df, users_df, test_df, k_list=k_list)
            per_fold_results["featuretools_lightfm"].append(metrics)
            logger.success(f"Featuretools+LightFM baseline completed. Metrics: {metrics}")
            if "precision_at_10" in metrics:
                writer.add_scalar("featuretools_lightfm/precision_at_10", metrics["precision_at_10"], fold_idx)
            if "n_clusters" in metrics:
                writer.add_scalar("featuretools_lightfm/n_clusters", metrics["n_clusters"], fold_idx)
        except Exception as e:
            logger.error(f"Featuretools+LightFM baseline failed on fold {fold_idx}: {e}")
            errors["featuretools_lightfm"].append(str(e))
            per_fold_results["featuretools_lightfm"].append(None)

        # SVD Baseline
        try:
            metrics = run_svd_baseline(train_df, test_df, k_list=k_list)
            per_fold_results["svd"].append(metrics)
            logger.success(f"SVD baseline completed. Metrics: {metrics}")
            if "rmse" in metrics:
                writer.add_scalar("svd/RMSE", metrics["rmse"], fold_idx)
        except Exception as e:
            logger.error(f"SVD baseline failed on fold {fold_idx}: {e}")
            errors["svd"].append(str(e))
            per_fold_results["svd"].append(None)

        # Popularity Baseline
        try:
            metrics = run_popularity_baseline(train_df, test_df, k_list=k_list)
            per_fold_results["popularity"].append(metrics)
            logger.success(f"Popularity baseline completed. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Popularity baseline failed on fold {fold_idx}: {e}")
            errors["popularity"].append(str(e))
            per_fold_results["popularity"].append(None)

        # DeepFM Baseline
        try:
            metrics = run_deepfm_baseline(train_df, test_df, k_list=k_list)
            per_fold_results["deepfm"].append(metrics)
            logger.success(f"DeepFM baseline completed. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"DeepFM baseline failed on fold {fold_idx}: {e}")
            errors["deepfm"].append(str(e))
            per_fold_results["deepfm"].append(None)

    # Aggregate results: mean and stddev for each metric and baseline
    import numpy as np
    aggregate_results = {}
    for baseline, results in per_fold_results.items():
        metrics_by_key = {}
        for fold_result in results:
            if fold_result is None:
                continue
            for k, v in fold_result.items():
                if v is None:
                    continue
                metrics_by_key.setdefault(k, []).append(v)
        baseline_agg = {}
        for k, v_list in metrics_by_key.items():
            if not k.startswith('ndcg@5'):
                continue  # Only keep ndcg@5
            arr = np.array(v_list)
            baseline_agg[f"{k}_mean"] = float(np.mean(arr))
            baseline_agg[f"{k}_std"] = float(np.std(arr))
            # Log mean to TensorBoard
            writer.add_scalar(f"{baseline}/{k}_mean", baseline_agg[f"{k}_mean"], 0)
            writer.add_scalar(f"{baseline}/{k}_std", baseline_agg[f"{k}_std"], 0)
        aggregate_results[baseline] = baseline_agg
        if errors[baseline]:
            aggregate_results[baseline]["errors"] = errors[baseline]

    all_results = {
        "per_fold": per_fold_results,
        "aggregate": aggregate_results,
        "n_folds": n_folds
    }

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
    main(sampling_fraction=0.1, k_list=[5])
