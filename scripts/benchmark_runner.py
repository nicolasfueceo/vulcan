"""
Benchmark Runner for FUEGO Recommender Systems

This module provides a simplified interface for running benchmarks on different
recommender systems using standardized metrics and evaluation procedures.
"""

import os
import logging
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Import recommender interfaces
from models.baselines.base_recommender import BaseRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for benchmarking recommender systems.

    This class provides a simplified interface to evaluate and compare different
    recommender systems using standardized metrics and evaluation procedures.
    """

    def __init__(self, results_dir: Optional[str] = None, random_seed: int = 42):
        """
        Initialize the benchmark runner.

        Args:
            results_dir (Optional[str], optional): Directory to store benchmark results.
                                                 Defaults to project's results/benchmarks directory.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Set up directories
        if results_dir is None:
            self.results_dir = project_root / "results" / "benchmarks"
        else:
            self.results_dir = Path(results_dir)

        # Create directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Set random seed
        np.random.seed(random_seed)
        self.random_seed = random_seed

        # Initialize results storage
        self.results = {}

        logger.info(
            f"Benchmark runner initialized with results_dir: {self.results_dir}"
        )

    def load_data(
        self,
        train_path: str,
        test_path: str,
        items_path: Optional[str] = None,
        user_id_col: str = "user_id_mapped",
        item_id_col: str = "book_id_mapped",
        rating_col: str = "rating",
        timestamp_col: Optional[str] = "timestamp",
    ) -> Dict[str, Any]:
        """
        Load dataset for benchmarking.

        Args:
            train_path (str): Path to training data CSV file.
            test_path (str): Path to test data CSV file.
            items_path (Optional[str], optional): Path to items data CSV file. Defaults to None.
            user_id_col (str, optional): Name of user ID column. Defaults to 'user_id_mapped'.
            item_id_col (str, optional): Name of item ID column. Defaults to 'book_id_mapped'.
            rating_col (str, optional): Name of rating column. Defaults to 'rating'.
            timestamp_col (Optional[str], optional): Name of timestamp column. Defaults to 'timestamp'.

        Returns:
            Dict[str, Any]: Dataset dictionary with train, test, and metadata.
        """
        logger.info(f"Loading dataset from {train_path} and {test_path}")

        # Load training data
        train_df = pd.read_csv(train_path)

        # Load test data
        test_df = pd.read_csv(test_path)

        # Load items data if available
        items_df = None
        if items_path and os.path.exists(items_path):
            items_df = pd.read_csv(items_path)

        # Rename columns to standard format
        column_mapping = {user_id_col: "userId", item_id_col: "itemId"}

        if rating_col in train_df.columns:
            column_mapping[rating_col] = "rating"

        if timestamp_col and timestamp_col in train_df.columns:
            column_mapping[timestamp_col] = "timestamp"

        train_df = train_df.rename(columns=column_mapping)
        test_df = test_df.rename(columns=column_mapping)

        # Create dataset dictionary
        dataset = {
            "train": train_df,
            "test": test_df,
            "items": items_df,
            "n_users": len(train_df["userId"].unique()),
            "n_items": len(train_df["itemId"].unique()),
            "n_train_ratings": len(train_df),
            "n_test_ratings": len(test_df),
        }

        logger.info(
            f"Loaded dataset with {dataset['n_users']} users, {dataset['n_items']} items, "
            f"{dataset['n_train_ratings']} train ratings, and {dataset['n_test_ratings']} test ratings"
        )

        return dataset

    def prepare_training_data(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare training data for recommenders.

        Args:
            dataset (Dict[str, Any]): Dataset dictionary from load_data.

        Returns:
            Dict[str, Any]: Training data dictionary.
        """
        logger.info("Preparing training data")

        training_data = {"ratings": dataset["train"], "items": dataset["items"]}

        return training_data

    def train_recommender(
        self, recommender: BaseRecommender, dataset: Dict[str, Any]
    ) -> None:
        """
        Train a recommender on the dataset.

        Args:
            recommender (BaseRecommender): Recommender to train.
            dataset (Dict[str, Any]): Dataset dictionary from load_data.
        """
        logger.info(f"Training recommender: {recommender.name}")

        # Prepare training data
        training_data = self.prepare_training_data(dataset)

        # Train the recommender
        start_time = time.time()
        recommender.train(training_data)
        training_time = time.time() - start_time

        logger.info(
            f"Recommender {recommender.name} trained in {training_time:.2f} seconds"
        )

        return training_time

    def evaluate_recommender(
        self,
        recommender: BaseRecommender,
        dataset: Dict[str, Any],
        k_values: List[int] = [5, 10, 20],
        n_users: Optional[int] = None,
        n_predictions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a recommender on the dataset.

        Args:
            recommender (BaseRecommender): Recommender to evaluate.
            dataset (Dict[str, Any]): Dataset dictionary from load_data.
            k_values (List[int], optional): List of k values for evaluation metrics.
                                          Defaults to [5, 10, 20].
            n_users (Optional[int], optional): Number of users to evaluate.
                                             Defaults to None (use all users).
            n_predictions (Optional[int], optional): Number of predictions to make.
                                                   Defaults to None (use all test ratings).

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info(f"Evaluating recommender: {recommender.name}")

        # Check if recommender is trained
        if not recommender.is_trained:
            logger.warning(f"Recommender {recommender.name} is not trained")
            return {}

        # Evaluate rating prediction
        rating_results = self.evaluate_rating_prediction(
            recommender=recommender, dataset=dataset, n_predictions=n_predictions
        )

        # Evaluate recommendation
        recommendation_results = self.evaluate_recommendation(
            recommender=recommender, dataset=dataset, k_values=k_values, n_users=n_users
        )

        # Combine results
        results = {
            "name": recommender.name,
            "description": recommender.description,
            "rating_prediction": rating_results,
            "recommendation": recommendation_results,
        }

        return results

    def evaluate_rating_prediction(
        self,
        recommender: BaseRecommender,
        dataset: Dict[str, Any],
        n_predictions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a recommender's rating prediction performance.

        Args:
            recommender (BaseRecommender): Recommender to evaluate.
            dataset (Dict[str, Any]): Dataset dictionary from load_data.
            n_predictions (Optional[int], optional): Number of predictions to make.
                                                   Defaults to None (use all test ratings).

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info(f"Evaluating rating prediction for {recommender.name}")

        test_df = dataset["test"]

        # Sample test ratings if specified
        if n_predictions is not None and n_predictions < len(test_df):
            test_sample = test_df.sample(n=n_predictions, random_state=self.random_seed)
        else:
            test_sample = test_df

        # Make predictions
        start_time = time.time()

        predictions = []
        actuals = []

        for _, row in tqdm(
            test_sample.iterrows(), total=len(test_sample), desc="Predicting ratings"
        ):
            user_id = row["userId"]
            item_id = row["itemId"]

            # Skip if rating column is not available
            if "rating" not in row:
                continue

            actual_rating = row["rating"]

            try:
                predicted_rating = recommender.predict(user_id, item_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except Exception as e:
                logger.warning(
                    f"Error predicting rating for user {user_id}, item {item_id}: {e}"
                )

        prediction_time = time.time() - start_time

        # Calculate metrics
        if not predictions:
            logger.warning(f"No predictions made for {recommender.name}")
            return {
                "rmse": None,
                "mae": None,
                "prediction_time": prediction_time,
                "predictions_count": 0,
            }

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))

        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Return results
        results = {
            "rmse": float(rmse),
            "mae": float(mae),
            "prediction_time": prediction_time,
            "predictions_count": len(predictions),
        }

        return results

    def evaluate_recommendation(
        self,
        recommender: BaseRecommender,
        dataset: Dict[str, Any],
        k_values: List[int] = [5, 10, 20],
        n_users: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a recommender's top-N recommendation performance.

        Args:
            recommender (BaseRecommender): Recommender to evaluate.
            dataset (Dict[str, Any]): Dataset dictionary from load_data.
            k_values (List[int], optional): List of k values for evaluation metrics.
                                          Defaults to [5, 10, 20].
            n_users (Optional[int], optional): Number of users to evaluate.
                                             Defaults to None (use all users).

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info(f"Evaluating recommendation for {recommender.name}")

        train_df = dataset["train"]
        test_df = dataset["test"]

        # Get unique users in test set
        test_users = test_df["userId"].unique()

        # Sample users if specified
        if n_users is not None and n_users < len(test_users):
            sampled_users = np.random.choice(test_users, size=n_users, replace=False)
        else:
            sampled_users = test_users

        # Create user-item sets
        train_user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            train_user_items[row["userId"]].add(row["itemId"])

        test_user_items = defaultdict(set)
        for _, row in test_df.iterrows():
            # Consider all test items as relevant if no rating column
            if (
                "rating" not in row or row["rating"] >= 4.0
            ):  # Consider ratings >= 4 as relevant
                test_user_items[row["userId"]].add(row["itemId"])

        # Initialize metrics
        precision = {k: [] for k in k_values}
        recall = {k: [] for k in k_values}
        ndcg = {k: [] for k in k_values}
        hit_rate = {k: [] for k in k_values}

        # Generate recommendations and calculate metrics
        start_time = time.time()

        for user_id in tqdm(sampled_users, desc="Generating recommendations"):
            # Skip users with no test items
            if user_id not in test_user_items or not test_user_items[user_id]:
                continue

            # Get recommendations
            try:
                recommendations = recommender.recommend(
                    user_id=user_id, n=max(k_values), filter_already_liked=True
                )

                # Extract item IDs
                rec_items = [rec["item_id"] for rec in recommendations]

                # Calculate metrics for each k
                for k in k_values:
                    # Precision@k
                    n_rel_and_rec_k = len(set(rec_items[:k]) & test_user_items[user_id])
                    precision[k].append(n_rel_and_rec_k / k if k > 0 else 0)

                    # Recall@k
                    n_rel = len(test_user_items[user_id])
                    recall[k].append(n_rel_and_rec_k / n_rel if n_rel > 0 else 0)

                    # Hit Rate@k
                    hit_rate[k].append(1.0 if n_rel_and_rec_k > 0 else 0.0)

                    # NDCG@k
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, k)))
                    dcg = sum(
                        1.0 / np.log2(i + 2)
                        for i, item in enumerate(rec_items[:k])
                        if item in test_user_items[user_id]
                    )
                    ndcg[k].append(dcg / idcg if idcg > 0 else 0)

            except Exception as e:
                logger.warning(
                    f"Error generating recommendations for user {user_id}: {e}"
                )

        recommendation_time = time.time() - start_time

        # Calculate average metrics
        avg_precision = {
            k: np.mean(vals) if vals else None for k, vals in precision.items()
        }
        avg_recall = {k: np.mean(vals) if vals else None for k, vals in recall.items()}
        avg_ndcg = {k: np.mean(vals) if vals else None for k, vals in ndcg.items()}
        avg_hit_rate = {
            k: np.mean(vals) if vals else None for k, vals in hit_rate.items()
        }

        # Log results
        for k in k_values:
            if avg_precision[k] is not None:
                logger.info(
                    f"Precision@{k}: {avg_precision[k]:.4f}, "
                    f"Recall@{k}: {avg_recall[k]:.4f}, "
                    f"NDCG@{k}: {avg_ndcg[k]:.4f}, "
                    f"Hit Rate@{k}: {avg_hit_rate[k]:.4f}"
                )

        # Return results
        results = {
            "precision": {
                str(k): float(avg_precision[k])
                if avg_precision[k] is not None
                else None
                for k in k_values
            },
            "recall": {
                str(k): float(avg_recall[k]) if avg_recall[k] is not None else None
                for k in k_values
            },
            "ndcg": {
                str(k): float(avg_ndcg[k]) if avg_ndcg[k] is not None else None
                for k in k_values
            },
            "hit_rate": {
                str(k): float(avg_hit_rate[k]) if avg_hit_rate[k] is not None else None
                for k in k_values
            },
            "recommendation_time": recommendation_time,
            "users_evaluated": len(sampled_users),
        }

        return results

    def run_benchmark(
        self,
        recommenders: List[BaseRecommender],
        dataset: Dict[str, Any],
        k_values: List[int] = [5, 10, 20],
        n_users: Optional[int] = None,
        n_predictions: Optional[int] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark on multiple recommenders.

        Args:
            recommenders (List[BaseRecommender]): List of recommenders to benchmark.
            dataset (Dict[str, Any]): Dataset dictionary from load_data.
            k_values (List[int], optional): List of k values for evaluation metrics.
                                          Defaults to [5, 10, 20].
            n_users (Optional[int], optional): Number of users to evaluate.
                                             Defaults to None (use all users).
            n_predictions (Optional[int], optional): Number of predictions to make.
                                                   Defaults to None (use all test ratings).
            save_results (bool, optional): Whether to save results to disk. Defaults to True.

        Returns:
            Dict[str, Any]: Benchmark results.
        """
        logger.info(f"Running benchmark on {len(recommenders)} recommenders")

        # Initialize results
        benchmark_results = {
            "dataset": {
                "n_users": dataset["n_users"],
                "n_items": dataset["n_items"],
                "n_train_ratings": dataset["n_train_ratings"],
                "n_test_ratings": dataset["n_test_ratings"],
            },
            "recommenders": [],
        }

        # Train and evaluate each recommender
        for recommender in recommenders:
            logger.info(f"Benchmarking recommender: {recommender.name}")

            # Train the recommender
            training_time = self.train_recommender(recommender, dataset)

            # Evaluate the recommender
            evaluation_results = self.evaluate_recommender(
                recommender=recommender,
                dataset=dataset,
                k_values=k_values,
                n_users=n_users,
                n_predictions=n_predictions,
            )

            # Add training time to results
            evaluation_results["training_time"] = training_time

            # Add to benchmark results
            benchmark_results["recommenders"].append(evaluation_results)

        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            results_path = self.results_dir / f"benchmark_results_{timestamp}.json"

            with open(results_path, "w") as f:
                json.dump(benchmark_results, f, indent=2)

            logger.info(f"Benchmark results saved to {results_path}")

        # Store results
        self.results = benchmark_results

        return benchmark_results

    def plot_results(
        self,
        metric: str = "ndcg",
        k: int = 10,
        results: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot benchmark results.

        Args:
            metric (str, optional): Metric to plot. Defaults to "ndcg".
            k (int, optional): K value for the metric. Defaults to 10.
            results (Optional[Dict[str, Any]], optional): Results to plot. Defaults to None (use stored results).
            save_path (Optional[str], optional): Path to save the plot. Defaults to None (don't save).
        """
        # Use stored results if none provided
        if results is None:
            results = self.results

        if not results or "recommenders" not in results:
            logger.warning("No results to plot")
            return

        # Extract metric values
        recommender_names = []
        metric_values = []

        for recommender_result in results["recommenders"]:
            name = recommender_result["name"]

            # Skip if metric is not available
            if (
                metric not in recommender_result["recommendation"]
                or str(k) not in recommender_result["recommendation"][metric]
                or recommender_result["recommendation"][metric][str(k)] is None
            ):
                continue

            value = recommender_result["recommendation"][metric][str(k)]

            recommender_names.append(name)
            metric_values.append(value)

        if not recommender_names:
            logger.warning(f"No data available for metric {metric}@{k}")
            return

        # Create plot
        plt.figure(figsize=(10, 6))

        # Sort by metric value
        sorted_indices = np.argsort(metric_values)[::-1]
        sorted_names = [recommender_names[i] for i in sorted_indices]
        sorted_values = [metric_values[i] for i in sorted_indices]

        # Plot bars
        bars = plt.bar(sorted_names, sorted_values)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        # Add labels and title
        plt.xlabel("Recommender")
        plt.ylabel(f"{metric.upper()}@{k}")
        plt.title(f"Comparison of {metric.upper()}@{k} across Recommenders")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()

        # Save plot if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        # Show plot
        plt.show()

    def save_recommender(
        self, recommender: BaseRecommender, path: Optional[str] = None
    ) -> str:
        """
        Save a trained recommender to disk.

        Args:
            recommender (BaseRecommender): Recommender to save.
            path (Optional[str], optional): Path to save the recommender. Defaults to None (auto-generate).

        Returns:
            str: Path where the recommender was saved.
        """
        if not recommender.is_trained:
            logger.warning(f"Recommender {recommender.name} is not trained")
            return ""

        # Generate path if not provided
        if path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            path = self.results_dir / f"{recommender.name}_{timestamp}.pkl"
        else:
            path = Path(path)

        # Create directory if it doesn't exist
        os.makedirs(path.parent, exist_ok=True)

        # Save the recommender
        try:
            recommender.save_model(str(path))
            logger.info(f"Recommender {recommender.name} saved to {path}")
            return str(path)
        except NotImplementedError:
            # If save_model is not implemented, use pickle
            try:
                with open(path, "wb") as f:
                    pickle.dump(recommender, f)
                logger.info(
                    f"Recommender {recommender.name} saved to {path} using pickle"
                )
                return str(path)
            except Exception as e:
                logger.error(f"Error saving recommender {recommender.name}: {e}")
                return ""

    def load_recommender(self, path: str) -> Optional[BaseRecommender]:
        """
        Load a trained recommender from disk.

        Args:
            path (str): Path to load the recommender from.

        Returns:
            Optional[BaseRecommender]: Loaded recommender or None if loading failed.
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"File not found: {path}")
            return None

        # Try to load using pickle
        try:
            with open(path, "rb") as f:
                recommender = pickle.load(f)
            logger.info(f"Recommender loaded from {path}")
            return recommender
        except Exception as e:
            logger.error(f"Error loading recommender from {path}: {e}")
            return None
