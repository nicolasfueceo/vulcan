"""
End-to-end test for the VULCAN optimization agent with real book data.
"""

import inspect
import time
import pandas as pd
import pytest
from loguru import logger

from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.data.cv_data_manager import CVDataManager
from src.utils.feature_registry import feature_registry
from src.utils.run_utils import init_run, terminate_pipeline
from src.utils.session_state import SessionState

# Configure logging
logger.add("logs/test_optimization_end_to_end.log", rotation="10 MB")


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_module():
    """Setup and teardown for the entire test module."""
    # Setup: Any setup code needed before all tests in this module run
    yield  # This is where the test runs

    # Teardown: Clean up
    terminate_pipeline()


def test_optimization_end_to_end():
    """Run end-to-end test of the optimization agent."""
    logger.info("Starting optimization agent end-to-end test")
    init_run()
    session_state = SessionState()

    # Define and register the feature needed for the test
    def average_rating(df: pd.DataFrame, scale: float = 1.0) -> pd.Series:
        """
        Calculates the average rating for each user and applies a scaling factor.
        The resulting feature is merged back to the original dataframe shape.
        """
        if df.empty:
            return pd.Series(name="average_rating", dtype=float)

        # Calculate average ratings per user and create a DataFrame from it.
        user_avg_ratings_df = (
            df.groupby("user_id")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "average_rating_temp"})
        )

        # Merge this back to the original DataFrame to broadcast the user-level feature.
        merged_df = pd.merge(df, user_avg_ratings_df, on="user_id", how="left")

        # Extract the feature column, fill NaNs, apply scaling, and return as a named Series.
        feature_series = merged_df["average_rating_temp"].fillna(0) * scale
        feature_series.name = "average_rating"
        return feature_series

    # Register the feature directly into the registry
    feature_data = {
        "func": average_rating,
        "params": {"scale": 1.0},
        "source_code": inspect.getsource(average_rating),
        "passed_test": True,
        "type": "code",
        "source_candidate_id": None,
    }
    feature_registry.register(name="average_rating", feature_data=feature_data)

    try:
        # Step 1: Initialize CVDataManager with production data
        try:
            cv_manager = CVDataManager(
                db_path="data/goodreads_curated.duckdb",
                splits_dir="data/processed/cv_splits",
            )
            cv_manager.load_cv_folds()  # Verify it can load the data
            logger.info("Successfully initialized CVDataManager with production data.")
        except Exception as e:
            logger.error(f"Failed to initialize CVDataManager: {e}")
            pytest.fail("Could not initialize CVDataManager with production data.")

        # Step 2: Configure optimizer
        optimizer = VULCANOptimizer(n_jobs=1, random_state=42, session=session_state)

        # Step 3: Run optimization
        logger.info("Starting optimization...")
        start_time = time.time()

        features = [
            {
                "name": "average_rating",
                "description": "User's average book rating",
                "parameters": {"scale": {"type": "float", "low": 0.1, "high": 2.0}},
                "target": "user",  # Indicates this is a user-level feature
            }
        ]

        result = optimizer.optimize(
            features=features,
            n_trials=3,  # Reduced for testing
            timeout=300,  # 5-minute timeout
        )
        end_time = time.time()
        logger.info(f"Optimization finished in {end_time - start_time:.2f} seconds.")

        # Step 4: Validate results
        assert result is not None
        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        logger.info(f"Best AUC: {result.best_score}")
        logger.info(f"Best params: {result.best_params}")

        # We expect the model to do better than random guessing
        assert result.best_score > 0.5

    finally:
        # Ensure the session's database connection is always closed
        session_state.close_connection()
