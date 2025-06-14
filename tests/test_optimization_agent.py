"""Tests for the VULCAN optimization agent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.strategy_team.optimization_agent_v2 import (
    VULCANOptimizer,
)
from src.utils.run_utils import init_run, terminate_pipeline


@pytest.fixture(autouse=True)
def run_around_tests():
    """Fixture to handle test setup and teardown with run context."""
    # Setup: Initialize run
    init_run()
    
    yield  # This is where the test runs
    
    # Teardown: Clean up
    terminate_pipeline()


def test_optimization_agent_init(tmp_path):
    """Test that the optimizer initializes correctly."""
    optimizer = VULCANOptimizer(data_dir=tmp_path, n_jobs=1, random_state=42)
    assert optimizer.data_dir == Path(tmp_path)
    assert optimizer.n_jobs == 1
    assert optimizer.random_state == 42


def test_sample_parameters():
    """Test parameter sampling logic."""
    import optuna
    
    features = [
        {
            "name": "test_feature",
            "parameters": {
                "param1": {"type": "int", "low": 1, "high": 10},
                "param2": {"type": "float", "low": 0.1, "high": 1.0},
                "param3": {"type": "categorical", "choices": ["a", "b", "c"]},
            },
        }
    ]
    
    optimizer = VULCANOptimizer(n_jobs=1, random_state=42)
    
    # Create a mock trial
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    
    # Sample parameters
    params = optimizer._sample_parameters(trial, features)
    
    # Check that all parameters were sampled
    assert "test_feature__param1" in params
    assert "test_feature__param2" in params
    assert "test_feature__param3" in params
    
    # Check parameter types and ranges
    assert isinstance(params["test_feature__param1"], int)
    assert 1 <= params["test_feature__param1"] <= 10
    
    assert isinstance(params["test_feature__param2"], float)
    assert 0.1 <= params["test_feature__param2"] <= 1.0
    
    assert params["test_feature__param3"] in ["a", "b", "c"]


def test_optimization_smoke_test(tmp_path, monkeypatch):
    """Smoke test for the optimization process with mock data."""
    # Create a simple mock dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create a simple CV split
    splits_dir = data_dir / "splits"
    splits_dir.mkdir()
    
    # Create a simple CV split with 2 folds
    cv_folds = [
        {
            "train": ["user1", "user2"],
            "validation": ["user3", "user4"],
        },
        {
            "train": ["user3", "user4"],
            "validation": ["user1", "user2"],
        },
    ]
    
    with open(splits_dir / "cv_folds.json", "w") as f:
        json.dump(cv_folds, f)
    
    # Create mock review data
    reviews_dir = data_dir / "curated_reviews_partitioned"
    reviews_dir.mkdir()
    
    # Create a simple review dataset
    np.random.seed(42)
    users = ["user1", "user2", "user3", "user4"]
    items = [f"item{i}" for i in range(10)]
    
    reviews = []
    for user in users:
        user_items = np.random.choice(items, size=5, replace=False)
        for item in user_items:
            reviews.append({
                "user_id": user,
                "item_id": item,
                "rating": np.random.randint(1, 6),
            })
    
    # Save as parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pa.Table.from_pandas(pd.DataFrame(reviews))
    pq.write_table(table, reviews_dir / "part-0.parquet")
    
    # Define test features
    features = [
        {
            "name": "test_feature",
            "parameters": {
                "param1": {"type": "float", "low": 0.1, "high": 1.0},
            },
        }
    ]
    
    # Mock the evaluation to make the test faster
    def mock_evaluate_fold(*args, **kwargs):
        return {"fold_idx": 0, "val_score": 0.8, "params": {"test_feature__param1": 0.5}}
    
    # Run optimization with a small number of trials
    optimizer = VULCANOptimizer(data_dir=data_dir, n_jobs=1, random_state=42)
    
    # Patch the evaluation method
    original_eval = optimizer._evaluate_fold
    optimizer._evaluate_fold = mock_evaluate_fold
    
    try:
        result = optimizer.optimize(
            features=features,
            n_trials=2,
            use_fast_mode=True,
        )
        
        # Check that we got a result
        assert result is not None
        assert "test_feature__param1" in result.best_params
        assert 0.0 <= result.best_score <= 1.0
        
    finally:
        # Restore original method
        optimizer._evaluate_fold = original_eval


if __name__ == "__main__":
    pytest.main(["-x", __file__])
