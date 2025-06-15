"""Pytest configuration and fixtures for VULCAN tests."""
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Load environment variables from .env file before any tests run."""
    load_dotenv()


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create and return a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_db_path(test_data_dir: Path) -> Path:
    """Create and return a path to a test database."""
    return test_data_dir / "test_db.duckdb"


@pytest.fixture(scope="session")
def create_test_database(test_db_path: Path) -> Path:
    """Create a test database with sample data.
    
    Returns:
        Path to the created database file
    """
    # Create sample data
    users = pd.DataFrame({
        "user_id": [f"user{i}" for i in range(1, 11)],
        "age": [25, 30, 35, 40, 45, 22, 28, 33, 38, 42],
    })
    
    items = pd.DataFrame({
        "item_id": [f"item{i}" for i in range(1, 21)],
        "popularity": range(1, 21),
    })
    
    interactions = []
    for user_idx, user_id in enumerate(users["user_id"]):
        # Each user interacts with 5 items
        for item_idx in range(1, 6):
            item_id = f"item{(user_idx * 2 + item_idx) % 20 + 1}"  # Distribute items
            interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "rating": (user_idx + item_idx) % 5 + 1,  # Rating 1-5
                "timestamp": 1000 + user_idx * 10 + item_idx,
            })
    
    pd.DataFrame(interactions)
    
    # Create database and load data
    conn = duckdb.connect(str(test_db_path))
    
    # Register DataFrames to DuckDB connection
    conn.register("users", users)
    conn.register("items", items)
    conn.register("interactions_df", pd.DataFrame(interactions))

    # Create tables
    conn.execute("CREATE TABLE users AS SELECT * FROM users")
    conn.execute("CREATE TABLE items AS SELECT * FROM items")
    conn.execute("CREATE TABLE interactions AS SELECT * FROM interactions_df")
    
    # Create CV splits table
    cv_splits = [
        {"fold_idx": 0, "user_id": f"user{i+1}", "split": "train" if i < 8 else "test"}
        for i in range(10)
    ]
    cv_splits_df = pd.DataFrame(cv_splits)
    conn.register("cv_splits_df", cv_splits_df)
    conn.execute("CREATE TABLE cv_splits AS SELECT * FROM cv_splits_df")
    
    conn.close()
    return test_db_path


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, test_data_dir: Path, create_test_database: Path):
    """Set up the test environment."""
    # Set environment variables
    monkeypatch.setenv("VULCAN_DATA_DIR", str(test_data_dir))
    
    # Create necessary directories
    (test_data_dir / "splits").mkdir(exist_ok=True)
    
    # Create a simple CV split file
    cv_splits = {
        "folds": [
            {
                "train": [f"user{i+1}" for i in range(8)],
                "validation": ["user9", "user10"],
            }
        ]
    }
    
    with open(test_data_dir / "splits" / "cv_folds.json", "w") as f:
        import json
        json.dump(cv_splits, f)
    
    yield  # Test runs here
    
    # Cleanup (handled by tmp_path_factory)
