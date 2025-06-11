import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb

from src.utils.run_utils import get_run_dir, get_run_id

# This is the main, raw data source, not a run-specific database.
DB_PATH = "data/goodreads.duckdb"


def get_run_database_file() -> Path:
    """Gets the path to the run-specific JSON database file."""
    return get_run_dir() / "database.json"


def _init_database():
    """Initializes the JSON database file for the run if it doesn't exist."""
    db_file = get_run_database_file()
    if not db_file.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)
        initial_db = {
            "run_id": get_run_id(),
            "started_at": datetime.utcnow().isoformat(),
            "features": {},
            "metrics": {},
            "models": {},
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(db_file, "w") as f:
            json.dump(initial_db, f, indent=2)


def get_db(key: str, default: Any = None) -> Any:
    """Get a value from the run's JSON database."""
    _init_database()
    try:
        with open(get_run_database_file(), "r") as f:
            db = json.load(f)
            return db.get(key, default)
    except Exception as e:
        print(f"Error reading database: {e}")
        return default


def set_db(key: str, value: Any) -> None:
    """Set a value in the run's JSON database."""
    _init_database()
    try:
        with open(get_run_database_file(), "r") as f:
            db = json.load(f)

        db[key] = value
        db["last_updated"] = datetime.utcnow().isoformat()

        with open(get_run_database_file(), "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Error writing to database: {e}")


def store_feature(feature_name: str, feature_data: Dict) -> None:
    """Store feature data in the run's database."""
    features = get_db("features", {})
    feature_data["run_id"] = get_run_id()
    feature_data["stored_at"] = datetime.utcnow().isoformat()
    features[feature_name] = feature_data
    set_db("features", features)


def store_metric(metric_name: str, metric_data: Dict) -> None:
    """Store metric data in the run's database."""
    metrics = get_db("metrics", {})
    metric_data["run_id"] = get_run_id()
    metric_data["stored_at"] = datetime.utcnow().isoformat()
    metrics[metric_name] = metric_data
    set_db("metrics", metrics)


def store_model(model_name: str, model_data: Dict) -> None:
    """Store model data in the run's database."""
    models = get_db("models", {})
    model_data["run_id"] = get_run_id()
    model_data["stored_at"] = datetime.utcnow().isoformat()
    models[model_name] = model_data
    set_db("models", models)


def get_feature(feature_name: str) -> Optional[Dict]:
    """Get feature data from the run's database."""
    features = get_db("features", {})
    return features.get(feature_name)


def get_metric(metric_name: str) -> Optional[Dict]:
    """Get metric data from the run's database."""
    metrics = get_db("metrics", {})
    return metrics.get(metric_name)


def get_model(model_name: str) -> Optional[Dict]:
    """Get model data from the run's database."""
    models = get_db("models", {})
    return models.get(model_name)


def get_db_connection():
    """
    Returns a connection to the main DuckDB data warehouse.
    """
    return duckdb.connect(DB_PATH, read_only=False)
