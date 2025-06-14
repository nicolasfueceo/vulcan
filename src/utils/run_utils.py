#!/usr/bin/env python3
"""
Utilities for managing run IDs and run-specific paths.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.cv_data_manager import CVDataManager

# Base directories
RUNTIME_DIR = Path("runtime")
RUNS_DIR = RUNTIME_DIR / "runs"

# Global variable to store current run ID
_run_id: Optional[str] = None
_run_dir: Optional[Path] = None

logger = logging.getLogger(__name__)


def init_run() -> Tuple[str, Path]:
    """
    Initializes a new run, setting a unique run ID and creating run-specific directories.
    This function should be called once at the start of a pipeline run.
    """
    global _run_id, _run_dir
    if _run_id:
        raise RuntimeError(f"Run has already been initialized with RUN_ID: {_run_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    _run_id = f"run_{timestamp}_{unique_id}"

    runtime_path = Path(__file__).resolve().parent.parent.parent / "runtime" / "runs"
    _run_dir = runtime_path / _run_id

    # Create all necessary subdirectories for the run
    (_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (_run_dir / "data").mkdir(parents=True, exist_ok=True)
    (_run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (_run_dir / "generated_code").mkdir(parents=True, exist_ok=True)

    return _run_id, _run_dir


def get_run_id() -> str:
    """Returns the unique identifier for the current run."""
    if _run_id is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_id


def get_run_dir() -> Path:
    """Returns the absolute path to the directory for the current run."""
    if _run_dir is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_dir


def get_run_artifact_path(*path_parts: str) -> Path:
    """Constructs an absolute path for an artifact within the current run's directory."""
    return get_run_dir().joinpath(*path_parts)


def get_run_logs_dir() -> Path:
    """Get the logs directory for the current run."""
    return get_run_dir() / "logs"


def get_run_tensorboard_dir() -> Path:
    """Get the TensorBoard directory for the current run."""
    return get_run_dir() / "tensorboard"


def get_run_generated_code_dir() -> Path:
    """Get the generated code directory for the current run."""
    return get_run_dir() / "generated_code"


def get_run_memory_file() -> Path:
    """Get the memory file path for the current run."""
    return get_run_dir() / "memory.json"


def get_run_database_file() -> Path:
    """Get the database file path for the current run."""
    return get_run_dir() / "database.json"


def get_run_log_file() -> Path:
    """Get the log file for the current run."""
    return get_run_logs_dir() / f"pipeline_{get_run_id()}.log"


def get_run_db_file() -> Path:
    """Get the database file for the current run."""
    return get_run_dir() / f"data_{get_run_id()}.duckdb"


def get_feature_code_path(feature_name: str) -> Path:
    """Get the path for a realized feature's code file."""
    return get_run_generated_code_dir() / f"{feature_name}.py"


def get_tensorboard_writer(agent_name: str):
    """Get a TensorBoard writer for the current run and agent."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(get_run_tensorboard_dir() / agent_name))


def format_log_message(message: str) -> str:
    """Format a log message with run context."""
    return f"[{get_run_id()}] {message}"


def config_list_from_json(file_path: str) -> List[Dict]:
    """Load OpenAI config list from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config list from {file_path}: {e}")
        return []


def restart_pipeline(config: Dict[str, Any] = None) -> None:
    """
    Restarts the pipeline with an optional configuration update.
    This function should be called by the ReflectionAgent when deciding to continue.

    Args:
        config: Optional dictionary of configuration parameters for the next run
    """
    global _run_id, _run_dir

    # Save current run ID
    old_run_id = _run_id

    # Initialize a new run
    new_run_id, new_run_dir = init_run()

    # If config is provided, save it
    if config:
        config_path = new_run_dir / "config" / "next_cycle_config.json"
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    logger.info(f"Pipeline restarted. Old run: {old_run_id}, New run: {new_run_id}")
    return new_run_id, new_run_dir


def terminate_pipeline() -> None:
    """
    Terminates the pipeline gracefully.
    This function should be called by the ReflectionAgent when deciding to stop.
    """
    # Close any open database connections
    CVDataManager.close_global_connection_pool()

    global _run_id, _run_dir

    if _run_id:
        logger.info(f"Pipeline terminated. Final run: {_run_id}")

        # Create a termination marker file
        termination_file = _run_dir / "pipeline_terminated.txt"
        with open(termination_file, "w", encoding="utf-8") as f:
            f.write(f"Pipeline terminated at {datetime.now().isoformat()}\n")

        # Reset global variables
        _run_id = None
        _run_dir = None
    else:
        logger.warning("Attempted to terminate pipeline but no run was active.")
