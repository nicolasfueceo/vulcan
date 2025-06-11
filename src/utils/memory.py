import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.run_utils import get_current_run, get_run_memory_file


def _init_memory():
    """Initializes the memory file if it doesn't exist."""
    memory_file = get_run_memory_file()
    if not memory_file.exists():
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        initial_memory = {
            "current_run_id": get_current_run(),
            "started_at": datetime.utcnow().isoformat(),
            "hypotheses": [],
            "candidate_feature_runs": [],  # List of feature proposal runs with timestamps
            "realized_functions": {},  # Map of feature name to function metadata
            "bo_history": {},  # Bayesian optimization history
            "best_params": {},  # Best parameters found
            "reflections": [],  # List of reflection insights
            "eda_runs": [],  # List of EDA runs with timestamps
            "version": "1.0",  # Schema version
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(memory_file, "w") as f:
            json.dump(initial_memory, f, indent=2)


def get_mem(key: str, default: Any = None) -> Any:
    """Get a value from memory."""
    _init_memory()
    try:
        with open(get_run_memory_file(), "r") as f:
            memory = json.load(f)
            return memory.get(key, default)
    except Exception as e:
        print(f"Error reading memory: {e}")
        return default


def set_mem(key: str, value: Any) -> None:
    """Set a value in memory."""
    _init_memory()
    try:
        with open(get_run_memory_file(), "r") as f:
            memory = json.load(f)

        memory[key] = value
        memory["last_updated"] = datetime.utcnow().isoformat()

        with open(get_run_memory_file(), "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Error writing to memory: {e}")


def append_to_list(key: str, value: Any) -> None:
    """Append a value to a list in memory."""
    _init_memory()
    try:
        with open(get_run_memory_file(), "r") as f:
            memory = json.load(f)

        if key not in memory:
            memory[key] = []
        memory[key].append(value)
        memory["last_updated"] = datetime.utcnow().isoformat()

        with open(get_run_memory_file(), "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Error appending to memory: {e}")


def get_latest_run(key: str) -> Optional[Dict]:
    """Get the latest run from a list of runs."""
    runs = get_mem(key, [])
    if not runs:
        return None
    return sorted(runs, key=lambda x: x.get("timestamp", ""))[-1]


def add_feature_proposal_run(features: List[Dict]) -> None:
    """Add a new feature proposal run with timestamp."""
    run = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": get_current_run(),
        "features": features,
    }
    append_to_list("candidate_feature_runs", run)


def add_eda_run(eda_results: Dict) -> None:
    """Add a new EDA run with timestamp."""
    run = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": get_current_run(),
        "results": eda_results,
    }
    append_to_list("eda_runs", run)


def get_eda_delta() -> Dict:
    """Get the delta between the last two EDA runs."""
    runs = get_mem("eda_runs", [])
    if len(runs) < 2:
        return runs[-1]["results"] if runs else {}

    current = runs[-1]["results"]
    previous = runs[-2]["results"]

    # Compute delta (new insights only)
    delta = {}
    for key, value in current.items():
        if key not in previous or previous[key] != value:
            delta[key] = value

    return delta


def persist_realized_functions(functions: Dict) -> None:
    """Persist realized functions to memory."""
    # Add run context to each function
    for func_name, func_data in functions.items():
        func_data["run_id"] = get_current_run()
        func_data["realized_at"] = datetime.utcnow().isoformat()

    set_mem("realized_functions", functions)
