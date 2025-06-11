import json
import os

import pytest

from src.agents.optimization_agent import OptimizationAgent
from src.utils.memory import load_memory
from src.utils.testing_utils import assert_json_schema


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure memory file is clean before each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")

    # Create a dummy realized function for the agent to consume
    realized_functions = {
        "code": [
            {
                "name": "DummyFeature",
                "param_names": ["ParamA", "ParamB"],
                "status": "valid",
                "error": "",
            }
        ],
        "llm": [],
    }

    memory = load_memory()
    memory["realized_functions"] = realized_functions
    with open("project_memory.json", "w") as f:
        json.dump(memory, f, indent=2)

    yield

    # Teardown: clean up memory file after each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")


def test_optimization_agent_smoke_test():
    """
    A smoke test for the OptimizationAgent.
    Checks:
    1. The agent runs without errors.
    2. `bo_history` is created in memory.
    3. The history is valid against the schema.
    4. At least one trial is recorded.
    """
    # 1. Setup
    agent = OptimizationAgent()

    # 2. Action
    agent.run_optimization(n_calls=10)  # Run for 10 trials to satisfy skopt

    # 3. Assertions
    memory = load_memory()
    bo_history = memory.get("bo_history")

    assert bo_history is not None, "`bo_history` should be in memory."

    # Check schema validity
    assert_json_schema(bo_history, "tests/schemas/bo_history_schema.json")

    # Check that trials were recorded
    assert len(bo_history["trials"]) == 10, "Expected 10 trials to be recorded."
    assert "best_rmse" in bo_history
    assert bo_history["best_rmse"] is not None
