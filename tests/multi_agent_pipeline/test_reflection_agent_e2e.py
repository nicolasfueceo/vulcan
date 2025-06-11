
import json
import os

import autogen
import pytest

from src.agents.reflection_agent import ReflectionAgent
from src.utils.memory import load_memory
from src.utils.testing_utils import assert_json_schema

# Load configuration for the Gemini model
api_key = os.environ.get("GOOGLE_API_KEY")
model = "gemini-1.5-pro-latest"

# Skip all tests in this file if the API key is not available
if not api_key:
    pytest.skip("GOOGLE_API_KEY not found, skipping tests.", allow_module_level=True)

config_list = autogen.config_list_from_json(
    "config/OAI_CONFIG_LIST.json",
    filter_dict={"model": [model]},
)
llm_config = config_list[0] if config_list else None
if llm_config:
    llm_config["api_key"] = api_key


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
class MockFMModel:
    """A mock factorization machine model for testing."""

    def __init__(self):
        self.linear_weights = {
            "DummyFeature_ParamA": 5.0,
            "DummyFeature_ParamB": -3.0,
            "LowValueFeature_scale": 0.001,
        }
        self.interaction_weights = {
            ("DummyFeature_ParamA", "DummyFeature_ParamB"): 0.2,
            ("DummyFeature_ParamA", "K"): -0.1,
        }


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure memory file is clean before each test
    if os.path.exists("runtime"):
        import shutil

        shutil.rmtree("runtime")
    os.makedirs("runtime/generated_code", exist_ok=True)

    # Create a dummy BO history
    bo_history = {
        "trials": [
            {
                "timestamp": "2025-06-12T10:00:00Z",
                "params": {"DummyFeature_ParamA": 10.0, "fm_dim": 16},
                "rmse": 0.9,
            },
            {
                "timestamp": "2025-06-12T10:01:00Z",
                "params": {"DummyFeature_ParamA": 9.8, "fm_dim": 32},
                "rmse": 0.88,
            },
        ],
        "best_params": {"DummyFeature_ParamA": 9.8, "fm_dim": 32},
        "best_rmse": 0.88,
    }

    memory = load_memory()
    memory["bo_history"] = bo_history
    with open("project_memory.json", "w") as f:
        json.dump(memory, f, indent=2)

    yield

    # Teardown: clean up memory file after each test
    if os.path.exists("runtime"):
        import shutil

        shutil.rmtree("runtime")


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
def test_reflection_agent():
    """
    Tests that the ReflectionAgent can:
    1. Consume optimization history from memory.
    2. Analyze the results and a mock model.
    3. Generate a valid reflection note.
    4. Correctly identify top and low-value features.
    """
    # 1. Setup
    agent = ReflectionAgent(llm_config=llm_config)
    mock_model = MockFMModel()

    # 2. Action
    agent.run_reflection(fm_model=mock_model)

    # 3. Assertions
    memory = load_memory()
    reflections = memory.get("reflections", [])

    assert len(reflections) == 1, "Expected one reflection to be added."

    note = reflections[0]
    assert_json_schema(note, "tests/schemas/reflection_schema.json")

    # Check content of the reflection
    assert len(note["top_features"]) > 0
    assert "DummyFeature_ParamA" in [f["feature"] for f in note["top_features"]]

    assert len(note["low_value_features"]) > 0
    assert "LowValueFeature_scale" in [f["feature"] for f in note["low_value_features"]]

    assert len(note["bo_insights"]) > 0
    assert any("always at" in s for s in note["bo_insights"])

    assert len(note["next_steps"]) >= 3
