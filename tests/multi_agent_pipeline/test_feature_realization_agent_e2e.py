import os

import autogen
import pytest

from src.agents.feature_realization_agent import FeatureRealizationAgent
from src.utils.memory import append_to_memory, load_memory
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
@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure memory file is clean before each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")

    # Create a dummy feature proposal for the agent to consume
    pass2_proposals = {
        "timestamp": "2025-06-11T11:00:00Z",
        "pass": 2,
        "proposals": [
            {
                "name": "GrimdarkMentionCount",
                "type": "code",
                "dsl": "COUNT(review_text contains 'grimdark')",
                "chain_of_thought": "...",
                "rationale": "...",
            },
            {
                "name": "InvalidDSLFeature",
                "type": "code",
                "dsl": "INVALID_SYNTAX(foo bar)",
                "chain_of_thought": "...",
                "rationale": "...",
            },
            {
                "name": "DarknessAffinityScore",
                "type": "llm",
                "prompt": "Given the last 10 reviews: <USER_REVIEWS>, rate from 0 to 1 how much the user prefers dark fantasy.",
                "chain_of_thought": "...",
                "rationale": "...",
            },
        ],
    }
    append_to_memory("feature_proposals", pass2_proposals)

    yield

    # Teardown: clean up memory file after each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
def test_feature_realization_agent():
    """
    Tests that the FeatureRealizationAgent can:
    1. Consume proposals from memory.
    2. Generate valid code for a correct DSL.
    3. Identify and flag an invalid DSL.
    4. Create a wrapper for an LLM feature.
    5. Update the memory with the results, which are valid against the schema.
    """
    # 1. Setup
    agent = FeatureRealizationAgent(llm_config=llm_config)

    # 2. Action
    agent.run_realization()

    # 3. Assertions
    memory = load_memory()
    realized_funcs = memory.get("realized_functions")

    assert realized_funcs is not None, "realized_functions should be in memory."

    # Check schema validity
    assert_json_schema(realized_funcs, "tests/schemas/realized_functions_schema.json")

    # Check code functions
    code_funcs = realized_funcs["code"]
    assert len(code_funcs) == 2, "Expected two code functions to be processed."

    valid_func = next(
        (f for f in code_funcs if f["name"] == "GrimdarkMentionCount"), None
    )
    assert valid_func is not None
    assert valid_func["status"] == "valid"

    invalid_func = next(
        (f for f in code_funcs if f["name"] == "InvalidDSLFeature"), None
    )
    assert invalid_func is not None
    assert invalid_func["status"] == "error"
    assert "Generated code failed to execute" in invalid_func["error"]

    # Check LLM functions
    llm_funcs = realized_funcs["llm"]
    assert len(llm_funcs) == 1, "Expected one LLM function to be processed."

    llm_func = llm_funcs[0]
    assert llm_func["name"] == "DarknessAffinityScore"
    assert llm_func["status"] == "valid"
