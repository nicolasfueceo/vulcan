import os

import autogen
import pytest

from src.agents.feature_ideation_agent import FeatureIdeationAgent
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
    if os.path.exists("runtime"):
        import shutil

        shutil.rmtree("runtime")
    os.makedirs("runtime/generated_code", exist_ok=True)

    # Create some dummy hypotheses for the agent to consume
    append_to_memory(
        "hypotheses",
        {
            "timestamp": "2025-06-11T10:00:00Z",
            "hypothesis": "Users who mention 'grimdark' have a preference for darker fantasy.",
            "priority": 4,
            "notes": "High signal observed in EDA.",
        },
    )
    append_to_memory(
        "hypotheses",
        {
            "timestamp": "2025-06-11T10:01:00Z",
            "hypothesis": "Review length is correlated with user engagement.",
            "priority": 3,
            "notes": "Easy to compute, potentially useful.",
        },
    )

    yield

    # Teardown: clean up memory file after each test
    if os.path.exists("runtime"):
        import shutil

        shutil.rmtree("runtime")


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
def test_feature_ideation_pass_1():
    """
    Tests that the FeatureIdeationAgent can successfully complete Pass 1.
    Checks:
    1. A new entry is added to "feature_proposals" in memory.
    2. The entry has pass=1.
    3. The proposals are valid against the schema.
    4. At least one code and one llm proposal are generated.
    """
    # 1. Setup
    agent = FeatureIdeationAgent(llm_config=llm_config)

    # 2. Action
    agent.run_ideation_pass(pass_number=1)

    # 3. Assertions
    memory = load_memory()
    proposals = memory.get("feature_proposals", [])

    assert len(proposals) == 1, "Expected one feature proposal entry to be added."

    pass1_result = proposals[0]
    assert pass1_result["pass"] == 1

    assert_json_schema(pass1_result, "tests/schemas/feature_proposal_schema.json")

    proposal_types = [p["type"] for p in pass1_result["proposals"]]
    assert "code" in proposal_types
    assert "llm" in proposal_types


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
def test_feature_ideation_pass_2():
    """
    Tests that the FeatureIdeationAgent can successfully complete Pass 2.
    Checks:
    1. A new entry with pass=2 is added.
    2. Proposals include "expected_effort" and "expected_impact".
    """
    # 1. Setup: Run pass 1 first to generate initial proposals
    agent = FeatureIdeationAgent(llm_config=llm_config)
    agent.run_ideation_pass(pass_number=1)

    # 2. Action
    agent.run_ideation_pass(pass_number=2)

    # 3. Assertions
    memory = load_memory()
    proposals = memory.get("feature_proposals", [])

    assert len(proposals) == 2, "Expected two feature proposal entries."

    pass2_result = proposals[1]
    assert pass2_result["pass"] == 2
    assert_json_schema(pass2_result, "tests/schemas/feature_proposal_schema.json")

    # Check that the proposals have the new fields
    for proposal in pass2_result["proposals"]:
        assert "expected_effort" in proposal
        assert "expected_impact" in proposal
        assert "notes" in proposal
