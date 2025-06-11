import os

import autogen
import pytest

from src.agents.research_agent import ResearchAgent

# Load environment variables from .env file is handled by conftest.py

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
def test_research_agent_initialization():
    """Tests if the ResearchAgent and its sub-agents can be initialized."""
    agent = ResearchAgent(llm_config=llm_config)
    assert agent.analyst is not None
    assert agent.scientist is not None
    assert agent.user_proxy is not None
    assert agent.analyst.name == "Data_Analyst"
    assert agent.scientist.name == "Data_Scientist"


@pytest.mark.skipif(not llm_config, reason="LLM configuration not found.")
def test_full_eda_workflow():
    """
    Tests the full EDA workflow with a live LLM call to Gemini:
    1. Analyst generates code.
    2. User proxy executes code.
    3. Scientist interprets the results.
    """
    # 1. Setup
    agent = ResearchAgent(llm_config=llm_config)
    task = "Calculate 2 + 2 and tell me the result."

    # 2. Action
    messages = agent.run_eda_task(None, None, task)

    # 3. Assertion
    assert len(messages) > 2, "Expected at least 3 messages in the chat."

    # The final message should be from the scientist with the interpretation
    final_message = messages[-1]
    assert final_message["name"] == "Data_Scientist"

    # Check that the final message contains the correct answer.
    # LLMs can be verbose, so we check for the presence of '4'.
    assert "4" in final_message["content"]
