import os
from unittest.mock import MagicMock

import pytest

from src.orchestrator import Orchestrator


# Mock the agents to avoid live LLM calls
@pytest.fixture
def mock_agents(monkeypatch):
    monkeypatch.setattr("src.orchestrator.ResearchAgent", MagicMock())
    monkeypatch.setattr("src.orchestrator.FeatureIdeationAgent", MagicMock())
    monkeypatch.setattr("src.orchestrator.FeatureRealizationAgent", MagicMock())
    monkeypatch.setattr("src.orchestrator.OptimizationAgent", MagicMock())
    monkeypatch.setattr("src.orchestrator.ReflectionAgent", MagicMock())


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure memory file is clean before each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")
    yield
    # Teardown: clean up memory file after each test
    if os.path.exists("project_memory.json"):
        os.remove("project_memory.json")


def test_orchestrator_smoke_test(mock_agents):
    """
    A smoke test for the Orchestrator.
    Checks that the orchestrator can run one full cycle without errors.
    """
    # 1. Setup
    # The llm_config can be a dummy for this test as the agents are mocked
    dummy_llm_config = {"model": "dummy", "api_key": "dummy"}
    orchestrator = Orchestrator(dummy_llm_config)

    # 2. Action
    # We are calling the internal method directly for this test
    try:
        orchestrator.run_full_cycle()
    except Exception as e:
        pytest.fail(f"Orchestrator smoke test failed with an exception: {e}")

    # 3. Assertion
    # The main assertion is that the code runs without errors.
    pass
