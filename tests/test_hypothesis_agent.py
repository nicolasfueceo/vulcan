import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.utils.db_api import ingest_sqlite_to_duckdb
from src.utils.memory import get_mem, set_mem


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure db file is clean before each test
    ingest_sqlite_to_duckdb("tests/dummy_goodreads.db")
    set_mem("eda", {})  # Clear memory
    set_mem("hypotheses", [])
    yield


def test_hypothesis_agent_smoke():
    """
    Smoke test for the HypothesisAgent.
    """
    # 1. Setup
    llm_config = {"config_list": [{"model": "stub"}]}  # No LLM call in this test
    data_analysis_agent = DataAnalysisAgent(llm_config=llm_config)
    hypothesis_agent = HypothesisAgent(llm_config=llm_config)

    # 2. Action
    data_analysis_agent.run()
    hypothesis_agent.run()

    # 3. Assertions
    hypotheses = get_mem("hypotheses")
    assert hypotheses is not None
    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0
    assert "id" in hypotheses[0]
    assert "text" in hypotheses[0]
