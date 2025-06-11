import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.utils.db_api import ingest_sqlite_to_duckdb
from src.utils.memory import get_mem, set_mem


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure db file is clean before each test
    ingest_sqlite_to_duckdb("tests/dummy_goodreads.db")
    set_mem("eda", {})
    set_mem("hypotheses", [])
    set_mem("prioritized_hypotheses", [])
    yield


def test_reasoning_agent_smoke():
    """
    Smoke test for the ReasoningAgent.
    """
    # 1. Setup
    llm_config = {"config_list": [{"model": "stub"}]}
    DataAnalysisAgent(llm_config=llm_config).run()
    HypothesisAgent(llm_config=llm_config).run()
    reasoning_agent = ReasoningAgent(llm_config=llm_config)

    # 2. Action
    reasoning_agent.run()

    # 3. Assertions
    prioritized_hypotheses = get_mem("prioritized_hypotheses")
    assert prioritized_hypotheses is not None
    assert isinstance(prioritized_hypotheses, list)
    assert len(prioritized_hypotheses) > 0
    assert "id" in prioritized_hypotheses[0]
    assert "priority" in prioritized_hypotheses[0]
    assert "feasibility" in prioritized_hypotheses[0]
