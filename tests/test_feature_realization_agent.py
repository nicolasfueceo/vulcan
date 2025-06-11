import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.agents.feature_ideation_agent import FeatureIdeationAgent
from src.agents.feature_realization_agent import FeatureRealizationAgent
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
    set_mem("candidate_features", [])
    set_mem("realized_code_fns", {})
    set_mem("realized_llm_fns", {})
    yield


def test_feature_realization_agent_smoke():
    """
    Smoke test for the FeatureRealizationAgent.
    """
    # 1. Setup
    llm_config = {"config_list": [{"model": "stub"}]}
    DataAnalysisAgent(llm_config=llm_config).run()
    HypothesisAgent(llm_config=llm_config).run()
    ReasoningAgent(llm_config=llm_config).run()
    FeatureIdeationAgent(llm_config=llm_config).run()
    feature_realization_agent = FeatureRealizationAgent(llm_config=llm_config)

    # 2. Action
    feature_realization_agent.run()

    # 3. Assertions
    realized_code_fns = get_mem("realized_code_fns")
    realized_llm_fns = get_mem("realized_llm_fns")
    assert realized_code_fns is not None
    assert realized_llm_fns is not None
    assert isinstance(realized_code_fns, dict)
    assert isinstance(realized_llm_fns, dict)
    # This is a weak assertion, as the dummy LLM call will likely not produce any features.
    # A more robust test would mock the LLM call to return some dummy features.
