import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.agents.feature_ideation_agent import FeatureIdeationAgent
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
    yield


def test_feature_ideation_agent_smoke():
    """
    Smoke test for the FeatureIdeationAgent.
    """
    # 1. Setup
    llm_config = {"config_list": [{"model": "stub"}]}
    DataAnalysisAgent(llm_config=llm_config).run()
    HypothesisAgent(llm_config=llm_config).run()
    ReasoningAgent(llm_config=llm_config).run()
    feature_ideation_agent = FeatureIdeationAgent(llm_config=llm_config)

    # 2. Action
    feature_ideation_agent.run()

    # 3. Assertions
    candidate_features = get_mem("candidate_features")
    assert candidate_features is not None
    assert isinstance(candidate_features, list)
    assert len(candidate_features) > 0
    assert "name" in candidate_features[0]
    assert "type" in candidate_features[0]
    assert "spec" in candidate_features[0]
