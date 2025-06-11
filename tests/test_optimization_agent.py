import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.agents.feature_ideation_agent import FeatureIdeationAgent
from src.agents.feature_realization_agent import FeatureRealizationAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.optimization_agent import OptimizationAgent
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
    set_mem("best_params", {})
    set_mem("best_rmse", 0.0)
    yield


def test_optimization_agent_smoke():
    """
    Smoke test for the OptimizationAgent.
    """
    # 1. Setup
    llm_config = {"config_list": [{"model": "stub"}]}
    DataAnalysisAgent(llm_config=llm_config).run()
    HypothesisAgent(llm_config=llm_config).run()
    ReasoningAgent(llm_config=llm_config).run()
    FeatureIdeationAgent(llm_config=llm_config).run()
    FeatureRealizationAgent(llm_config=llm_config).run()
    optimization_agent = OptimizationAgent()

    # 2. Action
    optimization_agent.run()

    # 3. Assertions
    best_params = get_mem("best_params")
    best_rmse = get_mem("best_rmse")
    assert best_params is not None
    assert best_rmse is not None
    assert isinstance(best_params, dict)
    assert isinstance(best_rmse, float)
    # This is a weak assertion, as the optimization will likely not find a great score.
    # A more robust test would check if the RMSE is within a reasonable range.
