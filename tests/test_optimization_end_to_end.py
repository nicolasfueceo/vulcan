"""
End-to-end test for the VULCAN optimization agent with real book data.
"""

import inspect
import time
import pandas as pd
import pytest
from loguru import logger

from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.data.cv_data_manager import CVDataManager
from src.utils.feature_registry import feature_registry
from src.utils.run_utils import init_run, terminate_pipeline
from src.utils.session_state import SessionState

# Configure logging
logger.add("logs/test_optimization_end_to_end.log", rotation="10 MB")


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_module():
    """Setup and teardown for the entire test module."""
    # Setup: Any setup code needed before all tests in this module run
    yield  # This is where the test runs

    # Teardown: Clean up
    terminate_pipeline()


def test_vulcan_orchestrator_end_to_end():
    """
    True end-to-end test of the VULCAN pipeline: runs orchestrator.main() for one epoch with LLM calls (no mocks).
    Checks that the pipeline completes and produces a report.
    Adds extra logging and prints progress to stdout for visibility.
    """
    print("\n\n[TEST] Starting VULCAN orchestrator end-to-end test (real LLM calls, full pipeline, FAST MODE ENABLED)...\n\n")
    from src.orchestrator import main
    import logging
    logging.getLogger("loguru").setLevel(logging.INFO)
    # Activate fast mode with 10% sample
    fast_mode_frac = 0.1
    report = main(epochs=1, fast_mode_frac=fast_mode_frac)
    print(f"\n[TEST] VULCAN orchestrator run complete. Report snippet:\n{report[:600]}")
    assert "VULCAN Run Complete" in report
    assert "Epoch Reports" in report
    assert "Final Strategy Refinement Report" in report
    print("\n[TEST] Full VULCAN run report (truncated):\n" + report[:2000])
