"""
End-to-end test for the VULCAN orchestrator.

This test validates the full pipeline execution, ensuring all agents and loops run
successfully and that logging and output artifacts are generated correctly.
"""

import os
import subprocess
import sys
import pytest
from unittest.mock import patch

from src.orchestrator import main as run_orchestrator

# Skip tests that require an OpenAI API key if it's not set
requires_openai_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Test requires OPENAI_API_KEY environment variable to be set",
)


@requires_openai_api_key
@patch("src.orchestrator.SessionState")
@patch("src.orchestrator.run_strategy_loop")
@patch("src.orchestrator.run_discovery_loop")
def test_orchestrator_mocked_run(
    mock_discovery_loop,
    mock_strategy_loop,
    MockSessionState,
):
    """
    Tests a mocked run of the orchestrator's main function.

    This test isolates the orchestrator's core logic by mocking the main agent loops,
    ensuring that session management and reporting work correctly without the overhead
    of full agent execution.
    """
    # Arrange
    mock_discovery_loop.return_value = "Mocked discovery report"
    mock_strategy_loop.return_value = {"should_continue": False}

    # Mock session state to ensure the main loop doesn't exit prematurely
    mock_session_instance = MockSessionState.return_value
    mock_session_instance.get_final_hypotheses.return_value = ["dummy_hypothesis"]

    # Act
    final_report = run_orchestrator()

    # Assert
    assert mock_discovery_loop.call_count == 1, "Discovery loop should be called once"
    assert mock_strategy_loop.call_count == 1, "Strategy loop should be called once"
    assert "VULCAN Run Complete" in final_report, "Final report title is missing"
    assert "Discovery Loop Report" in final_report, "Discovery report section is missing"
    assert "Strategy Refinement Report" in final_report, "Strategy report section is missing"


@requires_openai_api_key
def test_orchestrator_e2e_subprocess_run():
    """
    Tests a full end-to-end run of the orchestrator via a subprocess.

    This test executes the main orchestrator script using the specified conda
    environment, ensuring the full pipeline runs from start to finish. It validates
    success based on the exit code and the presence of key log messages.
    """
    # Arrange: Set up the command and environment
    project_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    command = [
        "conda",
        "run",
        "-n",
        "vulcan",
        "--no-capture-output",
        sys.executable,
        "-m",
        "src.orchestrator",
    ]

    # Act: Run the orchestrator as a subprocess
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=project_root, timeout=600
    )

    # Assert: Check for successful execution and key log messages
    assert result.returncode == 0, f"Orchestrator script failed with exit code {result.returncode}\nStderr: {result.stderr}"

    stdout = result.stdout
    assert "--- Running Insight Discovery Loop ---" in stdout
    assert "--- Running Strategy Refinement Loop ---" in stdout
    assert "VULCAN has completed its run" in stdout
