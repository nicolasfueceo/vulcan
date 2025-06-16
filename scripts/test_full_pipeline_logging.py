import json
import os
import pytest
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import main as run_orchestrator
from src.utils.run_utils import get_run_dir

def test_full_pipeline_end_to_end_logging():
    """
    Tests that a full run of the orchestrator produces a non-empty tool_calls.jsonl file
    with entries from both the discovery and strategy loops.
    """
    # 1. Run the full pipeline for one epoch in fast mode
    try:
        run_orchestrator(epochs=1, fast_mode_frac=0.01)
    except Exception as e:
        pytest.fail(f"Orchestrator main function failed: {e}")

    # 2. Get the path to the most recent run directory
    run_dir = get_run_dir(specific_run_id="latest")
    assert run_dir is not None, "Could not find the latest run directory."

    # 3. Verify the tool_calls.jsonl file
    log_file = Path(run_dir) / "logs" / "tool_calls.jsonl"

    assert log_file.exists(), f"Log file not found at {log_file}"

    with open(log_file, 'r') as f:
        log_entries = [json.loads(line) for line in f]

    assert len(log_entries) > 0, "The tool log file is empty."

    # 4. Check for key tool calls from both loops
    tool_names_logged = {entry['tool_name'] for entry in log_entries}

    # Tools expected from the discovery loop
    discovery_tools = {'run_sql_query', 'execute_python', 'finalize_hypotheses'}
    # Tools expected from the strategy loop
    strategy_tools = {'save_candidate_features'}

    found_discovery_tool = any(tool in tool_names_logged for tool in discovery_tools)
    found_strategy_tool = any(tool in tool_names_logged for tool in strategy_tools)

    assert found_discovery_tool, f"No discovery tool calls (e.g., {discovery_tools}) were found in the log."
    assert found_strategy_tool, f"No strategy tool calls (e.g., {strategy_tools}) were found in the log."

    print(f"Successfully verified {len(log_entries)} tool calls in {log_file}")
    print(f"Logged tools found: {sorted(list(tool_names_logged))}")

if __name__ == "__main__":
    pytest.main([__file__])
