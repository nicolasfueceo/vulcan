import pytest
import tempfile
import shutil
from pathlib import Path
from src.utils.session_state import SessionState
from src.utils.tools import get_add_to_central_memory_tool
import json

def make_fresh_session_state(tmp_path):
    # Use a fresh run directory for each test
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return SessionState(run_dir=run_dir)

def test_add_to_central_memory_basic(tmp_path):
    state = make_fresh_session_state(tmp_path)
    add_to_central_memory = get_add_to_central_memory_tool(state)
    note = "Test note"
    reasoning = "Test reasoning"
    agent = "UnitTestAgent"
    result = add_to_central_memory(note=note, reasoning=reasoning, agent=agent)
    assert "Note added to central memory" in result
    assert len(state.central_memory) == 1
    entry = state.central_memory[0]
    assert entry["note"] == note
    assert entry["reasoning"] == reasoning
    assert entry["agent"] == agent
    assert "timestamp" in entry
    # Check persistence
    state2 = SessionState(run_dir=state.run_dir)
    assert len(state2.central_memory) == 1
    assert state2.central_memory[0]["note"] == note

def test_add_to_central_memory_with_metadata(tmp_path):
    state = make_fresh_session_state(tmp_path)
    add_to_central_memory = get_add_to_central_memory_tool(state)
    note = "Metadata note"
    reasoning = "Metadata reasoning"
    agent = "MetaAgent"
    metadata = {"foo": "bar", "tables": ["users", "items"]}
    result = add_to_central_memory(note=note, reasoning=reasoning, agent=agent, metadata=metadata)
    assert "Note added to central memory" in result
    assert len(state.central_memory) == 1
    entry = state.central_memory[0]
    assert entry["metadata"] == metadata
    # Check that other fields are present
    assert entry["note"] == note
    assert entry["agent"] == agent
    assert entry["reasoning"] == reasoning
    assert "timestamp" in entry

def test_add_to_central_memory_multiple_entries(tmp_path):
    state = make_fresh_session_state(tmp_path)
    add_to_central_memory = get_add_to_central_memory_tool(state)
    add_to_central_memory(note="First", reasoning="First reason", agent="A")
    add_to_central_memory(note="Second", reasoning="Second reason", agent="B")
    assert len(state.central_memory) == 2
    notes = [e["note"] for e in state.central_memory]
    assert notes == ["First", "Second"]
    agents = [e["agent"] for e in state.central_memory]
    assert agents == ["A", "B"]
    # Check persistence
    state2 = SessionState(run_dir=state.run_dir)
    assert len(state2.central_memory) == 2
    assert state2.central_memory[1]["note"] == "Second"
