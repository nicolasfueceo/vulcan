import pytest
from agentic.core.session_state import SessionState

class DummyLogger:
    def __init__(self):
        self.logs = []
    def info(self, msg):
        self.logs.append(msg)
    def error(self, msg):
        self.logs.append(msg)

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.mark.parametrize("init_state", [None, {"foo": 1, "bar": 2}])
def test_session_state_init_and_set_get(init_state, dummy_logger):
    state = SessionState(logger=dummy_logger)
    if init_state:
        for k, v in init_state.items():
            state.set(k, v)
    for k, v in (init_state or {}).items():
        assert state.get(k) == v
    # Test default value
    assert state.get("nonexistent", default=42) == 42

# Add more tests for save/load, error handling, etc.
