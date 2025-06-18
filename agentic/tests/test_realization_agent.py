import pytest
from agentic.agents.realization import RealizationAgent

class DummyLogger:
    def __init__(self):
        self.logs = []
    def info(self, msg):
        self.logs.append(msg)
    def error(self, msg):
        self.logs.append(msg)

class DummySession:
    def __init__(self):
        self.state = {}
    def set(self, k, v):
        self.state[k] = v
    def get(self, k, default=None):
        return self.state.get(k, default)

class DummyDB:
    def __init__(self):
        self.kv = {}
    def set(self, k, v):
        self.kv[k] = v
    def get(self, k):
        return self.kv.get(k)

@pytest.fixture
def agent():
    return RealizationAgent(name="RealizationAgent", logger=DummyLogger(), session=DummySession(), db=DummyDB())

def test_realization_agent_inherits(agent):
    assert agent.name == "RealizationAgent"
    agent.log("realization running")
    assert any("realization running" in msg for msg in agent.logger.logs)

def test_realization_agent_execute(agent):
    result = agent.execute(context={"baz": 3})
    assert result == "realization executed"
