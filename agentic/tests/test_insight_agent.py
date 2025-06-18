import pytest
from agentic.agents.insight import InsightAgent

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
    return InsightAgent(name="InsightAgent", logger=DummyLogger(), session=DummySession(), db=DummyDB())

def test_insight_agent_inherits(agent):
    assert agent.name == "InsightAgent"
    agent.log("insight running")
    assert any("insight running" in msg for msg in agent.logger.logs)

def test_insight_agent_generate(agent):
    result = agent.generate(context={"bar": 2})
    assert result == "insight generated"
