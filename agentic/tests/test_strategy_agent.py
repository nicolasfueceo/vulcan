import pytest
from agentic.agents.strategy import StrategyAgent

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
    return StrategyAgent(name="StratAgent", logger=DummyLogger(), session=DummySession(), db=DummyDB())

def test_strategy_agent_inherits(agent):
    assert agent.name == "StratAgent"
    agent.log("strategy running")
    assert any("strategy running" in msg for msg in agent.logger.logs)

def test_strategy_agent_decide(agent):
    result = agent.decide(context={"foo": 1})
    assert result == "decision made"
