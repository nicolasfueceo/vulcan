import pytest
from agentic.agents.base import AgentBase

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
    return AgentBase(name="TestAgent", logger=DummyLogger(), session=DummySession(), db=DummyDB())

def test_agent_init(agent):
    assert agent.name == "TestAgent"
    assert hasattr(agent, "logger")
    assert hasattr(agent, "session")
    assert hasattr(agent, "db")

def test_agent_log_and_state(agent):
    agent.logger.info("hi!")
    assert "hi!" in agent.logger.logs[0]
    agent.session.set("foo", 123)
    assert agent.session.get("foo") == 123
    agent.db.set("bar", "baz")
    assert agent.db.get("bar") == "baz"
