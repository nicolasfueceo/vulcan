import pytest
from agentic.agents.factory import AgentFactory

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

class DummyBackend:
    def run(self, prompt, context):
        return f"run:{prompt}:{context}"

@pytest.fixture
def factory():
    return AgentFactory(
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=DummyBackend()
    )

def test_create_strategy_agent(factory):
    agent = factory.create("strategy", name="Strat")
    assert agent.name == "Strat"
    assert hasattr(agent, "decide")
    assert agent.backend is factory.backend


def test_create_insight_agent(factory):
    agent = factory.create("insight", name="Insight")
    assert agent.name == "Insight"
    assert hasattr(agent, "generate")
    assert agent.backend is factory.backend


def test_create_realization_agent(factory):
    agent = factory.create("realization", name="Realize")
    assert agent.name == "Realize"
    assert hasattr(agent, "execute")
    assert agent.backend is factory.backend
