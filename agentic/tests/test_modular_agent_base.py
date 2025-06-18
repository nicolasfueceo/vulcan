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

class DummyBackend:
    def __init__(self):
        self.calls = []
    def run(self, prompt, context):
        self.calls.append((prompt, context))
        return f"backend:{prompt}:{context}"

@pytest.fixture
def modular_agent():
    return AgentBase(
        name="ModAgent",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=DummyBackend()
    )

def test_backend_injection(modular_agent):
    result = modular_agent.run_with_backend(prompt="foo", context={"x": 1})
    assert result == "backend:foo:{'x': 1}"
    assert modular_agent.backend.calls[0] == ("foo", {"x": 1})

def test_backend_swap():
    class AltBackend:
        def run(self, prompt, context):
            return "alt"
    agent = AgentBase(
        name="A",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=AltBackend()
    )
    assert agent.run_with_backend("bar", {}) == "alt"
