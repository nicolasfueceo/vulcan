import pytest
from agentic.core.orchestrator_factory import OrchestratorFactory

class DummyLogger:
    def info(self, msg):
        pass
    def error(self, msg):
        pass

class DummySession:
    def set(self, k, v):
        pass
    def get(self, k, default=None):
        return default

class DummyDB:
    def set(self, k, v):
        pass
    def get(self, k):
        return None

class DummyBackend:
    def run(self, prompt, context):
        return f"run:{prompt}:{context}"

class DummyAgentFactory:
    def create(self, agent_type, name):
        return f"agent:{agent_type}:{name}"

@pytest.fixture
def orchestrator_factory():
    return OrchestratorFactory(
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=DummyBackend(),
        agent_factory=DummyAgentFactory()
    )

def test_orchestrator_factory_create(orchestrator_factory):
    orchestrator = orchestrator_factory.create("strategy_loop")
    assert orchestrator == "orchestrator:strategy_loop"
