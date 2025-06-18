import pytest
from agentic.core.orchestration import Orchestrator

class DummyAgent:
    def __init__(self, name):
        self.name = name
        self.ran = False
    def run(self, context):
        self.ran = True
        return f"{self.name} ran with {context}"

@pytest.fixture
def orchestrator():
    agents = [DummyAgent("A"), DummyAgent("B")]
    # Provide required args for new Orchestrator signature
    return Orchestrator(agents=agents, logger=None, session=None, db=None)

def test_orchestrator_run(orchestrator):
    results = orchestrator.run_all(context="foo")
    assert results == ["A ran with foo", "B ran with foo"]
    for agent in orchestrator.agents:
        assert agent.ran is True
