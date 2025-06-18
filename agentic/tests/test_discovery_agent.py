import pytest
from agentic.agents.discovery import DiscoveryAgent

class DummyLogger:
    def info(self, msg): pass
    def error(self, msg): pass
class DummySession:
    def set(self, k, v): pass
    def get(self, k, default=None): return default
class DummyDB:
    def set(self, k, v): pass
    def get(self, k): return None

class DummyBackend:
    def run(self, prompt, context):
        return [{"title": "Insight", "finding": f"Found: {context['data']}", "rationale": "Backend logic"}]

def test_discovery_agent_backend():
    agent = DiscoveryAgent(
        name="disco",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=DummyBackend()
    )
    result = agent.discover({"data": "foo"})
    assert isinstance(result, list)
    assert result[0]["finding"] == "Found: foo"

def test_discovery_agent_fallback():
    agent = DiscoveryAgent(
        name="disco",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB()
    )
    result = agent.discover({"data": "bar"})
    assert result[0]["title"] == "Placeholder Insight"
