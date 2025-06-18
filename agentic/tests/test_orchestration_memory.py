from agentic.core.orchestration import Orchestrator
from agentic.core.agent_memory import AgentMemory

class DummyAgent:
    def __init__(self):
        self.called = False
    def run(self):
        self.called = True
        return "ran"

def test_orchestrator_memory_save_load(tmp_path):
    mem = AgentMemory({"foo": "bar"})
    orch = Orchestrator(agents=[], logger=None, session=None, db=None, memory=mem)
    path = tmp_path / "mem.json"
    orch.save_memory(str(path))
    orch.memory.set("foo", "baz")  # mutate
    orch.load_memory(str(path))
    assert orch.memory.get("foo") == "bar"

def test_orchestrator_memory_isolation(tmp_path):
    mem1 = AgentMemory({"run": 1})
    mem2 = AgentMemory({"run": 2})
    orch1 = Orchestrator(agents=[], logger=None, session=None, db=None, memory=mem1)
    orch2 = Orchestrator(agents=[], logger=None, session=None, db=None, memory=mem2)
    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    orch1.save_memory(str(p1))
    orch2.save_memory(str(p2))
    assert AgentMemory.load(str(p1)).get("run") == 1
    assert AgentMemory.load(str(p2)).get("run") == 2
