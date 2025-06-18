import os
from agentic.core.agent_memory import AgentMemory

def test_agent_memory_basic():
    mem = AgentMemory()
    mem.set("foo", 123)
    assert mem.get("foo") == 123
    mem.set("bar", [1, 2, 3])
    assert mem.get("bar") == [1, 2, 3]
    mem.clear()
    assert mem.get("foo") is None

def test_agent_memory_save_load(tmp_path):
    mem = AgentMemory({"a": 1, "b": 2})
    path = tmp_path / "memory.json"
    mem.save(str(path))
    mem2 = AgentMemory.load(str(path))
    assert mem2.get("a") == 1
    assert mem2.get("b") == 2

def test_agent_memory_auto_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mem = AgentMemory({"x": 42})
    path = mem.save()
    assert os.path.exists(path)
    mem2 = AgentMemory.load(path)
    assert mem2.get("x") == 42
