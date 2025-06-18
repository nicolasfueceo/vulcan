import logging
from agentic.langgraph.memory.memory import AgentMemory

def test_agent_memory():
    logger = logging.getLogger("test")
    mem = AgentMemory(logger=logger)
    mem.add_insight({"title": "Test Insight"})
    mem.add_feature({"name": "Test Feature"})
    assert len(mem.get_insights()) == 1
    assert mem.get_insights()[0]["title"] == "Test Insight"
    assert len(mem.get_features()) == 1
    assert mem.get_features()[0]["name"] == "Test Feature"
    assert len(mem.get_history()) == 2
    mem.clear()
    assert mem.get_insights() == []
    assert mem.get_features() == []
    assert mem.get_history() == []
