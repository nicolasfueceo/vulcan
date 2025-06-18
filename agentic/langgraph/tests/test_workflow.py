import logging
from agentic.langgraph.workflow import build_workflow

def test_insight_discovery_workflow():
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    from agentic.core.agent_memory import AgentMemory
    memory = AgentMemory()
    pipeline = build_workflow(memory=memory)
    final_state = pipeline.run({})
    assert "insights" in final_state
    assert isinstance(final_state["insights"], list)
    assert "stop" in final_state
