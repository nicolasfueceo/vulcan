from agentic.langgraph.workflow import build_workflow
from agentic.core.agent_memory import AgentMemory

def test_workflow_memory_isolation(tmp_path):
    mem1 = AgentMemory({"insights": [1, 2]})
    mem2 = AgentMemory({"insights": [3]})
    wf1 = build_workflow(memory=mem1)
    wf2 = build_workflow(memory=mem2)
    # Simulate run
    state1 = wf1.run({})
    state2 = wf2.run({})
    # Robust check: originals are preserved, last is new insight
    insights1 = wf1.state_manager.memory.get("insights")
    insights2 = wf2.state_manager.memory.get("insights")
    assert insights1[:2] == [1, 2], f"wf1: expected [1, 2] as prefix, got {insights1}"
    assert isinstance(insights1[-1], dict) and insights1[-1]["title"] == "Mock Insight #3", f"wf1: last insight mismatch: {insights1[-1]}"
    assert insights2[:1] == [3], f"wf2: expected [3] as prefix, got {insights2}"
    assert isinstance(insights2[-1], dict) and insights2[-1]["title"] == "Mock Insight #2", f"wf2: last insight mismatch: {insights2[-1]}"
