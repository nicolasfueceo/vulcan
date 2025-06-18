from agentic.core.orchestration import Orchestrator
from agentic.utils.logger import AgenticLogger

def test_orchestrator_workflow_isolation(tmp_path):
    logger = AgenticLogger()
    # Orchestrator 1
    orch1 = Orchestrator(agents=[], logger=logger, session=None, db=None)
    state1, mem_path1 = orch1.run_workflow(initial_state={"insights": ["A"]})
    # Orchestrator 2
    orch2 = Orchestrator(agents=[], logger=logger, session=None, db=None)
    state2, mem_path2 = orch2.run_workflow(initial_state={"insights": ["B"]})
    # Load memories
    from agentic.core.agent_memory import AgentMemory
    mem1 = AgentMemory.load(mem_path1)
    mem2 = AgentMemory.load(mem_path2)
    # Each memory should contain its own initial insight and a new one
    insights1 = mem1.get("insights", [])
    insights2 = mem2.get("insights", [])
    assert insights1[0] == "A", f"Orch1: expected 'A' as first, got {insights1}"
    assert isinstance(insights1[-1], dict), f"Orch1: last insight not a dict: {insights1[-1]}"
    assert insights2[0] == "B", f"Orch2: expected 'B' as first, got {insights2}"
    assert isinstance(insights2[-1], dict), f"Orch2: last insight not a dict: {insights2[-1]}"
    assert mem_path1 != mem_path2, "Memory files should be different for each orchestrator run."
