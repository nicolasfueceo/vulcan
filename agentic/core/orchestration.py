from typing import List, Any, Optional

from agentic.core.agent_memory import AgentMemory

from agentic.langgraph.workflow import build_workflow
from agentic.core.metric_logger import MetricLogger
from typing import Optional

class Orchestrator:
    """Main orchestrator for agentic pipelines, now with integrated AgentMemory and LangGraph workflow support."""
    def __init__(self, agents: List[Any], logger, session, db, backend=None, memory: Optional[AgentMemory] = None):
        self.agents = agents
        self.logger = logger
        self.session = session
        self.db = db
        self.backend = backend
        self.memory = memory if memory is not None else AgentMemory()

    def save_memory(self, path=None, run_id=None):
        return self.memory.save(path=path, run_id=run_id)

    def load_memory(self, path):
        self.memory = AgentMemory.load(path)
        return self.memory

    def run_all(self, context: Any) -> list:
        results = []
        for agent in self.agents:
            results.append(agent.run(context))
        return results

    def run_workflow(self, initial_state=None, tool_api=None, reward_manager=None, metric_logger: Optional[MetricLogger] = None):
        """
        Run the LangGraph workflow with a fresh AgentMemory for this run.
        Optionally pass tool_api or reward_manager for extensibility.
        Returns final state and path to saved memory.
        """
        memory = AgentMemory()
        state = initial_state or {}
        # Pre-load user-provided insights into memory if present
        if "insights" in state:
            memory.set("insights", state["insights"])
        workflow = build_workflow(memory=memory, metric_logger=metric_logger)
        # You can extend build_workflow to accept tool_api, reward_manager, etc.
        final_state = workflow.run(state)
        import uuid
        run_id = str(uuid.uuid4())
        mem_path = memory.save(run_id=run_id)
        self.logger.info(f"Workflow completed, memory saved to {mem_path}")
        return final_state, mem_path
