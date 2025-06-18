import logging

from agentic.core.agent_memory import AgentMemory

class StateManagerNode:
    """Node for managing agentic pipeline state and stopping conditions."""
    def __init__(self, memory: AgentMemory, logger=None, max_insights=3):
        """Initialize with memory, logger, and max insights."""
        self.memory = memory
        self.logger = logger or logging.getLogger("StateManagerNode")
        self.max_insights = max_insights

    def run(self, state: dict) -> dict:
        """Update state with stop condition based on memory's insights."""
        self.logger.info(f"[StateManagerNode] Input state: {state}")
        insights = self.memory.get("insights", []) if hasattr(self.memory, "get") else []
        # Example stopping condition: stop after max_insights
        if len(insights) >= self.max_insights:
            state["stop"] = True
            self.logger.info(f"Stopping: {len(insights)} insights reached.")
        else:
            state["stop"] = False
        self.logger.info(f"[StateManagerNode] Output state: {state}")
        return state
