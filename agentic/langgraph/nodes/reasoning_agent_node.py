import logging

class ReasoningAgentNode:
    """
    Node for reasoning/exploration. Expects memory to be a dict-like AgentMemory (agentic.core.agent_memory.AgentMemory).
    Use memory.get("insights", []) for access; append new insights directly to the list.
    """
    def __init__(self, memory, tool_api=None, logger=None):
        self.memory = memory
        self.tool_api = tool_api
        self.logger = logger or logging.getLogger("ReasoningAgentNode")

    def run(self, state: dict) -> dict:
        self.logger.info(f"[ReasoningAgentNode] Input state: {state}")
        # Example: Agent explores DB, runs code, plots, etc.
        # For now, just mock an insight discovery step
        insights = self.memory.get("insights", [])
        new_insight = {
            "title": f"Mock Insight #{len(insights) + 1}",
            "evidence": "(mock evidence)",
            "code": "# code here",
            "params": {}
        }
        insights.append(new_insight)
        self.memory.set("insights", insights)
        self.logger.info(f"[ReasoningAgentNode] Discovered: {new_insight}")
        state["last_insight"] = new_insight
        return state
