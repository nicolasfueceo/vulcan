from agentic.agents.base import AgentBase
from typing import Any

class StrategyAgent(AgentBase):
    """Agent for making strategic decisions from provided context."""
    def decide(self, context: Any) -> Any:
        """Make a decision using the backend or return a placeholder."""
        if self.backend is not None:
            return self.backend.run(prompt=None, context=context)
        self.log(f"Making decision with context: {context}")
        return "decision made"
