from agentic.agents.base import AgentBase
from typing import Any

class InsightAgent(AgentBase):
    """Agent for generating insights from provided context."""
    def generate(self, context: Any) -> Any:
        """Generate insight using the backend or return a placeholder."""
        if self.backend is not None: 
            return self.backend.run(prompt=None, context=context)
        self.log(f"Generating insight with context: {context}")
        return "insight generated"
