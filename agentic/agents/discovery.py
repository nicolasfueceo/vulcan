from agentic.agents.base import AgentBase
from typing import Any, List, Dict

class DiscoveryAgent(AgentBase):
    """Agent for discovering insights from provided context."""
    def discover(self, context: Any) -> List[Dict[str, Any]]:
        """Discover insights using the backend or return a placeholder."""
        if self.backend is not None:
            return self.backend.run(prompt=None, context=context)
        self.log(f"Discovering insights with context: {context}")
        return [
            {
                "title": "Placeholder Insight",
                "finding": "This is a placeholder insight.",
                "rationale": "No backend provided."
            }
        ]
