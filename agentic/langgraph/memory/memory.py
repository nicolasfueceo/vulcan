import logging
from typing import Any, List, Dict

class AgentMemory:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("AgentMemory")
        self._insights: List[Dict[str, Any]] = []
        self._features: List[Dict[str, Any]] = []
        self._history: List[Dict[str, Any]] = []

    def add_insight(self, insight: Dict[str, Any]):
        self.logger.info(f"[AgentMemory] Adding insight: {insight}")
        self._insights.append(insight)
        self._history.append({"type": "insight", "data": insight})

    def get_insights(self) -> List[Dict[str, Any]]:
        return list(self._insights)

    def add_feature(self, feature: Dict[str, Any]):
        self.logger.info(f"[AgentMemory] Adding feature: {feature}")
        self._features.append(feature)
        self._history.append({"type": "feature", "data": feature})

    def get_features(self) -> List[Dict[str, Any]]:
        return list(self._features)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear(self):
        self.logger.info("[AgentMemory] Clearing memory")
        self._insights.clear()
        self._features.clear()
        self._history.clear()
