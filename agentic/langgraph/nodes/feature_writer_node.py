import logging

class FeatureWriterNode:
    """
    Node for feature synthesis. Expects memory to be a dict-like AgentMemory (agentic.core.agent_memory.AgentMemory).
    Use memory.get("insights", []) and memory.get("features", []); set values with memory.set(key, value).
    """
    def __init__(self, memory, logger=None):
        self.memory = memory
        self.logger = logger or logging.getLogger("FeatureWriterNode")

    def run(self, state: dict) -> dict:
        self.logger.info(f"[FeatureWriterNode] Input state: {state}")
        insights = self.memory.get("insights", [])
        if not insights:
            self.logger.warning("No insights to write feature for.")
            return state
        last_insight = insights[-1]
        feature_code = f"def feature_{len(insights)}(df, params):\n    # Feature logic for: {last_insight['title']}\n    return df"
        feature = {
            "name": f"feature_{len(insights)}",
            "code": feature_code,
            "params": last_insight.get("params", {})
        }
        features = self.memory.get("features", [])
        features.append(feature)
        self.memory.set("features", features)
        state["features"] = self.memory.get("features", [])
        self.logger.info(f"[FeatureWriterNode] Output state: {state}")
        return state
