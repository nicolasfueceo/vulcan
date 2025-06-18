import logging
from agentic.langgraph.nodes.reasoning_agent_node import ReasoningAgentNode
from agentic.langgraph.nodes.feature_writer_node import FeatureWriterNode
"""
Workflow assembly for the agentic pipeline. Provides build_workflow to construct the pipeline with a given AgentMemory.
"""
from agentic.langgraph.nodes.reasoning_agent_node import ReasoningAgentNode
from agentic.langgraph.nodes.feature_writer_node import FeatureWriterNode
from agentic.langgraph.nodes.state_manager_node import StateManagerNode
from agentic.core.agent_memory import AgentMemory
from agentic.langgraph.nodes.evaluation_node import EvaluationNode
from agentic.core.metric_logger import MetricLogger


from typing import Optional

def build_workflow(memory: AgentMemory, metric_logger: Optional[MetricLogger] = None):
    reasoner = ReasoningAgentNode(memory=memory, tool_api=None)
    feature_writer = FeatureWriterNode(memory=memory)
    state_manager = StateManagerNode(memory=memory)
    evaluation_node = EvaluationNode(memory=memory, metric_logger=metric_logger or MetricLogger())

    class Pipeline:
        def __init__(self):
            self.state_manager = state_manager
            self.evaluation_node = evaluation_node
        def run(self, state):
            # Reasoner -> FeatureWriter -> StateManager -> EvaluationNode
            state = reasoner.run(state)
            state = feature_writer.run(state)
            state = state_manager.run(state)
            state = self.evaluation_node.run(state)
            # Ensure insights/features from memory are included in the returned state
            state["insights"] = memory.get("insights", [])
            state["features"] = memory.get("features", [])
            return state

    return Pipeline()

