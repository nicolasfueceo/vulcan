import pytest
from agentic.utils.langgraph_backend import LangGraphBackend

class DummyGraph:
    def __init__(self, context):
        self.context = context
    def run(self):
        return {"insight": f"LangGraph: {self.context['data']}"}

def test_langgraph_backend_run():
    backend = LangGraphBackend(graph_builder=lambda ctx: DummyGraph(ctx))
    result = backend.run(prompt="", context={"data": "foo"})
    assert result["insight"] == "LangGraph: foo"
