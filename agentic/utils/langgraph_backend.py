from typing import Callable, Dict, Any

class LangGraphBackend:
    def __init__(self, graph_builder: Callable[[Dict[str, Any]], Any]):
        self.graph_builder = graph_builder

    def run(self, prompt: str, context: Dict[str, Any]) -> Any:
        # prompt is ignored; context is used to build the graph and run it
        graph = self.graph_builder(context)
        return graph.run()
