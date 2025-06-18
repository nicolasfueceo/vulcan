from typing import Callable, Dict, Any

class AutogenBackend:
    def __init__(self, autogen_func: Callable[[str, Dict[str, Any]], str]):
        self.autogen_func = autogen_func

    def run(self, prompt: str, context: Dict[str, Any]) -> str:
        return self.autogen_func(prompt, context)
