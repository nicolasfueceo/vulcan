from typing import Callable, Dict, Any

class LangChainBackend:
    def __init__(self, langchain_func: Callable[[str, Dict[str, Any]], str]):
        self.langchain_func = langchain_func

    def run(self, prompt: str, context: Dict[str, Any]) -> str:
        return self.langchain_func(prompt, context)
