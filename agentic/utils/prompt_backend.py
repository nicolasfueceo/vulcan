from typing import Callable, Dict, Any

class PromptBackend:
    def __init__(self, template: str, engine: Callable[[str], str]):
        self.template = template
        self.engine = engine

    def run(self, prompt: str, context: Dict[str, Any]) -> str:
        try:
            rendered = self.template.format(**context)
        except KeyError as e:
            raise KeyError(f"Missing key for prompt template: {e}")
        return self.engine(rendered)
