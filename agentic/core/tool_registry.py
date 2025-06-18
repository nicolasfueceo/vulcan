from typing import Callable, Dict, Any

class ToolRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self._registry[name] = func

    def call(self, name: str, *args, **kwargs) -> Any:
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not registered.")
        return self._registry[name](*args, **kwargs)
