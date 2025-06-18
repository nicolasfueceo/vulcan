from typing import Callable, Any, Dict

class RewardFunctionManager:
    """Manager for registering and retrieving reward functions for BO."""
    def __init__(self):
        self._registry: Dict[str, Callable[..., float]] = {}

    def register(self, name: str, fn: Callable[..., float]) -> None:
        self._registry[name] = fn

    def get(self, name: str) -> Callable[..., float]:
        if name not in self._registry:
            raise ValueError(f"No reward function registered as '{name}'")
        return self._registry[name]

    def list_functions(self):
        return list(self._registry.keys())
