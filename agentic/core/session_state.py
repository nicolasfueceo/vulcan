from typing import Any, Dict

class SessionState:
    """
    Minimal session state management for agentic core logic.
    Stores arbitrary key-value pairs and supports get/set operations.
    """
    def __init__(self, logger=None):
        self._state: Dict[str, Any] = {}
        self.logger = logger

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value
        if self.logger:
            self.logger.info(f"Set {key} to {value}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def save(self, path: str) -> None:
        import json
        try:
            with open(path, 'w') as f:
                json.dump(self._state, f)
            if self.logger:
                self.logger.info(f"Session state saved to {path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save session state: {e}")
            raise

    def load(self, path: str) -> None:
        import json
        try:
            with open(path, 'r') as f:
                self._state = json.load(f)
            if self.logger:
                self.logger.info(f"Session state loaded from {path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load session state: {e}")
            raise
