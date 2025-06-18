from typing import Any, Dict, Optional
import json
import os
from datetime import datetime

class AgentMemory:
    """
    Serializable memory for agent state, supporting save/load to file or dict.
    Each run/session can have its own memory file for isolation and reproducibility.
    """
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state = initial_state.copy() if initial_state else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def clear(self) -> None:
        self._state.clear()

    def to_dict(self) -> Dict[str, Any]:
        return self._state.copy()

    def update(self, other: Dict[str, Any]) -> None:
        self._state.update(other)

    def save(self, path: Optional[str] = None, run_id: Optional[str] = None) -> str:
        """
        Save memory to a JSON file. If path is None, auto-generate a path using run_id and timestamp.
        Returns the path used.
        """
        if path is None:
            if run_id is None:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("data", f"agent_memory_{run_id}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._state, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "AgentMemory":
        with open(path, "r") as f:
            state = json.load(f)
        return cls(state)
