from typing import Any, Optional


class AgentBase:
    def __init__(self, name: str, logger: Any, session: Any, db: Any, backend: Optional[Any] = None):
        self.name = name
        self.logger = logger
        self.session = session
        self.db = db
        self.backend = backend

    def log(self, msg: str) -> None:
        self.logger.info(f"[{self.name}] {msg}")

    def get_state(self, key: str, default=None):
        return self.session.get(key, default)

    def set_state(self, key: str, value: Any):
        self.session.set(key, value)

    def db_get(self, key: str):
        return self.db.get(key)

    def db_set(self, key: str, value: Any):
        self.db.set(key, value)

    def run_with_backend(self, prompt: str, context: Any) -> Any:
        if not self.backend or not hasattr(self.backend, "run"):
            raise RuntimeError("No valid backend provided to AgentBase.")
        return self.backend.run(prompt, context)
