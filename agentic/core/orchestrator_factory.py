class OrchestratorFactory:
    def __init__(self, logger, session, db, backend, agent_factory):
        self.logger = logger
        self.session = session
        self.db = db
        self.backend = backend
        self.agent_factory = agent_factory

    def create(self, orchestrator_type: str):
        # For demonstration: return a string, in real use return orchestrator instance
        return f"orchestrator:{orchestrator_type}"
