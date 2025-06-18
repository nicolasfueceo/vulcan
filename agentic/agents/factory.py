from agentic.agents.strategy import StrategyAgent
from agentic.agents.insight import InsightAgent
from agentic.agents.realization import RealizationAgent

class AgentFactory:
    def __init__(self, logger, session, db, backend):
        self.logger = logger
        self.session = session
        self.db = db
        self.backend = backend

    def create(self, agent_type: str, name: str):
        if agent_type == "strategy":
            return StrategyAgent(name=name, logger=self.logger, session=self.session, db=self.db, backend=self.backend)
        elif agent_type == "insight":
            return InsightAgent(name=name, logger=self.logger, session=self.session, db=self.db, backend=self.backend)
        elif agent_type == "realization":
            return RealizationAgent(name=name, logger=self.logger, session=self.session, db=self.db, backend=self.backend)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
