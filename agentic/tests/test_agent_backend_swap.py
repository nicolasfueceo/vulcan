import pytest
from agentic.agents.strategy import StrategyAgent
from agentic.agents.insight import InsightAgent
from agentic.agents.realization import RealizationAgent

class DummyLogger:
    def info(self, msg): pass
    def error(self, msg): pass
class DummySession:
    def set(self, k, v): pass
    def get(self, k, default=None): return default
class DummyDB:
    def set(self, k, v): pass
    def get(self, k): return None

class PromptBackend:
    def __init__(self, template):
        self.template = template
    def run(self, prompt, context):
        return self.template.format(**context)

class FuncBackend:
    def run(self, prompt, context):
        return f"func:{context['x']}"

def test_strategy_agent_backend_swap():
    agent = StrategyAgent(
        name="strat",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=PromptBackend(template="decision:{foo}")
    )
    assert agent.decide({"foo": 42}) == "decision:42"
    agent.backend = FuncBackend()
    assert agent.decide({"x": 7}) == "func:7"

def test_insight_agent_backend_swap():
    agent = InsightAgent(
        name="ins",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=PromptBackend(template="insight:{bar}")
    )
    assert agent.generate({"bar": 1}) == "insight:1"
    agent.backend = FuncBackend()
    assert agent.generate({"x": 99}) == "func:99"

def test_realization_agent_backend_swap():
    agent = RealizationAgent(
        name="real",
        logger=DummyLogger(),
        session=DummySession(),
        db=DummyDB(),
        backend=PromptBackend(template="realize:{baz}")
    )
    assert agent.execute({"baz": 5}) == "realize:5"
    agent.backend = FuncBackend()
    assert agent.execute({"x": 123}) == "func:123"
