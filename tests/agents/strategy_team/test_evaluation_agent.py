# tests/agents/strategy_team/test_evaluation_agent.py
import pytest

from src.agents.strategy_team.evaluation_agent import EvaluationAgent
from src.utils.session_state import SessionState


def test_evaluation_agent_instantiation():
    agent = EvaluationAgent()
    assert agent is not None


def test_evaluation_agent_run_smoke(tmp_path):
    # Minimal SessionState mock for smoke test
    session_state = SessionState(run_dir=tmp_path)
    agent = EvaluationAgent()
    # Should not raise (will do nothing as no optimization results)
    agent.run(session_state)
