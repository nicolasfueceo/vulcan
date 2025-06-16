# scripts/test_schema_validation.py
"""
Test that the finalize_hypotheses and save_candidate_features tools enforce Pydantic schema validation.
"""
import pytest
from src.schemas.models import Hypothesis, CandidateFeature
from src.utils.tools import get_finalize_hypotheses_tool, get_save_candidate_features_tool

class DummySessionState:
    def __init__(self):
        self.hypotheses = None
        self.candidate_features = None
    def finalize_hypotheses(self, hyps):
        self.hypotheses = hyps
    def set_candidate_features(self, feats):
        self.candidate_features = feats

def test_finalize_hypotheses_valid():
    session = DummySessionState()
    tool = get_finalize_hypotheses_tool(session)
    data = [
        {"summary": "Users who review more books tend to give higher ratings.", "rationale": "Observed a positive correlation in the sample."},
        {"summary": "Standalone books are rated higher than series books.", "rationale": "Series books have more variance and lower means in ratings."}
    ]
    result = tool(data)
    assert result.startswith("SUCCESS")
    assert session.hypotheses is not None
    assert all(isinstance(h, Hypothesis) for h in session.hypotheses)

def test_finalize_hypotheses_invalid():
    session = DummySessionState()
    tool = get_finalize_hypotheses_tool(session)
    # Missing rationale
    data = [
        {"summary": "Incomplete hypothesis."}
    ]
    result = tool(data)
    assert result.startswith("ERROR")
    assert session.hypotheses is None

def test_save_candidate_features_valid():
    session = DummySessionState()
    tool = get_save_candidate_features_tool(session)
    data = [
        {
            "name": "feature1",
            "type": "code",
            "spec": "df['a'] + df['b']",
            "depends_on": [],
            "params": {"alpha": 0.1},
            "rationale": "Captures additive signal."
        }
    ]
    result = tool(data)
    assert result.startswith("SUCCESS")
    assert session.candidate_features is not None
    assert all(isinstance(f, dict) for f in session.candidate_features)

def test_save_candidate_features_invalid():
    session = DummySessionState()
    tool = get_save_candidate_features_tool(session)
    # Missing name
    data = [
        {
            "type": "code",
            "spec": "df['a'] + df['b']",
            "depends_on": [],
            "params": {"alpha": 0.1},
            "rationale": "Captures additive signal."
        }
    ]
    result = tool(data)
    assert result.startswith("ERROR")
    assert session.candidate_features is None

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
