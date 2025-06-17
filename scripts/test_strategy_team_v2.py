"""
Integration test for the Python-centric strategy_team_v2 pipeline.
- Sets up a fake session state with mock insights and hypotheses
- Runs the strategy_team_v2 pipeline
- Asserts that features are realized and results are printed
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.run_utils import init_run

from src.agents.strategy_team.strategy_team_v2 import run_strategy_team_v2
from src.utils.session_state import SessionState
from src.schemas.models import CandidateFeature, Hypothesis

def make_mock_data():
    # Create pre-generated hypotheses and candidate features
    candidate_features = [
        CandidateFeature(
            name="user_rating_count",
            type="code",
            spec="df.groupby('user_id').size()",
            depends_on=["reviews.user_id"],
            parameters={},
            rationale="Counts how many books each user has rated."
        )
    ]
    hypotheses = [
        Hypothesis(
            summary="Users who rate more books tend to have higher engagement.",
            rationale="Book rating frequency reflects user engagement.",
            depends_on=["reviews.user_id", "reviews.rating"]
        )
    ]
    return candidate_features, hypotheses

def main():
    init_run()
    llm_config = {"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY", "sk-test")}
    db_path = "data/goodreads_curated.duckdb"
    candidate_features, hypotheses = make_mock_data()
    session_state = run_strategy_team_v2(llm_config, db_path, candidate_features, hypotheses)
    realized = session_state.get_state("realized_features", [])
    print(f"[TEST] Realized Features: {len(realized)}")
    for f in realized:
        print(f)
    metrics = session_state.get_state("final_evaluation_metrics", {})
    print("[TEST] Evaluation Metrics:", metrics)
    reflection = session_state.get_state("reflection_result", {})
    print("[TEST] Reflection Result:", reflection)
    assert realized, "No features were realized."
    assert metrics, "No evaluation metrics produced."
    assert reflection and "should_continue" in reflection, "No reflection result."
    print("[TEST] strategy_team_v2 integration test PASSED.")

if __name__ == "__main__":
    main()
