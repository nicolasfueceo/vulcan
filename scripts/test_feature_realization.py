# scripts/test_feature_realization.py
"""
Test script for FeatureRealizationAgent: ensures that candidate features are realized as valid Python functions and saved to session_state.features.
"""
import sys
import os
import ast
from src.utils.session_state import SessionState
from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent

# Mock LLM config (adjust as needed for your environment)
llm_config = {
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 512,
    "api_key": os.environ.get("OPENAI_API_KEY", "sk-test")
}

def make_candidate_feature(name, rationale, depends_on, parameters=None):
    return {
        "name": name,
        "type": "code",
        "spec": "# placeholder spec",
        "depends_on": depends_on,
        "parameters": parameters or {},
        "rationale": rationale
    }

def main():
    # Initialize run context
    from src.utils.run_utils import init_run
    init_run()
    # Initialize session state
    session_state = SessionState()
    session_state.set_state("db_path", "data/goodreads_curated.duckdb")

    # Mock candidate features
    candidate_features = [
     
        make_candidate_feature(
            name="authorship_diversity_score",
            rationale="Evaluates the diversity score of authorship on a book.",
            depends_on=["book_id", "author_ids"]
        ),
    ]
    session_state.set_state("candidate_features", candidate_features)

    # Run feature realization
    agent = FeatureRealizationAgent(llm_config, session_state)
    agent.run()

    # Check realized features
    features = session_state.get_state("features")
    assert features, "No features were realized!"
    print(f"[TEST] Realized features: {len(features)}")
    for feature in features:
        print("---")
        print("Feature name:", feature.name)
        print("Code string:\n", feature.code_str)
        assert feature.code_str.strip(), f"Feature {feature.name} has empty code!"
        # Check that code is valid Python
        try:
            ast.parse(feature.code_str)
        except Exception as e:
            raise AssertionError(f"Feature {feature.name} code is not valid Python: {e}")
    print("[TEST] Feature realization test PASSED.")

if __name__ == "__main__":
    main()
