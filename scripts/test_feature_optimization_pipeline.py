"""
Test script for the refactored feature optimization pipeline.
- Simulates a finalized hypothesis.
- Runs the FeatureEngineer agent (with updated prompt) to generate CandidateFeatures with full parameter specs.
- Runs the FeatureRealizationAgent to realize features.
- Runs the VULCANOptimizer to optimize features.
- Prints out the resulting optimization report.
"""

import inspect
import json
from pathlib import Path

from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.schemas.models import CandidateFeature, ParameterSpec
from src.utils.session_state import SessionState

# --- Step 1: Simulate finalized hypotheses and candidate features ---
# For now, we bypass LLM feature engineering and inject a candidate feature manually


def get_dummy_candidate_feature():
    """Return a CandidateFeature with full ParameterSpec schema."""
    return CandidateFeature(
        name="user_review_count",
        type="code",
        spec="df.groupby('user_id').size() if len(df) > 0 else 0",
        depends_on=["user_id"],
        parameters={
            "min_reviews": ParameterSpec(type="int", low=1, high=20, step=1, default=1),
            "agg_method": ParameterSpec(
                type="categorical", choices=["sum", "mean", "max"], default="sum"
            ),
        },
        rationale="Counts the number of reviews per user, tunable by minimum review count and aggregation method.",
    )


def validate_realized_feature_signature(realized_feature):
    """Ensure all parameters are in the function signature."""
    code_str = realized_feature.code_str
    name = realized_feature.name
    params = realized_feature.parameters
    try:
        local_vars = {}
        exec(code_str, {}, local_vars)
        func = local_vars.get(name)
        if func is None:
            raise ValueError(f"Function '{name}' not found in realized code.")
        sig = inspect.signature(func)
        arg_names = set(sig.parameters.keys())
        expected = set(params.keys()) | {"df"}
        missing = expected - arg_names
        if missing:
            raise ValueError(f"Missing parameters in function signature: {missing}")
        print(f"[Signature Validation] {name}: PASS")
    except Exception as e:
        print(f"[Signature Validation] {name}: FAIL - {e}")


def main():
    # Setup session state
    session_state = SessionState(Path("test_run_dir"))
    session_state.set_state("db_path", "data/goodreads_curated.duckdb")
    session_state.set_state("fast_mode_sample_frac", 0.1)

    # Step 2: Inject candidate features
    candidate = get_dummy_candidate_feature()
    print("[Candidate Feature]")
    print(json.dumps(candidate.model_dump(), indent=2, default=str))
    session_state.set_state("candidate_features", [candidate.model_dump()])

    # Step 3: Realize features
    llm_config = {"model": "gpt-4", "temperature": 0.0}
    realization_agent = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)
    realization_agent.run()
    realized_features = session_state.get_state("features")
    print("\n[Realized Features]")
    for f in realized_features.values():
        print(json.dumps(f.model_dump(), indent=2, default=str))
        validate_realized_feature_signature(f)

    # Step 4: Run optimization
    optimizer = VULCANOptimizer(session=session_state)
    features_list = [f.model_dump() for f in realized_features.values()]
    print("\n[Running Optimization]")
    result = optimizer.optimize(features=features_list, n_trials=2, use_fast_mode=True)
    print("\n[Optimization Result]")
    print(result.json(indent=2))


if __name__ == "__main__":
    main()
