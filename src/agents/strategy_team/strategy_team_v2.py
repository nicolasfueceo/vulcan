"""
Python-centric orchestration for Strategy Team agents.
This script demonstrates how to instantiate and use the core agent classes directly (not via LLM group chat):
- FeatureRealizationAgent
- FeatureAuditorAgent
- VULCANOptimizer
- EvaluationAgent
- ReflectionAgent

This is a minimal pipeline for demonstration and testing.
"""
from typing import List, Dict
from pathlib import Path

from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.agents.strategy_team.evaluation_agent import EvaluationAgent
from src.agents.strategy_team.reflection_agent import ReflectionAgent
from src.utils.session_state import SessionState
from src.utils.prompt_utils import load_prompt

# Dummy vision tool for the auditor
def dummy_vision_tool(plot_path: str) -> str:
    return f"Vision summary for {plot_path}"

def run_strategy_team_v2(llm_config: Dict, db_path: str, candidate_features: List, hypotheses: List, db_schema: str = ""):
    """
    Orchestrate the feature realization, auditing, optimization, evaluation, and reflection steps.
    """
    # Setup session state
    session_state = SessionState()
    session_state.set_state("db_path", db_path)
    # Convert Pydantic models to dicts for session state serialization
    session_state.set_state("candidate_features", [cf.model_dump() if hasattr(cf, 'model_dump') else dict(cf) for cf in candidate_features])
    session_state.set_state("final_hypotheses", [h.model_dump() if hasattr(h, 'model_dump') else dict(h) for h in hypotheses])
    session_state.set_state("realized_features", [])
    session_state.set_state("optimization_results", {})
    session_state.set_state("run_dir", "runtime/runs/test_strategy_team_v2")

    # 1. Feature Realization
    feature_engineer = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)
    feature_engineer.run()  # Realizes features and registers them
    realized_features = session_state.get_state("realized_features", [])


    # 3. Optimization
    optimizer = VULCANOptimizer(db_path=db_path, session=session_state)
    # The optimizer expects feature dicts, not objects
    features_for_optimization = [f.source_candidate if hasattr(f, 'source_candidate') else f for f in realized_features]
    opt_result = optimizer.optimize(features=features_for_optimization, n_trials=2, use_fast_mode=True)
    session_state.set_state("optimization_results", {"best_trial": opt_result})

    # 4. Evaluation
    evaluator = EvaluationAgent()
    evaluator.run(session_state)

    # 5. Reflection
    reflection_agent = ReflectionAgent(llm_config=llm_config)
    reflection_result = reflection_agent.run(session_state)
    session_state.set_state("reflection_result", reflection_result)

    return session_state

# If run as a script, demonstrate with dummy data
def main():
    llm_config = {"model": "gpt-4", "api_key": "sk-..."}  # Replace with real key
    db_path = "data/goodreads_curated.duckdb"
    # Minimal dummy CandidateFeature and Hypothesis objects
    from src.schemas.models import CandidateFeature, Hypothesis
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
    session_state = run_strategy_team_v2(llm_config, db_path, candidate_features, hypotheses)
    print("Final evaluation metrics:", session_state.get_state("final_evaluation_metrics"))
    print("Reflection result:", session_state.get_state("reflection_result"))

if __name__ == "__main__":
    import argparse
    import os
    import json
    import sys
    from src.schemas.models import CandidateFeature, Hypothesis

    parser = argparse.ArgumentParser(description="Run Strategy Team v2 Pipeline (Python agents, real LLM)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--db_path", type=str, default="data/goodreads_curated.duckdb", help="Path to DuckDB file")
    parser.add_argument("--candidate_features", type=str, help="Path to JSON file with CandidateFeature list")
    parser.add_argument("--hypotheses", type=str, help="Path to JSON file with Hypothesis list")
    parser.add_argument("--model", type=str, default="gpt-4", help="LLM model name (default: gpt-4)")
    args = parser.parse_args()

    # --- Load LLM config ---
    if not args.api_key or args.api_key.startswith("sk-test"):
        print("[ERROR] Please provide a real OpenAI API key via --api_key or OPENAI_API_KEY env var.")
        sys.exit(1)
    llm_config = {"model": args.model, "api_key": args.api_key}

    # --- Load candidate features and hypotheses ---
    def load_json_or_exit(path, cls):
        if not path:
            print(f"[ERROR] Please provide --{cls.__name__.lower()}s=<path.json>")
            sys.exit(1)
        with open(path, "r") as f:
            data = json.load(f)
        # Accept list of dicts or list of models
        return [cls(**item) if not isinstance(item, cls) else item for item in data]

    candidate_features = load_json_or_exit(args.candidate_features, CandidateFeature)
    hypotheses = load_json_or_exit(args.hypotheses, Hypothesis)

    # --- Run pipeline ---
    session_state = run_strategy_team_v2(llm_config, args.db_path, candidate_features, hypotheses)
    print("\n[RESULT] Final evaluation metrics:")
    print(json.dumps(session_state.get_state("final_evaluation_metrics", {}), indent=2, default=str))
    print("\n[RESULT] Reflection result:")
    print(json.dumps(session_state.get_state("reflection_result", {}), indent=2, default=str))
    print("\n[RESULT] Realized features:")
    for f in session_state.get_state("realized_features", []):
        print(f"- {getattr(f, 'name', str(f))}")
    print("\n[INFO] Pipeline run complete.")
