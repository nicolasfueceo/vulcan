import json
import logging
from typing import Any, Dict, List

import autogen

from src.schemas.models import CandidateFeature
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir
from src.utils.session_state import SessionState
from src.utils.tools import get_table_sample

logger = logging.getLogger(__name__)


def validate_and_filter_features(
    features: List[Dict[str, Any]],
) -> List[CandidateFeature]:
    """
    Validates a list of candidate feature dictionaries and filters them.
    - Ensures name uniqueness
    - Validates spec syntax for 'code' types
    - (Future) Checks dependencies
    - (Future) Scores and ranks features
    """
    validated_features = []
    seen_names = set()

    for feat_data in features:
        # 1. Deduplicate by name
        name = feat_data.get("name")
        if not name or name in seen_names:
            logger.warning(f"Skipping feature with duplicate or missing name: {name}")
            continue
        seen_names.add(name)

        # 2. Validate using Pydantic model and custom validation
        try:
            feature = CandidateFeature(**feat_data)
            feature.validate_spec()
            validated_features.append(feature)
        except Exception as e:
            logger.error(f"Validation failed for candidate feature '{name}': {e}")

    # 3. (Future) Add scoring and filtering logic here

    return validated_features


def run_feature_ideation(session_state: SessionState, llm_config: Dict):
    """Orchestrates the Feature Ideation phase."""
    logger.info("--- Running Feature Ideation Step ---")

    hypotheses = session_state.get_final_hypotheses()
    if not hypotheses:
        logger.warning("No vetted hypotheses found. Skipping feature ideation.")
        return

    # 1. Prepare context for the prompt
    hypotheses_context = "\n".join(
        [f"- {h.id}: {h.summary} (Rationale: {h.rationale})" for h in hypotheses]
    )

    # Load view descriptions
    views_file = get_run_dir() / "generated_views.json"
    view_descriptions = "No views created in the previous step."
    if views_file.exists():
        with open(views_file, "r") as f:
            views_data = json.load(f).get("views", [])
            view_descriptions = "\n".join(
                [f"- {v['name']}: {v['rationale']}" for v in views_data]
            )

    # Get table samples
    tables_to_sample = ["curated_books", "curated_reviews", "user_stats_daily"]
    table_samples = "\n".join([get_table_sample(table) for table in tables_to_sample])

    system_prompt = load_prompt(
        "agents/feature_ideator.j2",
        hypotheses_context=hypotheses_context,
        view_descriptions=view_descriptions,
        table_samples=table_samples,
    )

    # 2. Initialize and run the agent
    def save_candidate_features(features: List[Dict[str, Any]]) -> str:
        """Callback tool for the agent to save its generated features."""
        logger.info(
            f"FeatureIdeationAgent proposed {len(features)} candidate features."
        )

        validated = validate_and_filter_features(features)

        session_state.set_candidate_features([f.model_dump() for f in validated])
        logger.info(
            f"Saved {len(validated)} valid candidate features to session state."
        )

        # Print summary
        for feature in validated:
            logger.info(f"  - Feature: {feature.name}, Rationale: {feature.rationale}")

        return "SUCCESS"

    # We assume a simple agent setup for now
    ideation_agent = autogen.AssistantAgent(
        name="FeatureIdeationAgent", system_message=system_prompt, llm_config=llm_config
    )
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Ideation",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

    user_proxy.register_function(
        function_map={"save_candidate_features": save_candidate_features}
    )

    user_proxy.initiate_chat(
        ideation_agent,
        message="Please generate candidate features based on the provided hypotheses and context. Call the `save_candidate_features` tool with your final list.",
    )

    logger.info("--- Feature Ideation Step Complete ---")
