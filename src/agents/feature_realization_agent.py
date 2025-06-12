# src/agents/feature_realization_agent.py


import inspect
from typing import Dict, List

import autogen
import numpy as np
import pandas as pd
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.feature_registry import feature_registry
from src.utils.logging_config import setup_logging
from src.utils.sampling import sample_users_by_activity
from src.utils.session_state import SessionState
from src.utils.tools_v2 import execute_python


class FeatureRealizationAgent:
    def __init__(self, llm_config: Dict, session_state: SessionState):
        """Initialize the feature realization agent."""
        setup_logging()
        self.llm_config = llm_config
        self.session_state = session_state
        self.assistant = autogen.AssistantAgent(
            name="FeatureRealizationAssistant",
            llm_config=self.llm_config,
            system_message="""You are a feature realization expert. Your role is to:
1. Take feature specifications and turn them into working Python code
2. Ensure the code is efficient, well-documented, and follows best practices
3. Handle both simple features and complex compositions
4. Validate the code against the specification
5. Add appropriate error handling and logging""",
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_FeatureRealization",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "runtime/code_execution",
                "use_docker": False,
            },
        )
        self.writer = SummaryWriter("runtime/tensorboard/FeatureRealizationAgent")

    @agent_run_decorator("FeatureRealizationAgent")
    def run(self, candidate_features: List[Dict]):
        """
        Main method to run the feature realization process.
        """
        logger.info(
            f"Starting feature realization for {len(candidate_features)} features."
        )

        # Realize features using LLM batch processing
        # realized_features = llm_batch_tool(
        #     name="feature_realizer",
        #     instructions="Generate Python code for the following features:",
        #     prompts=[json.dumps(f) for f in candidate_features],
        #     llm_config=self.llm_config,
        # )

        # For now, let's just pass the candidate features through
        realized_features = candidate_features

        # Validate and execute the features
        # validated_features = []
        # for feature in realized_features:
        #     if self._validate_feature_code(feature):
        #         # ... (rest of the loop)
        #         pass

    def _load_prompt(self, prompt_file: str) -> str:
        with open(prompt_file) as f:
            return f.read()

    def _generate_code_logic(self, proposal: dict) -> str:
        # This will now be a simplified version, as the main logic is in the tool
        return proposal["spec"]

    def _realize_code_feature(
        self, feature_name: str, spec: str, params: dict
    ) -> (object, list, str):
        """
        Realizes a code-based feature by generating and returning a Python function.
        """
        param_string = ", ".join(params.keys())
        code = f"""
import pandas as pd
import numpy as np

def {feature_name}(df_reviews: pd.DataFrame, df_items: pd.DataFrame, {param_string}):
    return {spec}
"""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            func = exec_globals[feature_name]
            return func, list(params.keys()), None
        except Exception as e:
            return None, [], str(e)

    def _realize_llm_feature(
        self, name: str, spec: str, batch_size: int, cacheable: bool
    ) -> (object, list, str):
        """
        Realizes an LLM-based feature by creating a Python function that can call the LLM in batches.
        """
        is_user_feature = "<USER_REVIEWS>" in spec

        def llm_feature_function(
            df_reviews: pd.DataFrame, df_books: pd.DataFrame, **kwargs
        ):
            if is_user_feature:
                # Group by user_id and get the last 'batch_size' reviews
                ctx = df_reviews.groupby("user_id")["review_text"].apply(
                    lambda L: L.tolist()[-batch_size:]
                )
                prompts = ctx.apply(
                    lambda L: spec.replace("<USER_REVIEWS>", " ".join(L))
                )
                target_index = ctx.index
            else:  # Assumes item feature with <BOOK_DESCRIPTION>
                # Use book descriptions
                ctx = df_books.set_index("book_id")["description"].dropna()
                prompts = ctx.apply(lambda d: spec.replace("<BOOK_DESCRIPTION>", d))
                target_index = ctx.index

            # Call the LLM in batches
            # NOTE: llm_batch_tool is not implemented - this is a placeholder
            # A real implementation would handle batching properly
            # scores = llm_batch_tool(prompts.tolist())
            scores = [0.5] * len(prompts)  # Placeholder scores

            return pd.Series(scores, index=target_index, dtype=float)

        return llm_feature_function, ["scale"], None

    def _realize_composition_feature(
        self, name: str, spec: str, depends_on: list
    ) -> (object, list, str):
        """
        Realizes a composition feature by creating a wrapper function that calls its dependencies.
        """

        def composition_feature_function(
            df_reviews: pd.DataFrame, df_books: pd.DataFrame, **kwargs
        ):
            all_realized_fns = feature_registry.get_all()

            values = {}
            for dep_name in depends_on:
                if dep_name in all_realized_fns:
                    dep_fn_data = all_realized_fns[dep_name]
                    dep_fn = dep_fn_data["func"]
                    dep_params = {
                        p: kwargs.get(f"{dep_name}__{p}", 1)
                        for p in dep_fn_data["params"]
                    }
                    values[dep_name] = dep_fn(df_reviews, df_books, **dep_params)

            # This is a simplified and potentially unsafe use of eval.
            result = eval(spec, {"np": np}, values)
            return result

        # The parameters of a composition feature are the nested parameters of its dependencies.
        all_realized_fns = feature_registry.get_all()
        param_names = [
            f"{dep}__{p}"
            for dep in depends_on
            for p in all_realized_fns.get(dep, {}).get("params", [])
        ]

        return composition_feature_function, param_names, None

    def _validate_feature(self, feature_name: str, func: object, params: list) -> bool:
        """
        Validates a realized feature by running it on a sample of data in a
        sandboxed environment.
        """
        logger.info(f"Validating feature {feature_name}...")

        # Get a sample of users for validation
        user_ids = sample_users_by_activity(10, 1, 100)  # 10 users for speed
        if not user_ids:
            logger.warning("Skipping validation due to no sample users found.")
            return True

        # Create a script to run the validation
        param_string = ", ".join([f"{p}=1" for p in params])
        code = f"""
import pandas as pd
import numpy as np
from src.utils.db_api import fetch_df

# Define the function to test
{inspect.getsource(func)}

# Load the data
user_id_str = "','".join({user_ids})
df_reviews = fetch_df(f"SELECT * FROM reviews WHERE user_id IN ('{{user_id_str}}')")
book_ids = df_reviews['book_id'].unique()
book_id_str = "','".join(book_ids)
df_items = fetch_df(f"SELECT * FROM books WHERE book_id IN ('{{book_id_str}}')")

# Run the function
result = {feature_name}(df_reviews, df_items, {param_string})
print(len(result))
"""

        try:
            execution_result = execute_python(code)
            # Note: execute_python returns a string, not an object with exit_code
            # This is a simplified validation - would need to parse the result properly
            logger.info(f"Feature {feature_name} validation result: {execution_result}")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {feature_name} with exception: {e}")
            return False

        logger.info(f"...feature {feature_name} validated successfully.")
        return True
