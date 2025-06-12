# Source Code Documentation

Generated on: 2025-06-12 20:28:44

This document contains the complete source code structure and contents of the `src` directory.

## 📁 Full Directory Structure

```
├── .gitignore
├── Dockerfile.dataanalysis
├── README.md
├── config/
│   └── OAI_CONFIG_LIST.json
├── data/
│   ├── .gitkeep
│   └── __init__.py
├── data_curation/
│   ├── README.md
│   ├── clean_data.py
│   ├── run.py
│   ├── sql/
│   │   ├── 00_setup.sql
│   │   └── 01_curate_goodreads.sql
│   └── steps/
│       ├── analyze_db.py
│       ├── drop_useless_tables.py
│       ├── get_curated_schema.py
│       ├── inspect_raw_dates.py
│       └── verify_curated_dates.py
├── docker-compose.yml
├── docs/
│   ├── context.md
│   ├── core_mission.md
│   ├── experiments.md
│   ├── project_status_report.md
│   └── test.md
├── generate_src_docs.py
├── requirements.txt
├── scripts/
│   └── test_prompt.py
├── src/
│   ├── agents/
│   │   ├── discovery_team/
│   │   │   ├── __pycache__/
│   │   │   └── insight_discovery_agents.py
│   │   └── strategy_team/
│   │       ├── __pycache__/
│   │       ├── evaluation_agent.py
│   │       ├── feature_realization_agent.py
│   │       ├── hypothesis_agents.py
│   │       ├── optimization_agent.py
│   │       ├── reasoning_agent.py
│   │       └── reflection_agent.py
│   ├── config/
│   │   ├── __pycache__/
│   │   ├── logging.py
│   │   ├── settings.py
│   │   └── tensorboard.py
│   ├── core/
│   │   ├── __pycache__/
│   │   ├── database.py
│   │   ├── llm.py
│   │   └── tools.py
│   ├── orchestrator.py
│   ├── prompts/
│   │   ├── agents/
│   │   │   ├── discovery_team/
│   │   │   │   ├── base_analyst.j2
│   │   │   │   ├── data_representer.j2
│   │   │   │   ├── pattern_seeker.j2
│   │   │   │   └── quantitative_analyst.j2
│   │   │   ├── feature_ideator.j2
│   │   │   ├── reflection_agent.j2
│   │   │   └── strategy_team/
│   │   │       ├── engineer_agent.j2
│   │   │       ├── hypothesis_agent.j2
│   │   │       └── strategist_agent.j2
│   │   ├── globals/
│   │   │   ├── base_agent.j2
│   │   │   ├── base_analyst.j2
│   │   │   ├── base_strategy.j2
│   │   │   └── core_mission.j2
│   │   └── helpers/
│   │       ├── db_schema.j2
│   │       └── tool_usage.j2
│   ├── schemas/
│   │   ├── __pycache__/
│   │   ├── eda_report_schema.json
│   │   └── models.py
│   └── utils/
│       ├── __pycache__/
│       ├── data_utils.py
│       ├── decorators.py
│       ├── feature_registry.py
│       ├── logging_utils.py
│       ├── prompt_utils.py
│       ├── pubsub.py
│       ├── run_utils.py
│       ├── sampling.py
│       ├── session_state.py
│       ├── testing_utils.py
│       └── tools.py
├── src_documentation.md
├── src_documentation_with_prompts.md
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── multi_agent_pipeline/
    │   ├── test_feature_ideation_agent_e2e.py
    │   ├── test_feature_realization.py
    │   ├── test_feature_realization_agent_e2e.py
    │   ├── test_optimization_agent_e2e.py
    │   ├── test_orchestrator_smoke.py
    │   ├── test_reflection_agent_e2e.py
    │   └── test_research_agent_e2e.py
    ├── schemas/
    │   └── feature_proposal_schema.json
    └── test_feature_realization_agent.py
```

## 📄 File Contents (src directory only)

### `agents/discovery_team/insight_discovery_agents.py`

**File size:** 1,116 bytes

```python
"""
Insight Discovery Team agents for exploratory data analysis.
This team is responsible for discovering patterns and insights in the data.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_insight_discovery_agents(
    llm_config: Dict,
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the insight discovery loop.
    Uses Jinja2 templates from src/prompts/agents/discovery_team/
    """

    # Load agent prompts from Jinja2 templates
    agent_prompts = {
        "DataRepresenter": load_prompt("agents/discovery_team/data_representer.j2"),
        "QuantitativeAnalyst": load_prompt(
            "agents/discovery_team/quantitative_analyst.j2"
        ),
        "PatternSeeker": load_prompt("agents/discovery_team/pattern_seeker.j2"),
    }

    # Create agents with loaded prompts
    agents = {
        name: autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=llm_config,
        )
        for name, prompt in agent_prompts.items()
    }

    return agents
```

### `agents/strategy_team/evaluation_agent.py`

**File size:** 1,943 bytes

```python
# src/agents/evaluation_agent.py
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem
from src.utils.pubsub import acquire_lock, publish, release_lock


class EvaluationAgent:
    def __init__(self, llm_config: dict = None):
        self.writer = SummaryWriter("runtime/tensorboard/EvaluationAgent")
        self.run_count = 0

    @agent_run_decorator("EvaluationAgent")
    def run(self, message: dict = {}):
        """
        Runs a final evaluation on the best model and logs metrics.
        Triggered by 'opt_complete'.
        """
        lock_name = "lock:EvaluationAgent"
        if not acquire_lock(lock_name):
            logger.info("EvaluationAgent is already running. Skipping.")
            return

        try:
            best_rmse = get_mem("best_rmse")
            best_params = get_mem("best_params")

            if not best_params:
                logger.warning("No best parameters found. Skipping evaluation.")
                return

            # In a real implementation, this agent would:
            # 1. Retrain the model on the full training data using best_params.
            # 2. Evaluate on a held-out test set.
            # 3. Calculate various metrics (e.g., NDCG, precision@k).
            # 4. Log everything to TensorBoard.

            logger.info(f"Final evaluation with best params: {best_params}")
            logger.info(f"Best validation RMSE: {best_rmse}")

            # For now, just log the best RMSE from validation as the final metric
            self.writer.add_scalar("final_test_rmse", best_rmse, self.run_count)
            self.writer.add_hparams(best_params, {"hparam/final_test_rmse": best_rmse})

            self.run_count += 1
            publish("evaluation_done", {"status": "success"})

        finally:
            release_lock(lock_name)
            self.writer.close()
```

### `agents/strategy_team/feature_realization_agent.py`

**File size:** 8,745 bytes

```python
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
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "runtime/code_execution",
                "use_docker": False,
            },
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
```

### `agents/strategy_team/hypothesis_agents.py`

**File size:** 1,580 bytes

```python
"""
Hypothesis & Strategy Team agents for refining insights into concrete hypotheses.
This team is responsible for strategic analysis and hypothesis generation.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_hypothesis_agents(
    llm_config: Dict, insights_report: str
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the hypothesis and strategy loop.
    Uses Jinja2 templates from src/prompts/agents/strategy_team/
    """

    # Load agent prompts from Jinja2 templates
    hypothesis_prompt = load_prompt(
        "agents/strategy_team/hypothesis_agent.j2", insights_report=insights_report
    )
    strategist_prompt = load_prompt("agents/strategy_team/strategist_agent.j2")
    engineer_prompt = load_prompt("agents/strategy_team/engineer_agent.j2")

    agent_defs = [
        ("HypothesisAgent", hypothesis_prompt),
        ("StrategistAgent", strategist_prompt),
        ("EngineerAgent", engineer_prompt),
    ]

    # Create agents with loaded prompts
    agents = {
        name: autogen.AssistantAgent(
            name=name, system_message=prompt, llm_config=llm_config
        )
        for name, prompt in agent_defs
    }

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Hypothesis",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "SUCCESS" in x.get("content", ""),
        code_execution_config={"use_docker": False},
    )

    agents["user_proxy"] = user_proxy
    return agents
```

### `agents/strategy_team/optimization_agent.py`

**File size:** 6,947 bytes

```python
# src/agents/optimization_agent.py
import numpy as np
import pandas as pd
from loguru import logger
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.feature_registry import feature_registry
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.session_state import SessionState
from src.utils.tools import execute_python


class OptimizationAgent:
    """
    An agent responsible for optimizing feature parameters and model hyperparameters.
    """

    def __init__(self, session_state: SessionState):
        logger.info("OptimizationAgent initialized.")
        self.session_state = session_state
        self.writer = SummaryWriter("runtime/tensorboard")
        self.trial_count = 0
        self.run_count = 0

    def _define_search_space(self) -> list:
        """Defines the search space for the optimization."""
        logger.info("Defining search space...")
        search_space = []

        # Add realized feature parameters to the search space
        all_realized_fns = feature_registry.get_all()
        for feature_name, data in all_realized_fns.items():
            for param_name in data.get("params", []):
                # For simplicity, we assume all code params are real-valued
                # and LLM params have a 'scale' name
                if "scale" in param_name:
                    search_space.append(
                        Real(0.1, 2.0, name=f"{feature_name}__{param_name}")
                    )
                else:
                    search_space.append(
                        Real(0.1, 10.0, name=f"{feature_name}__{param_name}")
                    )

        # Add FM hyperparameters to the search space
        search_space.extend(
            [
                Integer(8, 128, name="fm_n_factors"),
                Real(1e-5, 1.0, "log-uniform", name="fm_reg"),
                Categorical([True, False], name="fm_use_bias"),
            ]
        )

        logger.info(f"...search space defined with {len(search_space)} dimensions.")
        return search_space

    def _objective_function(self, params: list) -> float:
        """The objective function to minimize (e.g., validation RMSE)."""
        logger.info(f"Running trial {self.trial_count} with parameters: {params}")

        # 1. Parse parameters
        param_dict = {dim.name: value for dim, value in zip(self.search_space, params)}

        # 2. Generate features using execute_python instead of sql_tool
        code = """
# Get data for optimization
all_reviews = session_state.conn.execute("SELECT user_id, book_id, rating, timestamp FROM curated_reviews").df()
all_books = session_state.conn.execute("SELECT book_id, title, description FROM curated_books").df()
print(f"Loaded {len(all_reviews)} reviews and {len(all_books)} books")
"""

        try:
            result = execute_python(code)
            logger.info(f"Data loading result: {result}")
        except Exception as e:
            logger.error(f"Failed to load data for optimization: {e}")
            return float("inf")  # Return worst possible score

        all_realized_fns = feature_registry.get_all()
        user_features_dfs = []
        item_features_dfs = []

        for name, data in all_realized_fns.items():
            func = data["func"]
            func_params = {
                p: param_dict.get(f"{name}__{p}", 1.0) for p in data.get("params", [])
            }
            # This is a simplification: we assume the function returns a Series
            # and we can determine if it's a user or item feature by its index name
            try:
                # For now, create dummy features since the actual feature functions may not work
                if "user" in name.lower():
                    dummy_feature = pd.Series([0.5] * 100, name=name)
                    dummy_feature.index.name = "user_id"
                    user_features_dfs.append(dummy_feature)
                else:
                    dummy_feature = pd.Series([0.5] * 100, name=name)
                    dummy_feature.index.name = "book_id"
                    item_features_dfs.append(dummy_feature)
            except Exception as e:
                logger.warning(f"Failed to generate feature {name}: {e}")

        user_features = (
            pd.concat(user_features_dfs, axis=1).fillna(0)
            if user_features_dfs
            else None
        )
        item_features = (
            pd.concat(item_features_dfs, axis=1).fillna(0)
            if item_features_dfs
            else None
        )

        # 3. For now, return a dummy RMSE since the full optimization pipeline is complex
        dummy_rmse = np.random.uniform(1.0, 5.0)  # Random RMSE between 1 and 5

        # 4. Log results
        self.writer.add_scalar("Validation_RMSE", dummy_rmse, self.trial_count)
        for name, value in param_dict.items():
            if isinstance(value, str) or isinstance(value, bool):
                self.writer.add_text(f"param/{name}", str(value), self.trial_count)
            else:
                self.writer.add_scalar(f"param/{name}", value, self.trial_count)

        self.trial_count += 1
        return dummy_rmse

    @agent_run_decorator("OptimizationAgent")
    def run(self, message: dict = {}):
        """
        Runs the Bayesian optimization pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:OptimizationAgent"
        if not acquire_lock(lock_name):
            logger.info("OptimizationAgent is already running. Skipping.")
            return

        try:
            self.search_space = self._define_search_space()
            if not self.search_space:
                logger.warning("Search space is empty. Skipping optimization.")
                return

            result = gp_minimize(
                self._objective_function,
                self.search_space,
                n_calls=20,
                random_state=42,
            )

            best_params_list = result.x
            best_params = {
                dim.name: value
                for dim, value in zip(self.search_space, best_params_list)
            }
            best_rmse = result.fun

            self.session_state.set_best_params(best_params)
            self.session_state.set_best_rmse(best_rmse)

            self.writer.close()
            logger.info(
                f"OptimizationAgent: Best RMSE: {best_rmse}, Best Params: {best_params}"
            )

            # Publish an event to signal completion
            publish(
                "optimization_done",
                {
                    "status": "success",
                    "best_rmse": best_rmse,
                    "best_params": best_params,
                },
            )
        finally:
            self.run_count += 1
            release_lock(lock_name)
```

### `agents/strategy_team/reasoning_agent.py`

**File size:** 3,793 bytes

```python
# src/agents/reasoning_agent.py
import json
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem, set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.schemas import PrioritizedHypothesis


class ReasoningAgent:
    """
    An agent responsible for prioritizing hypotheses based on feasibility and impact.
    """

    def __init__(self, llm_config: dict):
        logger.info("ReasoningAgent initialized.")
        self.assistant = autogen.AssistantAgent(
            name="ReasoningAgent",
            system_message="You are an expert data strategist. Your goal is to prioritize hypotheses based on their feasibility and potential impact. You must call the `save_prioritized_hypotheses` function with your results.",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_ReasoningAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReasoningAgent")
        self.run_count = get_mem("reasoning_run_count") or 0

    @agent_run_decorator("ReasoningAgent")
    def run(self, message: dict = {}):
        """
        Runs the hypothesis prioritization pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:ReasoningAgent"
        if not acquire_lock(lock_name):
            logger.info("ReasoningAgent is already running. Skipping.")
            return

        try:
            hypotheses = get_mem("hypotheses")
            if not hypotheses:
                logger.warning(
                    "No hypotheses found in memory. Skipping prioritization."
                )
                return

            def save_prioritized_hypotheses(
                prioritized_hypotheses: List[PrioritizedHypothesis],
            ):
                """Saves the prioritized hypotheses to memory and publishes an event."""
                set_mem(
                    "prioritized_hypotheses",
                    [h.model_dump() for h in prioritized_hypotheses],
                )
                logger.info(
                    f"Saved {len(prioritized_hypotheses)} prioritized hypotheses to memory."
                )

                # Publish an event to trigger the next agent
                publish(
                    "priorities_ready",
                    {
                        "status": "success",
                        "prioritized_hypotheses": [
                            h.model_dump() for h in prioritized_hypotheses
                        ],
                    },
                )
                self.writer.add_scalar(
                    "hypotheses_prioritized",
                    len(prioritized_hypotheses),
                    self.run_count,
                )
                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={
                    "save_prioritized_hypotheses": save_prioritized_hypotheses
                }
            )

            prompt = f"""
            Given the following hypotheses, please assess each one for its
            feasibility (1-5) and potential impact (1-5). A higher score is better.
            Call the `save_prioritized_hypotheses` function with your results.

            Hypotheses:
            {json.dumps(hypotheses, indent=2)}
            """

            self.user_proxy.initiate_chat(self.assistant, message=prompt)
            self.run_count += 1
            set_mem("reasoning_run_count", self.run_count)
            self.writer.close()
        finally:
            release_lock(lock_name)
```

### `agents/strategy_team/reflection_agent.py`

**File size:** 4,056 bytes

```python
# src/agents/reflection_agent.py
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.schemas.models import Hypothesis
from src.utils.decorators import agent_run_decorator
from src.utils.prompt_utils import load_prompt
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.session_state import SessionState


class ReflectionAgent:
    """
    An agent responsible for reflecting on the optimization results and
    suggesting next steps.
    """

    def __init__(self, llm_config: dict, session_state: SessionState):
        logger.info("ReflectionAgent initialized.")
        self.session_state = session_state
        self.assistant = autogen.AssistantAgent(
            name="ReflectionAgent",
            system_message="",  # Will be set dynamically before each run.
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_ReflectionAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReflectionAgent")

    @agent_run_decorator("ReflectionAgent")
    def run(self, message: dict = {}):
        """
        Runs the reflection pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:ReflectionAgent"
        if not acquire_lock(lock_name):
            logger.info("ReflectionAgent is already running. Skipping.")
            return

        try:
            bo_history = self.session_state.get_bo_history()
            best_params = self.session_state.get_best_params()

            if not bo_history and not best_params:
                logger.warning(
                    "No optimization history or best parameters found. Skipping reflection."
                )
                publish(
                    "pipeline_done",
                    {
                        "status": "success",
                        "reason": "No optimization results to reflect on.",
                    },
                )
                return

            def save_reflections(reflections: List[str], next_steps: List[Hypothesis]):
                """Saves the generated reflections and next steps to session state and publishes an event."""
                reflection_data = {
                    "reflections": reflections,
                    "next_steps": [h.model_dump() for h in next_steps],
                }
                self.session_state.add_reflection(reflection_data)

                if next_steps:
                    self.session_state.finalize_hypotheses(next_steps)

                logger.info("Saved reflections and next steps to session state.")

                if next_steps:
                    publish(
                        "start_eda",
                        {"reason": "Reflection suggested new hypotheses."},
                    )
                else:
                    publish(
                        "pipeline_done",
                        {"reason": "Pipeline converged."},
                    )

                run_count = self.session_state.increment_reflection_run_count()
                self.writer.add_scalar(
                    "new_hypotheses_from_reflection", len(next_steps), run_count
                )

                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_reflections": save_reflections}
            )

            prompt = load_prompt(
                "agents/reflection_agent.j2",
                bo_history=bo_history,
                best_params=best_params,
            )

            self.assistant.update_system_message(prompt)
            self.user_proxy.initiate_chat(
                self.assistant,
                message="Reflect on the provided results and suggest next steps. You must call the `save_reflections` function with your results.",
            )
            self.writer.close()
        finally:
            release_lock(lock_name)
```

### `config/logging.py`

**File size:** 1,715 bytes

```python
import logging
import sys

from src.utils.run_utils import format_log_message, get_run_logs_dir


class RunContextFormatter(logging.Formatter):
    """Custom formatter that includes run context in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with run context."""
        record.msg = format_log_message(str(record.msg))
        return super().format(record)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up logging configuration with run-specific paths."""
    # Create run-specific log directory
    log_dir = get_run_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = RunContextFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = RunContextFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
```

### `config/settings.py`

**File size:** 1,132 bytes

```python
"""
Configuration settings for the VULCAN project.
Contains database paths, LLM configurations, and other global constants.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

# Database configuration
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "goodreads_curated.duckdb"))

# LLM Configuration - Default configuration that can be used across agents
# This will be overridden by the orchestrator with actual API keys and config lists
LLM_CONFIG = {
    "config_list": [],  # Will be populated by orchestrator from OAI_CONFIG_LIST.json
    "cache_seed": None,
    "temperature": 0.7,
    "timeout": 120,
}

# Agent configuration
MAX_CONSECUTIVE_AUTO_REPLY = 10
CODE_EXECUTION_TIMEOUT = 120

# Plotting configuration
PLOT_DPI = 300
PLOT_STYLE = "default"
PLOT_PALETTE = "husl"

# OpenAI configuration
OPENAI_MODEL_VISION = "gpt-4o"
OPENAI_MODEL_TEXT = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 1000

# Database connection settings
DB_READ_ONLY = False  # Allow writes for temporary views
DB_TIMEOUT = 30
```

### `config/tensorboard.py`

**File size:** 1,364 bytes

```python
import subprocess
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from src.utils.run_utils import get_run_tensorboard_dir


def start_tensorboard() -> None:
    """Start TensorBoard in the background."""
    log_dir = get_run_tensorboard_dir()
    try:
        # Start TensorBoard in the background
        subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir), "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Warning: Could not start TensorBoard: {e}")


def get_tensorboard_writer() -> SummaryWriter:
    """Get a TensorBoard writer for the current run."""
    log_dir = get_run_tensorboard_dir()
    return SummaryWriter(log_dir=str(log_dir))


def log_metric(
    writer: SummaryWriter, tag: str, value: float, step: Optional[int] = None
) -> None:
    """Log a metric to TensorBoard."""
    writer.add_scalar(tag, value, step)


def log_metrics(
    writer: SummaryWriter, metrics: dict, step: Optional[int] = None
) -> None:
    """Log multiple metrics to TensorBoard."""
    for tag, value in metrics.items():
        log_metric(writer, tag, value, step)


def log_hyperparams(writer: SummaryWriter, hparams: dict) -> None:
    """Log hyperparameters to TensorBoard."""
    writer.add_hparams(hparams, {})
```

### `core/database.py`

**File size:** 7,204 bytes

```python
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config.settings import DB_PATH

logger = logging.getLogger(__name__)


def check_db_schema() -> bool:
    """
    Checks if the database has the required tables and they are not empty.
    """
    db_file = Path(DB_PATH)
    if not db_file.exists() or db_file.stat().st_size == 0:
        return False
    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            required_tables = {"books", "reviews", "users"}

            if not required_tables.issubset(tables):
                return False

            for table in required_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count == 0:
                    return False
        return True
    except duckdb.Error as e:
        logger.warning(f"Database schema check failed, will attempt to rebuild: {e}")
        return False


def ingest_json_to_duckdb():
    """
    Ingests data from gzipped JSON files into DuckDB, creating the schema.
    """
    books_json_path = "data/books.json.gz"
    reviews_json_path = "data/reviews.json.gz"

    logger.info(f"Starting ingestion from {books_json_path} and {reviews_json_path}")

    with duckdb.connect(database=DB_PATH, read_only=False) as conn:
        logger.info("Creating 'books' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE books AS 
            SELECT * 
            FROM read_json_auto('{books_json_path}', format='newline_delimited');
        """)
        logger.info("'books' table created.")

        logger.info("Creating 'reviews' table...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE reviews AS 
            SELECT *
            FROM read_json_auto('{reviews_json_path}', format='newline_delimited');
        """)
        logger.info("'reviews' table created.")

        logger.info("Creating 'users' table from distinct reviewers...")
        conn.execute("""
            CREATE OR REPLACE TABLE users AS
            SELECT DISTINCT user_id FROM reviews;
        """)
        logger.info("'users' table created.")

    logger.info("Data ingestion from JSON files to DuckDB complete.")


def fetch_df(query: str) -> pd.DataFrame:
    """
    Connects to the database, executes a query, and returns a DataFrame.
    """
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        return conn.execute(query).fetchdf()


def get_db_schema_string() -> str:
    """
    Introspects the database using SUMMARIZE and returns a detailed schema string
    with summary statistics. Connects in-process to avoid file locking issues.
    """
    schema_parts = []
    db_path = str(DB_PATH)  # Ensure it's a string for DuckDB

    try:
        logger.debug(f"Generating database schema from: {db_path}")

        # Connect in-process to an in-memory database to avoid file locks
        with duckdb.connect() as conn:
            # Attach the main database file in READ_ONLY mode, giving it an alias 'db'
            conn.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")

            # Query the information_schema to find tables in the attached database's 'main' schema
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_catalog = 'db';"
            ).fetchdf()

            if tables_df.empty:
                # Fallback to a simpler SHOW TABLES if the schema query fails
                try:
                    tables_df = conn.execute("SHOW TABLES FROM db;").fetchdf()
                    logger.debug("Used SHOW TABLES fallback method")
                except Exception:
                    logger.error(
                        "Failed to list tables via both information_schema and SHOW TABLES"
                    )
                    return "ERROR: No tables found in the attached database. Could not list tables via information_schema or SHOW TABLES."

            if tables_df.empty:
                logger.warning("No tables found in the database")
                return "ERROR: No tables found in the attached database."

            logger.debug(f"Found {len(tables_df)} tables in database")

            for _, row in tables_df.iterrows():
                table_name = row["table_name"] if "table_name" in row else row["name"]

                # We must use the 'db' alias to refer to tables in the attached database
                qualified_table_name = f'db."{table_name}"'

                try:
                    row_count_result = conn.execute(
                        f"SELECT COUNT(*) FROM {qualified_table_name};"
                    ).fetchone()
                    row_count = row_count_result[0] if row_count_result else 0
                    schema_parts.append(f"TABLE: {table_name} ({row_count:,} rows)")

                    # Use the SUMMARIZE command to get schema and statistics
                    summary_df = conn.execute(
                        f"SUMMARIZE {qualified_table_name};"
                    ).fetchdf()

                    for _, summary_row in summary_df.iterrows():
                        col_name = summary_row["column_name"]
                        col_type = summary_row["column_type"]
                        null_pct = summary_row["null_percentage"]

                        stats = [f"NULLs: {null_pct}%"]

                        # Add type-specific stats for a richer summary
                        if "VARCHAR" in col_type.upper():
                            unique_count = summary_row.get("approx_unique")
                            if unique_count is not None:
                                stats.append(f"~{int(unique_count)} unique values")
                        elif any(
                            t in col_type.upper()
                            for t in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
                        ):
                            min_val = summary_row.get("min")
                            max_val = summary_row.get("max")
                            if min_val is not None and max_val is not None:
                                stats.append(f"range: [{min_val}, {max_val}]")

                        schema_parts.append(
                            f"  - {col_name} ({col_type}) [{', '.join(stats)}]"
                        )
                    schema_parts.append("")

                except Exception as table_error:
                    logger.warning(
                        f"Failed to analyze table {table_name}: {table_error}"
                    )
                    schema_parts.append(f"TABLE: {table_name} (analysis failed)")
                    schema_parts.append("")

        result = "\n".join(schema_parts)
        logger.debug(f"Generated schema string with {len(result)} characters")
        return result

    except Exception as e:
        logger.error(f"Failed to get database schema using SUMMARIZE method: {e}")
        logger.exception(e)
        return (
            f"ERROR: Could not retrieve database schema from {db_path}. Error: {str(e)}"
        )
```

### `core/llm.py`

**File size:** 427 bytes

```python
def call_llm_batch(prompts: list) -> list:
    """
    A placeholder for a utility that calls an LLM with a batch of prompts.
    """
    # In a real implementation, this would use a library like `litellm`
    # to handle batching and API calls.
    print(f"Calling LLM with a batch of {len(prompts)} prompts.")

    # For now, return random scores for testing.
    import random

    return [random.random() for _ in prompts]
```

### `core/tools.py`

**File size:** 2,013 bytes

```python
"""
This module provides a collection of custom tools for the VULCAN agents to use
within their Python execution environment. These functions are designed to be
injected into the context available to the `execute_python` tool.
"""

import duckdb
from loguru import logger

from src.config.settings import DB_PATH


def get_table_sample(table_name: str, n_samples: int = 5) -> str:
    """
    Retrieves a random sample of rows from a specified table in the database.

    Args:
        table_name (str): The name of the table to sample from.
        n_samples (int): The number of rows to retrieve. Defaults to 5.

    Returns:
        str: A string representation of the sampled data in a markdown-friendly format,
             or an error message if the table cannot be accessed.
    """
    if not isinstance(table_name, str) or not table_name.isidentifier():
        return f"ERROR: Invalid table name '{table_name}'. Table names must be valid Python identifiers."

    if not isinstance(n_samples, int) or n_samples <= 0:
        return f"ERROR: Invalid number of samples '{n_samples}'. Must be a positive integer."

    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            # Use DuckDB's built-in TABLESAMPLE for efficient random sampling
            query = f'SELECT * FROM "{table_name}" USING SAMPLE {n_samples} ROWS;'
            sample_df = conn.execute(query).fetchdf()

            if sample_df.empty:
                return f"No data returned for table '{table_name}'. It may be empty."

            return (
                f"Sample of '{table_name}' ({n_samples} rows):\n"
                f"{sample_df.to_markdown(index=False)}"
            )
    except duckdb.CatalogException:
        return f"ERROR: Table '{table_name}' not found in the database."
    except Exception as e:
        logger.error(f"Failed to sample table '{table_name}': {e}")
        return (
            f"ERROR: An unexpected error occurred while sampling table '{table_name}'."
        )
```

### `orchestrator.py`

**File size:** 25,771 bytes

```python
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# This is no longer needed with a proper project structure and installation.
# # Add project root to Python path for proper imports
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
import autogen
from loguru import logger

from src.agents.discovery_team.insight_discovery_agents import (
    get_insight_discovery_agents,
)
from src.agents.strategy_team.hypothesis_agents import get_hypothesis_agents
from src.config.logging import setup_logging
from src.utils.run_utils import config_list_from_json, init_run
from src.utils.session_state import SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    execute_python,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    vision_tool,
)

# Load environment variables


def _safe_execute_python(code: str) -> str:
    """Wrapper for execute_python that guarantees a string return and catches exceptions."""
    try:
        result = execute_python(code)
        logging.debug(
            f"_safe_execute_python result length: {len(result) if result else 'NONE'}"
        )
        if result is None or result == "":
            return "ERROR: execute_python returned no output."
        return result
    except Exception as e:
        import traceback

        logging.error(f"_safe_execute_python caught exception: {e}")
        return f"ERROR: execute_python failed: {e}\n{traceback.format_exc()}"


def _safe_vision_tool(image_path: str, prompt: str) -> str:
    """Wrapper for vision_tool that guarantees a string return and catches exceptions."""
    try:
        result = vision_tool(image_path=image_path, prompt=prompt)
        if result is None or result == "":
            return "ERROR: vision_tool returned no output."
        return result
    except Exception as e:
        import traceback

        return f"ERROR: vision_tool failed: {e}\n{traceback.format_exc()}"


def _make_safe_add_insight(add_insight_func):
    def _safe_add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        supporting_code: str = "",
        plot_path: str = "",
        plot_interpretation: str = "",
    ) -> str:
        try:
            result = add_insight_func(
                title=title,
                finding=finding,
                source_representation=source_representation,
                supporting_code=supporting_code,
                plot_path=plot_path,
                plot_interpretation=plot_interpretation,
            )
            if result is None or result == "":
                return "ERROR: add_insight_to_report returned no output."
            return result
        except Exception as e:
            import traceback

            return f"ERROR: add_insight_to_report failed: {e}\n{traceback.format_exc()}"

    return _safe_add_insight_to_report


def run_discovery_loop(session_state: SessionState):
    """Orchestrates the Insight Discovery Team to find patterns in the data."""
    logging.info("--- Running Insight Discovery Loop ---")

    config_list = config_list_from_json(
        os.getenv("OAI_CONFIG_LIST", "config/OAI_CONFIG_LIST.json")
    )

    # Validate and substitute the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("OPENAI_API_KEY environment variable is empty or not set.")

    logging.info(f"API key loaded with length: {len(api_key)}")

    # Manually substitute the API key to ensure it's loaded
    for config in config_list:
        if config.get("api_key") == "${OPENAI_API_KEY}":
            config["api_key"] = api_key
            logging.info("Substituted API key in config")
        elif not config.get("api_key"):
            raise ValueError(f"No API key found in config: {config}")

    # Validate final config
    for i, config in enumerate(config_list):
        if not config.get("api_key") or config.get("api_key").strip() == "":
            raise ValueError(
                f"Config {i} has empty API key after substitution: {config}"
            )

    logging.info(f"Final config list validated with {len(config_list)} configurations")

    llm_config = {"config_list": config_list, "cache_seed": None}

    # Create proper UserProxy for tool execution
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=30,  # Per-agent limit, not total conversation limit
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    assistant_agents = get_insight_discovery_agents(llm_config)

    # Register tools using autogen.register_function for each assistant
    add_insight_func = get_add_insight_tool(session_state)
    safe_add_insight_func = _make_safe_add_insight(add_insight_func)

    for agent in assistant_agents.values():
        autogen.register_function(
            _safe_execute_python,
            caller=agent,
            executor=user_proxy,
            name="execute_python",
            description="Executes Python code in a sandboxed environment with database connection",
        )
        autogen.register_function(
            _safe_vision_tool,
            caller=agent,
            executor=user_proxy,
            name="vision_tool",
            description="Analyzes image files using vision AI",
        )
        autogen.register_function(
            safe_add_insight_func,
            caller=agent,
            executor=user_proxy,
            name="add_insight_to_report",
            description="Saves structured insights to the session report",
        )

    # Enhanced GroupChat with intelligent context management
    group_chat = autogen.GroupChat(
        agents=[user_proxy] + list(assistant_agents.values()),
        messages=[],
        max_round=500,  # High limit instead of -1 (uncapped), our SmartGroupChatManager will handle termination
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Close the session database connection to avoid lock conflicts during agent execution
    logging.info("Closing database connection for agent execution...")
    session_state.close_connection()

    def should_continue_exploration(
        session_state: SessionState, round_count: int
    ) -> bool:
        """Determines if exploration should continue based on insights and coverage."""
        insights = len(session_state.insights)
        min_insights = 8  # Minimum insights needed for comprehensive analysis

        # Always continue if we don't have enough insights
        if insights < min_insights:
            logging.info(
                f"Continuing exploration: {insights}/{min_insights} insights captured"
            )
            return True

        # After minimum insights, check for completeness every 10 rounds
        if round_count % 10 == 0 and insights >= min_insights:
            logging.info(
                f"Evaluating exploration completeness: {insights} insights, round {round_count}"
            )

            # Check if we have coverage across major areas
            insight_titles = [
                insight.title.lower() for insight in session_state.insights
            ]
            coverage_areas = {
                "rating": any("rating" in title for title in insight_titles),
                "genre": any(
                    any(term in title for term in ["genre", "shelf", "category"])
                    for title in insight_titles
                ),
                "author": any("author" in title for title in insight_titles),
                "temporal": any(
                    any(term in title for term in ["time", "year", "date", "temporal"])
                    for title in insight_titles
                ),
                "user": any("user" in title for title in insight_titles),
            }

            covered_areas = sum(coverage_areas.values())
            if covered_areas >= 4:  # Need coverage of at least 4/5 major areas
                logging.info(
                    f"Sufficient coverage achieved: {covered_areas}/5 areas covered"
                )
                return False

        # Safety limit - don't run indefinitely
        if round_count > 200:
            logging.warning(
                f"Reached maximum round limit ({round_count}), stopping exploration"
            )
            return False

        # Continue if we haven't reached the insight threshold or coverage
        return True

    def get_progress_prompt(session_state: SessionState, round_count: int) -> str:
        """Generate a progress prompt to guide agents when they seem stuck."""
        insights = len(session_state.insights)

        if insights == 0 and round_count > 5:
            return "\n\nIMPORTANT: No insights have been captured yet. Please ensure you call `add_insight_to_report()` after each analysis to record your findings. Focus on generating actual insights, not just data exploration."

        if insights < 4 and round_count > 15:
            return f"\n\nPROGRESS CHECK: Only {insights} insights captured after {round_count} rounds. Please focus on generating concrete insights using `add_insight_to_report()` and ensure comprehensive coverage of rating patterns, genres, authors, and user behavior."

        # Check coverage gaps
        insight_titles = [insight.title.lower() for insight in session_state.insights]
        missing_areas = []
        if not any("rating" in title for title in insight_titles):
            missing_areas.append("rating analysis")
        if not any(
            any(term in title for term in ["genre", "shelf", "category"])
            for title in insight_titles
        ):
            missing_areas.append("genre/category analysis")
        if not any("author" in title for title in insight_titles):
            missing_areas.append("author analysis")
        if not any(
            any(term in title for term in ["time", "year", "date", "temporal"])
            for title in insight_titles
        ):
            missing_areas.append("temporal analysis")
        if not any("user" in title for title in insight_titles):
            missing_areas.append("user behavior analysis")

        if missing_areas and round_count % 20 == 0:
            return f"\n\nCOVERAGE GAP: Missing analysis in: {', '.join(missing_areas)}. Please prioritize these areas in your next analysis."

        return ""

    def compress_conversation_context(
        messages: list, keep_recent: int = 20, llm_config: dict = None
    ) -> list:
        """Intelligently compress conversation context using LLM summarization."""
        if len(messages) <= keep_recent * 2:
            return messages  # No compression needed if conversation is still short

        logging.info(
            f"Compressing context with LLM: {len(messages)} -> target ~{keep_recent * 2} messages"
        )

        # Always preserve system messages and recent messages
        system_messages = [
            msg for msg in messages[:3] if msg.get("role") in ["user", "system"]
        ]
        recent_messages = messages[-keep_recent:]
        middle_messages = messages[len(system_messages) : -keep_recent]

        if not middle_messages:
            return messages

        try:
            # Create an LLM client for compression
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Group middle messages into chunks for summarization
            chunk_size = 15  # Process in chunks to avoid token limits
            compressed_summaries = []

            for i in range(0, len(middle_messages), chunk_size):
                chunk = middle_messages[i : i + chunk_size]

                # Create conversation text for summarization
                conversation_text = ""
                for msg in chunk:
                    role = msg.get("name", msg.get("role", "unknown"))
                    content = msg.get("content", "")
                    conversation_text += f"{role}: {content}\n\n"

                # Ask LLM to compress this chunk
                compression_prompt = f"""You are compressing a conversation between data analysis agents exploring a book recommendation database. 

Please create a concise summary that preserves:
1. All specific insights and findings discovered
2. Key analysis results, correlations, and statistics  
3. Important database views created
4. Significant plots generated and their interpretations
5. Critical decision points and conclusions

Focus on preserving factual discoveries and actionable insights while removing redundant discussion.

CONVERSATION CHUNK TO COMPRESS:
{conversation_text}

COMPRESSED SUMMARY (preserve all insights and key findings):"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use mini for cost efficiency in compression
                    messages=[{"role": "user", "content": compression_prompt}],
                    max_tokens=64000,
                    temperature=0.1,  # Low temperature for consistent compression
                )

                summary = response.choices[0].message.content

                # Create a compressed message
                compressed_msg = {
                    "role": "assistant",
                    "name": "ContextCompressor",
                    "content": f"[COMPRESSED SUMMARY of rounds {i + len(system_messages) + 1}-{i + len(system_messages) + len(chunk)}]\n\n{summary}",
                }
                compressed_summaries.append(compressed_msg)

            logging.info(
                f"LLM compression: {len(middle_messages)} messages -> {len(compressed_summaries)} summaries"
            )

            # Combine preserved and compressed content
            compressed_context = (
                system_messages + compressed_summaries + recent_messages
            )
            logging.info(
                f"Context compression completed: {len(messages)} -> {len(compressed_context)} messages"
            )
            return compressed_context

        except Exception as e:
            logging.warning(
                f"LLM compression failed: {e}. Falling back to keyword-based compression."
            )
            return _fallback_compression(messages, keep_recent)

    def _fallback_compression(messages: list, keep_recent: int = 20) -> list:
        """Fallback keyword-based compression if LLM compression fails."""
        if len(messages) <= keep_recent * 2:
            return messages

        # Always keep the initial system message and recent messages
        system_messages = [
            msg for msg in messages[:3] if msg.get("role") in ["user", "system"]
        ]
        recent_messages = messages[-keep_recent:]

        # Extract key insights and tool execution results from middle messages
        key_messages = []
        for msg in messages[len(system_messages) : -keep_recent]:
            content = msg.get("content", "")
            # Keep messages with insights, analysis results, or tool outputs
            if any(
                keyword in content.lower()
                for keyword in [
                    "insight:",
                    "analysis:",
                    "correlation:",
                    "finding:",
                    "plot_saved:",
                    "view_created:",
                    "stdout:",
                    "pattern:",
                    "hypothesis:",
                    "recommendation:",
                    "conclusion:",
                ]
            ):
                key_messages.append(msg)

        # Combine and return compressed context
        compressed = (
            system_messages + key_messages[-8:] + recent_messages
        )  # Keep max 8 key messages
        logging.info(
            f"Fallback compression completed: retained {len(compressed)} messages"
        )
        return compressed

    try:
        round_count = 0
        initial_message = "Team, let's begin our analysis. The database schema and our mission are in your system prompts. Please start by planning your first exploration step."

        # Enhanced conversation with custom termination logic
        class SmartGroupChatManager(autogen.GroupChatManager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.round_count = 0

            def run_chat(self, messages, sender, config=None):
                """Override the main chat runner to add our smart features."""
                self.round_count += 1

                # Log progress
                if self.round_count % 10 == 0:
                    insights_count = len(session_state.insights)
                    logging.info(
                        f"Exploration progress: Round {self.round_count}, {insights_count} insights captured"
                    )
                    if session_state.insights:
                        recent_insights = session_state.insights[-3:]
                        logging.info(
                            f"Recent insights: {[insight.title for insight in recent_insights]}"
                        )

                # Apply context compression periodically
                if self.round_count % 25 == 0:
                    try:
                        self.groupchat.messages = compress_conversation_context(
                            self.groupchat.messages, llm_config=llm_config
                        )
                        logging.info(
                            f"Applied LLM context compression at round {self.round_count}"
                        )
                    except Exception as e:
                        logging.warning(f"Context compression failed: {e}")

                # Check termination only after sufficient rounds
                if self.round_count > 15:
                    if not should_continue_exploration(session_state, self.round_count):
                        logging.info(
                            "Exploration criteria met, terminating conversation"
                        )
                        # Add termination message to conversation
                        termination_msg = {
                            "role": "assistant",
                            "content": "TERMINATE - Exploration criteria met. Sufficient insights captured with good coverage.",
                            "name": "SystemCoordinator",
                        }
                        self.groupchat.messages.append(termination_msg)
                        return True  # Signal termination

                # Add progress prompts when needed
                if self.round_count > 5 and self.round_count % 15 == 0:
                    progress_prompt = get_progress_prompt(
                        session_state, self.round_count
                    )
                    if progress_prompt:
                        logging.info(
                            f"Adding progress guidance at round {self.round_count}"
                        )
                        guidance_msg = {
                            "role": "user",
                            "content": progress_prompt,
                            "name": "SystemCoordinator",
                        }
                        self.groupchat.messages.append(guidance_msg)

                # Call the parent implementation
                res = super().run_chat(messages, sender, config)
                # super() returns a tuple (final, reply). If reply is None, substitute a fallback.
                if isinstance(res, tuple) and len(res) == 2:
                    final, reply = res
                    if reply is None:
                        logging.warning(
                            "Parent run_chat returned None reply; substituting fallback message to avoid crash."
                        )
                        reply = {
                            "role": "assistant",
                            "content": "ERROR: Reply generation failed (empty). Please review the previous tool call outputs and ensure a proper assistant response.",
                        }
                    return final, reply
                # In unexpected cases, just return res as-is
                return res

        # Create the enhanced manager
        smart_manager = SmartGroupChatManager(
            groupchat=group_chat, llm_config=llm_config
        )

        # Start the conversation using AutoGen's standard pattern
        user_proxy.initiate_chat(smart_manager, message=initial_message)

        insights_count = len(session_state.insights)
        logging.info(
            f"Exploration completed after {smart_manager.round_count} rounds with {insights_count} insights"
        )

    finally:
        # Reopen the connection after agent execution
        logging.info("Reopening database connection...")
        session_state.reconnect()

    logging.info("--- Insight Discovery Loop Complete ---")


def run_strategy_loop(session_state: SessionState):
    """Orchestrates the Hypothesis & Strategy Team to refine insights."""
    logging.info("--- Running Strategy Loop ---")
    insights_report = session_state.get_final_insight_report()
    if "No insights" in insights_report:
        logging.warning("Skipping strategy loop as no insights were generated.")
        return None

    config_list = config_list_from_json(
        os.getenv("OAI_CONFIG_LIST", "config/OAI_CONFIG_LIST.json")
    )

    # Validate and substitute the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("OPENAI_API_KEY environment variable is empty or not set.")

    logging.info(f"API key loaded with length: {len(api_key)}")

    # Manually substitute the API key to ensure it's loaded
    for config in config_list:
        if config.get("api_key") == "${OPENAI_API_KEY}":
            config["api_key"] = api_key
            logging.info("Substituted API key in config")
        elif not config.get("api_key"):
            raise ValueError(f"No API key found in config: {config}")

    # Validate final config
    for i, config in enumerate(config_list):
        if not config.get("api_key") or config.get("api_key").strip() == "":
            raise ValueError(
                f"Config {i} has empty API key after substitution: {config}"
            )

    logging.info(f"Final config list validated with {len(config_list)} configurations")

    llm_config = {"config_list": config_list, "cache_seed": None}

    agents_and_proxy = get_hypothesis_agents(llm_config, insights_report)
    user_proxy = agents_and_proxy.pop("user_proxy")
    agents = list(agents_and_proxy.values())

    user_proxy.register_function(
        function_map={
            "execute_python": execute_python,
            "finalize_hypotheses": get_finalize_hypotheses_tool(session_state),
        }
    )

    group_chat = autogen.GroupChat(
        agents=[user_proxy] + agents,
        messages=[],
        max_round=20,
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Close the session database connection to avoid lock conflicts during agent execution
    logging.info("Closing database connection for strategy agent execution...")
    session_state.close_connection()

    try:
        user_proxy.initiate_chat(
            manager,
            message=f"""Welcome strategists. Your task is to refine the following insights into a set of concrete, testable hypotheses. You can use `execute_python` to re-verify findings. When the final list is agreed upon, the EngineerAgent must call `finalize_hypotheses`.

--- INSIGHTS REPORT ---
{insights_report}
""",
        )
    finally:
        # Reopen the connection after agent execution
        logging.info("Reopening database connection...")
        session_state.reconnect()

    logging.info("--- Strategy Loop Complete ---")
    return session_state.get_final_hypotheses()


def main():
    """
    Main function to run the VULCAN agent orchestration.
    """
    # Initialize the run
    run_id, run_dir = init_run()
    logger.info(f"Starting VULCAN Run ID: {run_id}")

    # Setup logging
    setup_logging()

    # Initialize session state
    session_state = SessionState(run_dir)

    try:
        run_discovery_loop(session_state)
        # final_hypotheses = run_strategy_loop(session_state)

        logging.info("Orchestration complete.")
        # if final_hypotheses is not None:
        #     logging.info("--- FINAL VETTED HYPOTHESES ---")
        #     for h in final_hypotheses:
        #         logging.info(f"- {h.model_dump_json(indent=2)}")
        # else:
        #     logging.info("--- No hypotheses were finalized. ---")

    except Exception as e:
        logging.error(f"An error occurred during orchestration: {e}", exc_info=True)
    finally:
        session_state.close_connection()

        # Clean up any generated views
        cleanup_result = cleanup_analysis_views(Path(session_state.run_dir))
        logging.info(f"View cleanup: {cleanup_result}")

        logging.info("Run finished. Session state saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"VULCAN run failed: {e}")
        logger.exception(e)
        # Optional: Add any cleanup logic here
        sys.exit(1)
```

### `schemas/eda_report_schema.json`

**File size:** 1,225 bytes

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["schema_overview", "global_stats", "samples", "insights", "plots", "hypotheses"],
  "properties": {
    "schema_overview": {
      "type": "object",
      "description": "Database schema information including tables and their columns"
    },
    "global_stats": {
      "type": "object",
      "description": "Summary statistics for each table"
    },
    "samples": {
      "type": "object",
      "description": "Representative samples from each table"
    },
    "insights": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["metric", "value", "comment"],
        "properties": {
          "metric": {"type": "string"},
          "value": {"type": ["number", "string"]},
          "comment": {"type": "string"}
        }
      }
    },
    "plots": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "caption"],
        "properties": {
          "path": {"type": "string"},
          "caption": {"type": "string"}
        }
      }
    },
    "hypotheses": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
} 
```

### `schemas/models.py`

**File size:** 2,887 bytes

```python
# src/utils/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field


class Insight(BaseModel):
    title: str = Field(description="A concise, descriptive title for the insight.")
    finding: str = Field(
        description="The detailed finding or observation, explaining what was discovered."
    )
    supporting_code: Optional[str] = Field(
        None, description="The exact SQL or Python code used to generate the finding."
    )
    source_representation: str = Field(
        description="The name of the SQL View or Graph used for analysis (e.g., 'vw_user_review_summary' or 'g_user_book_bipartite')."
    )
    plot_path: Optional[str] = Field(
        None, description="The absolute path to the plot that visualizes the finding."
    )
    plot_interpretation: Optional[str] = Field(
        None,
        description="A detailed, LLM-generated analysis of what the plot shows and its implications.",
    )


class Hypothesis(BaseModel):
    id: str = Field(
        ..., description="A unique identifier for the hypothesis, e.g., 'H-01'."
    )
    description: str = Field(
        ..., description="The full text of the hypothesis, clearly stated."
    )
    strategic_critique: str = Field(
        ...,
        description="A detailed critique from the Strategist on how this hypothesis aligns with the Core Objective.",
    )
    feasibility_critique: str = Field(
        ...,
        description="A detailed critique from the Engineer on the technical feasibility and computational cost.",
    )


class PrioritizedHypothesis(BaseModel):
    id: str = Field(..., description="The unique identifier for the hypothesis.")
    priority: int = Field(
        ..., ge=1, le=5, description="The priority score from 1 to 5."
    )
    feasibility: int = Field(
        ..., ge=1, le=5, description="The feasibility score from 1 to 5."
    )
    notes: str = Field(..., description="A brief justification for the scores.")


class CandidateFeature(BaseModel):
    name: str = Field(
        ..., description="A descriptive, camel-case name for the feature."
    )
    type: str = Field(
        ..., description="The type of feature: 'code', 'llm', or 'composition'."
    )
    spec: str = Field(
        ...,
        description="The feature specification (e.g., DSL for code, prompt for llm).",
    )
    depends_on: Optional[List[str]] = Field(
        [], description="A list of other features this feature depends on."
    )
    rationale: str = Field(
        ..., description="A brief explanation of why this feature is useful."
    )
    effort: int = Field(
        ...,
        ge=1,
        le=5,
        description="The estimated effort to implement this feature (1-5).",
    )
    impact: int = Field(
        ...,
        ge=1,
        le=5,
        description="The estimated impact of this feature on the model (1-5).",
    )
```

### `utils/data_utils.py`

**File size:** 414 bytes

```python
import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_size: float = 0.8
) -> (pd.DataFrame, pd.DataFrame):
    """Splits a DataFrame into training and validation sets based on a timestamp."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    return train_df, val_df
```

### `utils/decorators.py`

**File size:** 913 bytes

```python
# src/utils/decorators.py
import time
from functools import wraps

from loguru import logger


def agent_run_decorator(agent_name: str):
    """
    A decorator to log the duration of an agent's run method and write it to TensorBoard.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger.info(f"{agent_name} started.")
            start_time = time.time()

            result = func(self, *args, **kwargs)

            end_time = time.time()
            duration = end_time - start_time

            if hasattr(self, "writer") and self.writer is not None:
                run_count = getattr(self, "run_count", 0)
                self.writer.add_scalar("run_duration_seconds", duration, run_count)

            logger.info(f"{agent_name} finished in {duration:.2f} seconds.")
            return result

        return wrapper

    return decorator
```

### `utils/feature_registry.py`

**File size:** 900 bytes

```python
class FeatureRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, feature_data: dict):
        """Registers a feature function and its metadata."""
        if name in self._registry:
            print(f"Warning: Feature {name} is already registered. Overwriting.")
        self._registry[name] = feature_data

    def get(self, name: str) -> dict:
        """Retrieves a feature function and its metadata."""
        return self._registry.get(name)

    def get_all(self) -> dict:
        """Retrieves the entire registry."""
        return self._registry.copy()


# Global instance of the registry
feature_registry = FeatureRegistry()


def get_feature(name: str):
    """Public method to get a feature from the global registry."""
    feature_data = feature_registry.get(name)
    if feature_data:
        return feature_data.get("func")
    return None
```

### `utils/logging_utils.py`

**File size:** 3,871 bytes

```python
# src/utils/logging_utils.py
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Agent colors mapping
AGENT_COLORS = {
    "DataAnalysisAgent": "cyan",
    "HypothesisAgent": "green",
    "ReasoningAgent": "yellow",
    "FeatureIdeationAgent": "magenta",
    "FeatureRealizationAgent": "blue",
    "OptimizationAgent": "red",
    "EvaluationAgent": "white",
    "ReflectionAgent": "bright_blue",
    "ResearchAgent": "bright_green",
}


class InterceptHandler(logging.Handler):
    """
    A handler to intercept standard logging messages and redirect them to loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_agent_logger(agent_name: str) -> None:
    """Set up logging for a specific agent with its own color and file."""
    # Remove default handler
    logger.remove()

    # Add console handler with agent-specific color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<{AGENT_COLORS.get(agent_name, 'white')}>{agent_name}</{AGENT_COLORS.get(agent_name, 'white')}> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Add file handler for this agent
    logger.add(
        LOGS_DIR / f"{agent_name.lower()}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
    )


def log_agent_context(agent_name: str, context: Dict[str, Any]) -> None:
    """Log the context passed to an agent."""
    logger.info(f"Context received: {context}")


def log_agent_response(agent_name: str, response: Dict[str, Any]) -> None:
    """Log the response from an agent."""
    logger.info(f"Response generated: {response}")


def log_agent_error(agent_name: str, error: Exception) -> None:
    """Log an error that occurred in an agent."""
    logger.error(f"Error occurred: {str(error)}")


def log_llm_prompt(agent_name: str, prompt: str) -> None:
    """Log the prompt sent to the LLM."""
    logger.info(f"📤 LLM PROMPT:\n{'-' * 50}\n{prompt}\n{'-' * 50}")


def log_llm_response(agent_name: str, response: str) -> None:
    """Log the response from the LLM."""
    logger.info(f"📥 LLM RESPONSE:\n{'-' * 50}\n{response}\n{'-' * 50}")


def log_tool_call(agent_name: str, tool_name: str, tool_args: Dict[str, Any]) -> None:
    """Log a tool call being made."""
    logger.info(f"🔧 TOOL CALL: {tool_name} with args: {tool_args}")


def log_tool_result(agent_name: str, tool_name: str, result: Any) -> None:
    """Log the result of a tool call."""
    logger.info(f"🔧 TOOL RESULT from {tool_name}: {result}")


# Example usage in an agent:
"""
from src.utils.logging_utils import setup_agent_logger, log_agent_context, log_agent_response

class MyAgent:
    def __init__(self):
        setup_agent_logger(self.__class__.__name__)
    
    def run(self, context):
        log_agent_context(self.__class__.__name__, context)
        # ... agent logic ...
        log_agent_response(self.__class__.__name__, response)
"""
```

### `utils/prompt_utils.py`

**File size:** 2,893 bytes

```python
import logging
from pathlib import Path

import jinja2

from src.core.database import get_db_schema_string

logger = logging.getLogger(__name__)

_prompt_dir = Path(__file__).parent.parent / "prompts"

# Initialize the Jinja environment
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_prompt_dir),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _refresh_database_schema():
    """Refresh the database schema in the Jinja environment globals."""
    try:
        db_schema = get_db_schema_string()
        _jinja_env.globals["db_schema"] = db_schema
        logger.debug(
            f"Database schema refreshed successfully. Schema length: {len(db_schema)} characters"
        )
        return db_schema
    except Exception as e:
        logger.error(f"Failed to refresh database schema: {e}")
        _jinja_env.globals["db_schema"] = "ERROR: Could not load database schema"
        return None


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads and renders a Jinja2 template from the prompts directory.
    Refreshes database schema and logs the rendered prompt for debugging.

    Args:
        template_name: The name of the template file (e.g., 'agents/strategist.j2').
        **kwargs: The context variables to render the template with.

    Returns:
        The rendered prompt as a string.
    """
    try:
        # Refresh database schema to ensure it's current
        schema = _refresh_database_schema()

        # Load and render the template
        template = _jinja_env.get_template(template_name)
        rendered_prompt = template.render(**kwargs)

        # Log the template being loaded and key info
        logger.info(f"Loaded prompt template: {template_name}")
        if kwargs:
            logger.debug(f"Template variables: {list(kwargs.keys())}")

        # Log the rendered prompt for debugging (truncated to avoid spam)
        prompt_preview = (
            rendered_prompt[:500] + "..."
            if len(rendered_prompt) > 500
            else rendered_prompt
        )
        logger.debug(f"Rendered prompt preview (first 500 chars):\n{prompt_preview}")

        # Log full prompt length
        logger.info(f"Full rendered prompt length: {len(rendered_prompt)} characters")

        return rendered_prompt

    except jinja2.TemplateNotFound as e:
        logger.error(f"Template not found: {template_name}")
        raise ValueError(f"Prompt template '{template_name}' not found") from e
    except jinja2.TemplateError as e:
        logger.error(f"Template rendering error for {template_name}: {e}")
        raise ValueError(f"Error rendering template '{template_name}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading prompt {template_name}: {e}")
        raise


# Initialize the database schema on module load
_refresh_database_schema()
```

### `utils/pubsub.py`

**File size:** 1,386 bytes

```python
# src/utils/pubsub.py
import json
import threading

import redis
from loguru import logger

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)


def publish(channel: str, message: dict):
    """Publishes a message to a Redis channel."""
    logger.info(f"Publishing to channel '{channel}': {message}")
    redis_client.publish(channel, json.dumps(message))


def subscribe(channel: str, callback):
    """Subscribes to a Redis channel and runs a callback for each message."""

    def worker():
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        logger.info(f"Subscribed to channel '{channel}'.")
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                logger.info(f"Received message on channel '{channel}': {data}")
                callback(data)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def acquire_lock(lock_name: str, timeout: int = 60) -> bool:
    """
    Acquires a lock in Redis. Returns True if the lock was acquired,
    False otherwise. The lock will expire after the timeout.
    """
    return redis_client.set(lock_name, "locked", ex=timeout, nx=True)


def release_lock(lock_name: str):
    """Releases a lock in Redis."""
    redis_client.delete(lock_name)
```

### `utils/run_utils.py`

**File size:** 3,903 bytes

```python
#!/usr/bin/env python3
"""
Utilities for managing run IDs and run-specific paths.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import autogen

# Base directories
RUNTIME_DIR = Path("runtime")
RUNS_DIR = RUNTIME_DIR / "runs"

# Global variable to store current run ID
_run_id: Optional[str] = None
_run_dir: Optional[Path] = None


def init_run() -> Tuple[str, Path]:
    """
    Initializes a new run, setting a unique run ID and creating run-specific directories.
    This function should be called once at the start of a pipeline run.
    """
    global _run_id, _run_dir
    if _run_id:
        raise RuntimeError(f"Run has already been initialized with RUN_ID: {_run_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    _run_id = f"run_{timestamp}_{unique_id}"

    runtime_path = Path(__file__).resolve().parent.parent.parent / "runtime" / "runs"
    _run_dir = runtime_path / _run_id

    # Create all necessary subdirectories for the run
    (_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (_run_dir / "data").mkdir(parents=True, exist_ok=True)
    (_run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (_run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (_run_dir / "generated_code").mkdir(parents=True, exist_ok=True)

    return _run_id, _run_dir


def get_run_id() -> str:
    """Returns the unique identifier for the current run."""
    if _run_id is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_id


def get_run_dir() -> Path:
    """Returns the absolute path to the directory for the current run."""
    if _run_dir is None:
        raise RuntimeError("Run context is not initialized. Call init_run() first.")
    return _run_dir


def get_run_artifact_path(*path_parts: str) -> Path:
    """Constructs an absolute path for an artifact within the current run's directory."""
    return get_run_dir().joinpath(*path_parts)


def get_run_logs_dir() -> Path:
    """Get the logs directory for the current run."""
    return get_run_dir() / "logs"


def get_run_tensorboard_dir() -> Path:
    """Get the TensorBoard directory for the current run."""
    return get_run_dir() / "tensorboard"


def get_run_generated_code_dir() -> Path:
    """Get the generated code directory for the current run."""
    return get_run_dir() / "generated_code"


def get_run_memory_file() -> Path:
    """Get the memory file path for the current run."""
    return get_run_dir() / "memory.json"


def get_run_database_file() -> Path:
    """Get the database file path for the current run."""
    return get_run_dir() / "database.json"


def get_run_log_file() -> Path:
    """Get the log file for the current run."""
    return get_run_logs_dir() / f"pipeline_{get_run_id()}.log"


def get_run_db_file() -> Path:
    """Get the database file for the current run."""
    return get_run_dir() / f"data_{get_run_id()}.duckdb"


def get_feature_code_path(feature_name: str) -> Path:
    """Get the path for a realized feature's code file."""
    return get_run_generated_code_dir() / f"{feature_name}.py"


def get_tensorboard_writer(agent_name: str):
    """Get a TensorBoard writer for the current run and agent."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(get_run_tensorboard_dir() / agent_name))


def format_log_message(message: str) -> str:
    """Format a log message with run context."""
    return f"[{get_run_id()}] {message}"


def config_list_from_json(file_path: str) -> List[Dict]:
    """
    Loads an AutoGen configuration list from a JSON file, resolving environment variables.
    """
    return autogen.config_list_from_json(
        env_or_file=file_path,
    )
```

### `utils/sampling.py`

**File size:** 1,205 bytes

```python
from src.utils import db_api


def sample_users_by_activity(n: int, min_rev: int, max_rev: int) -> list[str]:
    sql = f"""
      SELECT user_id FROM (
        SELECT user_id, COUNT(*) AS cnt
        FROM reviews
        GROUP BY user_id
      ) sub
      WHERE cnt BETWEEN {min_rev} AND {max_rev}
      ORDER BY RANDOM()
      LIMIT {n};
    """
    return db_api.conn.execute(sql).fetchdf()["user_id"].tolist()


def sample_users_stratified(n_total: int, strata: dict) -> list[str]:
    """
    Samples users from different activity strata.

    Args:
        n_total (int): The total number of users to sample.
        strata (dict): A dictionary where keys are strata names and values are
                       tuples of (min_reviews, max_reviews, proportion).
                       Proportions should sum to 1.

    Returns:
        list[str]: A list of sampled user IDs.
    """
    all_user_ids = []
    for stratum, (min_rev, max_rev, proportion) in strata.items():
        n_sample = int(n_total * proportion)
        if n_sample == 0:
            continue

        user_ids = sample_users_by_activity(n_sample, min_rev, max_rev)
        all_user_ids.extend(user_ids)

    return all_user_ids
```

### `utils/session_state.py`

**File size:** 12,814 bytes

```python
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from src.schemas.models import Hypothesis, Insight
from src.utils.run_utils import get_run_dir


class SessionState:
    """Manages the state and artifacts of a single pipeline run."""

    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or get_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize default state
        self.insights: List[Insight] = []
        self.hypotheses: List[Hypothesis] = []

        # Additional state for complete pipeline management
        self.prioritized_hypotheses: List[Dict] = []
        self.candidate_features: List[Dict] = []
        self.best_params: Dict = {}
        self.best_rmse: Optional[float] = None
        self.bo_history: Dict = {}
        self.reflections: List[Dict] = []

        # Run counters for agents
        self.ideation_run_count: int = 0
        self.feature_realization_run_count: int = 0
        self.reflection_run_count: int = 0

        # Load existing state if available
        self._load_from_disk()

        self.db_path = "data/goodreads_curated.duckdb"
        try:
            # Connect in read-write mode to allow for TEMP view creation
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            print(f"Successfully connected to {self.db_path} in read-write mode.")
        except Exception as e:
            print(f"FATAL: Failed to connect to database at {self.db_path}: {e}")
            self.conn = None
            raise e

    def _load_from_disk(self):
        """Loads existing session state from disk if available."""
        state_file = self.run_dir / "session_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                # Load insights and hypotheses using proper model classes
                if "insights" in data:
                    self.insights = [Insight(**insight) for insight in data["insights"]]
                if "hypotheses" in data:
                    self.hypotheses = [
                        Hypothesis(**hypothesis) for hypothesis in data["hypotheses"]
                    ]

                # Load simple state fields
                self.prioritized_hypotheses = data.get("prioritized_hypotheses", [])
                self.candidate_features = data.get("candidate_features", [])
                self.best_params = data.get("best_params", {})
                self.best_rmse = data.get("best_rmse")
                self.bo_history = data.get("bo_history", {})
                self.reflections = data.get("reflections", [])
                self.set_state("features", data.get("features", {}))
                self.set_state("metrics", data.get("metrics", {}))
                self.set_state("models", data.get("models", {}))

                # Load run counters
                self.ideation_run_count = data.get("ideation_run_count", 0)
                self.feature_realization_run_count = data.get(
                    "feature_realization_run_count", 0
                )
                self.reflection_run_count = data.get("reflection_run_count", 0)

                print(
                    f"Loaded existing session state with {len(self.insights)} insights and {len(self.hypotheses)} hypotheses."
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load existing session state: {e}. Starting with fresh state."
                )
        else:
            print("No existing session state found. Starting with fresh state.")

    def add_insight(self, insight: Insight):
        self.insights.append(insight)
        self.save_to_disk()
        print(f"Added and saved new insight: '{insight.title}'")

    def finalize_hypotheses(self, hypotheses: List[Hypothesis]):
        self.hypotheses.extend(hypotheses)
        self.save_to_disk()
        print(f"Finalized and saved {len(hypotheses)} hypotheses.")

    # Prioritized hypotheses management
    def set_prioritized_hypotheses(self, hypotheses: List[Dict]):
        self.prioritized_hypotheses = hypotheses
        self.save_to_disk()

    def get_prioritized_hypotheses(self) -> List[Dict]:
        return self.prioritized_hypotheses

    # Candidate features management
    def set_candidate_features(self, features: List[Dict]):
        self.candidate_features = features
        self.save_to_disk()

    def get_candidate_features(self) -> List[Dict]:
        return self.candidate_features

    # Optimization results management
    def set_best_params(self, params: Dict):
        self.best_params = params
        self.save_to_disk()

    def get_best_params(self) -> Dict:
        return self.best_params

    def set_best_rmse(self, rmse: float):
        self.best_rmse = rmse
        self.save_to_disk()

    def get_best_rmse(self) -> Optional[float]:
        return self.best_rmse

    def set_bo_history(self, history: Dict):
        self.bo_history = history
        self.save_to_disk()

    def get_bo_history(self) -> Dict:
        return self.bo_history

    # Reflections management
    def add_reflection(self, reflection: Dict):
        self.reflections.append(reflection)
        self.save_to_disk()

    def get_reflections(self) -> List[Dict]:
        return self.reflections

    # Run counters management
    def increment_ideation_run_count(self) -> int:
        self.ideation_run_count += 1
        self.save_to_disk()
        return self.ideation_run_count

    def get_ideation_run_count(self) -> int:
        return self.ideation_run_count

    def increment_feature_realization_run_count(self) -> int:
        self.feature_realization_run_count += 1
        self.save_to_disk()
        return self.feature_realization_run_count

    def get_feature_realization_run_count(self) -> int:
        return self.feature_realization_run_count

    def increment_reflection_run_count(self) -> int:
        self.reflection_run_count += 1
        self.save_to_disk()
        return self.reflection_run_count

    def get_reflection_run_count(self) -> int:
        return self.reflection_run_count

    # Feature, metric, and model storage
    def store_feature(self, feature_name: str, feature_data: Dict):
        """Store feature data in the session state."""
        features = self.get_state("features", {})
        features[feature_name] = feature_data
        self.set_state("features", features)

    def get_feature(self, feature_name: str) -> Optional[Dict]:
        """Get feature data from the session state."""
        features = self.get_state("features", {})
        return features.get(feature_name)

    def store_metric(self, metric_name: str, metric_data: Dict):
        """Store metric data in the session state."""
        metrics = self.get_state("metrics", {})
        metrics[metric_name] = metric_data
        self.set_state("metrics", metrics)

    def get_metric(self, metric_name: str) -> Optional[Dict]:
        """Get metric data from the session state."""
        metrics = self.get_state("metrics", {})
        return metrics.get(metric_name)

    def store_model(self, model_name: str, model_data: Dict):
        """Store model data in the session state."""
        models = self.get_state("models", {})
        models[model_name] = model_data
        self.set_state("models", models)

    def get_model(self, model_name: str) -> Optional[Dict]:
        """Get model data from the session state."""
        models = self.get_state("models", {})
        return models.get(model_name)

    # Generic get/set methods for backward compatibility and any additional state
    def get_state(self, key: str, default: Any = None) -> Any:
        """Generic getter for any state attribute."""
        return getattr(self, key, default)

    def set_state(self, key: str, value: Any):
        """Generic setter for any state attribute."""
        setattr(self, key, value)
        self.save_to_disk()

    def get_final_insight_report(self) -> str:
        """Returns a string report of all insights generated."""
        if not self.insights:
            return "No insights were generated during this run."

        report = "--- INSIGHTS REPORT ---\n\n"
        for i, insight in enumerate(self.insights, 1):
            report += f"Insight {i}: {insight.title}\n"
            report += f"  Finding: {insight.finding}\n"
            if insight.source_representation:
                report += f"  Source: {insight.source_representation}\n"
            if insight.supporting_code:
                report += f"  Code:\n```\n{insight.supporting_code}\n```\n"
            if insight.plot_path:
                report += f"  Plot: {insight.plot_path}\n"
            report += "\n"
        return report

    def get_final_hypotheses(self) -> Optional[List[Hypothesis]]:
        """Returns the final list of vetted hypotheses."""
        return self.hypotheses if self.hypotheses else None

    def vision_tool(self, image_path: str, prompt: str) -> str:
        """
        Analyzes an image file using OpenAI's GPT-4o vision model.
        This tool automatically resolves the path relative to the run's output directory.
        """
        import base64
        import os

        from openai import OpenAI

        try:
            full_path = self.get_run_output_dir() / image_path
            if not full_path.exists():
                logger.error(f"Vision tool failed: File not found at {full_path}")
                return f"ERROR: File not found at '{image_path}'. Please ensure the file was saved correctly."

            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Read and encode the image
            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except ImportError:
            return "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
        except Exception as e:
            logger.error(f"Vision tool failed with an unexpected error: {e}")
            return f"ERROR: An unexpected error occurred while analyzing the image: {e}"

    def save_to_disk(self):
        """Saves the current session state to disk."""
        output = {
            "insights": [i.model_dump() for i in self.insights],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "prioritized_hypotheses": self.prioritized_hypotheses,
            "candidate_features": self.candidate_features,
            "best_params": self.best_params,
            "best_rmse": self.best_rmse,
            "bo_history": self.bo_history,
            "reflections": self.reflections,
            "features": self.get_state("features", {}),
            "metrics": self.get_state("metrics", {}),
            "models": self.get_state("models", {}),
            "ideation_run_count": self.ideation_run_count,
            "feature_realization_run_count": self.feature_realization_run_count,
            "reflection_run_count": self.reflection_run_count,
        }
        output_path = self.run_dir / "session_state.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def reconnect(self):
        """Reopens the database connection."""
        try:
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            print(f"Successfully reconnected to {self.db_path} in read-write mode.")
        except Exception as e:
            print(f"FATAL: Failed to reconnect to database at {self.db_path}: {e}")
            self.conn = None
            raise e

    @property
    def db_connection(self):
        """Returns the current database connection."""
        return self.conn
```

### `utils/testing_utils.py`

**File size:** 1,542 bytes

```python
import json

import numpy as np
import pandas as pd
from jsonschema import validate


def assert_json_schema(instance: dict, schema_path: str) -> None:
    """Raises AssertionError if instance doesn't match schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        validate(instance=instance, schema=schema)
    except Exception as e:
        raise AssertionError(f"JSON schema validation failed: {e}")


def load_test_data(
    n_reviews: int, n_items: int, n_users: int
) -> (pd.DataFrame, pd.DataFrame):
    """Creates a synthetic toy dataset with random ratings, random words."""
    # Create reviews data
    review_data = {
        "user_id": np.random.randint(0, n_users, n_reviews),
        "book_id": np.random.randint(0, n_items, n_reviews),
        "rating": np.random.randint(1, 6, n_reviews),
        "review_text": [
            " ".join(
                np.random.choice(
                    ["good", "bad", "fantasy", "sci-fi", "grimdark"], size=10
                )
            )
            for _ in range(n_reviews)
        ],
        "timestamp": pd.to_datetime(
            np.random.randint(1577836800, 1609459200, n_reviews), unit="s"
        ),
    }
    df_reviews = pd.DataFrame(review_data)

    # Create items data
    item_data = {
        "book_id": np.arange(n_items),
        "author": [f"Author_{i}" for i in range(n_items)],
        "genre": np.random.choice(["Fantasy", "Sci-Fi"], size=n_items),
    }
    df_items = pd.DataFrame(item_data)

    return df_reviews, df_items
```

### `utils/tools.py`

**File size:** 13,677 bytes

```python
# -*- coding: utf-8 -*-
import json
import subprocess
from pathlib import Path

import duckdb

from src.config.settings import DB_PATH
from src.utils.run_utils import get_run_dir

# The preamble provides the helper functions and the database connection
# that will be available to the agent's code.
_PYTHON_TOOL_PREAMBLE = f"""
import pandas as pd
import duckdb
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime

# Use absolute path to the database with read-write access (orchestrator connection is closed)
DB_PATH = "{DB_PATH}"

# Set better plotting defaults
plt.style.use('default')
sns.set_palette("husl")

def get_table_sample(table_name: str, n_samples: int = 5) -> str:
    '''Retrieves a random sample of rows from a specified table in the database.'''
    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as sample_conn:
            query = f'SELECT * FROM "{{table_name}}" USING SAMPLE {{n_samples}} ROWS;'
            sample_df = sample_conn.execute(query).fetchdf()
            
            if sample_df.empty:
                return f"No data returned for table '{{table_name}}'. It may be empty."
            
            return f"Sample of '{{table_name}}' ({{n_samples}} rows):\\n{{sample_df.to_markdown(index=False)}}"
    except Exception as e:
        return f"ERROR: Failed to sample table '{{table_name}}': {{e}}"

def save_plot(filename: str):
    """Save the current matplotlib figure to the *run-local* ``plots`` directory.

    - Always resolves to ``runtime/runs/<RUN_ID>/plots/<filename>.png``.
    - Any directory components in *filename* are stripped to avoid accidental path traversal.
    - Automatically appends ``.png`` if missing.
    """

    # Ensure we are always inside the run directory (``execute_python`` sets cwd appropriately).
    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True)

    # Disallow sub-directories in filename for safety; keep only the basename.
    basename = Path(filename).name

    # Append .png if not provided
    if not basename.lower().endswith(".png"):
        basename += ".png"

    path = plots_dir / basename
    
    # Apply better plot formatting
    plt.tight_layout()
    
    # Auto-set reasonable axis limits if not already set
    ax = plt.gca()
    if hasattr(ax, 'get_xlim'):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Only adjust if using default limits
        if xlim == (0.0, 1.0) or any(abs(x) > 1e6 for x in xlim):
            # Get data bounds and add small margin
            try:
                lines = ax.get_lines()
                if lines:
                    xdata = np.concatenate([line.get_xdata() for line in lines])
                    if len(xdata) > 0:
                        x_margin = (np.max(xdata) - np.min(xdata)) * 0.05
                        ax.set_xlim(np.min(xdata) - x_margin, np.max(xdata) + x_margin)
            except:
                pass
    
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close() # Close the plot to free memory
    abs_path = path.resolve()
    print(f"PLOT_SAVED:{abs_path}")  # Special token for downstream parsing
    return str(abs_path)

def create_analysis_view(view_name: str, sql_query: str):
    '''Creates a permanent view for analysis and tracks it for cleanup.'''
    # Use a non-read-only connection just for view creation
    with duckdb.connect(database=str(DB_PATH), read_only=False) as write_conn:
        # Create the view
        full_sql = f"CREATE OR REPLACE VIEW {{view_name}} AS {{sql_query}}"
        write_conn.execute(full_sql)
        
        # Track the view for cleanup
        views_file = Path("./generated_views.json")
        if views_file.exists():
            with open(views_file, 'r') as f:
                views_data = json.load(f)
        else:
            views_data = {"views": []}
        
        views_data["views"].append({
            "name": view_name,
            "sql": full_sql,
            "created_at": datetime.now().isoformat(),
        })
        
        with open(views_file, 'w') as f:
            json.dump(views_data, f, indent=2)
            
        print(f"VIEW_CREATED:{{view_name}}")
        return f"Successfully created view: {{view_name}}"

def analyze_and_plot(data_df, title="Data Analysis", x_col=None, y_col=None, plot_type="auto"):
    '''Helper function to both analyze data numerically AND create bounded visualizations.'''
    print(f"\\n=== NUMERICAL ANALYSIS: {{title}} ===")
    print(f"Dataset shape: {{data_df.shape}}")
    print(f"\\nSummary statistics:")
    print(data_df.describe())
    
    if len(data_df.columns) > 1:
        print(f"\\nCorrelation matrix:")
        numeric_cols = data_df.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 1:
            print(numeric_cols.corr())
    
    # Create visualization with bounded axes
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter" and x_col and y_col:
        plt.scatter(data_df[x_col], data_df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        # Set bounded limits
        x_min, x_max = data_df[x_col].min(), data_df[y_col].max()
        y_min, y_max = data_df[y_col].min(), data_df[y_col].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    elif plot_type == "hist" or (plot_type == "auto" and len(data_df.columns) == 1):
        col = data_df.columns[0] if not x_col else x_col
        data_df[col].hist(bins=30, alpha=0.7)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        # Set bounded x limits
        x_min, x_max = data_df[col].min(), data_df[col].max()
        x_margin = (x_max - x_min) * 0.05
        plt.xlim(x_min - x_margin, x_max + x_margin)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return title.lower().replace(" ", "_").replace(":", "")

# The user's code will run inside this 'with' block with read-write access
with duckdb.connect(database=str(DB_PATH), read_only=False) as conn:
"""


def execute_python(code: str) -> str:
    """
    Executes a string of Python code in a sandboxed environment with a pre-configured
    database connection (`conn`) and helper functions (`save_plot`).

    Args:
        code: The string of Python code to execute.

    Returns:
        A string containing the captured STDOUT and STDERR from the execution.
    """

    # Always use the standard preamble for now (simplified approach)
    indented_code = "    " + code.replace("\n", "\n    ")
    full_script = f"{_PYTHON_TOOL_PREAMBLE}\n{indented_code}"

    # Use the run-specific directory for sandboxing
    run_dir = get_run_dir()
    script_path = run_dir / "temp_agent_script.py"

    with open(script_path, "w") as f:
        f.write(full_script)

    try:
        # Execute the script in a new process for isolation
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            check=True,  # This will raise CalledProcessError if the script fails (exit code != 0)
            timeout=120,  # Add a timeout to prevent runaway processes
            cwd=str(run_dir),  # Set the working directory to the run directory
        )
        output = f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if not result.stdout and not result.stderr:
            output += "\nNo output produced by code."
        return output

    except subprocess.CalledProcessError as e:
        # This block catches script failures (e.g., Python exceptions)
        return f"EXECUTION FAILED:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "EXECUTION FAILED: Code execution timed out after 120 seconds."
    finally:
        # Clean up the temporary script
        if script_path.exists():
            script_path.unlink()


def cleanup_analysis_views(run_dir: Path):
    """Cleans up any database views created during a run."""
    views_file = run_dir / "generated_views.json"
    if not views_file.exists():
        print("No views to clean up.")
        return

    try:
        with open(views_file, "r") as f:
            views_data = json.load(f)

        views_to_drop = [view["name"] for view in views_data["views"]]

        if not views_to_drop:
            print("No views to clean up.")
            return

        with duckdb.connect(database=DB_PATH, read_only=False) as conn:
            for view_name in views_to_drop:
                try:
                    conn.execute(f"DROP VIEW IF EXISTS {view_name};")
                    print(f"Successfully dropped view: {view_name}")
                except Exception as e:
                    print(f"Warning: Could not drop view {view_name}: {e}")
        # Optionally remove the tracking file after cleanup
        # views_file.unlink()
    except Exception as e:
        print(f"Error during view cleanup: {e}")


def get_add_insight_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to add insights to the session state."""

    def add_insight_to_report(
        title: str,
        finding: str,
        source_representation: str,
        supporting_code: str = None,
        plot_path: str = None,
        plot_interpretation: str = None,
    ) -> str:
        """
        Adds a structured insight to the session report.

        Args:
            title: A concise, descriptive title for the insight
            finding: The detailed finding or observation
            source_representation: The name of the SQL View or Graph used for analysis
            supporting_code: The exact SQL or Python code used to generate the finding
            plot_path: The path to the plot that visualizes the finding
            plot_interpretation: LLM-generated analysis of what the plot shows

        Returns:
            Confirmation message
        """
        from src.schemas.models import Insight

        insight = Insight(
            title=title,
            finding=finding,
            source_representation=source_representation,
            supporting_code=supporting_code,
            plot_path=plot_path,
            plot_interpretation=plot_interpretation,
        )

        session_state.add_insight(insight)
        return f"Successfully added insight: '{title}' to the report."

    return add_insight_to_report


def get_finalize_hypotheses_tool(session_state):
    """Returns a function that can be used as an AutoGen tool to finalize hypotheses."""

    def finalize_hypotheses(hypotheses_data: list) -> str:
        """
        Finalizes the list of vetted hypotheses.

        Args:
            hypotheses_data: List of dictionaries containing hypothesis information
            Each dict should have: id, description, strategic_critique, feasibility_critique

        Returns:
            Confirmation message
        """
        from src.schemas.models import Hypothesis

        hypotheses = []
        for h_data in hypotheses_data:
            hypothesis = Hypothesis(
                id=h_data["id"],
                description=h_data["description"],
                strategic_critique=h_data["strategic_critique"],
                feasibility_critique=h_data["feasibility_critique"],
            )
            hypotheses.append(hypothesis)

        session_state.finalize_hypotheses(hypotheses)
        return f"Successfully finalized {len(hypotheses)} hypotheses."

    return finalize_hypotheses


def vision_tool(image_path: str, prompt: str) -> str:
    """
    Analyzes an image file using OpenAI's GPT-4o vision model.
    This is a standalone wrapper around the SessionState vision_tool method.
    """
    import base64
    import os
    from pathlib import Path

    from openai import OpenAI

    try:
        # Try to resolve path relative to current working directory
        full_path = Path(image_path)
        if not full_path.exists():
            # Try relative to run directory if available
            from src.utils.run_utils import get_run_dir

            run_dir = get_run_dir()
            full_path = run_dir / image_path

        if not full_path.exists():
            return f"ERROR: File not found at '{image_path}'. Please ensure the file was saved correctly."

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Read and encode the image
        with open(full_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except ImportError:
        return "ERROR: OpenAI library is not installed. Please install it with `pip install openai`."
    except Exception as e:
        return f"ERROR: An unexpected error occurred while analyzing the image: {e}"
```

## 📊 Summary

- **Total files processed:** 27
- **Directory:** `src`
- **Generated:** 2025-06-12 20:28:44

---

*This documentation was generated automatically. It includes all text-based source files and their complete contents.*


# 📜 Prompts Directory (src/prompts)

This document contains the complete source code structure and contents of the `src/prompts` directory.

## 📁 Full Directory Structure

```
├── .gitignore
├── Dockerfile.dataanalysis
├── README.md
├── config/
│   └── OAI_CONFIG_LIST.json
├── data/
│   ├── .gitkeep
│   └── __init__.py
├── data_curation/
│   ├── README.md
│   ├── clean_data.py
│   ├── run.py
│   ├── sql/
│   │   ├── 00_setup.sql
│   │   └── 01_curate_goodreads.sql
│   └── steps/
│       ├── analyze_db.py
│       ├── drop_useless_tables.py
│       ├── get_curated_schema.py
│       ├── inspect_raw_dates.py
│       └── verify_curated_dates.py
├── docker-compose.yml
├── docs/
│   ├── context.md
│   ├── core_mission.md
│   ├── experiments.md
│   ├── project_status_report.md
│   └── test.md
├── generate_src_docs.py
├── requirements.txt
├── scripts/
│   └── test_prompt.py
├── src/
│   ├── agents/
│   │   ├── discovery_team/
│   │   │   ├── __pycache__/
│   │   │   └── insight_discovery_agents.py
│   │   └── strategy_team/
│   │       ├── __pycache__/
│   │       ├── evaluation_agent.py
│   │       ├── feature_realization_agent.py
│   │       ├── hypothesis_agents.py
│   │       ├── optimization_agent.py
│   │       ├── reasoning_agent.py
│   │       └── reflection_agent.py
│   ├── config/
│   │   ├── __pycache__/
│   │   ├── logging.py
│   │   ├── settings.py
│   │   └── tensorboard.py
│   ├── core/
│   │   ├── __pycache__/
│   │   ├── database.py
│   │   ├── llm.py
│   │   └── tools.py
│   ├── orchestrator.py
│   ├── prompts/
│   │   ├── agents/
│   │   │   ├── discovery_team/
│   │   │   │   ├── base_analyst.j2
│   │   │   │   ├── data_representer.j2
│   │   │   │   ├── pattern_seeker.j2
│   │   │   │   └── quantitative_analyst.j2
│   │   │   ├── feature_ideator.j2
│   │   │   ├── reflection_agent.j2
│   │   │   └── strategy_team/
│   │   │       ├── engineer_agent.j2
│   │   │       ├── hypothesis_agent.j2
│   │   │       └── strategist_agent.j2
│   │   ├── globals/
│   │   │   ├── base_agent.j2
│   │   │   ├── base_analyst.j2
│   │   │   ├── base_strategy.j2
│   │   │   └── core_mission.j2
│   │   └── helpers/
│   │       ├── db_schema.j2
│   │       └── tool_usage.j2
│   ├── schemas/
│   │   ├── __pycache__/
│   │   ├── eda_report_schema.json
│   │   └── models.py
│   └── utils/
│       ├── __pycache__/
│       ├── data_utils.py
│       ├── decorators.py
│       ├── feature_registry.py
│       ├── logging_utils.py
│       ├── prompt_utils.py
│       ├── pubsub.py
│       ├── run_utils.py
│       ├── sampling.py
│       ├── session_state.py
│       ├── testing_utils.py
│       └── tools.py
├── src_documentation.md
├── src_documentation_with_prompts.md
├── temp_prompts.md
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── multi_agent_pipeline/
    │   ├── test_feature_ideation_agent_e2e.py
    │   ├── test_feature_realization.py
    │   ├── test_feature_realization_agent_e2e.py
    │   ├── test_optimization_agent_e2e.py
    │   ├── test_orchestrator_smoke.py
    │   ├── test_reflection_agent_e2e.py
    │   └── test_research_agent_e2e.py
    ├── schemas/
    │   └── feature_proposal_schema.json
    └── test_feature_realization_agent.py
```

## 📄 File Contents (src directory only)

## 📊 Summary

- **Total files processed:** 0
- **Directory:** `src/prompts`
- **Generated:** 2025-06-12 20:28:44

---

*This documentation was generated automatically. It includes all text-based source files and their complete contents.*
