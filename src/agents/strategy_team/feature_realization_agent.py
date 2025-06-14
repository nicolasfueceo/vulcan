# src/agents/strategy_team/feature_realization_agent.py
import json
from typing import Dict, List, Tuple

import autogen
from loguru import logger

from src.schemas.models import CandidateFeature, RealizedFeature
from src.utils.feature_registry import feature_registry
from src.utils.prompt_utils import load_prompt
from src.utils.session_state import SessionState
from src.utils.tools import execute_python
from src.utils.decorators import agent_run_decorator


class FeatureRealizationAgent:
    def __init__(self, llm_config: Dict, session_state: SessionState):
        """Initialize the feature realization agent."""
        logger.info("Initializing FeatureRealizationAgent")
        self.llm_config = llm_config
        self.session_state = session_state
        self.db_path = session_state.get_state("db_path")
        if not self.db_path:
            raise ValueError(
                "db_path not found in SessionState. It must be initialized by the Orchestrator."
            )

        self.llm_agent = autogen.AssistantAgent(
            name="FeatureRealizationAssistant",
            llm_config=self.llm_config,
            system_message="""You are an expert Python programmer. Given a feature specification, you write clean, efficient, and correct Python code that follows the provided function signature and relies only on standard libraries like pandas and numpy.""",
        )

    @agent_run_decorator("FeatureRealizationAgent")
    def run(self) -> None:
        """
        Main method to realize candidate features from the session state,
        with a self-correction loop for validation failures.
        """
        logger.info("Starting feature realization...")
        candidate_features_data = self.session_state.get_candidate_features()
        if not candidate_features_data:
            logger.warning("No candidate features found to realize.")
            self.session_state.set_state("realized_features", [])
            return

        candidate_features = [CandidateFeature(**f) for f in candidate_features_data]
        realized_features: List[RealizedFeature] = []
        MAX_RETRIES = 2  # Allow up to 2 correction attempts per feature

        # Create a user proxy agent for the chat
        user_proxy = autogen.UserProxyAgent(
            name="TempProxy",
            human_input_mode="NEVER",
            code_execution_config=False,  # We execute code in our own sandbox
        )

        for candidate in candidate_features:
            logger.info(f"Attempting to realize feature: {candidate.name}")
            
            is_realized = False
            last_error = ""
            code_str = ""

            for attempt in range(MAX_RETRIES + 1): # +1 to include initial attempt
                if attempt == 0:
                    # First attempt: Generate code from the original spec
                    message = load_prompt("agents/feature_realization.j2", **candidate.model_dump())
                else:
                    # Retry attempt: Provide the error context and ask for a fix
                    logger.warning(f"Retrying realization for '{candidate.name}' (Attempt {attempt}/{MAX_RETRIES}). Error: {last_error}")
                    message = f"""The previous code you wrote for the feature '{candidate.name}' failed validation with the following error:
---
ERROR:
{last_error}
---
ORIGINAL CODE:
```python
{code_str}
```
Please analyze the error and the original spec, then provide a corrected version of the full Python function.
The function MUST be complete, including all necessary imports and the function signature.

Original Spec: {candidate.spec}
"""
                # Initiate a chat to generate/fix the code
                user_proxy.initiate_chat(self.llm_agent, message=message, max_turns=1, silent=True)
                last_message = user_proxy.last_message(self.llm_agent)
                
                if not last_message or "content" not in last_message:
                    last_error = "LLM response was empty or invalid."
                    code_str = ""
                    continue # Go to the next retry attempt

                response_msg = last_message["content"]

                try:
                    code_str = response_msg.split("```python")[1].split("```")[0].strip()
                except IndexError:
                    code_str = ""
                    last_error = "LLM response did not contain a valid Python code block."
                    continue  # Go to the next retry attempt

                # --- Validation ---
                passed, last_error = self._validate_feature(candidate.name, code_str, candidate.params)

                if passed:
                    logger.success(f"Successfully validated feature '{candidate.name}' on attempt {attempt + 1}.")
                    is_realized = True
                    break  # Exit the retry loop on success

            # After the loop, create the final RealizedFeature object
            realized = RealizedFeature(
                name=candidate.name,
                code_str=code_str,
                params=candidate.params,
                passed_test=is_realized,
                type=candidate.type,
                source_candidate=candidate,
            )
            realized_features.append(realized)
            if is_realized:
                self._register_feature(realized)

        self.session_state.set_state("realized_features", [r.model_dump() for r in realized_features])
        successful_count = len([r for r in realized_features if r.passed_test])
        logger.info(f"Finished feature realization. Successfully realized and validated {successful_count} features.")

    def _register_feature(self, feature: RealizedFeature):
        """Registers a validated feature in the feature registry."""
        logger.info(f"Registering feature '{feature.name}' in the feature registry.")
        try:
            # Compile the code string into a function object
            temp_namespace = {}
            exec(feature.code_str, globals(), temp_namespace)
            feature_func = temp_namespace[feature.name]

            if not callable(feature_func):
                raise ValueError(f"The compiled code for feature '{feature.name}' is not a callable function.")

            feature_data = {
                "type": feature.type,
                "func": feature_func,  # Store the compiled function
                "params": feature.params,
                "source_candidate": feature.source_candidate,
            }
        except Exception as e:
            logger.error(f"Failed to compile and register feature '{feature.name}': {e}")
            return  # Do not register a broken feature
        feature_registry.register(name=feature.name, feature_data=feature_data)

    def _realize_code_feature(self, candidate: CandidateFeature) -> RealizedFeature:
        """Realizes a feature based on a code spec by wrapping it in a function."""
        logger.info(f"Realizing code feature: {candidate.name}")
        param_string = ", ".join(candidate.params.keys())
        code_str = f"""
import pandas as pd
import numpy as np

def {candidate.name}(df: pd.DataFrame, {param_string}):
    # This feature was generated based on the spec:
    # {candidate.spec}
    try:
        return {candidate.spec}
    except Exception as e:
        # Add context to the error. The ERROR: prefix is for the validator.
        print(f"ERROR: Error executing feature '{candidate.name}': {{e}}")
        return None
"""
        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            params=candidate.params,
            passed_test=False,  # Will be set after validation
            type=candidate.type,
            source_candidate=candidate,
        )

    def _realize_llm_feature(self, candidate: CandidateFeature) -> RealizedFeature:
        """Realizes a feature using an LLM call to generate the code."""
        logger.info(f"Realizing LLM feature: {candidate.name}")
        prompt = load_prompt(
            "realize_feature_from_spec",
            feature_name=candidate.name,
            feature_rationale=candidate.rationale,
            feature_spec=candidate.spec,
            feature_params=json.dumps(candidate.params, indent=2),
        )

        response = self.llm_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )

        if not isinstance(response, str):
            logger.error(
                f"LLM did not return a valid string response. Got: {type(response)}"
            )
            code_str = (
                f"# LLM FAILED: Response was not a string for feature {candidate.name}"
            )
        elif "```python" in response:
            code_str = response.split("```python")[1].split("```")[0].strip()
        else:
            code_str = response  # Assume the whole response is code

        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            params=candidate.params,
            passed_test=False,
            type=candidate.type,
            source_candidate=candidate,
        )

    def _realize_composition_feature(
        self, candidate: CandidateFeature
    ) -> RealizedFeature:
        """Realizes a feature by composing existing features. (Simplified)"""
        logger.info(f"Realizing composition feature: {candidate.name}")
        # This is a placeholder for a much more complex logic.
        # A robust implementation would require a dependency graph and careful parameter mapping.
        logger.warning(
            f"Composition feature '{candidate.name}' uses simplified realization logic."
        )

        dep_calls = []
        all_params = {}
        for dep_name in candidate.depends_on:
            dep_feature_data = feature_registry.get(dep_name)
            if not dep_feature_data:
                raise ValueError(f"Dependency '{dep_name}' not found in registry.")

            dep_params = dep_feature_data.get("params", {})
            all_params.update({f"{dep_name}__{k}": v for k, v in dep_params.items()})
            param_arg_str = ", ".join(dep_params.keys())
            dep_calls.append(
                f"    values['{dep_name}'] = {dep_name}(df, {param_arg_str})"
            )

        dep_calls_str = "\n".join(dep_calls)

        code_str = f"""
import pandas as pd
import numpy as np

# This is a simplified composition. A real implementation would need to handle imports.

def {candidate.name}(df: pd.DataFrame, {", ".join(all_params.keys())}):
    values = {{}}
{dep_calls_str}

    # The spec is a formula like 'feat_a * feat_b'
    result = eval('{candidate.spec}', {{"np": np}}, values)
    return result
"""

        return RealizedFeature(
            name=candidate.name,
            code_str=code_str,
            params=all_params,
            passed_test=False,
            type=candidate.type,
            source_candidate=candidate,
        )

    def _validate_feature(self, name: str, code_str: str, params: Dict) -> Tuple[bool, str]:
        """
        Validates the feature's code by executing it in a sandbox.
        Returns a tuple of (bool, str) for (pass/fail, error_message).
        """
        logger.info(f"Validating feature: {name}")
        try:
            # The new prompt template ensures the generated function can handle an empty dataframe.
            param_args = ", ".join(
                [f"{key}={repr(value)}" for key, value in params.items()]
            )
            validation_code = (
                code_str
                + "\nimport pandas as pd\nimport numpy as np\n# Validation Call\n"
                + f"print({name}(pd.DataFrame(), {param_args}))"
            )

            output = execute_python(validation_code)

            if "ERROR:" in output:
                logger.warning(f"Validation failed for {name}. Error:\n{output}")
                return False, output

            logger.success(f"Validation successful for {name}")
            return True, ""
        except Exception as e:  # pylint: disable=broad-except
            error_message = f"Exception during validation for {name}: {e}"
            logger.error(error_message, exc_info=True)
            return False, str(e)
