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
            system_message=(
                "You are an expert Python programmer. You must ONLY output the Python function code matching the provided template. "
                "Do NOT include any tool directives (e.g., @UserProxy_Strategy please run ...), object/class instantiations, or extra markdown/code blocks. "
                "Fill in ONLY the logic section marked in the template. Do NOT alter the function signature or imports. "
                "Your code must be clean, efficient, robust, and use only standard libraries like pandas and numpy."
            ),
        )

    @agent_run_decorator("FeatureRealizationAgent")
    def run(self) -> None:
        """
        Main method to realize candidate features from the session state, enforcing the contract-based template.
        Each candidate is realized using a strict function template, validated, and registered. Redundant/legacy code is removed.
        """
        logger.info("Starting feature realization...")
        candidate_features_data = self.session_state.get_candidate_features()
        if not candidate_features_data:
            logger.warning("No candidate features found to realize.")
            self.session_state.set_state("realized_features", [])
            self.session_state.set_state("features", {})
            return

        # Patch: fill missing fields with defaults to avoid validation errors
        for f in candidate_features_data:
            if "type" not in f or f["type"] is None:
                f["type"] = "code"
            if "spec" not in f or f["spec"] is None:
                f["spec"] = ""
            if "rationale" not in f or f["rationale"] is None:
                f["rationale"] = f.get("description", "")
        candidate_features = [CandidateFeature(**f) for f in candidate_features_data]
        realized_features: List[RealizedFeature] = []
        MAX_RETRIES = 2

        user_proxy = autogen.UserProxyAgent(
            name="TempProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        fast_mode_sample_frac = self.session_state.get_state("fast_mode_sample_frac")
        logger.info(f"[FeatureRealizationAgent] fast_mode_sample_frac before set: {fast_mode_sample_frac}")
        self.session_state.set_state("optimizer_sample_frac", fast_mode_sample_frac)
        optimizer_sample_frac = self.session_state.get_state("optimizer_sample_frac")
        logger.info(f"[FeatureRealizationAgent] optimizer_sample_frac after set: {optimizer_sample_frac}")

        for candidate in candidate_features:
            logger.info(f"Attempting to realize feature: {candidate.name}")
            is_realized = False
            last_error = ""
            code_str = ""
            for attempt in range(MAX_RETRIES + 1):
                if attempt == 0:
                    template_kwargs = dict(
                        feature_name=candidate.name,
                        description=candidate.description,
                        dependencies=candidate.depends_on if hasattr(candidate, 'depends_on') else candidate.dependencies,
                        params=list(candidate.params.keys()) if hasattr(candidate, 'params') else [],
                    )
                    message = load_prompt("agents/strategy_team/feature_realization_agent.j2", **template_kwargs)
                    prompt = (
                        "Your only job is to fill in the Python function template for this feature. "
                        "Do NOT add any extra markdown or explanations. Output ONLY the function code block."
                    )
                    message = f"{message}\n\n{prompt}"
                else:
                    message = (
                        f"The previous code you wrote for the feature '{candidate.name}' failed validation with the following error:\n---\nERROR:\n{last_error}\n---\n"
                        "Please provide a corrected version of the LOGIC BLOCK ONLY to be inserted into the function."
                    )
                user_proxy.initiate_chat(self.llm_agent, message=message, max_turns=1, silent=True)
                last_message = user_proxy.last_message(self.llm_agent)
                if not last_message or "content" not in last_message:
                    last_error = "LLM response was empty or invalid."
                    code_str = ""
                    continue
                response_msg = last_message["content"]
                try:
                    code_str = response_msg.split("```python")[1].split("```", 1)[0].strip()
                except IndexError:
                    code_str = response_msg.strip()
                passed, last_error = self._validate_feature(candidate.name, code_str, candidate.params)
                if passed:
                    logger.success(f"Successfully validated feature '{candidate.name}' on attempt {attempt + 1}.")
                    is_realized = True
                    break
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
        # --- Correlation-based feature pruning ---
        import pandas as pd
        import numpy as np
        feature_series = {}
        for r in realized_features:
            if r.passed_test:
                try:
                    temp_namespace = {}
                    exec(r.code_str, globals(), temp_namespace)
                    func = temp_namespace[r.name]
                    dummy_df = pd.DataFrame({"user_id": [1, 2, 3], "book_id": [10, 20, 30], "rating": [4, 5, 3]})
                    s = func(dummy_df, **r.params)
                    if isinstance(s, pd.Series):
                        feature_series[r.name] = s.reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Could not compute feature '{r.name}' for correlation analysis: {e}")
        if len(feature_series) > 1:
            feature_matrix = pd.DataFrame(feature_series)
            corr = feature_matrix.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                for row in upper.index:
                    if upper.loc[row, col] > 0.95:
                        var_row = feature_matrix[row].var()
                        var_col = feature_matrix[col].var()
                        drop = row if var_row < var_col else col
                        to_drop.add(drop)
            pruned_realized = [r for r in realized_features if r.name not in to_drop]
            if to_drop:
                logger.info(f"Pruned highly correlated features: {sorted(list(to_drop))}")
        else:
            pruned_realized = realized_features
        # Save to both realized_features and features for downstream use
        self.session_state.set_state("realized_features", [r.model_dump() for r in pruned_realized])
        self.session_state.set_state("features", {r.name: r.model_dump() for r in pruned_realized if r.passed_test})
        successful_count = len([r for r in pruned_realized if r.passed_test])
        logger.info(f"Finished feature realization. Successfully realized and validated {successful_count} features after correlation pruning.")

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
            type=candidate.type or "code",
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
