import math
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import structlog
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax

from vulcan.prompt_templates import (
    CREATE_FEATURE_PROMPT,
    MATH_FEATURE_PROMPT,
    MUTATE_FEATURE_PROMPT,
    REFINE_TOP_FEATURE_PROMPT,
    REFLECT_AND_REFINE_PROMPT,
    REPAIR_FEATURE_PROMPT,
    ROW_INFERENCE_PROMPT,
    SYSTEM_PROMPT,
)
from vulcan.schemas import FeatureDefinition
from vulcan.schemas.agent_schemas import LLMFeatureOutput, LLMInteractionLog
from vulcan.schemas.evolution_types import FeatureCandidate

logger = structlog.get_logger(__name__)


class BaseUCBAgent(ABC):
    def __init__(self, name: str, llm_kwargs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.count = 0
        self.reward_sum = 0.0
        self.llm_kwargs = llm_kwargs or {}
        # Assuming API key is set in the environment
        self.llm_client = ChatOpenAI(model="gpt-4-turbo-preview", **self.llm_kwargs)
        self.prompt_template: Optional[str] = None

    def _create_chain(self):
        """Creates the LangChain prompt and parser."""
        if not self.prompt_template:
            raise ValueError("Agent's prompt_template is not set.")

        parser = PydanticOutputParser(pydantic_object=LLMFeatureOutput)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", self.prompt_template),
            ]
        )
        return prompt, parser

    def _invoke_chain_with_logging(
        self, prompt, parser, prompt_input: Dict[str, Any]
    ) -> Tuple[Optional[LLMFeatureOutput], LLMInteractionLog]:
        """Invokes the LLM chain, logs the interaction, and prints to console."""
        console = Console()
        parsed_response = None
        error_message = None
        raw_response_content = ""

        # Get raw response by invoking prompt and then client
        formatted_prompt = prompt.invoke(prompt_input)

        # --- New Prompt Logging ---
        prompt_panel = Panel(
            str(formatted_prompt),
            title="📬 Full Prompt to LLM",
            border_style="dim blue",
            expand=True,
        )
        console.print(prompt_panel)
        # --- End New Prompt Logging ---

        with console.status(f"[bold green]Asking {self.name}...[/bold green]"):
            try:
                raw_response = self.llm_client.invoke(formatted_prompt)
                raw_response_content = raw_response.content
                # Parse response
                parsed_response = parser.parse(raw_response_content)
            except Exception as e:
                error_message = str(e)

        log = LLMInteractionLog(
            agent_name=self.name,
            timestamp=time.time(),
            prompt_input=prompt_input,
            raw_response=raw_response_content,
            parsed_response=parsed_response,
            error_message=error_message,
        )

        # Pretty print the interaction to the console
        if parsed_response:
            title = f"🤖 [bold green]Agent:[/] [bright_cyan]{self.name}[/bright_cyan] | [bold green]Feature:[/] [bright_white]{parsed_response.name}[/bright_white]"

            context_str = ""
            for k, v in prompt_input.items():
                if k == "format_instructions":
                    continue
                # Truncate long values for display
                v_str = str(v)
                if len(v_str) > 1000:
                    v_str = v_str[:500] + "\\n...\\n" + v_str[-500:]
                context_str += f"[bold cyan]{k}:[/bold cyan]\\n{v_str}\\n\\n"

            context_panel = Panel(
                context_str.strip(),
                title="📝 Prompt Context",
                border_style="cyan",
                expand=True,
            )

            reasoning_panel = Panel(
                parsed_response.chain_of_thought_reasoning,
                title="🤔 Chain-of-Thought Reasoning",
                border_style="yellow",
                expand=True,
            )

            main_panel_group = Group(context_panel, reasoning_panel)

            code_body = parsed_response.code or parsed_response.llm_prompt or "N/A"
            code_panel = Panel(
                Syntax(code_body, "python", theme="monokai", line_numbers=True),
                title="💡 Generated Feature Code/Prompt",
                border_style="blue",
                expand=True,
            )
            console.print(
                Panel(main_panel_group, title=title, border_style="green", expand=True)
            )
            console.print(code_panel)

        elif error_message:
            error_panel = Panel(
                f"[bold red]Error during LLM call:[/bold red]\\n{error_message}",
                title=f"❌ Agent: {self.name} Failed",
                border_style="red",
            )
            console.print(error_panel)

        return parsed_response, log

    def get_ucb1(self, total_pulls: int) -> float:
        if self.count == 0:
            return float("inf")
        average_reward = self.reward_sum / self.count
        exploration_term = math.sqrt(2 * math.log(total_pulls) / self.count)
        return average_reward + exploration_term

    def update_reward(self, reward: float):
        self.count += 1
        self.reward_sum += reward

    def _create_feature_definition_from_llm_output(
        self, llm_output: LLMFeatureOutput
    ) -> FeatureDefinition:
        """Helper to convert LLM output to a FeatureDefinition."""
        return FeatureDefinition(
            name=llm_output.name,
            description=llm_output.description,
            feature_type=llm_output.feature_type,
            code=llm_output.code,
            llm_prompt=llm_output.llm_prompt,
            llm_chain_of_thought_reasoning=llm_output.chain_of_thought_reasoning,
        )

    @abstractmethod
    def select(self, *args, **kwargs) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        """Selects or creates a feature, returning the definition and the interaction log."""
        pass


class IdeateNewAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = CREATE_FEATURE_PROMPT

    def select(
        self, context: str, existing_features: List[FeatureCandidate]
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        prompt, parser = self._create_chain()
        existing_feature_summary = [
            f'{{name: "{f.feature.name}", desc: "{f.feature.description}"}}'
            for f in existing_features
        ]
        prompt_input = {
            "context": context,
            "existing_features": str(existing_feature_summary),
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for IdeateNewAgent. Log: {log}")
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class RefineTopAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = REFINE_TOP_FEATURE_PROMPT

    def select(
        self, context: str, existing_features: List[FeatureCandidate]
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        if not existing_features:
            raise ValueError("RefineTopAgent requires at least one existing feature.")

        top_feature = max(existing_features, key=lambda f: f.score)
        prompt, parser = self._create_chain()
        prompt_input = {
            "context": context,
            "feature_name": top_feature.feature.name,
            "feature_description": top_feature.feature.description,
            "feature_implementation": top_feature.feature.code
            or top_feature.feature.llm_prompt,
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for RefineTopAgent. Log: {log}")
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class MutateExistingAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = MUTATE_FEATURE_PROMPT

    def select(
        self, context: str, existing_features: List[FeatureCandidate]
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        if not existing_features:
            raise ValueError(
                "MutateExistingAgent requires at least one existing feature."
            )

        # Tournament selection
        tournament_size = min(5, len(existing_features))
        tournament = random.sample(existing_features, tournament_size)
        feature_to_mutate = max(tournament, key=lambda x: x.score)

        prompt, parser = self._create_chain()
        prompt_input = {
            "context": context,
            "feature_name": feature_to_mutate.feature.name,
            "feature_description": feature_to_mutate.feature.description,
            "feature_implementation": feature_to_mutate.feature.code
            or feature_to_mutate.feature.llm_prompt,
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for MutateExistingAgent. Log: {log}")
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class LLMRowAgent(BaseUCBAgent):
    def __init__(self, name: str, batch_size: int = 20, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.prompt_template = ROW_INFERENCE_PROMPT

    def select(
        self, context: str, data_rows: List[Dict], text_columns: List[str]
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        prompt, parser = self._create_chain()
        prompt_input = {
            "context": context,
            "data_rows": str(data_rows),
            "text_columns": str(text_columns),
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for LLMRowAgent. Log: {log}")
        # For LLMRowAgent, the "code" is not applicable, and the "llm_prompt" is the key output.
        # The LLMFeatureOutput schema handles this with optional fields.
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class ReflectAndRefineAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = REFLECT_AND_REFINE_PROMPT

    def select(
        self, evaluated_feature: FeatureCandidate
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        prompt, parser = self._create_chain()
        feature = evaluated_feature.feature
        metrics = (
            evaluated_feature.evaluation_result.metrics.dict()
            if evaluated_feature.evaluation_result
            else {}
        )

        prompt_input = {
            "feature_name": feature.name,
            "feature_description": feature.description,
            "feature_implementation": feature.code or feature.llm_prompt,
            "feature_score": evaluated_feature.score,
            "feature_metrics": str(metrics),
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for ReflectAndRefineAgent. Log: {log}")
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class RepairAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = REPAIR_FEATURE_PROMPT

    def select(
        self, faulty_code: str, error_message: str
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        prompt, parser = self._create_chain()

        prompt_input = {
            "faulty_code": faulty_code,
            "error_message": error_message,
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(f"LLM output failed for RepairAgent. Log: {log}")
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log


class MathematicalFeatureAgent(BaseUCBAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.prompt_template = MATH_FEATURE_PROMPT

    def select(
        self, context: str, existing_features: List[FeatureCandidate]
    ) -> Tuple[FeatureDefinition, LLMInteractionLog]:
        prompt, parser = self._create_chain()
        existing_feature_summary = [
            f'{{name: "{f.feature.name}", desc: "{f.feature.description}"}}'
            for f in existing_features
        ]
        prompt_input = {
            "context": context,
            "existing_features": str(existing_feature_summary),
            "format_instructions": parser.get_format_instructions(),
        }
        llm_output, log = self._invoke_chain_with_logging(prompt, parser, prompt_input)
        if not llm_output:
            raise ValueError(
                f"LLM output failed for MathematicalFeatureAgent. Log: {log}"
            )
        feature_def = self._create_feature_definition_from_llm_output(llm_output)
        return feature_def, log
