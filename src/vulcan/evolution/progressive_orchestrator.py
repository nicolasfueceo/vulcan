"""Progressive Feature Evolution Orchestrator for VULCAN."""

import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

# Attempt to import for plotting, but make it optional
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from vulcan.agents.ucb_agents import (
    IdeateNewAgent,
    LLMRowAgent,
    MathematicalFeatureAgent,
    MutateExistingAgent,
    RefineTopAgent,
    ReflectAndRefineAgent,
    RepairAgent,
)
from vulcan.evaluation.recommendation_evaluator import RecommendationEvaluator
from vulcan.features import FeatureExecutor
from vulcan.schemas import (
    DataContext,
    EvolutionAction,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureSet,
    FeatureType,
    VulcanConfig,
)
from vulcan.schemas.agent_schemas import LLMInteractionLog
from vulcan.schemas.evolution_types import FeatureCandidate, GenerationStats
from vulcan.utils import ExperimentTracker, PerformanceTracker, get_vulcan_logger

logger = structlog.get_logger(__name__)


class ProgressiveEvolutionOrchestrator:
    """Orchestrates progressive feature evolution with RL-guided actions."""

    def __init__(
        self,
        config: VulcanConfig,
        results_manager: Any,
        performance_tracker: Optional[PerformanceTracker] = None,
        websocket_callback: Optional[callable] = None,
    ) -> None:
        """Initialize the progressive evolution orchestrator."""
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.websocket_callback = websocket_callback
        self.results_manager = results_manager

        # Core components
        self.agents = {
            "IdeateNew": IdeateNewAgent("IdeateNew"),
            "RefineTop": RefineTopAgent("RefineTop"),
            "MutateExisting": MutateExistingAgent("MutateExisting"),
            "LLMRowInference": LLMRowAgent("RowInference", batch_size=20),
            "MathematicalFeature": MathematicalFeatureAgent("MathematicalFeature"),
        }
        self.repair_agent = RepairAgent("RepairAgent")
        self.reflection_agent = ReflectAndRefineAgent("ReflectAndRefine")
        self.feature_executor = FeatureExecutor(config)
        self.feature_evaluator = RecommendationEvaluator(config)
        self.experiment_tracker: Optional[ExperimentTracker] = None

        # Performance tracking
        self.performance_tracker = performance_tracker or PerformanceTracker(
            max_history=1000
        )

        # Evolution parameters
        # Note: Parameters like population_size, generation_size, etc., are sourced from
        # the config.experiment section now.
        self.population_size = self.config.experiment.population_size
        self.generation_size = self.config.experiment.generation_size
        self.max_repair_attempts = self.config.experiment.max_repair_attempts
        self.mutation_rate = self.config.experiment.mutation_rate
        # UCB exploration constant will be added later as per step 2 instructions

        # Population management
        self.population: List[FeatureCandidate] = []
        self.generation_history: List[GenerationStats] = []
        self.feature_registry: Dict[str, FeatureCandidate] = {}  # Prevent duplicates
        self.llm_interactions: List[LLMInteractionLog] = []

        # RL State is being replaced by UCB stats on agents
        self.current_generation = 0
        self.total_features_generated = 0
        self.total_ucb_pulls = 0

        # Tracking
        self.best_score = 0.0
        self.best_candidate: Optional[FeatureCandidate] = None

        # Create experiment folder, TensorBoard writer, and metadata.json
        # Ensure experiments_root exists (taken from config.experiment.output_dir)
        exp_root_path = Path(self.config.experiment.output_dir)
        exp_root_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure base experiments dir exists

        self.experiment_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Added UUID for more uniqueness
        exp_dir_path = exp_root_path / self.experiment_id
        exp_dir_path.mkdir(parents=True, exist_ok=True)

        # Write metadata.json
        metadata_path = exp_dir_path / "metadata.json"
        try:
            # Convert Pydantic config to dict for JSON serialization
            config_dict = self.config.dict()
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "experiment_id": self.experiment_id,
                        "config": config_dict,  # Store the whole config
                        "start_time": str(datetime.utcnow()),
                        "status": "initialized",  # Initial status
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            self.logger.error(
                f"Failed to write metadata.json to {metadata_path}", error=str(e)
            )

        # TensorBoard logging subfolder:
        self.tensorboard_logdir = exp_dir_path / "tensorboard"
        self.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
        try:
            self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_logdir))
            self.logger.info(
                f"TensorBoard writer initialized. Log directory: {self.tensorboard_logdir}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize TensorBoard SummaryWriter at {self.tensorboard_logdir}",
                error=str(e),
            )
            self.tb_writer = None  # Ensure tb_writer is None if init fails

        self.run_dir = exp_dir_path

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # No agent initialization needed for UCB agents for now
            await self.feature_executor.initialize()
            await self.feature_evaluator.initialize()

            if self.config.experiment.wandb_enabled:
                self.experiment_tracker = ExperimentTracker(self.config.experiment)

            self.logger.info("Progressive Evolution Orchestrator initialized")
            return True

        except Exception as e:
            self.logger.error("Failed to initialize orchestrator", error=str(e))
            return False

    async def run_evolution(
        self,
        data_context: DataContext,
        max_generations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run progressive feature evolution."""
        logger.info(
            "run_evolution received DataContext",
            data_context_type=type(data_context),
            train_data_type=type(data_context.train_data),
        )
        max_generations = max_generations or self.config.experiment.max_generations

        self.logger.info(
            "Starting progressive feature evolution",
            max_generations=max_generations,
            population_size=self.population_size,
            generation_size=self.generation_size,
        )

        start_time = time.time()

        try:
            # Initialize with seed population
            await self._initialize_population(data_context)

            for generation in range(max_generations):
                self.current_generation = generation + 1
                self.total_ucb_pulls += 1

                # Choose action using UCB policy
                action_name, chosen_agent = self._choose_action_ucb()

                # Generate new candidates based on action
                candidates = await self._generate_candidates(chosen_agent, data_context)

                # Evaluate candidates (with automatic repair)
                evaluated_candidates = await self._evaluate_candidates(
                    candidates, data_context
                )

                # Add refined candidates from reflection step
                refined_candidates = await self._reflect_and_refine(
                    evaluated_candidates, data_context
                )
                evaluated_candidates.extend(refined_candidates)

                # Update population
                self._update_population(evaluated_candidates)

                # Record generation statistics
                gen_stats = self._record_generation_stats(
                    action_name, evaluated_candidates
                )

                # Update UCB rewards for the agent
                reward_for_ucb = gen_stats.best_score
                chosen_agent.update_reward(reward_for_ucb)

                # Log to TensorBoard
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        f"reward/arm/{action_name}",
                        reward_for_ucb,
                        self.current_generation,
                    )
                    self.tb_writer.add_scalar(
                        "reward/best_in_gen", reward_for_ucb, self.current_generation
                    )
                    self.tb_writer.add_scalar(
                        "score/best_overall", self.best_score, self.current_generation
                    )
                    self.tb_writer.add_scalar(
                        "population/size", len(self.population), self.current_generation
                    )

                if self._check_stopping_criteria(max_generations):
                    self.logger.info("Stopping criteria met. Finalizing evolution.")
                    break

            execution_time = time.time() - start_time
            results = await self._generate_evolution_results(
                data_context, execution_time
            )

            self.logger.info("Progressive evolution completed", **results)

            # Optional: Save plot of score curve if matplotlib is available and TensorBoard is not used
            if (
                MATPLOTLIB_AVAILABLE
                and not self.tb_writer
                and self.config.experiment.save_artifacts
            ):
                try:
                    # We need the specific experiment output directory here.
                    # This should be passed from VulcanOrchestrator or accessed via a shared context.
                    # Assuming self.config might hold a more specific path after VO sets it.
                    # This is a bit of a hack; ideally VO provides this path explicitly to run_evolution.
                    # Let's try to get it from a potentially updated config by VO or a new attribute.
                    # This path should resolve to .../experiments/YYYYMMDD_HHMMSS_exp_name_uuid/
                    # For now, let's assume the current_experiment_dir_path is made available to this instance,
                    # e.g., self.current_experiment_dir_path set by VulcanOrchestrator before calling this.
                    # This is not ideal. A better way is for VO to pass this path.
                    # HACK: Attempting to use a convention if VO updated a shared config or similar.
                    # This requires careful coordination with VulcanOrchestrator.
                    # Let's assume the main logger's file handler is in the correct experiment dir.
                    plot_save_path = None
                    root_logger = logging.getLogger()  # Standard logging root
                    for handler in root_logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            plot_save_path = (
                                Path(handler.baseFilename).parent / "score_curve.png"
                            )
                            break

                    if plot_save_path and self.generation_history:
                        generations = [s.generation for s in self.generation_history]
                        best_scores = [s.best_score for s in self.generation_history]
                        avg_scores = [s.avg_score for s in self.generation_history]

                        plt.figure(figsize=(10, 6))
                        plt.plot(
                            generations,
                            best_scores,
                            label="Best Score per Generation",
                            marker="o",
                        )
                        plt.plot(
                            generations,
                            avg_scores,
                            label="Average Score per Generation",
                            marker="x",
                        )
                        plt.xlabel("Generation")
                        plt.ylabel("Score")
                        plt.title("Evolution Score Curve")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(plot_save_path)
                        plt.close()
                        self.logger.info(f"Score curve plot saved to {plot_save_path}")
                    else:
                        self.logger.warning(
                            "Could not determine plot save path or no history to plot for score_curve.png"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate or save score curve plot: {e}",
                        exc_info=True,
                    )

            return results

        except Exception as e:
            self.logger.error("Evolution failed", error=str(e))
            raise

    def _check_stopping_criteria(self, max_generations: int) -> bool:
        """Check if evolution should stop."""
        if self.current_generation >= max_generations:
            return True

        # Stop if best score hasn't improved in N generations
        patience = self.config.experiment.get("early_stopping_patience", 10)
        if len(self.generation_history) > patience:
            recent_history = self.generation_history[-patience:]
            best_scores = [s.best_score for s in recent_history]
            # using a small tolerance
            if (max(best_scores) - recent_history[0].best_score) < 1e-4:
                self.logger.info("Early stopping due to no improvement.")
                return True

        return False

    def _choose_action_ucb(self) -> Tuple[str, Any]:
        """Choose an agent using the UCB1 algorithm."""
        ucb_values = {
            name: agent.get_ucb1(self.total_ucb_pulls)
            for name, agent in self.agents.items()
        }
        chosen_name = max(ucb_values, key=ucb_values.get)
        self.logger.info(
            "Chose action with UCB", action=chosen_name, ucb_scores=ucb_values
        )
        return chosen_name, self.agents[chosen_name]

    async def _initialize_population(self, data_context: DataContext) -> None:
        """Initialize the population with features from the IdeateNewAgent."""
        self.logger.info("Initializing population with new features")
        logger.info(
            "_initialize_population received DataContext",
            data_context_type=type(data_context),
            train_data_type=type(data_context.train_data),
        )
        initial_agent = self.agents["IdeateNew"]

        # --- Add data sample to context ---
        data_sample = data_context.train_data.head(10).to_string()
        context_summary = f"""Data Schema: {list(data_context.data_schema.keys())}\nUsers: {data_context.n_users}, Items: {data_context.n_items}\n\nData Sample:\n{data_sample}"""

        candidates = []
        for _ in range(self.population_size):
            feature_def, log = initial_agent.select(
                context=context_summary, existing_features=self.population
            )
            self.llm_interactions.append(log)
            candidate = FeatureCandidate(
                feature=feature_def,
                score=0.0,
                generation=0,
            )
            candidates.append(candidate)

        evaluated_candidates = await self._evaluate_candidates(candidates, data_context)
        self._update_population(evaluated_candidates)
        self.logger.info(
            f"Initialized population with {len(self.population)} features."
        )

    async def _generate_candidates(
        self, agent: Any, data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Generate new candidates using the selected UCB agent."""
        self.logger.info(f"Generating candidates with agent: {agent.name}")
        candidates = []

        # --- Add data sample to context ---
        data_sample = data_context.train_data.head(10).to_string()
        context_summary = f"""Data Schema: {list(data_context.data_schema.keys())}\nUsers: {data_context.n_users}, Items: {data_context.n_items}\n\nData Sample:\n{data_sample}"""

        for _ in range(self.generation_size):
            kwargs = {"context": context_summary}
            if agent.name in ["RefineTop", "MutateExisting"]:
                if not self.population:
                    self.logger.warning(
                        f"Agent {agent.name} requires a population, but it's empty. Skipping generation."
                    )
                    continue
                kwargs["existing_features"] = self.population

            if agent.name == "LLMRowInference":
                # This needs a sample of data rows. Using validation_data for this.
                # In a real scenario, this might need more careful handling.
                df = data_context.validation_data
                kwargs["data_rows"] = df.head(agent.batch_size).to_dict("records")
                kwargs["text_columns"] = data_context.text_columns

            feature_def, log = agent.select(**kwargs)
            self.llm_interactions.append(log)
            candidate = FeatureCandidate(
                feature=feature_def, score=0.0, generation=self.current_generation
            )
            candidates.append(candidate)
        return candidates

    async def _reflect_and_refine(
        self, evaluated_candidates: List[FeatureCandidate], data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Use reflection to refine successful features."""
        refined_features = []
        for candidate in evaluated_candidates:
            if (
                candidate.status == "evaluated"
                and candidate.score
                > self.config.experiment.get("reflection_threshold", 0.6)
            ):
                self.logger.info(
                    f"Reflecting on successful feature {candidate.feature.name} with score {candidate.score}"
                )
                try:
                    refined_feature_def, log = self.reflection_agent.select(
                        evaluated_feature=candidate
                    )
                    self.llm_interactions.append(log)

                    # Create a new candidate for the refined feature
                    refined_candidate = FeatureCandidate(
                        feature=refined_feature_def,
                        score=0.0,
                        generation=self.current_generation,
                        parent_id=candidate.feature.name,
                        mutation_type="reflection",
                    )

                    # Evaluate the new refined candidate
                    evaluated_refined_list = await self._evaluate_candidates(
                        [refined_candidate], data_context
                    )
                    if evaluated_refined_list:
                        refined_features.extend(evaluated_refined_list)

                except Exception as e:
                    self.logger.error(
                        f"Failed to refine feature {candidate.feature.name} after reflection",
                        error=str(e),
                    )

        return refined_features

    async def _evaluate_candidates(
        self, candidates: List[FeatureCandidate], data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Evaluate and repair feature candidates."""
        evaluated_candidates = []
        console = Console()

        for candidate in candidates:
            candidate.status = "evaluating"
            self.logger.info(
                "Evaluating candidate", feature_name=candidate.feature.name
            )

            # Execute feature to get values
            # This now returns a tuple: (success, error_message, feature_results_dict)
            success, error_msg, feature_results = await self._execute_with_repair(
                candidate, data_context
            )

            if not success:
                self.logger.warning(
                    "Execution failed after repairs",
                    feature_name=candidate.feature.name,
                    error=error_msg,
                )
                candidate.status = "failed"
                candidate.error_message = error_msg

                # --- Add explicit failure logging to console ---
                failure_panel = Panel(
                    f"[bold]Feature Name:[/] {candidate.feature.name}\\n\\n[bold]Error:[/]\\n[red]{error_msg}[/red]",
                    title="💀 Feature Execution Failed",
                    border_style="bold red",
                    expand=False,
                )
                console.print(failure_panel)
                # --- End failure logging ---

                evaluated_candidates.append(candidate)
                continue

            # Evaluate feature set (which now contains just this one feature)
            feature_set = FeatureSet(features=[candidate.feature])
            evaluation_result = await self.feature_evaluator.evaluate_feature_set(
                feature_set, feature_results, data_context, self.current_generation
            )

            candidate.evaluation_result = evaluation_result
            candidate.score = evaluation_result.overall_score
            candidate.status = "evaluated"

            # --- Detailed Console Logging ---
            title = f"📈 Evaluation: [bold bright_white]{candidate.feature.name}[/]"
            score_color = (
                "green"
                if candidate.score > 0.5
                else "yellow"
                if candidate.score > 0.2
                else "red"
            )

            summary_table = Table(show_header=False, box=None, padding=(0, 1))
            summary_table.add_row(
                "[bold]Overall Score[/]:",
                f"[{score_color}]{candidate.score:.4f}[/{score_color}]",
            )

            metrics_table = Table(
                title="[bold]Metric Breakdown[/]",
                box=None,
                show_header=True,
                header_style="bold cyan",
            )
            metrics_table.add_column("Metric", justify="left")
            metrics_table.add_column("Value", justify="right")

            if evaluation_result.metrics:
                metrics_dict = evaluation_result.metrics.dict()
                for key, value in metrics_dict.items():
                    if value is not None:
                        metrics_table.add_row(
                            key.replace("_", " ").title(), f"{value:.4f}"
                        )

            log_panel = Panel(
                Group(summary_table, metrics_table),
                title=title,
                border_style="blue",
                expand=False,
            )
            console.print(log_panel)
            # --- End Detailed Logging ---

            evaluated_candidates.append(candidate)

        self.logger.info(
            "Finished evaluating all candidates for this generation.",
            count=len(candidates),
        )
        return evaluated_candidates

    async def _execute_with_repair(
        self, candidate: FeatureCandidate, data_context: DataContext
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Execute feature, with a repair loop if execution fails."""
        for attempt in range(self.max_repair_attempts + 1):
            try:
                # Try to execute the feature
                results = await self.feature_executor.execute_feature_set(
                    features=[candidate.feature],
                    data_context=data_context,
                    target_split="validation",
                )
                return True, None, results  # Success!

            except Exception as e:
                error_msg = str(e)
                self.logger.warning(
                    "Feature execution failed",
                    feature_name=candidate.feature.name,
                    error=error_msg,
                    attempt=attempt + 1,
                )

                if attempt >= self.max_repair_attempts:
                    self.logger.error(
                        "Feature repair failed after max attempts",
                        feature_name=candidate.feature.name,
                    )
                    return False, error_msg, None

                # Attempt to repair the feature
                self.logger.info(
                    "Attempting to repair feature",
                    feature_name=candidate.feature.name,
                )
                try:
                    repaired_feature_def, log = self.repair_agent.select(
                        faulty_code=candidate.feature.code or "",
                        error_message=error_msg,
                    )
                    self.llm_interactions.append(log)
                    candidate.feature = repaired_feature_def
                    candidate.repair_attempts += 1
                    self.logger.info(
                        "Feature repaired successfully, retrying execution",
                        feature_name=candidate.feature.name,
                    )
                except Exception as repair_e:
                    self.logger.error(
                        "Agent-based repair failed",
                        feature_name=candidate.feature.name,
                        repair_error=str(repair_e),
                    )
                    return (
                        False,
                        f"Execution failed: {error_msg}. Repair failed: {repair_e}",
                        None,
                    )

        return False, "Reached end of repair loop unexpectedly.", None

    def _update_population(self, new_candidates: List[FeatureCandidate]) -> None:
        """Update population with new candidates, keeping only the best."""
        # Add successful candidates to population
        successful_candidates = [c for c in new_candidates if c.status == "evaluated"]
        self.population.extend(successful_candidates)

        # Sort by score (descending) and keep top performers
        self.population.sort(key=lambda x: x.score, reverse=True)
        self.population = self.population[: self.population_size]

        self._update_best_candidate()

        # --- Log Best Feature ---
        if self.best_candidate:
            self.logger.info(
                "Current Best Feature",
                name=self.best_candidate.feature.name,
                score=self.best_candidate.score,
                generation=self.best_candidate.generation,
            )
        # --- End Log Best Feature ---

        # Save the updated population to disk
        if self.results_manager:
            population_for_saving = [
                {
                    "id": cand.feature.name,
                    "score": cand.score,
                    "generation": cand.generation,
                    "parent_id": cand.parent_id,
                    "mutation_type": cand.mutation_type,
                    "status": cand.status,
                    "repair_attempts": cand.repair_attempts,
                }
                for cand in self.population
            ]
            self.results_manager.update_population(population_for_saving)

    def _update_best_candidate(self) -> None:
        """Update the best candidate tracker."""
        if self.population:
            current_best = self.population[0]
            if current_best.score > self.best_score:
                self.best_score = current_best.score
                self.best_candidate = current_best

    def _select_parent_feature(self) -> FeatureCandidate:
        """Select parent feature for mutation (tournament selection)."""
        tournament_size = min(5, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.score)

    def _get_feature_key(self, feature: FeatureDefinition) -> str:
        """Generate unique key for feature to prevent duplicates."""
        # Create a unique key based on feature type and implementation
        if feature.feature_type == FeatureType.CODE_BASED:
            implementation = feature.code or ""
        elif feature.feature_type == FeatureType.LLM_BASED:
            implementation = feature.llm_prompt or ""
        elif feature.feature_type == FeatureType.HYBRID:
            implementation = f"{feature.llm_prompt}_{feature.preprocessing_code}_{feature.postprocessing_code}"
        else:
            implementation = ""

        return f"{feature.name}_{hash(implementation)}"

    def _get_population_features(self) -> FeatureSet:
        """Get features from current population as a FeatureSet."""
        features = [candidate.feature for candidate in self.population]
        return FeatureSet(
            features=features,
            action_taken=EvolutionAction.GENERATE_NEW,
        )

    def _get_recent_evaluations(self) -> List[FeatureEvaluation]:
        """Get recent feature evaluations."""
        return []  # Simplified for now

    def _record_generation_stats(
        self, action: str, candidates: List[FeatureCandidate]
    ) -> GenerationStats:
        """Record statistics for the current generation."""
        successful = [
            c for c in candidates if c.status == "evaluated" and c.score is not None
        ]
        failed = len(candidates) - len(successful)
        repaired = sum(
            1 for c in candidates if c.repair_attempts > 0 and c.status == "evaluated"
        )

        avg_score = (
            sum(c.score for c in successful) / len(successful) if successful else 0.0
        )
        best_score_this_gen = max(c.score for c in successful) if successful else 0.0

        # Calculate average clustering and recommendation metrics for successful candidates
        avg_silhouette = 0.0
        avg_num_clusters = 0.0
        avg_ndcg_at_k = 0.0
        metrics_count = 0

        for cand in successful:
            if cand.evaluation_result and cand.evaluation_result.metrics:
                metrics = cand.evaluation_result.metrics
                avg_silhouette += (
                    getattr(metrics, "silhouette_score", 0) or 0
                )  # Handle None
                avg_num_clusters += (
                    getattr(metrics, "num_clusters", 0) or 0
                )  # Handle None
                avg_ndcg_at_k += (
                    getattr(metrics, "ndcg_at_k", {}).get("mean", 0) or 0
                )  # ndcg_at_k might be a dict
                metrics_count += 1

        if metrics_count > 0:
            avg_silhouette /= metrics_count
            avg_num_clusters /= metrics_count
            avg_ndcg_at_k /= metrics_count

        progress_log_data = {
            "generation": self.current_generation,
            "action_taken": action,
            "total_candidates": len(candidates),
            "successful_features": len(successful),
            "failed_features": failed,
            "repaired_features": repaired,
            "avg_score_this_gen": avg_score,
            "best_score_this_gen": best_score_this_gen,
            "current_best_overall_score": self.best_score,
            "population_size": len(self.population),
        }
        self.logger.info("Generation completed", **progress_log_data)

        if self.tb_writer:
            self.tb_writer.add_scalar(
                "Generation/AvgScore", avg_score, self.current_generation
            )
            self.tb_writer.add_scalar(
                "Generation/BestScoreThisGen",
                best_score_this_gen,
                self.current_generation,
            )
            self.tb_writer.add_scalar(
                "Generation/SuccessfulFeatures",
                len(successful),
                self.current_generation,
            )
            self.tb_writer.add_scalar(
                "Generation/FailedFeatures", failed, self.current_generation
            )
            self.tb_writer.add_scalar(
                "Generation/RepairedFeatures", repaired, self.current_generation
            )
            self.tb_writer.add_scalar(
                "Generation/PopulationSize",
                len(self.population),
                self.current_generation,
            )
            self.tb_writer.add_scalar(
                "OverallBestScore", self.best_score, self.current_generation
            )
            # Log UCB stats from agents
            for agent_name, agent in self.agents.items():
                self.tb_writer.add_scalar(
                    f"UCB/Pulls/{agent_name}", agent.count, self.current_generation
                )
                if agent.count > 0:
                    avg_reward = agent.reward_sum / agent.count
                    self.tb_writer.add_scalar(
                        f"UCB/AvgReward/{agent_name}",
                        avg_reward,
                        self.current_generation,
                    )

            # Log average cluster/rec metrics for the generation
            if metrics_count > 0:  # Only log if we had metrics to average
                self.tb_writer.add_scalar(
                    "GenerationMetrics/AvgSilhouette",
                    avg_silhouette,
                    self.current_generation,
                )
                self.tb_writer.add_scalar(
                    "GenerationMetrics/AvgNumClusters",
                    avg_num_clusters,
                    self.current_generation,
                )
                self.tb_writer.add_scalar(
                    "GenerationMetrics/AvgNDCG", avg_ndcg_at_k, self.current_generation
                )

        stats = GenerationStats(
            generation=self.current_generation,
            total_features=len(candidates),
            successful_features=len(successful),
            failed_features=failed,
            repaired_features=repaired,
            avg_score=avg_score,
            best_score=best_score_this_gen,
            action_taken=action,
            population_size=len(self.population),
        )
        self.generation_history.append(stats)
        return stats

    async def _send_evolution_update(self) -> None:
        """Send real-time evolution update."""
        if self.websocket_callback:
            update_data = {
                "generation": self.current_generation,
                "population_size": len(self.population),
                "best_score": self.best_score,
                "best_feature": self.best_candidate.feature.name
                if self.best_candidate
                else None,
            }
            await self.websocket_callback(update_data)

    async def _generate_evolution_results(
        self, data_context: DataContext, execution_time: float
    ) -> Dict[str, Any]:
        """Generate final evolution results."""
        # Safe calculation of average success rate to avoid division by zero
        avg_success_rate = 0.0
        if self.generation_history:
            success_rates = []
            for stats in self.generation_history:
                if stats.total_features > 0:
                    success_rates.append(
                        stats.successful_features / stats.total_features
                    )

            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)

        final_results_payload = {
            "best_score": self.best_score,
            "total_generations": self.current_generation,
            "execution_time_seconds": execution_time,
            "fold_id": data_context.fold_id,  # Ensure data_context is available here
            "best_features_count": len(self.best_candidate.feature)
            if self.best_candidate and hasattr(self.best_candidate.feature, "__len__")
            else (1 if self.best_candidate else 0),
            "best_feature_details": self.best_candidate.feature.dict()
            if self.best_candidate
            else None,
            "population_size_final": len(self.population),
            "total_features_considered_in_run": sum(
                stats.total_features for stats in self.generation_history
            ),
            "avg_candidate_success_rate": avg_success_rate,
            "visualization_data": self._get_evolution_visualization_data(),
            "llm_interactions": [log.dict() for log in self.llm_interactions],
        }
        self.logger.info("Progressive Evolution Run Summary", **final_results_payload)

        # Prepare output for VulcanOrchestrator (current structure seems fine)
        return {
            "best_score": self.best_score,
            "total_generations": self.current_generation,
            "execution_time": execution_time,
            "fold_id": data_context.fold_id,
            "best_features": [self.best_candidate.feature.dict()]
            if self.best_candidate
            else [],
            "population_size": len(self.population),
            "total_features_generated": sum(
                stats.total_features for stats in self.generation_history
            ),
            "avg_success_rate": avg_success_rate,
            "visualization_data": self._get_evolution_visualization_data(),
            "llm_interactions": [log.dict() for log in self.llm_interactions],
        }

    def _get_evolution_visualization_data(self) -> Dict[str, Any]:
        """Get evolution data for visualization."""
        agent_stats = {
            name: {"count": agent.count, "reward_sum": agent.reward_sum}
            for name, agent in self.agents.items()
        }

        return {
            "population": [
                {
                    "id": candidate.feature.name,
                    "score": candidate.score,
                    "generation": candidate.generation,
                    "parent_id": candidate.parent_id,
                    "mutation_type": candidate.mutation_type,
                    "status": candidate.status,
                    "repair_attempts": candidate.repair_attempts,
                }
                for candidate in self.population
            ],
            "generation_history": [
                {
                    "generation": stats.generation,
                    "total_features": stats.total_features,
                    "successful_features": stats.successful_features,
                    "avg_score": stats.avg_score,
                    "best_score": stats.best_score,
                    "action_taken": stats.action_taken,
                    "population_size": stats.population_size,
                }
                for stats in self.generation_history
            ],
            "agent_stats": agent_stats,
            "best_candidate": {
                "feature_name": self.best_candidate.feature.name,
                "score": self.best_candidate.score,
                "generation": self.best_candidate.generation,
            }
            if self.best_candidate
            else None,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.feature_executor:
            await self.feature_executor.cleanup()
        # if self.feature_evaluator: # RecommendationEvaluator might not have a cleanup method
        #     await self.feature_evaluator.cleanup()

        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed.")

        self.logger.info("Progressive Evolution Orchestrator cleaned up")

    def _log_feature_to_disk(self, feature: FeatureDefinition, metrics: dict):
        """Save feature metadata and metrics to a JSON file."""
        log_data = {
            "name": feature.name,
            "description": feature.description,
            "type": feature.feature_type.value,
            "reasoning_steps": feature.llm_chain_of_thought_reasoning,
            "code": feature.code if feature.feature_type == "code_based" else None,
            "prompt": feature.llm_prompt
            if feature.feature_type == "llm_based"
            else None,
            "metrics": metrics,
            "accepted": metrics.get("reward", 0)
            > self.config.experiment.get("acceptance_threshold", 0.5),
        }
        feature_dir = self.run_dir / "features"
        feature_dir.mkdir(exist_ok=True)
        file_path = feature_dir / f"{feature.name}.json"
        with open(file_path, "w") as f:
            json.dump(log_data, f, indent=2)
        self.logger.info(f"Logged feature to {file_path}")
