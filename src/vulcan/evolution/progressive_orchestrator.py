"""Progressive Feature Evolution Orchestrator for VULCAN."""

import json
import logging
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from torch.utils.tensorboard import SummaryWriter

# Attempt to import for plotting, but make it optional
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from vulcan.agents import FeatureAgent
from vulcan.evaluation.recommendation_evaluator import RecommendationEvaluator
from vulcan.features import FeatureExecutor
from vulcan.schemas import (
    ActionContext,
    DataContext,
    EvolutionAction,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureSet,
    FeatureType,
    VulcanConfig,
)
from vulcan.utils import ExperimentTracker, PerformanceTracker, get_vulcan_logger

logger = structlog.get_logger(__name__)


@dataclass
class FeatureCandidate:
    """A feature candidate in the population."""

    feature: FeatureDefinition
    score: float
    generation: int
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    execution_successful: bool = True
    error_message: Optional[str] = None
    repair_attempts: int = 0
    evaluation_result: Optional[FeatureEvaluation] = None


@dataclass
class GenerationStats:
    """Statistics for a generation."""

    generation: int
    total_features: int
    successful_features: int
    failed_features: int
    repaired_features: int
    avg_score: float
    best_score: float
    action_taken: EvolutionAction
    population_size: int


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
        self.feature_agent = FeatureAgent(config)
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

        # RL State
        self.action_rewards: Dict[EvolutionAction, List[float]] = defaultdict(list)
        self.current_generation = 0
        self.total_features_generated = 0

        # UCB statistics for two arms: 0 = GENERATE_NEW, 1 = MUTATE_EXISTING
        self.ucb_counts = [0, 0]  # number of times each arm was played
        self.ucb_values = [0.0, 0.0]  # cumulative reward for each arm
        self.total_ucb_pulls = 0  # total pulls so far
        self.ucb_exploration_const = self.config.experiment.ucb_exploration_constant

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

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            await self.feature_agent.initialize()
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

                # Choose action using RL policy
                action = self._choose_action()

                # Generate new candidates based on action
                candidates = await self._generate_candidates(action, data_context)

                # Evaluate candidates (with automatic repair)
                evaluated_candidates = await self._evaluate_candidates(
                    candidates, data_context
                )

                # Update population
                self._update_population(evaluated_candidates)

                # Record generation statistics
                gen_stats = self._record_generation_stats(action, evaluated_candidates)

                # Update RL rewards
                self._update_action_rewards(action, gen_stats)

                # Update UCB statistics
                # Map EvolutionAction back to arm index (0 for GENERATE_NEW, 1 for MUTATE_EXISTING)
                chosen_arm_for_ucb_update = (
                    0 if action == EvolutionAction.GENERATE_NEW else 1
                )
                # Use the best score from the generation as the reward for the chosen arm
                # If no successful features, best_score might be low or an initial value, which is fine for UCB.
                reward_for_ucb = gen_stats.best_score

                self.ucb_counts[chosen_arm_for_ucb_update] += 1
                self.total_ucb_pulls += 1
                self.ucb_values[chosen_arm_for_ucb_update] += reward_for_ucb
                self.logger.info(
                    f"UCB stats updated for arm {chosen_arm_for_ucb_update} ({action.value}): "
                    f"count={self.ucb_counts[chosen_arm_for_ucb_update]}, "
                    f"value_sum={self.ucb_values[chosen_arm_for_ucb_update]:.4f}, "
                    f"reward_this_gen={reward_for_ucb:.4f}, total_pulls={self.total_ucb_pulls}"
                )

                # Send real-time updates
                await self._send_evolution_update()

                # Log progress
                if generation % 5 == 0:
                    self.logger.info(
                        "Evolution progress",
                        generation=generation + 1,
                        best_score=self.best_score,
                        population_size=len(self.population),
                        action_taken=action.value,
                        avg_score=gen_stats.avg_score,
                    )

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

    def _choose_action_ucb(self) -> int:
        """
        Return 0 for GenerateNew, 1 for MutateExisting, based on UCB1 formula:
        UCB(a) = avg_reward(a) + c * sqrt(ln N / n_a)
        Returns an integer: 0 for GENERATE_NEW, 1 for MUTATE_EXISTING.
        """
        # If any arm has never been tried, try it first:
        for arm in (0, 1):
            if self.ucb_counts[arm] == 0:
                self.logger.info(f"UCB: Choosing arm {arm} (untried)")
                return arm

        total = self.total_ucb_pulls
        ucb_scores = []
        for arm in (0, 1):
            avg_reward = self.ucb_values[arm] / self.ucb_counts[arm]
            bonus = self.ucb_exploration_const * math.sqrt(
                math.log(total) / self.ucb_counts[arm]
            )
            ucb_scores.append(avg_reward + bonus)
            self.logger.debug(
                f"UCB Arm {arm}: avg_reward={avg_reward:.4f}, bonus={bonus:.4f}, score={ucb_scores[arm]:.4f}"
            )

        # Choose the arm with the higher UCB score
        chosen_arm = 0 if ucb_scores[0] >= ucb_scores[1] else 1
        self.logger.info(
            f"UCB: Scores GenNew={ucb_scores[0]:.4f}, MutEx={ucb_scores[1]:.4f}. Chosen arm: {chosen_arm}"
        )
        return chosen_arm

    def _choose_action(self) -> EvolutionAction:
        """Choose action using Upper Confidence Bound (UCB1) policy."""

        chosen_arm_index = self._choose_action_ucb()  # Returns 0 or 1

        if chosen_arm_index == 0:
            action = EvolutionAction.GENERATE_NEW
        else:  # chosen_arm_index == 1
            action = EvolutionAction.MUTATE_EXISTING

        self.logger.info(
            f"UCB translated arm {chosen_arm_index} to action: {action.value}"
        )

        # Can't mutate if population is empty
        if action == EvolutionAction.MUTATE_EXISTING and len(self.population) == 0:
            self.logger.warning(
                "UCB chose MUTATE_EXISTING, but population is empty. Defaulting to GENERATE_NEW."
            )
            action = EvolutionAction.GENERATE_NEW

        return action

    async def _initialize_population(self, data_context: DataContext) -> None:
        """Initialize the population with seed features."""
        self.logger.info("Initializing seed population", size=self.generation_size)

        # Generate initial population
        candidates = await self._generate_new_features(
            self.generation_size, data_context
        )
        evaluated_candidates = await self._evaluate_candidates(candidates, data_context)

        self.population = evaluated_candidates[: self.population_size]
        self._update_best_candidate()

    async def _generate_candidates(
        self, action: EvolutionAction, data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Generate new feature candidates based on action."""
        if action == EvolutionAction.GENERATE_NEW:
            return await self._generate_new_features(self.generation_size, data_context)
        elif action == EvolutionAction.MUTATE_EXISTING:
            return await self._mutate_existing_features(
                self.generation_size, data_context
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _generate_new_features(
        self, count: int, data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Generate completely new features."""
        candidates = []
        self.logger.info(
            "Attempting to generate new features",
            count=count,
            generation=self.current_generation,
        )

        for i in range(count):
            feature_name_candidate = f"new_feature_gen{self.current_generation}_cand{i}"  # Tentative name for logging
            try:
                # Build action context with proper FeatureSet
                action_context = ActionContext(
                    current_features=self._get_population_features(),
                    performance_history=self._get_recent_evaluations(),
                    available_actions=[EvolutionAction.GENERATE_NEW],
                    max_features=count,
                    max_cost=100.0,
                )

                # Generate feature
                agent_context = dict(
                    data_context=data_context,
                    action_context=action_context,
                    action_to_perform=EvolutionAction.GENERATE_NEW,
                    generation=self.current_generation,
                    candidate_index=i,
                )

                self.logger.info(
                    "Sending prompt to FeatureAgent for new feature",
                    generation=self.current_generation,
                    candidate_index=i,
                    action=EvolutionAction.GENERATE_NEW.value,
                )
                result = await self.feature_agent.execute(agent_context)

                if result.get("success"):
                    feature = result["feature"]
                    self.logger.info(
                        "FeatureAgent proposed new feature",
                        generation=self.current_generation,
                        proposed_feature_name=feature.name,
                        # Assuming FeatureAgent callbacks save detailed prompt/response, ref it generically
                        details_loc="FeatureAgent LLM outputs/callbacks",
                    )
                    # Check for duplicates
                    feature_key = self._get_feature_key(feature)
                    if feature_key not in self.feature_registry:
                        candidate = FeatureCandidate(
                            feature=feature,
                            score=0.0,
                            generation=self.current_generation,
                        )
                        candidates.append(candidate)
                        self.feature_registry[feature_key] = candidate

            except Exception as e:
                self.logger.warning(
                    "Failed to generate new feature candidate",
                    error=str(e),
                    generation=self.current_generation,
                    candidate_index=i,
                    exc_info=True,
                )
        self.logger.info(
            "Finished generating new feature candidates",
            generated_count=len(candidates),
            requested_count=count,
        )
        return candidates

    async def _mutate_existing_features(
        self, count: int, data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Mutate existing features from population."""
        candidates = []
        self.logger.info(
            "Attempting to mutate existing features",
            count=count,
            generation=self.current_generation,
            population_size=len(self.population),
        )

        if not self.population:
            self.logger.warning(
                "Mutation requested but population is empty. Falling back to generating new features."
            )
            return await self._generate_new_features(count, data_context)

        for i in range(count):
            try:
                # Select parent feature (bias towards better performers)
                parent = self._select_parent_feature()

                # Build mutation context with proper FeatureSet
                parent_feature_set = FeatureSet(
                    features=[parent.feature],
                    action_taken=EvolutionAction.MUTATE_EXISTING,
                )

                action_context = ActionContext(
                    current_features=parent_feature_set,
                    performance_history=self._get_recent_evaluations(),
                    available_actions=[EvolutionAction.MUTATE_EXISTING],
                    max_features=1,
                    max_cost=50.0,
                )
                agent_context = dict(
                    data_context=data_context,
                    action_context=action_context,
                    action_to_perform=EvolutionAction.MUTATE_EXISTING,
                    generation=self.current_generation,
                    candidate_index=i,
                    target_feature_name=parent.feature.name,
                )

                self.logger.info(
                    "Sending prompt to FeatureAgent for mutating feature",
                    generation=self.current_generation,
                    candidate_index=i,
                    action=EvolutionAction.MUTATE_EXISTING.value,
                    target_feature_name=parent.feature.name,
                )
                result = await self.feature_agent.execute(agent_context)

                if result.get("success"):
                    mutated_feature = result["feature"]
                    self.logger.info(
                        "FeatureAgent proposed mutated feature",
                        generation=self.current_generation,
                        proposed_feature_name=mutated_feature.name,
                        original_feature_name=parent.feature.name,
                        details_loc="FeatureAgent LLM outputs/callbacks",
                    )
                    # Check for duplicates
                    feature_key = self._get_feature_key(mutated_feature)
                    if feature_key not in self.feature_registry:
                        candidate = FeatureCandidate(
                            feature=mutated_feature,
                            score=0.0,
                            generation=self.current_generation,
                            parent_id=parent.feature.name,
                            mutation_type=result.get("mutation_type", "unknown"),
                        )
                        candidates.append(candidate)
                        self.feature_registry[feature_key] = candidate

            except Exception as e:
                self.logger.warning(
                    "Failed to mutate feature candidate",
                    error=str(e),
                    generation=self.current_generation,
                    candidate_index=i,
                    target_feature_name=parent.feature.name
                    if "parent" in locals()
                    else "unknown",
                    exc_info=True,
                )
        self.logger.info(
            "Finished mutating feature candidates",
            generated_count=len(candidates),
            requested_count=count,
        )
        return candidates

    async def _evaluate_candidates(
        self, candidates: List[FeatureCandidate], data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Evaluate candidates with automatic code repair."""
        evaluated_candidates_log = []
        if not candidates:
            self.logger.info(
                "No candidates to evaluate for current generation.",
                generation=self.current_generation,
            )
            return []

        self.logger.info(
            f"Starting evaluation of {len(candidates)} candidates for generation {self.current_generation}"
        )
        evaluated = []

        for idx, candidate in enumerate(candidates):
            feature_name = candidate.feature.name
            self.logger.debug(
                f"Evaluating candidate {idx + 1}/{len(candidates)}: {feature_name}"
            )
            try:
                # Execute feature with automatic repair
                success, error, feature_results = await self._execute_with_repair(
                    candidate, data_context
                )

                if success:
                    # Evaluate feature performance
                    feature_set = FeatureSet(features=[candidate.feature])

                    eval_result = await self.feature_evaluator.evaluate_feature_set(
                        feature_set=feature_set,
                        feature_results=feature_results,  # Pass actual feature execution results
                        data_context=data_context,
                        iteration=self.current_generation,
                    )

                    candidate.score = eval_result.score
                    candidate.execution_successful = True
                    candidate.evaluation_result = (
                        eval_result  # Store the full evaluation
                    )

                    # Log and store artifact if W&B enabled (or always if TensorBoard is primary)
                    if self.experiment_tracker:
                        self.experiment_tracker.log_feature_evaluation(eval_result)

                    log_payload = {
                        "event_type": "feature_evaluation_success",
                        "generation": self.current_generation,
                        "feature_name": feature_name,
                        "score": candidate.score,
                        "metrics": eval_result.metrics.dict(),  # Assuming metrics is a Pydantic model
                    }
                    self.logger.info("Feature evaluated successfully", **log_payload)
                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                            f"FeatureScores/{feature_name.replace('_', '/')}",
                            candidate.score,
                            self.current_generation,
                        )
                        # Log individual metrics if desired
                        for (
                            metric_name,
                            metric_value,
                        ) in eval_result.metrics.dict().items():
                            if isinstance(metric_value, (int, float)):
                                self.tb_writer.add_scalar(
                                    f"FeatureMetrics/{feature_name.replace('_', '/')}/{metric_name}",
                                    metric_value,
                                    self.current_generation,
                                )

                    # Track evaluation (already present)
                    # self.performance_tracker.record_evaluation(evaluation)
                    evaluated_candidates_log.append(
                        {
                            "name": feature_name,
                            "score": candidate.score,
                            "status": "success",
                        }
                    )

                else:
                    candidate.execution_successful = False
                    candidate.error_message = error
                    # Still add to evaluated list for tracking
                    self.logger.warning(
                        "Feature execution failed after repair attempts",
                        feature_name=feature_name,
                        error=error,
                        generation=self.current_generation,
                    )
                    evaluated_candidates_log.append(
                        {
                            "name": feature_name,
                            "error": error,
                            "status": "failed_execution",
                        }
                    )

            except Exception as e:
                candidate.execution_successful = False
                candidate.error_message = str(e)
                self.logger.warning(
                    "Feature evaluation process failed unexpectedly",
                    feature_name=feature_name,
                    error=str(e),
                    generation=self.current_generation,
                    exc_info=True,
                )
                evaluated_candidates_log.append(
                    {
                        "name": feature_name,
                        "error": str(e),
                        "status": "failed_evaluation_exception",
                    }
                )

        self.logger.info(
            "Finished evaluation of candidates for generation",
            generation=self.current_generation,
            evaluation_summary=evaluated_candidates_log,
        )
        return evaluated

    async def _execute_with_repair(
        self, candidate: FeatureCandidate, data_context: DataContext
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Execute feature with automatic code repair on failure."""
        for attempt in range(self.max_repair_attempts + 1):
            try:
                # Try to execute the feature
                results = await self.feature_executor.execute_feature_set(
                    features=[candidate.feature],
                    data_context=data_context,
                    target_split="validation",
                )

                return True, None, results  # Success! Return the results

            except Exception as e:
                error_msg = str(e)

                if attempt < self.max_repair_attempts:
                    # Attempt automatic repair
                    self.logger.info(
                        "Attempting feature repair",
                        feature_name=candidate.feature.name,  # Corrected: use feature_name consistently
                        attempt=attempt + 1,
                        error=error_msg,
                        generation=self.current_generation,
                    )

                    repaired_feature = await self._repair_feature(
                        candidate.feature, error_msg, data_context
                    )

                    if repaired_feature:
                        candidate.feature = repaired_feature
                        candidate.repair_attempts += 1
                        self.logger.info(
                            "Feature repaired successfully",
                            original_feature_name=candidate.feature.name,  # Corrected
                            repaired_feature_name=repaired_feature.name,  # Corrected
                            generation=self.current_generation,
                        )
                        return True, None, results

                return False, error_msg, None

    async def _repair_feature(
        self, feature: FeatureDefinition, error_msg: str, data_context: DataContext
    ) -> Optional[FeatureDefinition]:
        """Attempt to repair a broken feature."""
        try:
            repair_context = {
                "feature": feature,
                "error_message": error_msg,
                "data_context": data_context,
                "repair_instructions": "Fix the syntax/logic error in this feature",
            }

            result = await self.feature_agent.execute(repair_context)

            if result.get("success"):
                repaired_feature = result["feature"]
                self.logger.info(
                    "Feature repaired successfully",
                    original_feature_name=feature.name,
                    repaired_feature_name=repaired_feature.name,
                    generation=self.current_generation,
                )
                return repaired_feature

        except Exception as e:
            self.logger.warning("Feature repair failed", error=str(e))

        return None

    def _update_population(self, new_candidates: List[FeatureCandidate]) -> None:
        """Update population with new candidates, keeping only the best."""
        # Add successful candidates to population
        successful_candidates = [c for c in new_candidates if c.execution_successful]
        self.population.extend(successful_candidates)

        # Sort by score (descending) and keep top performers
        self.population.sort(key=lambda x: x.score, reverse=True)
        self.population = self.population[: self.population_size]

        self._update_best_candidate()

        # Save the updated population to disk
        if self.results_manager:
            population_for_saving = [
                {
                    "id": cand.feature.name,
                    "score": cand.score,
                    "generation": cand.generation,
                    "parent_id": cand.parent_id,
                    "mutation_type": cand.mutation_type,
                    "execution_successful": cand.execution_successful,
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
        self, action: EvolutionAction, candidates: List[FeatureCandidate]
    ) -> GenerationStats:
        """Record statistics for the current generation."""
        successful = [
            c for c in candidates if c.execution_successful and c.score is not None
        ]
        failed = len(candidates) - len(successful)
        repaired = sum(
            1 for c in candidates if c.repair_attempts > 0 and c.execution_successful
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
            "action_taken": action.value,
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
            # Log UCB stats
            self.tb_writer.add_scalar(
                "UCB/TotalPulls", self.total_ucb_pulls, self.current_generation
            )
            self.tb_writer.add_scalar(
                "UCB/ExplorationConstant",
                self.ucb_exploration_const,
                self.current_generation,
            )
            for arm_idx, arm_action in enumerate(
                [EvolutionAction.GENERATE_NEW, EvolutionAction.MUTATE_EXISTING]
            ):
                arm_name = arm_action.value
                self.tb_writer.add_scalar(
                    f"UCB/Arm_{arm_name}_Pulls",
                    self.ucb_counts[arm_idx],
                    self.current_generation,
                )
                if self.ucb_counts[arm_idx] > 0:
                    avg_reward_arm = self.ucb_values[arm_idx] / self.ucb_counts[arm_idx]
                    self.tb_writer.add_scalar(
                        f"UCB/Arm_{arm_name}_AvgReward",
                        avg_reward_arm,
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

    def _update_action_rewards(
        self, action: EvolutionAction, stats: GenerationStats
    ) -> None:
        """Update RL rewards for the taken action."""
        # Reward based on success rate and score improvement
        success_rate = (
            stats.successful_features / stats.total_features
            if stats.total_features > 0
            else 0.0
        )
        score_improvement = stats.best_score - (
            self.generation_history[-2].best_score
            if len(self.generation_history) > 1
            else 0.0
        )

        reward = success_rate * 0.5 + score_improvement * 0.5
        self.action_rewards[action].append(reward)

        # Keep only recent rewards
        if len(self.action_rewards[action]) > 20:
            self.action_rewards[action] = self.action_rewards[action][-20:]

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
            # Add path to log file and other artifacts if known here
            # This might be better logged by VulcanOrchestrator which knows the full experiment path
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
        }

    async def get_evolution_visualization_data(self) -> Dict[str, Any]:
        """Get evolution data for visualization."""
        # Ensure action_rewards always has the expected structure for frontend
        action_rewards_data = {
            "generate_new": self.action_rewards.get(EvolutionAction.GENERATE_NEW, []),
            "mutate_existing": self.action_rewards.get(
                EvolutionAction.MUTATE_EXISTING, []
            ),
        }

        return {
            "population": [
                {
                    "id": candidate.feature.name,
                    "score": candidate.score,
                    "generation": candidate.generation,
                    "parent_id": candidate.parent_id,
                    "mutation_type": candidate.mutation_type,
                    "execution_successful": candidate.execution_successful,
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
                    "action_taken": stats.action_taken.value,
                    "population_size": stats.population_size,
                }
                for stats in self.generation_history
            ],
            "action_rewards": action_rewards_data,
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
        if self.feature_agent:
            await self.feature_agent.cleanup()
        if self.feature_executor:
            await self.feature_executor.cleanup()
        # if self.feature_evaluator: # RecommendationEvaluator might not have a cleanup method
        #     await self.feature_evaluator.cleanup()

        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed.")

        self.logger.info("Progressive Evolution Orchestrator cleaned up")
