"""Progressive Feature Evolution Orchestrator for VULCAN."""

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

# Attempt to import for plotting, but make it optional
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from vulcan.agents import FeatureAgent
from vulcan.evaluation.recommendation_evaluator import RecommendationEvaluator
from vulcan.features import FeatureExecutor
from vulcan.types import (
    ActionContext,
    DataContext,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureSet,
    FeatureType,
    MCTSAction,
    VulcanConfig,
)
from vulcan.utils import ExperimentTracker, PerformanceTracker, get_vulcan_logger

logger = structlog.get_logger(__name__)


class EvolutionAction(Enum):
    """Actions available to the RL agent."""

    GENERATE_NEW = "generate_new"
    MUTATE_EXISTING = "mutate_existing"


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
        performance_tracker: Optional[PerformanceTracker] = None,
        websocket_callback: Optional[callable] = None,
        tensorboard_writer: Optional[Any] = None,  # Added for TensorBoard
    ) -> None:
        """Initialize the progressive evolution orchestrator."""
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.websocket_callback = websocket_callback
        self.tensorboard_writer = tensorboard_writer  # Store TensorBoard writer

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
        # the config.mcts section. While "mcts" might be a legacy naming convention,
        # these parameters are still fundamental to the progressive evolution algorithm.
        self.population_size = getattr(config.mcts, "population_size", 50)
        self.generation_size = getattr(config.mcts, "generation_size", 20)
        self.max_repair_attempts = getattr(config.mcts, "max_repair_attempts", 3)
        self.mutation_rate = getattr(config.mcts, "mutation_rate", 0.3)

        # Population management
        self.population: List[FeatureCandidate] = []
        self.generation_history: List[GenerationStats] = []
        self.feature_registry: Dict[str, FeatureCandidate] = {}  # Prevent duplicates

        # RL State
        self.action_rewards: Dict[EvolutionAction, List[float]] = defaultdict(list)
        self.current_generation = 0
        self.total_features_generated = 0

        # Tracking
        self.best_score = 0.0
        self.best_candidate: Optional[FeatureCandidate] = None

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
                and not self.tensorboard_writer
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

    def _choose_action(self) -> EvolutionAction:
        """Choose action using an epsilon-greedy RL policy.

        The policy balances exploration and exploitation:
        - Exploration: With a probability epsilon, a random action (generate new or mutate existing)
          is chosen. Epsilon decays over generations, reducing random exploration as the
          evolution progresses.
        - Exploitation: With probability 1-epsilon, the action with the highest average historical
          reward is chosen.
        If mutation is chosen but the population is empty, it defaults to generating a new feature.
        """
        # Epsilon-greedy exploration
        epsilon = max(0.1, 1.0 - (self.current_generation / 50))  # Decay exploration

        if (
            random.random() < epsilon
            or len(self.action_rewards[EvolutionAction.GENERATE_NEW]) == 0
        ):
            # Explore: random action
            action = random.choice(list(EvolutionAction))
        else:
            # Exploit: choose action with highest average reward
            action_scores = {}
            for action_type in EvolutionAction:
                rewards = self.action_rewards[action_type]
                action_scores[action_type] = (
                    sum(rewards) / len(rewards) if rewards else 0.0
                )

            action = max(action_scores, key=action_scores.get)

        # Can't mutate if population is empty
        if action == EvolutionAction.MUTATE_EXISTING and len(self.population) == 0:
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
                    available_actions=[MCTSAction.ADD],  # Use proper MCTSAction enum
                    max_features=count,
                    max_cost=100.0,
                )

                # Generate feature
                agent_context = {
                    "data_context": data_context,
                    "action_context": action_context,
                    "generation": self.current_generation,
                    "diversity_pressure": True,
                }

                self.logger.info(
                    "Sending prompt to FeatureAgent for new feature",
                    generation=self.current_generation,
                    candidate_index=i,
                    action=MCTSAction.ADD.value,  # Assuming ADD for new features
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
                    action_taken=MCTSAction.MUTATE,
                )

                action_context = ActionContext(
                    current_features=parent_feature_set,
                    performance_history=self._get_recent_evaluations(),
                    available_actions=[MCTSAction.MUTATE],  # Use proper MCTSAction enum
                    max_features=1,
                    max_cost=50.0,
                )

                agent_context = {
                    "data_context": data_context,
                    "action_context": action_context,
                    "parent_feature": parent.feature,
                    "generation": self.current_generation,
                    "mutation_target": parent.feature.name,
                }

                self.logger.info(
                    "Sending prompt to FeatureAgent for mutating feature",
                    generation=self.current_generation,
                    candidate_index=i,
                    action=MCTSAction.MUTATE.value,
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

                    evaluation = await self.feature_evaluator.evaluate_feature_set(
                        feature_set=feature_set,
                        feature_results=feature_results,  # Pass actual feature execution results
                        data_context=data_context,
                        iteration=self.current_generation,
                    )

                    candidate.score = evaluation.overall_score
                    candidate.execution_successful = True
                    evaluated.append(candidate)

                    log_payload = {
                        "event": "feature_evaluation_success",
                        "generation": self.current_generation,
                        "feature_name": feature_name,
                        "score": candidate.score,
                        "metrics": evaluation.metrics.dict(),  # Assuming metrics is a Pydantic model
                    }
                    self.logger.info("Feature evaluated successfully", **log_payload)
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar(
                            f"FeatureScores/{feature_name.replace('_', '/')}",
                            candidate.score,
                            self.current_generation,
                        )
                        # Log individual metrics if desired
                        for (
                            metric_name,
                            metric_value,
                        ) in evaluation.metrics.dict().items():
                            if isinstance(metric_value, (int, float)):
                                self.tensorboard_writer.add_scalar(
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
            action_taken=MCTSAction.ADD,  # Default action
        )

    def _get_recent_evaluations(self) -> List[FeatureEvaluation]:
        """Get recent feature evaluations."""
        return []  # Simplified for now

    def _record_generation_stats(
        self, action: EvolutionAction, candidates: List[FeatureCandidate]
    ) -> GenerationStats:
        """Record statistics for the current generation."""
        successful = [
            c for c in candidates if c.execution_successful and hasattr(c, "score")
        ]
        failed = len(candidates) - len(successful)
        repaired = sum(c.repair_attempts > 0 for c in candidates)

        avg_score = (
            sum(c.score for c in successful) / len(successful) if successful else 0.0
        )
        best_score_this_gen = max(c.score for c in successful) if successful else 0.0

        stats = GenerationStats(
            generation=self.current_generation,
            total_features=len(candidates),
            successful_features=len(successful),
            failed_features=failed,
            repaired_features=repaired,
            avg_score=avg_score,
            best_score=best_score_this_gen,  # Best score in *this* generation
            action_taken=action,
            population_size=len(self.population),
        )

        self.generation_history.append(stats)

        # Enhanced logging for progress
        progress_log_data = {
            "event": "generation_progress",
            "generation": self.current_generation,
            "action_taken": action.value,
            "candidates_generated": len(candidates),
            "successful_evaluations": len(successful),
            "failed_evaluations": failed,
            "features_repaired": repaired,
            "avg_score_this_gen": avg_score,
            "best_score_this_gen": best_score_this_gen,
            "current_population_size": len(self.population),
            "overall_best_score": self.best_score,  # Best score found so far across all gens
            # Consider adding time elapsed for this generation if easily trackable
        }
        self.logger.info("Generation completed", **progress_log_data)

        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(
                "Generation/AvgScore", avg_score, self.current_generation
            )
            self.tensorboard_writer.add_scalar(
                "Generation/BestScoreThisGen",
                best_score_this_gen,
                self.current_generation,
            )
            self.tensorboard_writer.add_scalar(
                "Generation/SuccessfulFeatures",
                len(successful),
                self.current_generation,
            )
            self.tensorboard_writer.add_scalar(
                "Generation/FailedFeatures", failed, self.current_generation
            )
            self.tensorboard_writer.add_scalar(
                "Generation/RepairedFeatures", repaired, self.current_generation
            )
            self.tensorboard_writer.add_scalar(
                "Generation/PopulationSize",
                len(self.population),
                self.current_generation,
            )
            self.tensorboard_writer.add_scalar(
                "OverallBestScore", self.best_score, self.current_generation
            )
            # Log action distribution if needed (e.g., using text or histogram)
            # self.tensorboard_writer.add_text("Generation/ActionTaken", action.value, self.current_generation)

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
        """Cleanup resources."""
        await self.feature_agent.cleanup()
        await self.feature_executor.cleanup()
        if hasattr(self.feature_evaluator, "cleanup"):
            await self.feature_evaluator.cleanup()
        self.logger.info("Progressive Evolution Orchestrator cleaned up")
