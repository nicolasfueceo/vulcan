"""Progressive Feature Evolution Orchestrator for VULCAN."""

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from vulcan.agents import FeatureAgent
from vulcan.evaluation import FeatureEvaluator
from vulcan.features import FeatureExecutor
from vulcan.types import (
    ActionContext,
    DataContext,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureSet,
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
    ) -> None:
        """Initialize the progressive evolution orchestrator."""
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.websocket_callback = websocket_callback

        # Core components
        self.feature_agent = FeatureAgent(config)
        self.feature_executor = FeatureExecutor(config)
        self.feature_evaluator = FeatureEvaluator(config)
        self.experiment_tracker: Optional[ExperimentTracker] = None

        # Performance tracking
        self.performance_tracker = performance_tracker or PerformanceTracker(
            max_history=1000
        )

        # Evolution parameters
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
        self.iteration_count = 0
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
        max_generations = max_generations or self.config.mcts.max_iterations

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
            return results

        except Exception as e:
            self.logger.error("Evolution failed", error=str(e))
            raise

    def _choose_action(self) -> EvolutionAction:
        """Choose action using epsilon-greedy RL policy."""
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

        for i in range(count):
            try:
                # Build action context
                action_context = ActionContext(
                    current_features=self._get_population_features(),
                    performance_history=self._get_recent_evaluations(),
                    available_actions=["generate"],
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

                result = await self.feature_agent.execute(agent_context)

                if result.get("success"):
                    feature = result["feature"]

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
                self.logger.warning("Failed to generate new feature", error=str(e))

        return candidates

    async def _mutate_existing_features(
        self, count: int, data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Mutate existing features from population."""
        candidates = []

        if not self.population:
            return await self._generate_new_features(count, data_context)

        for i in range(count):
            try:
                # Select parent feature (bias towards better performers)
                parent = self._select_parent_feature()

                # Build mutation context
                action_context = ActionContext(
                    current_features=[parent.feature],
                    performance_history=self._get_recent_evaluations(),
                    available_actions=["mutate"],
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

                result = await self.feature_agent.execute(agent_context)

                if result.get("success"):
                    mutated_feature = result["feature"]

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
                self.logger.warning("Failed to mutate feature", error=str(e))

        return candidates

    async def _evaluate_candidates(
        self, candidates: List[FeatureCandidate], data_context: DataContext
    ) -> List[FeatureCandidate]:
        """Evaluate candidates with automatic code repair."""
        evaluated = []

        for candidate in candidates:
            try:
                # Execute feature with automatic repair
                success, error = await self._execute_with_repair(
                    candidate, data_context
                )

                if success:
                    # Evaluate feature performance
                    feature_set = FeatureSet(features=[candidate.feature])

                    evaluation = await self.feature_evaluator.evaluate_feature_set(
                        feature_set=feature_set,
                        feature_results={candidate.feature.name: {}},  # Simplified
                        data_context=data_context,
                        iteration=self.current_generation,
                    )

                    candidate.score = evaluation.overall_score
                    candidate.execution_successful = True
                    evaluated.append(candidate)

                    # Track evaluation
                    self.performance_tracker.record_evaluation(evaluation)

                else:
                    candidate.execution_successful = False
                    candidate.error_message = error
                    # Still add to evaluated list for tracking

            except Exception as e:
                candidate.execution_successful = False
                candidate.error_message = str(e)
                self.logger.warning(
                    "Feature evaluation failed",
                    feature=candidate.feature.name,
                    error=str(e),
                )

        return evaluated

    async def _execute_with_repair(
        self, candidate: FeatureCandidate, data_context: DataContext
    ) -> Tuple[bool, Optional[str]]:
        """Execute feature with automatic code repair on failure."""
        for attempt in range(self.max_repair_attempts + 1):
            try:
                # Try to execute the feature
                results = await self.feature_executor.execute_feature_set(
                    features=[candidate.feature],
                    data_context=data_context,
                    target_split="validation",
                )

                return True, None  # Success!

            except Exception as e:
                error_msg = str(e)

                if attempt < self.max_repair_attempts:
                    # Attempt automatic repair
                    self.logger.info(
                        "Attempting feature repair",
                        feature=candidate.feature.name,
                        attempt=attempt + 1,
                        error=error_msg,
                    )

                    repaired_feature = await self._repair_feature(
                        candidate.feature, error_msg, data_context
                    )

                    if repaired_feature:
                        candidate.feature = repaired_feature
                        candidate.repair_attempts += 1
                        continue  # Try again with repaired feature

                return False, error_msg

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
                    original=feature.name,
                    repaired=repaired_feature.name,
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
        return f"{feature.name}_{hash(feature.implementation)}"

    def _get_population_features(self) -> List[FeatureDefinition]:
        """Get features from current population."""
        return [candidate.feature for candidate in self.population]

    def _get_recent_evaluations(self) -> List[FeatureEvaluation]:
        """Get recent feature evaluations."""
        return []  # Simplified for now

    def _record_generation_stats(
        self, action: EvolutionAction, candidates: List[FeatureCandidate]
    ) -> GenerationStats:
        """Record statistics for the current generation."""
        successful = [c for c in candidates if c.execution_successful]
        failed = [c for c in candidates if not c.execution_successful]
        repaired = [c for c in candidates if c.repair_attempts > 0]

        avg_score = (
            sum(c.score for c in successful) / len(successful) if successful else 0.0
        )
        best_score = max(c.score for c in successful) if successful else 0.0

        stats = GenerationStats(
            generation=self.current_generation,
            total_features=len(candidates),
            successful_features=len(successful),
            failed_features=len(failed),
            repaired_features=len(repaired),
            avg_score=avg_score,
            best_score=best_score,
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
            "avg_success_rate": sum(
                stats.successful_features / stats.total_features
                for stats in self.generation_history
            )
            / len(self.generation_history)
            if self.generation_history
            else 0.0,
        }

    async def get_evolution_visualization_data(self) -> Dict[str, Any]:
        """Get evolution data for visualization."""
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
            "action_rewards": {
                action.value: rewards[-10:]  # Last 10 rewards
                for action, rewards in self.action_rewards.items()
            },
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
