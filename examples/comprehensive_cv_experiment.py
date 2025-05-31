#!/usr/bin/env python3
"""
Comprehensive Cross-Validation Experiment Runner for VULCAN
Runs feature engineering across all CV folds with deep MCTS exploration.
"""

import asyncio

# Add project root to path
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

sys.path.append(str(Path(__file__).parent.parent))

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Experiment configuration
OUTER_FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = [1, 2, 3]
MAX_DEPTH = 30
MAX_ITERATIONS = 100  # More iterations for deeper exploration
EXPERIMENT_NAME_PREFIX = "comprehensive_cv_deep_mcts"


class ComprehensiveExperimentRunner:
    """Runs comprehensive cross-validation experiments across all folds."""

    def __init__(self, config_path: str = "../config/vulcan.yaml"):
        """Initialize the experiment runner.

        Args:
            config_path: Path to VULCAN configuration file.
        """
        self.config_path = config_path
        self.base_config = None
        self.results: List[Dict] = []
        self.start_time = None

    async def initialize(self) -> bool:
        """Initialize the experiment runner."""
        try:
            # Load base configuration using ConfigManager
            config_manager = ConfigManager(self.config_path)
            self.base_config = config_manager.config

            # Override configuration for comprehensive experiment
            self.base_config.mcts.max_depth = MAX_DEPTH
            self.base_config.mcts.max_iterations = MAX_ITERATIONS
            self.base_config.experiment.wandb_enabled = True
            self.base_config.experiment.wandb_project = "vulcan-comprehensive-cv"

            # Enhanced LLM configuration for comprehensive experiment
            self.base_config.llm.model_name = "gpt-4o"  # Use latest GPT-4o model
            self.base_config.llm.max_tokens = 8000  # Higher token limit
            self.base_config.llm.temperature = 0.7  # Balanced creativity

            # Enhanced logging for comprehensive experiment
            self.base_config.logging.level = "INFO"
            self.base_config.logging.structured = True

            logger.info(
                "Comprehensive experiment runner initialized",
                max_depth=MAX_DEPTH,
                max_iterations=MAX_ITERATIONS,
                total_cv_combinations=len(OUTER_FOLDS) * len(INNER_FOLDS),
                wandb_project=self.base_config.experiment.wandb_project,
            )

            return True

        except Exception as e:
            logger.error("Failed to initialize experiment runner", error=str(e))
            return False

    def get_fold_combinations(self) -> List[Tuple[int, int]]:
        """Get all outer-inner fold combinations.

        Returns:
            List of (outer_fold, inner_fold) tuples.
        """
        combinations = []
        for outer_fold in OUTER_FOLDS:
            for inner_fold in INNER_FOLDS:
                combinations.append((outer_fold, inner_fold))
        return combinations

    async def run_fold_experiment(
        self,
        outer_fold: int,
        inner_fold: int,
        experiment_index: int,
        total_experiments: int,
    ) -> Dict:
        """Run experiment on a specific fold combination.

        Args:
            outer_fold: Outer CV fold number.
            inner_fold: Inner CV fold number.
            experiment_index: Current experiment index (1-based).
            total_experiments: Total number of experiments.

        Returns:
            Experiment results dictionary.
        """
        fold_id = f"outer_{outer_fold}_inner_{inner_fold}"
        experiment_name = f"{EXPERIMENT_NAME_PREFIX}_{fold_id}"

        logger.info(
            "Starting fold experiment",
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            experiment_index=experiment_index,
            total_experiments=total_experiments,
            experiment_name=experiment_name,
            max_depth=MAX_DEPTH,
            max_iterations=MAX_ITERATIONS,
        )

        start_time = time.time()

        try:
            # Create orchestrator with fold-specific configuration
            fold_config = self.base_config.model_copy(deep=True)

            # Add fold-specific tags for W&B
            fold_config.experiment.tags = [
                f"outer_fold_{outer_fold}",
                f"inner_fold_{inner_fold}",
                f"cv_combination_{experiment_index}",
                "comprehensive_experiment",
                "deep_mcts",
                f"max_depth_{MAX_DEPTH}",
            ]

            orchestrator = VulcanOrchestrator(fold_config)

            # Initialize orchestrator
            init_success = await orchestrator.initialize_components()
            if not init_success:
                raise Exception("Failed to initialize orchestrator")

            # Create data context for this fold
            data_context = await self.create_data_context(
                orchestrator=orchestrator, outer_fold=outer_fold, inner_fold=inner_fold
            )

            logger.info(
                "Data context created for fold",
                fold_id=fold_id,
                n_users=data_context.n_users,
                n_items=data_context.n_items,
                sparsity=data_context.sparsity,
            )

            # Run the experiment
            experiment_id = await orchestrator.start_experiment(
                experiment_name=experiment_name,
                config_overrides={
                    "max_depth": MAX_DEPTH,
                    "max_iterations": MAX_ITERATIONS,
                },
                data_context=data_context,
            )

            # Wait for experiment completion
            while orchestrator.get_status().is_running:
                await asyncio.sleep(1)

            # Get experiment results
            history = orchestrator.get_experiment_history()
            if not history:
                raise Exception("No experiment results found")

            result = history[-1]  # Get the latest experiment result
            execution_time = time.time() - start_time

            # Prepare result summary
            result_summary = {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fold_id": fold_id,
                "experiment_name": experiment_name,
                "experiment_index": experiment_index,
                "execution_time": execution_time,
                "best_score": result.best_score,
                "total_iterations": result.total_iterations,
                "feature_count": len(result.best_features)
                if result.best_features
                else 0,
                "success": True,
                "n_users": data_context.n_users,
                "n_items": data_context.n_items,
                "sparsity": data_context.sparsity,
                "max_depth_used": MAX_DEPTH,
                "max_iterations_used": MAX_ITERATIONS,
            }

            # Add feature details if available
            if result.best_features:
                result_summary["best_features"] = [
                    {
                        "name": feature.name,
                        "type": feature.feature_type.value,
                        "description": feature.description,
                    }
                    for feature in result.best_features
                ]

            logger.info(
                "Fold experiment completed successfully",
                fold_id=fold_id,
                experiment_index=experiment_index,
                execution_time=f"{execution_time:.2f}s",
                best_score=result.best_score,
                total_iterations=result.total_iterations,
                feature_count=result_summary["feature_count"],
            )

            # Cleanup
            await orchestrator.cleanup()

            return result_summary

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Fold experiment failed",
                fold_id=fold_id,
                experiment_index=experiment_index,
                error=str(e),
                execution_time=f"{execution_time:.2f}s",
            )

            return {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fold_id": fold_id,
                "experiment_name": experiment_name,
                "experiment_index": experiment_index,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "max_depth_used": MAX_DEPTH,
                "max_iterations_used": MAX_ITERATIONS,
            }

    async def run_comprehensive_experiment(self) -> List[Dict]:
        """Run comprehensive experiment across all CV folds.

        Returns:
            List of experiment results.
        """
        fold_combinations = self.get_fold_combinations()
        total_experiments = len(fold_combinations)

        logger.info(
            "Starting comprehensive cross-validation experiment",
            total_cv_combinations=total_experiments,
            outer_folds=OUTER_FOLDS,
            inner_folds=INNER_FOLDS,
            max_depth=MAX_DEPTH,
            max_iterations=MAX_ITERATIONS,
            estimated_duration_hours=(total_experiments * 5)
            / 60,  # Rough estimate: 5 min per experiment
        )

        start_time = time.time()
        results = []

        # Run experiments sequentially to avoid resource conflicts
        for i, (outer_fold, inner_fold) in enumerate(fold_combinations, 1):
            logger.info(
                "Progress update",
                completed_experiments=i - 1,
                total_experiments=total_experiments,
                progress_percent=round((i - 1) / total_experiments * 100, 1),
                current_fold=f"outer_{outer_fold}_inner_{inner_fold}",
            )

            result = await self.run_fold_experiment(
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                experiment_index=i,
                total_experiments=total_experiments,
            )

            results.append(result)
            self.results.append(result)

            # Log intermediate progress
            successful_experiments = sum(1 for r in results if r.get("success", False))
            if successful_experiments > 0:
                avg_score = (
                    sum(
                        r.get("best_score", 0)
                        for r in results
                        if r.get("success", False)
                    )
                    / successful_experiments
                )
                logger.info(
                    "Intermediate results summary",
                    completed_experiments=i,
                    successful_experiments=successful_experiments,
                    failed_experiments=i - successful_experiments,
                    average_score=round(avg_score, 4),
                )

        total_time = time.time() - start_time

        # Final summary
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]

        logger.info(
            "Comprehensive experiment completed",
            total_experiments=total_experiments,
            successful_experiments=len(successful_results),
            failed_experiments=len(failed_results),
            total_execution_time=f"{total_time:.2f}s",
            average_execution_time_per_fold=f"{total_time / total_experiments:.2f}s",
        )

        if successful_results:
            avg_score = sum(r["best_score"] for r in successful_results) / len(
                successful_results
            )
            best_score = max(r["best_score"] for r in successful_results)
            worst_score = min(r["best_score"] for r in successful_results)

            logger.info(
                "Performance summary across all folds",
                average_score=round(avg_score, 4),
                best_score=round(best_score, 4),
                worst_score=round(worst_score, 4),
                score_std=round(
                    (
                        sum(
                            (r["best_score"] - avg_score) ** 2
                            for r in successful_results
                        )
                        / len(successful_results)
                    )
                    ** 0.5,
                    4,
                ),
            )

        return results

    def save_results_summary(self, results: List[Dict]) -> None:
        """Save comprehensive results summary to file.

        Args:
            results: List of experiment results.
        """
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_cv_results_{timestamp}.json"

        summary = {
            "experiment_info": {
                "timestamp": timestamp,
                "max_depth": MAX_DEPTH,
                "max_iterations": MAX_ITERATIONS,
                "total_experiments": len(results),
                "successful_experiments": sum(
                    1 for r in results if r.get("success", False)
                ),
                "failed_experiments": sum(
                    1 for r in results if not r.get("success", False)
                ),
            },
            "results": results,
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Results summary saved",
            results_file=results_file,
            total_experiments=len(results),
        )

    async def create_data_context(self, orchestrator, outer_fold: int, inner_fold: int):
        """Create data context for a specific CV fold.

        Args:
            orchestrator: VULCAN orchestrator instance.
            outer_fold: Outer fold number.
            inner_fold: Inner fold number.

        Returns:
            Data context for the specified fold.
        """
        try:
            from vulcan.data.goodreads_loader import GoodreadsDataLoader

            # Use the same paths as in the orchestrator
            db_path = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db"
            splits_dir = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits"

            # Create data loader for this specific fold
            loader = GoodreadsDataLoader(
                db_path=db_path,
                splits_dir=splits_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                batch_size=1000,
            )

            # Get data context - this will use the full data for the fold
            data_context = loader.get_data_context()

            return data_context

        except Exception as e:
            logger.error(
                "Failed to create data context",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                error=str(e),
            )
            raise


async def main():
    """Main experiment execution function."""
    print("=" * 80)
    print("🎯 VULCAN Comprehensive Cross-Validation Experiment")
    print(
        f"🌳 Deep MCTS Exploration (max_depth={MAX_DEPTH}, max_iterations={MAX_ITERATIONS})"
    )
    print(
        f"📊 Running on {len(OUTER_FOLDS)} outer folds × {len(INNER_FOLDS)} inner folds = {len(OUTER_FOLDS) * len(INNER_FOLDS)} total experiments"
    )
    print("📈 Full W&B tracking enabled for dashboard visualization")
    print("=" * 80)

    # Initialize experiment runner
    runner = ComprehensiveExperimentRunner()

    if not await runner.initialize():
        print("❌ Failed to initialize experiment runner")
        return

    print("✅ Experiment runner initialized successfully")
    print("🚀 Starting comprehensive experiment...")

    # Run comprehensive experiment
    start_time = time.time()
    results = await runner.run_comprehensive_experiment()
    total_time = time.time() - start_time

    # Save results
    runner.save_results_summary(results)

    # Final report
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"⏱️  Total Time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)")
    print(f"✅ Successful: {len(successful_results)}/{len(results)} experiments")
    print(f"❌ Failed: {len(failed_results)}/{len(results)} experiments")

    if successful_results:
        avg_score = sum(r["best_score"] for r in successful_results) / len(
            successful_results
        )
        best_experiment = max(successful_results, key=lambda x: x["best_score"])
        print(f"📊 Average Score: {avg_score:.4f}")
        print(
            f"🏆 Best Score: {best_experiment['best_score']:.4f} (fold: {best_experiment['fold_id']})"
        )
        print("📈 Dashboard: https://wandb.ai/your-entity/vulcan-comprehensive-cv")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
