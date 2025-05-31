#!/usr/bin/env python3
"""
Focused Cross-Validation Experiment Runner for VULCAN
Runs feature engineering on one outer fold with deep MCTS exploration.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List

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

# Focused experiment configuration
OUTER_FOLD = 1  # Focus on one outer fold
INNER_FOLDS = [1, 2, 3]  # Test all inner folds for this outer fold
MAX_DEPTH = 30
MAX_ITERATIONS = 50  # Reduced for faster iteration
EXPERIMENT_NAME_PREFIX = "focused_cv_deep_mcts"


class FocusedExperimentRunner:
    """Runs focused cross-validation experiments on one outer fold."""

    def __init__(self, config_path: str = "../config/vulcan.yaml"):
        """Initialize the focused experiment runner.

        Args:
            config_path: Path to VULCAN configuration file.
        """
        self.config_path = config_path
        self.base_config = None
        self.results: List[Dict] = []

    async def initialize(self) -> bool:
        """Initialize the experiment runner."""
        try:
            # Load base configuration using ConfigManager
            config_manager = ConfigManager(self.config_path)
            self.base_config = config_manager.config

            # Override configuration for focused experiment
            self.base_config.mcts.max_depth = MAX_DEPTH
            self.base_config.mcts.max_iterations = MAX_ITERATIONS
            self.base_config.experiment.wandb_enabled = False

            # Enhanced LLM configuration
            self.base_config.llm.model_name = "gpt-4o"
            self.base_config.llm.max_tokens = 8000
            self.base_config.llm.temperature = 0.7

            # Enhanced logging
            self.base_config.logging.level = "INFO"
            self.base_config.logging.structured = True

            logger.info(
                "Focused experiment runner initialized",
                outer_fold=OUTER_FOLD,
                inner_folds=INNER_FOLDS,
                max_depth=MAX_DEPTH,
                max_iterations=MAX_ITERATIONS,
                total_experiments=len(INNER_FOLDS),
            )

            return True

        except Exception as e:
            logger.error("Failed to initialize experiment runner", error=str(e))
            return False

    async def run_fold_experiment(
        self, outer_fold: int, inner_fold: int, experiment_index: int
    ) -> Dict:
        """Run experiment on a specific fold combination.

        Args:
            outer_fold: Outer CV fold number.
            inner_fold: Inner CV fold number.
            experiment_index: Current experiment index (1-based).

        Returns:
            Experiment results dictionary.
        """
        fold_id = f"outer_{outer_fold}_inner_{inner_fold}"
        experiment_name = f"{EXPERIMENT_NAME_PREFIX}_{fold_id}"

        logger.info(
            "Starting focused fold experiment",
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            experiment_index=experiment_index,
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
                f"experiment_{experiment_index}",
                "focused_experiment",
                "deep_mcts",
                f"max_depth_{MAX_DEPTH}",
                "enhanced_validation",
            ]

            orchestrator = VulcanOrchestrator(fold_config)

            # Initialize orchestrator
            init_success = await orchestrator.initialize_components()
            if not init_success:
                raise Exception("Failed to initialize orchestrator")

            # Create data context for this fold
            data_context = await self.create_data_context(
                outer_fold=outer_fold, inner_fold=inner_fold
            )

            logger.info(
                "Data context created for focused experiment",
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

            # Wait for experiment completion with timeout
            max_wait_time = 1800  # 30 minutes max per experiment
            wait_time = 0
            check_interval = 5

            while orchestrator.get_status().is_running and wait_time < max_wait_time:
                await asyncio.sleep(check_interval)
                wait_time += check_interval

                if wait_time % 60 == 0:  # Log every minute
                    logger.info(
                        "Experiment progress",
                        fold_id=fold_id,
                        elapsed_time=f"{wait_time}s",
                        status="running",
                    )

            if wait_time >= max_wait_time:
                logger.warning(
                    "Experiment timeout reached",
                    fold_id=fold_id,
                    max_wait_time=max_wait_time,
                )

            # Get experiment results
            history = orchestrator.get_experiment_history()
            if not history:
                raise Exception("No experiment results found")

            result = history[-1]
            execution_time = time.time() - start_time

            # Prepare comprehensive result summary
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
                "experiment_id": experiment_id,
                "timeout": wait_time >= max_wait_time,
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
                "Focused fold experiment completed successfully",
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
                "Focused fold experiment failed",
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

    async def run_focused_experiment(self) -> List[Dict]:
        """Run focused experiment on one outer fold across inner folds.

        Returns:
            List of experiment results.
        """
        total_experiments = len(INNER_FOLDS)

        logger.info(
            "Starting focused cross-validation experiment",
            outer_fold=OUTER_FOLD,
            inner_folds=INNER_FOLDS,
            total_experiments=total_experiments,
            max_depth=MAX_DEPTH,
            max_iterations=MAX_ITERATIONS,
            estimated_duration_minutes=total_experiments * 10,  # ~10 min per experiment
        )

        start_time = time.time()
        results = []

        # Run experiments sequentially
        for i, inner_fold in enumerate(INNER_FOLDS, 1):
            logger.info(
                "Progress update",
                completed_experiments=i - 1,
                total_experiments=total_experiments,
                progress_percent=round((i - 1) / total_experiments * 100, 1),
                current_fold=f"outer_{OUTER_FOLD}_inner_{inner_fold}",
            )

            result = await self.run_fold_experiment(
                outer_fold=OUTER_FOLD,
                inner_fold=inner_fold,
                experiment_index=i,
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
            "Focused experiment completed",
            outer_fold=OUTER_FOLD,
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
                "Performance summary for focused experiment",
                outer_fold=OUTER_FOLD,
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

    async def create_data_context(self, outer_fold: int, inner_fold: int):
        """Create data context for a specific CV fold.

        Args:
            outer_fold: Outer fold number.
            inner_fold: Inner fold number.

        Returns:
            Data context for the specified fold.
        """
        try:
            from vulcan.data.goodreads_loader import GoodreadsDataLoader

            # Use absolute paths
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

            # Get data context
            data_context = loader.get_data_context()

            return data_context

        except Exception as e:
            logger.error(
                "Failed to create data context for focused experiment",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                error=str(e),
            )
            raise

    def save_results_summary(self, results: List[Dict]) -> None:
        """Save focused experiment results summary to file.

        Args:
            results: List of experiment results.
        """
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"focused_cv_results_outer_{OUTER_FOLD}_{timestamp}.json"

        summary = {
            "experiment_info": {
                "experiment_type": "focused_cv",
                "timestamp": timestamp,
                "outer_fold": OUTER_FOLD,
                "inner_folds": INNER_FOLDS,
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
            "Focused experiment results saved",
            results_file=results_file,
            total_experiments=len(results),
        )


async def main():
    """Main focused experiment execution function."""
    print("=" * 80)
    print("🎯 VULCAN Focused Cross-Validation Experiment")
    print(f"📊 Testing Outer Fold {OUTER_FOLD} across Inner Folds {INNER_FOLDS}")
    print(
        f"🌳 Deep MCTS Exploration (max_depth={MAX_DEPTH}, max_iterations={MAX_ITERATIONS})"
    )
    print(f"🔬 Total Experiments: {len(INNER_FOLDS)}")
    print("📊 Local experiment tracking enabled")
    print("=" * 80)

    # Initialize experiment runner
    runner = FocusedExperimentRunner()

    if not await runner.initialize():
        print("❌ Failed to initialize focused experiment runner")
        return

    print("✅ Focused experiment runner initialized successfully")
    print("🚀 Starting focused experiment...")

    # Run focused experiment
    start_time = time.time()
    results = await runner.run_focused_experiment()
    total_time = time.time() - start_time

    # Save results
    runner.save_results_summary(results)

    # Final report
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    print("\n" + "=" * 80)
    print("🎉 FOCUSED EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"⏱️  Total Time: {total_time / 60:.2f} minutes ({total_time:.0f} seconds)")
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

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
