#!/usr/bin/env python3
"""
VULCAN Standalone Experiment Runner

Run VULCAN experiments from the command line and save results to folders.
Results can be viewed in the React dashboard by selecting the output folder.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vulcan.core import ConfigManager, VulcanOrchestrator
from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.types import VulcanConfig
from vulcan.types.config_types import ClusteringConfig
from vulcan.utils import get_vulcan_logger, setup_logging

logger = get_vulcan_logger(__name__)


class ExperimentRunner:
    """Standalone experiment runner with result saving."""

    def __init__(self, output_dir: str = "experiments"):
        """Initialize experiment runner.

        Args:
            output_dir: Base directory for saving experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.experiment_dir: Optional[Path] = None

    def create_experiment_folder(self, experiment_name: str) -> Path:
        """Create a timestamped folder for this experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{experiment_name.replace(' ', '_')}"

        self.experiment_dir = self.output_dir / folder_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "features").mkdir(exist_ok=True)
        (self.experiment_dir / "llm_outputs").mkdir(exist_ok=True)
        (self.experiment_dir / "evaluations").mkdir(exist_ok=True)
        (self.experiment_dir / "tree_snapshots").mkdir(exist_ok=True)

        logger.info(f"Created experiment folder: {self.experiment_dir}")
        return self.experiment_dir

    async def run_experiment(
        self,
        config: VulcanConfig,
        experiment_name: str,
        outer_fold: int = 1,
        inner_fold: int = 1,
        n_clusters: Optional[int] = None,
        cluster_range: Optional[tuple] = None,
        metric: str = "silhouette",
    ) -> Dict[str, Any]:
        """Run a VULCAN experiment with proper clustering evaluation.

        Args:
            config: VULCAN configuration
            experiment_name: Name for this experiment
            outer_fold: Outer fold for cross-validation
            inner_fold: Inner fold for cross-validation
            n_clusters: Fixed number of clusters (if None, will optimize)
            cluster_range: Range of clusters to try (default: (2, 20))
            metric: Clustering metric to optimize ("silhouette", "calinski_harabasz", "davies_bouldin")

        Returns:
            Experiment results dictionary
        """
        # Create experiment folder
        exp_dir = self.create_experiment_folder(experiment_name)

        # Save configuration
        config_dict = config.dict()
        config_dict["experiment_settings"] = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "n_clusters": n_clusters,
            "cluster_range": cluster_range,
            "metric": metric,
            "timestamp": datetime.now().isoformat(),
        }

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Initialize orchestrator with result saving callbacks
        orchestrator = VulcanOrchestrator(config)

        # Add callback to save LLM outputs
        def save_llm_output(
            iteration: int, prompt: str, response: str, feature_name: str
        ):
            """Save LLM interaction to file."""
            llm_file = (
                exp_dir / "llm_outputs" / f"iter_{iteration:04d}_{feature_name}.json"
            )
            with open(llm_file, "w") as f:
                json.dump(
                    {
                        "iteration": iteration,
                        "feature_name": feature_name,
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "response": response,
                    },
                    f,
                    indent=2,
                )

        # Add callback to save feature code
        def save_feature_code(
            iteration: int, feature_name: str, feature_code: str, feature_type: str
        ):
            """Save generated feature code."""
            feature_file = (
                exp_dir / "features" / f"iter_{iteration:04d}_{feature_name}.py"
            )
            with open(feature_file, "w") as f:
                f.write(f"# Feature: {feature_name}\n")
                f.write(f"# Type: {feature_type}\n")
                f.write(f"# Iteration: {iteration}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                f.write(feature_code)

        # Add callback to save evaluations
        def save_evaluation(iteration: int, evaluation: Dict[str, Any]):
            """Save feature evaluation results."""
            eval_file = exp_dir / "evaluations" / f"iter_{iteration:04d}.json"
            with open(eval_file, "w") as f:
                json.dump(evaluation, f, indent=2)

        # Add callback to save tree snapshots
        def save_tree_snapshot(iteration: int, tree_data: Dict[str, Any]):
            """Save MCTS tree snapshot."""
            tree_file = exp_dir / "tree_snapshots" / f"tree_iter_{iteration:04d}.json"
            with open(tree_file, "w") as f:
                json.dump(tree_data, f, indent=2)

        # Attach callbacks to orchestrator
        orchestrator.add_callback("llm_output", save_llm_output)
        orchestrator.add_callback("feature_code", save_feature_code)
        orchestrator.add_callback("evaluation", save_evaluation)
        orchestrator.add_callback("tree_snapshot", save_tree_snapshot)

        try:
            # Initialize components
            logger.info("Initializing VULCAN components...")
            await orchestrator.initialize_components()

            # Load data
            logger.info(
                f"Loading data for outer_fold={outer_fold}, inner_fold={inner_fold}"
            )
            loader = GoodreadsDataLoader(
                db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
                splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )

            data_context = loader.get_data_context()

            # Configure clustering evaluation
            cluster_config = ClusteringConfig(
                n_clusters=n_clusters,
                cluster_range=list(cluster_range) if cluster_range else [2, 20],
                metric=metric,
                optimize_n_clusters=n_clusters is None,
            )

            # Save evaluation configuration
            with open(exp_dir / "evaluation_config.json", "w") as f:
                json.dump(cluster_config.dict(), f, indent=2)

            # Update config with evaluation settings
            config.evaluation.clustering_config = cluster_config

            # Start experiment
            logger.info(f"Starting experiment: {experiment_name}")
            start_time = time.time()

            experiment_id = await orchestrator.start_experiment(
                experiment_name=experiment_name,
                data_context=data_context,
                config_overrides={
                    "max_iterations": config.mcts.max_iterations,
                    "evaluation_config": cluster_config.dict(),
                },
            )

            # Wait for experiment to complete
            while orchestrator.get_status().is_running:
                await asyncio.sleep(1.0)

            execution_time = time.time() - start_time

            # Get final results
            history = orchestrator.get_experiment_history()
            if history:
                final_result = history[-1]

                # Save final results
                results = {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "execution_time": execution_time,
                    "best_score": final_result.best_score,
                    "best_features": final_result.best_features,
                    "total_iterations": final_result.total_iterations,
                    "metric_used": metric,
                    "optimal_n_clusters": final_result.optimal_n_clusters
                    if hasattr(final_result, "optimal_n_clusters")
                    else None,
                    "feature_count": len(final_result.best_features)
                    if final_result.best_features
                    else 0,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(exp_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2)

                # Save performance data
                perf_data = orchestrator.export_performance_data()
                with open(exp_dir / "performance_data.json", "w") as f:
                    json.dump(perf_data, f, indent=2)

                logger.info("Experiment completed successfully!")
                logger.info(f"Results saved to: {exp_dir}")
                logger.info(f"Best score: {final_result.best_score:.4f}")
                logger.info(f"Execution time: {execution_time:.2f}s")

                return results
            else:
                raise RuntimeError("No experiment results available")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")

            # Save error information
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
            }

            with open(exp_dir / "error.json", "w") as f:
                json.dump(error_info, f, indent=2)

            raise

        finally:
            await orchestrator.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run VULCAN experiments")
    parser.add_argument("experiment_name", help="Name for this experiment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--output-dir", default="experiments", help="Output directory for results"
    )
    parser.add_argument("--outer-fold", type=int, default=1, help="Outer fold (1-5)")
    parser.add_argument("--inner-fold", type=int, default=3, help="Inner fold (1-3)")
    parser.add_argument(
        "--max-iterations", type=int, default=50, help="Maximum MCTS iterations"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Fixed number of clusters (if not set, will optimize)",
    )
    parser.add_argument(
        "--cluster-min", type=int, default=2, help="Minimum clusters to try"
    )
    parser.add_argument(
        "--cluster-max", type=int, default=30, help="Maximum clusters to try"
    )
    parser.add_argument(
        "--metric",
        choices=["silhouette", "calinski_harabasz", "davies_bouldin"],
        default="silhouette",
        help="Clustering metric to optimize",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "none"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
    else:
        config = VulcanConfig()

    # Override configuration with command line arguments
    config.mcts.max_iterations = args.max_iterations
    config.llm.provider = args.llm_provider
    if args.debug:
        config.logging.level = "DEBUG"

    # Setup logging
    setup_logging(config.logging)

    # Create and run experiment
    runner = ExperimentRunner(args.output_dir)

    try:
        await runner.run_experiment(
            config=config,
            experiment_name=args.experiment_name,
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
            n_clusters=args.n_clusters,
            cluster_range=(args.cluster_min, args.cluster_max),
            metric=args.metric,
        )
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
