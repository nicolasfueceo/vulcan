#!/usr/bin/env python3
"""
VULCAN Cold Start Feature Engineering Pipeline

This script demonstrates the complete VULCAN system for cold start
recommendation scenarios with baseline comparison and visualization.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autonomous_fe_env import (
    ConfigManager,
    FeatureEvaluator,
    MCTSOrchestrator,
    StateManager,
    get_agent,
)
from autonomous_fe_env.data import SqlDAL
from autonomous_fe_env.evaluation import ColdStartEvaluator
from autonomous_fe_env.feature import FeatureRegistry
from autonomous_fe_env.visualization import AgentMonitor, PipelineVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vulcan_cold_start.log")],
)
logger = logging.getLogger(__name__)


def setup_vulcan_system(config_path: str, db_path: str):
    """Set up the VULCAN system components."""
    logger.info("Setting up VULCAN system...")

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(config_path)

    # Initialize data access layer
    dal = SqlDAL(db_path)

    # Initialize core components
    feature_registry = FeatureRegistry(config)
    state_manager = StateManager(config)
    feature_evaluator = FeatureEvaluator(config)
    feature_evaluator.setup(dal)

    # Initialize agents
    feature_agent = get_agent("feature", config)
    reflection_agent = get_agent("reflection", config)

    # Initialize MCTS orchestrator
    mcts = MCTSOrchestrator(
        feature_agent=feature_agent,
        reflection_agent=reflection_agent,
        feature_evaluator=feature_evaluator,
        feature_registry=feature_registry,
        state_manager=state_manager,
        config=config,
    )

    # Initialize evaluation and visualization
    cold_start_evaluator = ColdStartEvaluator(config)
    cold_start_evaluator.setup(dal, db_path)

    pipeline_visualizer = PipelineVisualizer(config)
    agent_monitor = AgentMonitor(config)

    logger.info("VULCAN system setup complete")

    return {
        "config": config,
        "dal": dal,
        "feature_registry": feature_registry,
        "state_manager": state_manager,
        "feature_evaluator": feature_evaluator,
        "mcts": mcts,
        "cold_start_evaluator": cold_start_evaluator,
        "pipeline_visualizer": pipeline_visualizer,
        "agent_monitor": agent_monitor,
        "feature_agent": feature_agent,
        "reflection_agent": reflection_agent,
    }


def run_baseline_evaluation(cold_start_evaluator, sample_size=5000):
    """Run baseline model evaluation."""
    logger.info("Running baseline evaluation...")

    start_time = time.time()
    baseline_results = cold_start_evaluator.run_baseline_evaluation(
        sample_size=sample_size
    )
    eval_time = time.time() - start_time

    logger.info(f"Baseline evaluation completed in {eval_time:.2f} seconds")

    # Print baseline results
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)

    for model_name, score in baseline_results.items():
        print(f"{model_name.upper()}: {score:.4f}")

    best_baseline = max(baseline_results.items(), key=lambda x: x[1])
    print(f"\nBest Baseline: {best_baseline[0]} ({best_baseline[1]:.4f})")
    print("=" * 60)

    return baseline_results


def run_vulcan_pipeline(components, max_iterations=20, visualization_interval=5):
    """Run the VULCAN feature engineering pipeline."""
    logger.info("Starting VULCAN pipeline...")

    mcts = components["mcts"]
    pipeline_visualizer = components["pipeline_visualizer"]
    agent_monitor = components["agent_monitor"]
    cold_start_evaluator = components["cold_start_evaluator"]

    # Update visualizer with baseline scores
    baseline_results = cold_start_evaluator.baseline_results
    pipeline_visualizer.update_baseline_scores(baseline_results)

    best_score = -float("inf")
    best_features = []

    try:
        for iteration in range(max_iterations):
            logger.info(f"VULCAN Iteration {iteration + 1}/{max_iterations}")

            # Log agent activity
            agent_monitor.log_agent_activity(
                "MCTS_Orchestrator", "iteration_start", {"iteration": iteration + 1}
            )

            # Run MCTS iteration
            start_time = time.time()
            result = mcts.run_iteration()
            iteration_time = time.time() - start_time

            if result and "best_score" in result:
                current_score = result["best_score"]
                current_features = result.get("best_features", [])

                # Update best score
                if current_score > best_score:
                    best_score = current_score
                    best_features = current_features
                    logger.info(f"New best score: {best_score:.4f}")

                # Log to visualizers
                pipeline_visualizer.log_mcts_iteration(
                    iteration + 1,
                    best_score,
                    result.get("nodes_explored", 0),
                    [f.name for f in current_features],
                )

                # Log feature evaluations
                for feature in current_features:
                    pipeline_visualizer.log_feature_evaluation(
                        feature.name, current_score, iteration_time
                    )

                # Log agent performance
                agent_monitor.log_agent_performance(
                    "MCTS_Orchestrator",
                    "iteration_score",
                    current_score,
                    {"iteration": iteration + 1, "features": len(current_features)},
                )

            # Periodic status updates
            if (iteration + 1) % visualization_interval == 0:
                pipeline_visualizer.print_live_status()
                agent_monitor.print_live_status()

                # Compare with baselines
                if best_score > -float("inf"):
                    comparison = cold_start_evaluator.compare_with_baselines(best_score)
                    print("\nCurrent VULCAN vs Best Baseline:")
                    best_baseline = max(baseline_results.items(), key=lambda x: x[1])
                    improvement = (
                        (best_score - best_baseline[1]) / best_baseline[1]
                    ) * 100
                    print(f"  VULCAN: {best_score:.4f}")
                    print(f"  {best_baseline[0]}: {best_baseline[1]:.4f}")
                    print(f"  Improvement: {improvement:+.1f}%")

            # Small delay to prevent overwhelming
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

    return {
        "best_score": best_score,
        "best_features": best_features,
        "total_iterations": iteration + 1,
    }


def generate_final_report(components, vulcan_results, output_dir="results"):
    """Generate comprehensive final report with visualizations."""
    logger.info("Generating final report...")

    os.makedirs(output_dir, exist_ok=True)

    cold_start_evaluator = components["cold_start_evaluator"]
    pipeline_visualizer = components["pipeline_visualizer"]
    agent_monitor = components["agent_monitor"]

    vulcan_score = vulcan_results["best_score"]

    # Generate comparison report
    cold_start_evaluator.print_detailed_report(vulcan_score)

    # Generate visualizations
    try:
        # Pipeline summary plot
        summary_plot_path = os.path.join(output_dir, "vulcan_pipeline_summary.png")
        pipeline_visualizer.create_static_summary(save_path=summary_plot_path)

        # Baseline comparison plot
        comparison_plot_path = os.path.join(output_dir, "baseline_comparison.png")
        cold_start_evaluator.generate_visualization(
            vulcan_score, save_path=comparison_plot_path
        )

        # Interactive dashboard
        dashboard_path = os.path.join(output_dir, "vulcan_dashboard.html")
        pipeline_visualizer.create_live_dashboard(save_path=dashboard_path)

        # Export agent activity log
        activity_log_path = os.path.join(output_dir, "agent_activity_log.json")
        agent_monitor.export_activity_log(activity_log_path)

        logger.info(f"Reports and visualizations saved to {output_dir}/")

    except Exception as e:
        logger.warning(f"Error generating visualizations: {e}")

    # Print final summary
    print("\n" + "=" * 80)
    print("VULCAN COLD START EVALUATION - FINAL SUMMARY")
    print("=" * 80)

    print("VULCAN Results:")
    print(f"  Final Score: {vulcan_score:.4f}")
    print(f"  Best Features: {len(vulcan_results['best_features'])}")
    print(f"  Total Iterations: {vulcan_results['total_iterations']}")

    # Baseline comparison
    baseline_results = cold_start_evaluator.baseline_results
    if baseline_results:
        best_baseline = max(baseline_results.items(), key=lambda x: x[1])
        print("\nBaseline Comparison:")
        print(f"  Best Baseline: {best_baseline[0]} ({best_baseline[1]:.4f})")

        if best_baseline[1] > 0:
            improvement = ((vulcan_score - best_baseline[1]) / best_baseline[1]) * 100
            print(f"  VULCAN Improvement: {improvement:+.1f}%")

            if improvement > 0:
                print("  ✅ VULCAN outperformed all baselines!")
            else:
                print("  ❌ VULCAN did not beat the best baseline")

    # Pipeline statistics
    pipeline_stats = pipeline_visualizer.get_summary_stats()
    print("\nPipeline Statistics:")
    print(f"  Runtime: {pipeline_stats['runtime']:.1f} seconds")
    print(f"  Features Evaluated: {pipeline_stats['total_features_evaluated']}")
    print(f"  Agent Activities: {pipeline_stats['total_agent_activities']}")

    print("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run VULCAN Cold Start Pipeline")
    parser.add_argument(
        "--config",
        default="src/autonomous_fe_env/config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--db", default="data/goodreads.db", help="Path to database file"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Maximum number of MCTS iterations"
    )
    parser.add_argument(
        "--baseline-sample",
        type=int,
        default=5000,
        help="Sample size for baseline evaluation",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip baseline evaluation"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.db):
        logger.error(f"Database file not found: {args.db}")
        return 1

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1

    try:
        # Setup system
        components = setup_vulcan_system(args.config, args.db)

        # Run baseline evaluation
        if not args.skip_baselines:
            baseline_results = run_baseline_evaluation(
                components["cold_start_evaluator"], sample_size=args.baseline_sample
            )
        else:
            logger.info("Skipping baseline evaluation")

        # Run VULCAN pipeline
        vulcan_results = run_vulcan_pipeline(
            components, max_iterations=args.iterations, visualization_interval=5
        )

        # Generate final report
        generate_final_report(components, vulcan_results, args.output_dir)

        logger.info("VULCAN pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
