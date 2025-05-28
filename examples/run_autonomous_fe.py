#!/usr/bin/env python3
"""
Example script demonstrating the VULCAN autonomous feature engineering system.

This script shows how to:
1. Set up the complete system with all components
2. Run MCTS-based feature engineering
3. Evaluate and analyze results
4. Use reflection for strategic insights
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autonomous_fe_env import (
    ConfigManager,
    FeatureEvaluator,
    MCTSOrchestrator,
    ReflectionEngine,
    get_agent,
    get_dal,
)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("autonomous_fe.log")],
    )


async def main():
    """Main function demonstrating the autonomous feature engineering system."""

    # Set up logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting VULCAN Autonomous Feature Engineering Demo")

    try:
        # 1. Load Configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager()

        # You can also load from a file:
        # config_manager = ConfigManager("config/config.yaml")

        config = config_manager.get_config()
        logger.info("Configuration loaded successfully")

        # 2. Set up Data Access Layer
        logger.info("Setting up data access layer...")
        dal = get_dal(config)
        dal.connect()

        # Get database schema for context
        schema = dal.get_schema()
        logger.info(f"Database schema: {list(schema.keys())}")

        # 3. Set up Feature Evaluator
        logger.info("Setting up feature evaluator...")
        evaluator = FeatureEvaluator(config)
        evaluator.setup(dal)

        # 4. Set up Agents
        logger.info("Setting up agents...")
        feature_agent = get_agent(
            "feature", config=config.get("agents", {}).get("feature_agent", {})
        )
        reflection_agent = get_agent(
            "reflection", config=config.get("agents", {}).get("reflection_agent", {})
        )

        # 5. Set up Reflection Engine
        logger.info("Setting up reflection engine...")
        reflection_engine = ReflectionEngine(config)
        reflection_agent.setup_reflection_engine(reflection_engine)

        # 6. Set up MCTS Orchestrator
        logger.info("Setting up MCTS orchestrator...")
        orchestrator = MCTSOrchestrator(config)
        orchestrator.setup(
            dal=dal,
            agent=feature_agent,
            evaluator=evaluator,
            reflection_engine=reflection_engine,
        )

        # 7. Calculate Baseline Performance
        logger.info("Calculating baseline performance...")
        baseline_score = orchestrator.calculate_baseline()
        logger.info(f"Baseline score: {baseline_score:.4f}")

        # 8. Run MCTS Feature Engineering
        logger.info("Starting MCTS feature engineering process...")

        # Option 1: Run single MCTS
        best_node = orchestrator.run()

        # Option 2: Run parallel MCTS (uncomment to use)
        # parallel_mcts = ParallelMCTS(config)
        # worker_orchestrators = parallel_mcts.create_worker_orchestrators(orchestrator, num_workers=2)
        # best_node = await parallel_mcts.run_parallel_search(worker_orchestrators)

        # 9. Analyze Results
        logger.info("Analyzing results...")

        best_features = orchestrator.get_best_features()
        state_manager = orchestrator.state_manager

        logger.info(f"Best score achieved: {best_node.score:.4f}")
        logger.info(
            f"Improvement over baseline: {best_node.score - baseline_score:.4f}"
        )
        logger.info(f"Number of features in best set: {len(best_features)}")

        if best_features:
            logger.info("Best features found:")
            for i, feature in enumerate(best_features):
                logger.info(f"  {i + 1}. {feature.name}: {feature.description}")

        # 10. Generate Final Reflection
        logger.info("Generating final strategic reflection...")

        feature_history = []
        for feature_state in state_manager.get_feature_history():
            if feature_state.feature:
                feature_history.append(
                    {
                        "name": feature_state.feature.name,
                        "description": feature_state.feature.description,
                        "score": feature_state.score,
                    }
                )

        performance_history = []
        for i, feature_state in enumerate(state_manager.get_feature_history()):
            performance_history.append(
                {
                    "iteration": i + 1,
                    "score": feature_state.score,
                    "features": [feature_state.feature.name]
                    if feature_state.feature
                    else [],
                }
            )

        try:
            final_reflection = await reflection_engine.strategic_reflection(
                state_manager, feature_history, performance_history
            )
            logger.info("Final Strategic Reflection:")
            logger.info(final_reflection)
        except Exception as e:
            logger.warning(f"Could not generate final reflection: {e}")

        # 11. Print Summary Statistics
        logger.info("\n" + "=" * 50)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 50)

        mcts_stats = state_manager.get_mcts_stats()
        logger.info(f"Total MCTS iterations: {mcts_stats['total_iterations']}")
        logger.info(f"Successful iterations: {mcts_stats['successful_iterations']}")
        logger.info(f"Failed iterations: {mcts_stats['failed_iterations']}")
        logger.info(
            f"Success rate: {mcts_stats['successful_iterations'] / mcts_stats['total_iterations'] * 100:.1f}%"
        )

        if "duration_seconds" in mcts_stats:
            logger.info(f"Total duration: {mcts_stats['duration_seconds']:.2f} seconds")

        logger.info(f"Final best score: {best_node.score:.4f}")
        logger.info(f"Baseline score: {baseline_score:.4f}")
        logger.info(f"Total improvement: {best_node.score - baseline_score:.4f}")

        # 12. Save Results
        logger.info("Saving results...")
        state_manager.save_to_file()

        # Save best features to a readable format
        if best_features:
            results_dir = config.get("results_dir", "results")
            os.makedirs(results_dir, exist_ok=True)

            with open(f"{results_dir}/best_features.txt", "w") as f:
                f.write("Best Features Found by VULCAN\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Baseline Score: {baseline_score:.4f}\n")
                f.write(f"Best Score: {best_node.score:.4f}\n")
                f.write(f"Improvement: {best_node.score - baseline_score:.4f}\n\n")

                for i, feature in enumerate(best_features):
                    f.write(f"Feature {i + 1}: {feature.name}\n")
                    f.write(f"Description: {feature.description}\n")
                    f.write(f"Output Column: {feature.output_column_name}\n")
                    f.write(f"Required Columns: {feature.required_input_columns}\n")
                    f.write("Code:\n")
                    f.write(feature.code)
                    f.write("\n" + "-" * 40 + "\n\n")

        logger.info("Results saved successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

    finally:
        # Clean up
        if "dal" in locals():
            dal.disconnect()
        logger.info("VULCAN demo completed")


def run_simple_example():
    """Run a simplified example without async components."""

    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("Running simplified VULCAN example...")

    try:
        # Basic configuration
        config = {
            "data_source": {
                "type": "sql",
                "db_path": "data/goodreads.db",
                "splits": {
                    "directory": "data/splits",
                    "files": {"train": "train.csv"},
                    "id_column": "user_id",
                },
            },
            "mcts": {
                "max_iterations": 10,  # Reduced for demo
                "exploration_factor": 1.414,
            },
            "evaluation": {
                "n_clusters": 3,
                "sample_size": 500,  # Reduced for demo
            },
        }

        # Set up components
        dal = get_dal(config)
        dal.connect()

        evaluator = FeatureEvaluator(config)
        evaluator.setup(dal)

        feature_agent = get_agent("feature", config={})

        # Set up and run MCTS
        orchestrator = MCTSOrchestrator(config)
        orchestrator.setup(dal=dal, agent=feature_agent, evaluator=evaluator)

        # Calculate baseline and run
        baseline_score = orchestrator.calculate_baseline()
        best_node = orchestrator.run()

        # Print results
        logger.info(f"Baseline: {baseline_score:.4f}")
        logger.info(f"Best score: {best_node.score:.4f}")
        logger.info(f"Improvement: {best_node.score - baseline_score:.4f}")

        best_features = orchestrator.get_best_features()
        if best_features:
            logger.info("Best features:")
            for feature in best_features:
                logger.info(f"  - {feature.name}")

        dal.disconnect()
        logger.info("Simple example completed successfully")

    except Exception as e:
        logger.error(f"Error in simple example: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VULCAN Autonomous Feature Engineering Demo"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Run simplified example without async"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    if args.simple:
        run_simple_example()
    else:
        # Check if we have a database file
        db_path = "data/goodreads.db"
        if not os.path.exists(db_path):
            print(f"Warning: Database file {db_path} not found.")
            print(
                "Please ensure you have merged your databases using the merge_databases.py script."
            )
            print("Running with simplified example instead...")
            run_simple_example()
        else:
            asyncio.run(main())
