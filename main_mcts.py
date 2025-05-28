#!/usr/bin/env python3
"""
VULCAN Main Execution Script with MCTS Feature Selection

This script demonstrates the complete VULCAN system:
1. LLM-driven autonomous feature engineering using MCTS
2. Real-time monitoring and reflection
3. Comprehensive evaluation and analysis
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from autonomous_fe_env import (
    ConfigManager,
    FeatureEvaluator,
    MCTSOrchestrator,
    ParallelMCTS,
    ReflectionEngine,
    get_agent,
    get_dal,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


try:
    from autonomous_fe_env.visualization.pipeline_visualizer import PipelineVisualizer
except ImportError:
    PipelineVisualizer = None


class VulcanMCTSRunner:
    """Main runner for VULCAN MCTS-based feature engineering."""

    def __init__(
        self, config_path: str = "src/autonomous_fe_env/config/default_config.yaml"
    ):
        """Initialize the VULCAN runner."""
        self.config_path = config_path
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # Update config to use proper data paths
        self.config["database"]["path"] = "data/goodreads.db"
        self.config["data_source"] = {
            "type": "sql",
            "db_path": "data/goodreads.db",
            "splits": {
                "directory": "data/splits",
                "files": {
                    "train": "outer_fold_1_train_fe_users.csv",
                    "validation": "outer_fold_1_val_clusters_users.csv",
                    "test": "outer_test_users.csv",
                },
                "id_column": "user_id",
            },
        }

        # Initialize components
        self.dal = None
        self.evaluator = None
        self.feature_agent = None
        self.reflection_agent = None
        self.reflection_engine = None
        self.orchestrator = None
        self.state_manager = None
        self.visualizer = None

        # Create directories for organized output
        os.makedirs("states", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Update config to use organized directories
        self.config["state_dir"] = "states"
        self.config["results_dir"] = "results"

        self.logger.info("🚀 VULCAN MCTS Runner initialized")

    def setup_logging(self):
        """Set up comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("vulcan_mcts.log", mode="w"),
            ],
        )

    def initialize_components(self):
        """Initialize all VULCAN components."""
        self.logger.info("🔧 Initializing VULCAN components...")

        # 1. Data Access Layer
        self.logger.info("📊 Setting up data access layer...")
        self.dal = get_dal(self.config)

        # Check if database exists
        db_path = self.config.get("database", {}).get("path", "data/goodreads.db")
        if not os.path.exists(db_path):
            self.logger.warning(f"⚠️  Database not found at {db_path}")
            self.logger.info("📝 Creating mock database for demonstration...")
            self._create_mock_database()

        self.dal.connect()
        schema = self.dal.get_schema()
        self.logger.info(f"📋 Database schema: {list(schema.keys())}")

        # 2. Feature Evaluator
        self.logger.info("⚖️  Setting up feature evaluator...")
        self.evaluator = FeatureEvaluator(self.config)
        self.evaluator.setup(self.dal)

        # 3. LLM-based Feature Agent
        self.logger.info("🤖 Setting up LLM feature agent...")
        self.feature_agent = get_agent(
            "llm_feature",
            config={
                "model_name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2000,
                "max_retries": 3,
            },
        )

        # 4. Reflection Agent
        self.logger.info("💭 Setting up reflection agent...")
        self.reflection_agent = get_agent(
            "reflection",
            config=self.config.get("agents", {}).get("reflection_agent", {}),
        )

        # 5. Reflection Engine
        self.logger.info("🧠 Setting up reflection engine...")
        self.reflection_engine = ReflectionEngine(self.config)
        if hasattr(self.reflection_agent, "setup_reflection_engine"):
            self.reflection_agent.setup_reflection_engine(self.reflection_engine)

        # 6. MCTS Orchestrator
        self.logger.info("🌳 Setting up MCTS orchestrator...")
        self.orchestrator = MCTSOrchestrator(self.config)
        self.orchestrator.setup(
            dal=self.dal,
            agent=self.feature_agent,
            evaluator=self.evaluator,
            reflection_engine=self.reflection_engine,
            visualizer=self.visualizer,
        )

        # 7. State Manager
        self.state_manager = self.orchestrator.state_manager

        # 8. Pipeline Visualizer
        self.logger.info("📊 Setting up pipeline visualizer...")
        if PipelineVisualizer:
            self.visualizer = PipelineVisualizer(self.config)
        else:
            self.logger.warning(
                "⚠️  PipelineVisualizer not available, skipping visualization"
            )
            self.visualizer = None

        self.logger.info("✅ All components initialized successfully")

    def _create_mock_database(self):
        """Create a mock database for demonstration purposes."""
        import sqlite3

        import numpy as np
        import pandas as pd

        db_path = self.config.get("database", {}).get("path", "data/goodreads.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create mock data
        np.random.seed(42)
        n_users = 1000
        n_books = 500
        n_interactions = 5000

        # Generate mock interactions
        user_ids = np.random.randint(1, n_users + 1, n_interactions)
        book_ids = np.random.randint(1, n_books + 1, n_interactions)
        ratings = np.random.randint(1, 6, n_interactions)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "user_id": user_ids,
                "book_id": book_ids,
                "rating": ratings,
                "review_text": [f"Review {i}" for i in range(n_interactions)],
                "date_added": pd.date_range(
                    "2020-01-01", periods=n_interactions, freq="H"
                ),
                "book_title": [f"Book {book_id}" for book_id in book_ids],
                "author": [f"Author {book_id % 100}" for book_id in book_ids],
            }
        )

        # Save to database
        conn = sqlite3.connect(db_path)
        data.to_sql("interactions", conn, if_exists="replace", index=False)
        conn.close()

        self.logger.info(f"📝 Created mock database with {n_interactions} interactions")

    async def run_mcts_feature_engineering(self, use_parallel: bool = False):
        """Run MCTS-based feature engineering."""
        self.logger.info("🌳 Starting MCTS feature engineering process...")

        # Calculate baseline
        self.logger.info("📊 Calculating baseline performance...")
        baseline_score = self.orchestrator.calculate_baseline()
        self.logger.info(f"📈 Baseline score: {baseline_score:.4f}")

        # Update visualizer with baseline
        if self.visualizer:
            self.visualizer.update_baseline_scores({"baseline": baseline_score})

        start_time = time.time()

        if use_parallel:
            # Run parallel MCTS
            self.logger.info("🔄 Running parallel MCTS...")
            parallel_mcts = ParallelMCTS(self.config)
            num_workers = self.config.get("parallel_mcts", {}).get("num_workers", 2)

            worker_orchestrators = parallel_mcts.create_worker_orchestrators(
                self.orchestrator, num_workers=num_workers
            )
            best_node = await parallel_mcts.run_parallel_search(worker_orchestrators)
        else:
            # Run single MCTS
            self.logger.info("🌳 Running single-threaded MCTS...")
            best_node = self.orchestrator.run()

        end_time = time.time()
        duration = end_time - start_time

        # Log results
        self.logger.info("🎉 MCTS feature engineering completed!")
        self.logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        self.logger.info(f"🏆 Best score: {best_node.score:.4f}")
        self.logger.info(f"📈 Improvement: {best_node.score - baseline_score:.4f}")

        return best_node, baseline_score, duration

    def analyze_results(self, best_node, baseline_score: float):
        """Analyze and display results."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔍 ANALYZING RESULTS")
        self.logger.info("=" * 60)

        # Get best features
        best_features = self.orchestrator.get_best_features()

        self.logger.info(f"🏆 Best score achieved: {best_node.score:.4f}")
        self.logger.info(f"📊 Baseline score: {baseline_score:.4f}")
        self.logger.info(
            f"📈 Total improvement: {best_node.score - baseline_score:.4f}"
        )
        self.logger.info(f"🔢 Number of features: {len(best_features)}")

        if best_features:
            self.logger.info("\n🎯 Best features found:")
            for i, feature in enumerate(best_features, 1):
                self.logger.info(f"  {i}. {feature.name}")
                self.logger.info(f"     📝 {feature.description}")
                self.logger.info(f"     🔧 Columns: {feature.required_input_columns}")

        # MCTS Statistics
        mcts_stats = self.state_manager.get_mcts_stats()
        self.logger.info("\n📊 MCTS Statistics:")
        self.logger.info(f"  Total iterations: {mcts_stats.get('total_iterations', 0)}")
        self.logger.info(f"  Successful: {mcts_stats.get('successful_iterations', 0)}")
        self.logger.info(f"  Failed: {mcts_stats.get('failed_iterations', 0)}")

        total_iter = mcts_stats.get("total_iterations", 1)
        success_rate = mcts_stats.get("successful_iterations", 0) / total_iter * 100
        self.logger.info(f"  Success rate: {success_rate:.1f}%")

        return best_features

    async def generate_final_reflection(self, best_features: List):
        """Generate final strategic reflection."""
        self.logger.info("\n💭 Generating final strategic reflection...")

        try:
            # Prepare reflection context
            feature_history = []
            for feature_state in self.state_manager.get_feature_history():
                if feature_state.feature:
                    feature_history.append(
                        {
                            "name": feature_state.feature.name,
                            "description": feature_state.feature.description,
                            "score": feature_state.score,
                        }
                    )

            performance_history = []
            for i, feature_state in enumerate(self.state_manager.get_feature_history()):
                performance_history.append(
                    {
                        "iteration": i + 1,
                        "score": feature_state.score,
                        "features": [feature_state.feature.name]
                        if feature_state.feature
                        else [],
                    }
                )

            # Generate reflection
            final_reflection = await self.reflection_engine.strategic_reflection(
                self.state_manager, feature_history, performance_history
            )

            self.logger.info("🧠 Final Strategic Reflection:")
            self.logger.info("-" * 40)
            self.logger.info(final_reflection)
            self.logger.info("-" * 40)

        except Exception as e:
            self.logger.warning(f"⚠️  Could not generate final reflection: {e}")

    def save_results(
        self, best_features: List, best_score: float, baseline_score: float
    ):
        """Save results to files."""
        self.logger.info("💾 Saving results...")

        # Create results directory
        results_dir = self.config.get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save state to states directory
        states_dir = self.config.get("state_dir", "states")
        os.makedirs(states_dir, exist_ok=True)

        # Update state manager to save to states directory
        original_save_path = getattr(self.state_manager, "save_path", None)
        if hasattr(self.state_manager, "save_path"):
            self.state_manager.save_path = f"{states_dir}/state_manager.json"

        self.state_manager.save_to_file()

        # Restore original save path if it existed
        if original_save_path:
            self.state_manager.save_path = original_save_path

        # Save best features
        if best_features:
            with open(f"{results_dir}/best_features.txt", "w") as f:
                f.write("VULCAN MCTS Feature Engineering Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Baseline Score: {baseline_score:.4f}\n")
                f.write(f"Best Score: {best_score:.4f}\n")
                f.write(f"Improvement: {best_score - baseline_score:.4f}\n")
                f.write(f"Number of Features: {len(best_features)}\n\n")

                f.write("Best Features:\n")
                f.write("-" * 20 + "\n")
                for i, feature in enumerate(best_features, 1):
                    f.write(f"\n{i}. {feature.name}\n")
                    f.write(f"   Description: {feature.description}\n")
                    f.write(f"   Required Columns: {feature.required_input_columns}\n")
                    f.write("   Code:\n")
                    f.write("   " + "\n   ".join(feature.code.split("\n")) + "\n")

        # Generate and save visualizations
        if self.visualizer:
            self.logger.info("📊 Generating visualizations...")

            # Create live dashboard
            dashboard_path = f"{results_dir}/mcts_dashboard.html"
            self.visualizer.create_live_dashboard(save_path=dashboard_path)

            # Create static summary
            summary_path = f"{results_dir}/mcts_summary.png"
            self.visualizer.create_static_summary(save_path=summary_path)

            # Create MCTS graph visualization
            if self.visualizer.mcts_tree_snapshots:
                graph_path = f"{results_dir}/mcts_graph.html"
                self.visualizer.create_mcts_graph_visualization(save_path=graph_path)
                self.logger.info(f"📊 MCTS graph visualization saved to {graph_path}")

                # Create MCTS evolution animation if we have multiple snapshots
                if len(self.visualizer.mcts_tree_snapshots) > 1:
                    animation_path = f"{results_dir}/mcts_evolution.html"
                    self.visualizer.create_mcts_evolution_animation(
                        save_path=animation_path
                    )
                    self.logger.info(
                        f"🎬 MCTS evolution animation saved to {animation_path}"
                    )
            else:
                self.logger.warning(
                    "⚠️  No MCTS tree snapshots available for graph visualization"
                )

            # Print live status
            self.visualizer.print_live_status()

            self.logger.info(f"📊 Visualizations saved to {results_dir}/")

        self.logger.info(f"💾 Results saved to {results_dir}/")

    async def run(self, use_parallel: bool = False):
        """Run the complete VULCAN MCTS feature engineering process."""
        try:
            self.logger.info("🚀 Starting VULCAN MCTS Feature Engineering")

            # Initialize all components
            self.initialize_components()

            # Run MCTS feature engineering
            (
                best_node,
                baseline_score,
                duration,
            ) = await self.run_mcts_feature_engineering(use_parallel)

            # Analyze results
            best_features = self.analyze_results(best_node, baseline_score)

            # Generate final reflection
            await self.generate_final_reflection(best_features)

            # Save results
            self.save_results(best_features, best_node.score, baseline_score)

            self.logger.info(
                "🎉 VULCAN MCTS Feature Engineering completed successfully!"
            )

            return {
                "best_score": best_node.score,
                "baseline_score": baseline_score,
                "improvement": best_node.score - baseline_score,
                "best_features": best_features,
                "duration": duration,
            }

        except Exception as e:
            self.logger.error(f"❌ Error during execution: {e}")
            raise
        finally:
            # Cleanup
            if self.dal:
                self.dal.disconnect()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VULCAN MCTS Feature Engineering")
    parser.add_argument(
        "--config",
        default="src/autonomous_fe_env/config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--parallel", action="store_true", help="Use parallel MCTS")
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override max iterations"
    )

    args = parser.parse_args()

    # Create runner
    runner = VulcanMCTSRunner(args.config)

    # Override iterations if specified
    if args.iterations:
        runner.config["mcts"]["max_iterations"] = args.iterations
        runner.logger.info(f"🔧 Overriding max iterations to {args.iterations}")

    # Run the system
    results = await runner.run(use_parallel=args.parallel)

    print("\n" + "=" * 60)
    print("🎉 VULCAN MCTS FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"🏆 Best Score: {results['best_score']:.4f}")
    print(f"📊 Baseline: {results['baseline_score']:.4f}")
    print(f"📈 Improvement: {results['improvement']:.4f}")
    print(f"🔢 Features: {len(results['best_features'])}")
    print(f"⏱️  Duration: {results['duration']:.2f}s")
    print("=" * 60)


def run_sync():
    """Synchronous wrapper for running the async main function."""
    asyncio.run(main())


if __name__ == "__main__":
    run_sync()
