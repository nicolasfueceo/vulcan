"""Run a complete MCTS experiment with tuned baselines and rigorous evaluation."""

import asyncio
import json
import time

from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.mcts.mcts_orchestrator import MCTSOrchestrator
from vulcan.types import VulcanConfig
from vulcan.utils import PerformanceTracker


async def run_mcts_experiment():
    """Run a complete MCTS feature engineering experiment."""

    print("=" * 80)
    print("🚀 VULCAN MCTS Feature Engineering Experiment")
    print("=" * 80)

    # Create config with appropriate settings
    config = VulcanConfig()
    config.mcts.max_iterations = 10  # Start with 10 iterations for testing
    config.mcts.max_depth = 5  # Max 5 features deep
    config.llm.temperature = 0.7  # Good creativity/consistency balance

    print("\n📋 Configuration:")
    print(f"  - Max iterations: {config.mcts.max_iterations}")
    print(f"  - Max depth: {config.mcts.max_depth}")
    print(f"  - LLM model: {config.llm.model_name}")
    print(f"  - LLM temperature: {config.llm.temperature}")

    # Load data
    print("\n📊 Loading Goodreads data...")
    loader = GoodreadsDataLoader(
        db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
        splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
        outer_fold=1,
        inner_fold=1,
        batch_size=1000,
    )
    data_context = loader.get_data_context()
    print(f"  ✅ Loaded {data_context.n_users:,} users, {data_context.n_items:,} items")
    print(f"  📈 Sparsity: {data_context.sparsity:.4f}")

    # Create performance tracker
    performance_tracker = PerformanceTracker(max_history=100)

    # Create MCTS orchestrator
    print("\n🌳 Initializing MCTS orchestrator...")
    orchestrator = MCTSOrchestrator(config, performance_tracker)
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize MCTS orchestrator")
        return

    print("  ✅ MCTS orchestrator initialized")

    # Check if baselines are tuned
    print("\n🔧 Checking baseline hyperparameters...")
    try:
        with open("baseline_hyperparams.json") as f:
            hyperparams = json.load(f)
        print("  ✅ Found tuned baseline hyperparameters")
        for model, params in hyperparams.items():
            if isinstance(params, dict) and "no_components" in params:
                print(f"    - {model}: {params['no_components']} components")
            elif isinstance(params, dict) and "k_neighbors" in params:
                print(f"    - {model}: k={params['k_neighbors']}")
    except FileNotFoundError:
        print("  ⚠️  No tuned hyperparameters found - baselines will use defaults")

    # Run MCTS search
    print("\n🔍 Starting MCTS search for optimal features...")
    print("  📝 Note: Using rigorous evaluation at each step (on data subset)")

    experiment_start = time.time()

    try:
        # Run the search
        results = await orchestrator.run_search(
            data_context=data_context, max_iterations=config.mcts.max_iterations
        )

        experiment_time = time.time() - experiment_start

        # Display results
        print("\n" + "=" * 80)
        print("📊 MCTS Search Results")
        print("=" * 80)

        print(f"\n⏱️  Total time: {experiment_time:.1f} seconds")
        print(f"🔄 Iterations completed: {results['total_iterations']}")
        print("🌳 Tree statistics:")
        print(f"   - Total nodes: {results['tree_stats']['total_nodes']}")
        print(f"   - Max depth: {results['tree_stats']['max_depth']}")
        print(
            f"   - Avg branching factor: {results['tree_stats']['avg_branching_factor']:.2f}"
        )

        print(f"\n🏆 Best score achieved: {results['best_score']:.4f}")
        print(f"📊 Best features found ({results['best_feature_count']} features):")

        for i, feature in enumerate(results["best_features"], 1):
            print(f"\n  {i}. {feature['name']}")
            print(f"     Description: {feature['description']}")
            print(f"     Type: {feature['feature_type']}")
            if feature.get("code"):
                # Show first line of code
                first_line = feature["code"].strip().split("\n")[0]
                print(f"     Code: {first_line}...")

        # Performance analysis
        print("\n📈 Performance Tracking Summary:")
        perf_summary = orchestrator.performance_tracker.get_performance_summary()

        if perf_summary["best_features"]:
            print("\n  Top performing features:")
            for i, feat in enumerate(perf_summary["best_features"][:3], 1):
                print(
                    f"    {i}. {feat['feature_name']} (avg score: {feat['avg_score']:.4f})"
                )

        if perf_summary.get("performance_trend"):
            trend_data = perf_summary["performance_trend"]
            if len(trend_data) > 1:
                improvement = trend_data[-1] - trend_data[0]
                print(
                    f"\n  Performance trend: {'+' if improvement > 0 else ''}{improvement:.4f}"
                )

        # Save results
        print("\n💾 Saving experiment results...")
        with open("mcts_experiment_results.json", "w") as f:
            json.dump(
                {
                    "experiment_time": experiment_time,
                    "config": {
                        "max_iterations": config.mcts.max_iterations,
                        "max_depth": config.mcts.max_depth,
                        "llm_model": config.llm.model_name,
                        "llm_temperature": config.llm.temperature,
                    },
                    "results": results,
                    "performance_summary": perf_summary,
                },
                f,
                indent=2,
            )
        print("  ✅ Results saved to mcts_experiment_results.json")

        # Get tree visualization data
        tree_data = await orchestrator.get_tree_visualization_data()
        with open("mcts_tree_data.json", "w") as f:
            json.dump(tree_data, f, indent=2)
        print("  ✅ Tree data saved to mcts_tree_data.json")

    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await orchestrator.cleanup()
        print("  ✅ Cleanup complete")

    print("\n✨ Experiment complete!")


if __name__ == "__main__":
    # Set up OpenAI API key if needed
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using heuristic feature generation.")

    # Run the experiment
    asyncio.run(run_mcts_experiment())
