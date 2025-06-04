#!/usr/bin/env python3
"""Test the file-based experiment results system."""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vulcan.core import ConfigManager, VulcanOrchestrator
from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.utils.results_manager import ResultsManager


async def test_file_based_system():
    """Test the file-based experiment system."""

    print("🧪 Testing File-Based Experiment System")
    print("=" * 50)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.config
    print("✅ Configuration loaded")

    # Create results manager
    results_manager = ResultsManager(config)
    print("✅ Results manager created")

    # Create orchestrator
    orchestrator = VulcanOrchestrator(config)
    await orchestrator.initialize_components()
    print("✅ Orchestrator initialized")

    # Load data
    print("\n📊 Loading data...")
    loader = GoodreadsDataLoader(
        db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
        splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
        outer_fold=1,
        inner_fold=1,
        batch_size=1000,
    )

    data_context = loader.get_data_context(sample_size=5000)  # Small sample for testing
    print(
        f"✅ Data loaded: {data_context.n_users:,} users, {data_context.n_items:,} items"
    )

    # Test experiment start
    print("\n🚀 Starting test experiment...")
    experiment_config = {
        "experimentName": "file_based_test",
        "max_iterations": 3,  # Very short for testing
        "data_sample_size": 1000,
        "algorithm": "evolution",
    }

    try:
        experiment_id = await orchestrator.start_experiment(
            experiment_name=experiment_config["experimentName"],
            config_overrides=experiment_config,
            data_context=data_context,
            results_manager=results_manager,
        )

        print(f"✅ Experiment started: {experiment_id}")

        # Wait a bit for the experiment to progress
        print("\n⏳ Waiting for experiment to progress...")
        for i in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)

            # Check if we have any experiments in results manager
            experiments = results_manager.list_experiments()
            if experiments:
                latest_exp = experiments[0]
                print(
                    f"📊 Progress: {latest_exp.get('status', 'unknown')} - "
                    f"Iterations: {latest_exp.get('iterations_completed', 0)} - "
                    f"Best Score: {latest_exp.get('best_score', 0.0):.4f}"
                )

                if latest_exp.get("status") == "completed":
                    print("✅ Experiment completed!")
                    break

            if i % 5 == 0:
                print(f"   Still waiting... ({i}s)")

        # List all experiments
        print("\n📁 Experiment Results:")
        experiments = results_manager.list_experiments()
        for i, exp in enumerate(experiments[:3]):  # Show first 3
            print(
                f"  {i + 1}. {exp.get('experiment_name', 'Unknown')} - "
                f"Status: {exp.get('status', 'unknown')} - "
                f"Score: {exp.get('best_score', 0.0):.4f}"
            )

        # Test loading experiment data
        if experiments:
            latest_exp_name = experiments[0]["experiment_name"]
            print(f"\n📈 Loading data for experiment: {latest_exp_name}")

            exp_data = results_manager.load_experiment_data(latest_exp_name)
            if exp_data:
                print("✅ Data loaded:")
                print(f"  - Nodes: {len(exp_data.get('nodes', []))}")
                print(f"  - Stats: {exp_data.get('stats', {})}")
                print(f"  - Best candidate: {exp_data.get('best_candidate', 'None')}")
            else:
                print("❌ Failed to load experiment data")

        print("\n🎉 File-based system test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        await orchestrator.cleanup()
        print("🔄 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_file_based_system())
