#!/usr/bin/env python3
"""Run a full VULCAN experiment with real data and LLM inference."""

import asyncio
import time
from typing import Any, Dict

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator


async def run_full_experiment() -> Dict[str, Any]:
    """Run a complete VULCAN experiment with real data."""
    print("🚀 Starting full VULCAN experiment with real data and LLM inference...")

    start_time = time.time()
    orchestrator = None

    try:
        # 1. Load configuration
        print("\n1️⃣ Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.config
        print(
            f"✅ Configuration loaded: LLM={config.llm.provider}, Model={config.llm.model_name}"
        )

        # 2. Create and initialize orchestrator
        print("\n2️⃣ Creating and initializing orchestrator...")
        orchestrator = VulcanOrchestrator(config)

        # Initialize all components
        init_success = await orchestrator.initialize_components()
        if not init_success:
            raise RuntimeError("Failed to initialize orchestrator components")
        print("✅ All components initialized successfully")

        # 3. Start experiment
        print("\n3️⃣ Starting experiment...")
        experiment_config = {
            "experiment_name": "full_test_experiment",
            "config_overrides": {
                "max_iterations": 5,  # Start small for testing
            },
        }

        experiment_id = await orchestrator.start_experiment(**experiment_config)
        print(f"✅ Experiment started with ID: {experiment_id}")

        # 4. Monitor experiment progress
        print("\n4️⃣ Monitoring experiment progress...")

        # Wait for experiment to complete or check status periodically
        max_wait_time = 300  # 5 minutes max
        check_interval = 10  # Check every 10 seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            status = orchestrator.get_status()

            if not status.is_running:
                print("✅ Experiment completed!")
                break

            print(f"⏳ Experiment running... (elapsed: {elapsed_time}s)")
            print(f"   Experiment ID: {status.experiment_id}")

            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        if elapsed_time >= max_wait_time:
            print("⚠️ Experiment timeout reached")

        # 5. Get final results
        print("\n5️⃣ Getting final results...")
        final_status = orchestrator.get_status()
        experiment_history = orchestrator.get_experiment_history()

        if experiment_history:
            result = experiment_history[-1]  # Get the latest experiment
            print("✅ Final Results:")
            print(f"   Best Score: {result.best_score:.4f}")
            print(f"   Best Feature: {result.best_feature}")
            print(f"   Total Iterations: {result.total_iterations}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            if result.best_node_id:
                print(f"   Best Node ID: {result.best_node_id}")
        else:
            print("❌ No experiment results available")

        # 6. Get performance analytics
        print("\n6️⃣ Getting performance analytics...")
        try:
            performance_summary = orchestrator.get_performance_metrics()
            print("✅ Performance Summary:")
            print(f"   Total Evaluations: {performance_summary['total_evaluations']}")
            print(f"   Average Score: {performance_summary['average_score']:.4f}")
            print(f"   Best Score: {performance_summary['best_score']:.4f}")
            print(
                f"   Improvement: {performance_summary['improvement_from_baseline']:.4f}"
            )

            # Show top features
            top_features = orchestrator.get_best_features(top_k=3)
            if top_features:
                print("   Top Features:")
                for i, feature in enumerate(top_features, 1):
                    print(
                        f"     {i}. {feature['feature_name']} (score: {feature['avg_score']:.4f})"
                    )

        except Exception as e:
            print(f"⚠️ Could not get performance analytics: {e}")

        total_time = time.time() - start_time
        print(f"\n🎉 Full experiment completed in {total_time:.2f} seconds!")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "total_time": total_time,
            "final_status": final_status.dict() if final_status else None,
            "experiment_history": [result.dict() for result in experiment_history],
        }

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time,
        }

    finally:
        # Cleanup
        if orchestrator:
            try:
                await orchestrator.cleanup()
                print("✅ Cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("🎯 VULCAN Full Experiment Test")
    print("=" * 60)

    result = await run_full_experiment()

    print("\n" + "=" * 60)
    if result["success"]:
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    else:
        print("❌ EXPERIMENT FAILED!")
        print(f"Error: {result['error']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
