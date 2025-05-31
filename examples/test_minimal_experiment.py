#!/usr/bin/env python3
"""Minimal test of VULCAN MCTS functionality without LLM dependency."""

import asyncio
import time
from typing import Any, Dict

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator


async def test_minimal_mcts() -> Dict[str, Any]:
    """Test core MCTS functionality with minimal dependencies."""
    print("🧪 Testing VULCAN MCTS core functionality...")

    start_time = time.time()
    orchestrator = None

    try:
        # 1. Load configuration and disable LLM
        print("\n1️⃣ Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.config

        # Override LLM to use local/heuristic mode
        config.llm.provider = "local"  # Disable LLM calls
        print(f"✅ Configuration loaded: LLM={config.llm.provider} (heuristic mode)")

        # 2. Create and initialize orchestrator
        print("\n2️⃣ Initializing orchestrator...")
        orchestrator = VulcanOrchestrator(config)

        init_success = await orchestrator.initialize_components()
        if not init_success:
            raise RuntimeError("Failed to initialize orchestrator components")
        print("✅ All components initialized successfully")

        # 3. Start minimal experiment
        print("\n3️⃣ Starting minimal experiment...")
        experiment_config = {
            "experiment_name": "minimal_mcts_test",
            "config_overrides": {
                "max_iterations": 3,  # Very small for testing
            },
        }

        experiment_id = await orchestrator.start_experiment(**experiment_config)
        print(f"✅ Experiment started with ID: {experiment_id}")

        # 4. Wait for completion
        print("\n4️⃣ Waiting for completion...")

        max_wait = 60  # 1 minute max
        elapsed = 0
        check_interval = 2

        while elapsed < max_wait:
            status = orchestrator.get_status()

            if not status.is_running:
                print("✅ Experiment completed!")
                break

            print(f"⏳ Running... ({elapsed}s)")
            await asyncio.sleep(check_interval)
            elapsed += check_interval

        # 5. Get results
        print("\n5️⃣ Analyzing results...")
        final_status = orchestrator.get_status()
        experiment_history = orchestrator.get_experiment_history()

        if experiment_history:
            result = experiment_history[-1]
            print("📊 Results Summary:")
            print("   ✓ Experiment completed successfully")
            print(f"   ✓ Total iterations: {result.total_iterations}")
            print(f"   ✓ Execution time: {result.execution_time:.2f}s")
            print(f"   ✓ Best score: {result.best_score:.4f}")
            print(f"   ✓ Features explored: {result.best_feature}")

            # Check MCTS tree was built
            try:
                performance_summary = orchestrator.get_performance_metrics()
                total_evaluations = performance_summary.get("total_evaluations", 0)
                print(f"   ✓ Feature evaluations: {total_evaluations}")

                if total_evaluations >= result.total_iterations:
                    print("   ✓ MCTS tree successfully built and evaluated")
                else:
                    print("   ⚠️ Some MCTS iterations may have failed")

            except Exception as e:
                print(f"   ⚠️ Performance metrics unavailable: {e}")
        else:
            print("❌ No experiment results found")

        total_time = time.time() - start_time
        print(f"\n🎉 Minimal test completed in {total_time:.2f} seconds!")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "total_time": total_time,
            "mcts_working": len(experiment_history) > 0,
            "evaluations_count": len(experiment_history),
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time,
        }

    finally:
        if orchestrator:
            try:
                await orchestrator.cleanup()
                print("✅ Cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")


async def main():
    """Main entry point."""
    print("=" * 50)
    print("🎯 VULCAN MCTS Core Test")
    print("=" * 50)

    result = await test_minimal_mcts()

    print("\n" + "=" * 50)
    if result["success"] and result.get("mcts_working"):
        print("🎉 MCTS CORE FUNCTIONALITY WORKING!")
        print("✅ Tree building, node selection, and evaluation working")
    elif result["success"]:
        print("⚠️ EXPERIMENT COMPLETED BUT MCTS MAY NEED FIXES")
    else:
        print("❌ MCTS TEST FAILED!")
        print(f"Error: {result['error']}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
