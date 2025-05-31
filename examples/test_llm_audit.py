#!/usr/bin/env python3
"""Comprehensive LLM audit test for VULCAN with GPT-4o."""

import asyncio
import os
import time
from typing import Any, Dict

from vulcan.core.config_manager import ConfigManager
from vulcan.core.orchestrator import VulcanOrchestrator


def load_environment():
    """Load environment variables from .env file."""
    try:
        # Load from .env file
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    # Clean up the API key value (remove quotes and fix nested format)
                    if key == "OPENAI_API_KEY":
                        # Handle the nested format in the file
                        if value.startswith('"OPENAI_API_KEY='):
                            value = value.replace('"OPENAI_API_KEY=', "").rstrip('"')
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        os.environ[key] = value
                        print(f"✅ Loaded {key} from .env file")
                    else:
                        os.environ[key] = value
    except FileNotFoundError:
        print("⚠️ .env file not found, using existing environment variables")
    except Exception as e:
        print(f"⚠️ Error loading .env file: {e}")


async def audit_llm_responses() -> Dict[str, Any]:
    """Run comprehensive LLM audit with GPT-4o."""
    print("🔍 Starting comprehensive LLM audit with GPT-4o...")

    start_time = time.time()
    orchestrator = None

    try:
        # Load environment variables
        load_environment()

        # Verify API key is loaded
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        print(f"✅ API key loaded (length: {len(api_key)})")

        # 1. Load configuration
        print("\n1️⃣ Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.config

        # Ensure we're using GPT-4o with proper settings
        config.llm.provider = "openai"
        config.llm.model_name = "gpt-4o"
        config.llm.max_tokens = 8000
        config.llm.temperature = 0.7

        print(f"✅ Configuration set: {config.llm.provider} / {config.llm.model_name}")
        print(
            f"   Max tokens: {config.llm.max_tokens}, Temperature: {config.llm.temperature}"
        )

        # 2. Create and initialize orchestrator
        print("\n2️⃣ Initializing orchestrator...")
        orchestrator = VulcanOrchestrator(config)

        init_success = await orchestrator.initialize_components()
        if not init_success:
            raise RuntimeError("Failed to initialize orchestrator components")
        print("✅ All components initialized successfully")

        # 3. Start LLM-powered experiment
        print("\n3️⃣ Starting LLM-powered experiment...")
        experiment_config = {
            "experiment_name": "llm_audit_experiment",
            "config_overrides": {
                "max_iterations": 3,  # Small for detailed auditing
            },
        }

        experiment_id = await orchestrator.start_experiment(**experiment_config)
        print(f"✅ Experiment started with ID: {experiment_id}")

        # 4. Monitor with detailed LLM logging
        print("\n4️⃣ Monitoring LLM interactions...")

        max_wait = 180  # 3 minutes for LLM calls
        elapsed = 0
        check_interval = 5

        llm_interactions = []

        while elapsed < max_wait:
            status = orchestrator.get_status()

            if not status.is_running:
                print("✅ Experiment completed!")
                break

            print(f"⏳ Running... ({elapsed}s) - Monitoring LLM calls")
            await asyncio.sleep(check_interval)
            elapsed += check_interval

        # 5. Analyze results and LLM performance
        print("\n5️⃣ Analyzing LLM performance...")
        final_status = orchestrator.get_status()
        experiment_history = orchestrator.get_experiment_history()

        if experiment_history:
            result = experiment_history[-1]
            print("📊 Experiment Results:")
            print(f"   ✓ Total iterations: {result.total_iterations}")
            print(f"   ✓ Execution time: {result.execution_time:.2f}s")
            print(f"   ✓ Best score: {result.best_score:.4f}")
            print(f"   ✓ Best feature: {result.best_feature}")

            # Analyze performance tracking
            try:
                performance_summary = orchestrator.get_performance_metrics()
                total_evaluations = performance_summary.get("total_evaluations", 0)
                print(f"   ✓ Feature evaluations: {total_evaluations}")

                # Get top features to see LLM-generated ones
                top_features = orchestrator.get_best_features(top_k=5)
                if top_features:
                    print("   ✓ Generated features:")
                    for i, feature in enumerate(top_features, 1):
                        print(
                            f"     {i}. {feature['feature_name']} (score: {feature['avg_score']:.4f})"
                        )

            except Exception as e:
                print(f"   ⚠️ Performance metrics error: {e}")
        else:
            print("❌ No experiment results found")

        # 6. LLM Audit Summary
        print("\n6️⃣ LLM Audit Summary:")
        print("   📋 Check the logs above for:")
        print("   • 'Sending prompt to LLM' - Prompt details")
        print("   • 'Full LLM prompt' - Complete prompts sent")
        print("   • 'Received LLM response' - Response metadata")
        print("   • 'Full LLM response' - Complete responses")
        print("   • Token usage statistics")

        total_time = time.time() - start_time
        print(f"\n🎉 LLM audit completed in {total_time:.2f} seconds!")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "total_time": total_time,
            "llm_model": config.llm.model_name,
            "iterations": result.total_iterations if experiment_history else 0,
            "features_generated": len(top_features)
            if "top_features" in locals()
            else 0,
        }

    except Exception as e:
        print(f"❌ LLM audit failed: {e}")
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
    print("=" * 70)
    print("🎯 VULCAN GPT-4o LLM Audit & Performance Test")
    print("=" * 70)

    result = await audit_llm_responses()

    print("\n" + "=" * 70)
    if result["success"]:
        print("🎉 LLM AUDIT COMPLETED SUCCESSFULLY!")
        print(f"✅ Model: {result.get('llm_model', 'Unknown')}")
        print(f"✅ Iterations: {result.get('iterations', 0)}")
        print(f"✅ Features: {result.get('features_generated', 0)}")
        print(f"✅ Time: {result.get('total_time', 0):.2f}s")
        print("\n📋 Review the detailed logs above to audit:")
        print("   • LLM prompt quality and context")
        print("   • Response parsing and feature extraction")
        print("   • Token usage and efficiency")
        print("   • Feature generation creativity and relevance")
    else:
        print("❌ LLM AUDIT FAILED!")
        print(f"Error: {result['error']}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
