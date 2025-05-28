"""
Simple test script to verify feature generation works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autonomous_fe_env import ConfigManager, StateManager, get_agent


def test_feature_generation():
    """Test feature generation without dashboard."""
    print("🧪 Testing feature generation...")

    # Load configuration
    config_manager = ConfigManager("src/autonomous_fe_env/config/default_config.yaml")
    config = config_manager.get_config()

    # Initialize components
    state_manager = StateManager("test_state")
    state_manager.set_baseline_score(0.42)

    feature_agent = get_agent(
        "llm_feature",
        config={
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )

    print("✅ Components initialized")

    # Test feature generation
    context = {"state_manager": state_manager, "iteration": 1}

    print("🤖 Generating feature...")
    result = feature_agent.execute(context)
    feature = result.get("feature")

    if feature:
        print(f"✅ Generated feature: {feature.name}")
        print(f"📝 Description: {feature.description}")
        print(f"🔧 Required columns: {feature.required_input_columns}")
        print(f"💻 Code preview: {feature.code[:200]}...")
        return True
    else:
        print("❌ Failed to generate feature")
        return False


if __name__ == "__main__":
    success = test_feature_generation()
    if success:
        print("\n🎉 Feature generation test passed!")
    else:
        print("\n💥 Feature generation test failed!")
