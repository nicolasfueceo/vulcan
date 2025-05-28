#!/usr/bin/env python3
"""
Test script for VULCAN system components.

This script tests the basic functionality of the VULCAN system
to ensure all components are properly integrated.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all core components can be imported."""
    print("Testing imports...")

    try:
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_manager():
    """Test configuration manager."""
    print("\nTesting ConfigManager...")

    try:
        from autonomous_fe_env import ConfigManager

        config_manager = ConfigManager(
            "src/autonomous_fe_env/config/default_config.yaml"
        )
        config = config_manager.get_config()

        # Check some key configuration values
        assert "database" in config
        assert "mcts" in config
        assert "agents" in config

        print("✅ ConfigManager working correctly")
        return True
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        return False


def test_feature_definition():
    """Test feature definition creation."""
    print("\nTesting FeatureDefinition...")

    try:
        from autonomous_fe_env.feature import DataRequirement, FeatureDefinition

        # Create a simple feature
        feature = FeatureDefinition(
            name="test_feature",
            description="A test feature",
            code="def test_feature(df): return df['rating'].mean()",
            required_input_columns=["rating"],
            output_column_name="avg_rating",
            data_requirements=DataRequirement(
                type="horizontal",
                entity_type="user",
                columns=["rating"],
                lookup_id_field="user_id",
            ),
        )

        assert feature.name == "test_feature"
        assert feature.output_column_name == "avg_rating"
        assert feature.data_requirements is not None

        print("✅ FeatureDefinition working correctly")
        return True
    except Exception as e:
        print(f"❌ FeatureDefinition test failed: {e}")
        return False


def test_agents():
    """Test agent creation."""
    print("\nTesting Agents...")

    try:
        from autonomous_fe_env import get_agent

        # Create agents for testing
        feature_agent = get_agent("feature", config={"mode": "predefined"})
        reflection_agent = get_agent("reflection", config={"mode": "stub"})

        assert feature_agent is not None
        assert reflection_agent is not None

        print("✅ Agents created successfully")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False


def test_state_manager():
    """Test state manager."""
    print("\nTesting StateManager...")

    try:
        from autonomous_fe_env import StateManager

        state_manager = StateManager("test_state")

        # Test basic functionality
        state_manager.update_mcts_stats(iteration_success=True)
        stats = state_manager.get_mcts_stats()

        assert stats["total_iterations"] == 1
        assert stats["successful_iterations"] == 1

        print("✅ StateManager working correctly")
        return True
    except Exception as e:
        print(f"❌ StateManager test failed: {e}")
        return False


def test_visualization():
    """Test visualization components."""
    print("\nTesting Visualization...")

    try:
        from autonomous_fe_env.visualization import AgentMonitor, PipelineVisualizer

        config = {"visualization": {"update_interval": 5}}

        visualizer = PipelineVisualizer(config)
        monitor = AgentMonitor(config)

        # Test basic logging
        visualizer.log_mcts_iteration(1, 0.5, 10, ["feature1"])
        monitor.log_agent_activity("test_agent", "test_activity", {})

        print("✅ Visualization components working correctly")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("VULCAN SYSTEM TEST")
    print("=" * 60)

    tests = [
        test_imports,
        test_config_manager,
        test_feature_definition,
        test_agents,
        test_state_manager,
        test_visualization,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! VULCAN system is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
