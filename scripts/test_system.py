#!/usr/bin/env python3
"""System validation script for VULCAN 2.0."""

import asyncio
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def test_imports() -> bool:
    """Test that all core modules can be imported."""
    console.print("🧪 Testing imports...", style="blue")

    try:
        # Test core imports
        from vulcan import ConfigManager, VulcanOrchestrator, setup_logging
        from vulcan.api import create_app
        from vulcan.types import ExperimentRequest, VulcanConfig

        console.print("✅ All imports successful", style="green")
        return True

    except ImportError as e:
        console.print(f"❌ Import failed: {e}", style="red")
        return False


def test_configuration() -> bool:
    """Test configuration system."""
    console.print("🧪 Testing configuration...", style="blue")

    try:
        from vulcan.core import ConfigManager

        # Test default configuration
        manager = ConfigManager()
        config = manager.config

        # Validate configuration
        assert config.mcts.max_iterations > 0
        assert config.api.port >= 1000
        assert config.llm.provider in ["openai", "anthropic", "local"]

        console.print("✅ Configuration system working", style="green")
        return True

    except Exception as e:
        console.print(f"❌ Configuration test failed: {e}", style="red")
        return False


def test_types() -> bool:
    """Test type system."""
    console.print("🧪 Testing type system...", style="blue")

    try:
        from vulcan.types import ExperimentRequest, VulcanConfig

        # Test type creation
        config = VulcanConfig()
        request = ExperimentRequest(experiment_name="test")

        # Test serialization
        config_dict = config.dict()
        assert isinstance(config_dict, dict)

        console.print("✅ Type system working", style="green")
        return True

    except Exception as e:
        console.print(f"❌ Type system test failed: {e}", style="red")
        return False


async def test_orchestrator() -> bool:
    """Test orchestrator initialization."""
    console.print("🧪 Testing orchestrator...", style="blue")

    try:
        from vulcan.core import ConfigManager, VulcanOrchestrator

        # Initialize components
        config_manager = ConfigManager()
        orchestrator = VulcanOrchestrator(config_manager.config)

        # Test initialization
        await orchestrator.initialize_components()

        # Test status
        status = orchestrator.get_status()
        assert hasattr(status, "is_running")

        # Cleanup
        await orchestrator.cleanup()

        console.print("✅ Orchestrator working", style="green")
        return True

    except Exception as e:
        console.print(f"❌ Orchestrator test failed: {e}", style="red")
        return False


def test_api_creation() -> bool:
    """Test API server creation."""
    console.print("🧪 Testing API creation...", style="blue")

    try:
        from vulcan.api import create_app
        from vulcan.core import ConfigManager

        # Create app
        config_manager = ConfigManager()
        app = create_app(config_manager.config)

        # Validate app
        assert app is not None
        assert hasattr(app, "routes")

        console.print("✅ API creation working", style="green")
        return True

    except Exception as e:
        console.print(f"❌ API creation test failed: {e}", style="red")
        return False


async def main() -> None:
    """Run all system tests."""
    console.print("🚀 VULCAN 2.0 System Validation", style="bold blue")
    console.print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Type System", test_types),
        ("Orchestrator", test_orchestrator),
        ("API Creation", test_api_creation),
    ]

    results = []

    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))

    # Display results
    console.print("\n📊 Test Results", style="bold")

    table = Table()
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        table.add_row(test_name, status)
        if result:
            passed += 1

    console.print(table)

    # Summary
    console.print(f"\n📈 Summary: {passed}/{total} tests passed", style="bold")

    if passed == total:
        console.print("🎉 All tests passed! VULCAN 2.0 is ready.", style="bold green")
        sys.exit(0)
    else:
        console.print(
            "❌ Some tests failed. Please check the errors above.", style="bold red"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
