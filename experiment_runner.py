#!/usr/bin/env python3
"""
VULCAN Progressive Evolution Experiment Runner

This script provides easy ways to start different types of experiments
with the Progressive Evolution system.
"""

import argparse
import sys
import time
from typing import Any, Dict

import requests
import structlog

# Setup logging
logger = structlog.get_logger(__name__)

# API Configuration
API_BASE = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE}/api/health"
STATUS_ENDPOINT = f"{API_BASE}/api/status"
START_ENDPOINT = f"{API_BASE}/api/experiments/start"
STOP_ENDPOINT = f"{API_BASE}/api/experiments/stop"
TREE_ENDPOINT = f"{API_BASE}/api/tree"

# Experiment Presets
EXPERIMENT_PRESETS = {
    "quick": {
        "name": "Quick Test",
        "description": "Fast test run for development and debugging",
        "config": {
            "experiment_name": "Quick Progressive Evolution Test",
            "max_iterations": 5,
            "population_size": 10,
            "generation_size": 8,
            "data_sample_size": 500,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 2,
            "mutation_rate": 0.3,
            "epsilon": 0.1,
            "learning_rate": 0.01,
        },
        "estimated_time": "2-3 minutes",
    },
    "standard": {
        "name": "Standard Evolution",
        "description": "Balanced experiment for typical feature engineering tasks",
        "config": {
            "experiment_name": "Standard Progressive Evolution",
            "max_iterations": 20,
            "population_size": 30,
            "generation_size": 15,
            "data_sample_size": 2000,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 3,
            "mutation_rate": 0.3,
            "epsilon": 0.1,
            "learning_rate": 0.01,
        },
        "estimated_time": "10-15 minutes",
    },
    "intensive": {
        "name": "Intensive Evolution",
        "description": "Comprehensive experiment for complex datasets",
        "config": {
            "experiment_name": "Intensive Progressive Evolution",
            "max_iterations": 50,
            "population_size": 50,
            "generation_size": 25,
            "data_sample_size": 5000,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 3,
            "mutation_rate": 0.25,
            "epsilon": 0.05,
            "learning_rate": 0.005,
        },
        "estimated_time": "30-60 minutes",
    },
    "exploration": {
        "name": "High Exploration",
        "description": "Focus on exploring diverse feature space",
        "config": {
            "experiment_name": "High Exploration Evolution",
            "max_iterations": 30,
            "population_size": 40,
            "generation_size": 20,
            "data_sample_size": 3000,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 3,
            "mutation_rate": 0.5,
            "epsilon": 0.3,
            "learning_rate": 0.02,
        },
        "estimated_time": "20-30 minutes",
    },
    "exploitation": {
        "name": "High Exploitation",
        "description": "Focus on refining promising features",
        "config": {
            "experiment_name": "High Exploitation Evolution",
            "max_iterations": 25,
            "population_size": 35,
            "generation_size": 18,
            "data_sample_size": 2500,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 4,
            "mutation_rate": 0.15,
            "epsilon": 0.05,
            "learning_rate": 0.005,
        },
        "estimated_time": "15-25 minutes",
    },
    "repair_focused": {
        "name": "Repair-Focused",
        "description": "Test automatic code repair capabilities",
        "config": {
            "experiment_name": "Repair-Focused Evolution",
            "max_iterations": 15,
            "population_size": 25,
            "generation_size": 12,
            "data_sample_size": 1500,
            "outer_fold": 1,
            "inner_fold": 1,
            "max_repair_attempts": 5,
            "mutation_rate": 0.4,
            "epsilon": 0.2,
            "learning_rate": 0.015,
        },
        "estimated_time": "8-12 minutes",
    },
}


def check_system_health() -> bool:
    """Check if the VULCAN system is running and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            logger.info("System health check", status=health_data.get("status"))
            return health_data.get("status") == "healthy"
        else:
            logger.error("Health check failed", status_code=response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        logger.error("Failed to connect to VULCAN system", error=str(e))
        return False


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error("Status check failed", status_code=response.status_code)
            return {}
    except requests.exceptions.RequestException as e:
        logger.error("Failed to get system status", error=str(e))
        return {}


def start_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Start a new experiment with the given configuration."""
    try:
        payload = {
            "experiment_name": config.get("experiment_name", "Progressive Evolution"),
            "config_overrides": config,
        }

        response = requests.post(START_ENDPOINT, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                logger.info(
                    "Experiment started successfully",
                    experiment_id=result["data"]["experiment_id"],
                )
                return result
            else:
                logger.error(
                    "Failed to start experiment", message=result.get("message")
                )
                return result
        else:
            logger.error(
                "Start experiment request failed", status_code=response.status_code
            )
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logger.error("Failed to start experiment", error=str(e))
        return {"status": "error", "message": str(e)}


def stop_experiment() -> Dict[str, Any]:
    """Stop the currently running experiment."""
    try:
        response = requests.post(STOP_ENDPOINT, timeout=10)
        if response.status_code == 200:
            result = response.json()
            logger.info("Stop experiment request sent", status=result.get("status"))
            return result
        else:
            logger.error(
                "Stop experiment request failed", status_code=response.status_code
            )
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logger.error("Failed to stop experiment", error=str(e))
        return {"status": "error", "message": str(e)}


def get_evolution_data() -> Dict[str, Any]:
    """Get current evolution tree data."""
    try:
        response = requests.get(TREE_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(
                "Failed to get evolution data", status_code=response.status_code
            )
            return {}
    except requests.exceptions.RequestException as e:
        logger.error("Failed to get evolution data", error=str(e))
        return {}


def monitor_experiment(experiment_id: str, check_interval: int = 10) -> None:
    """Monitor the progress of a running experiment."""
    logger.info("Starting experiment monitoring", experiment_id=experiment_id)

    start_time = time.time()
    last_generation = 0

    while True:
        try:
            # Get system status
            status = get_system_status()
            if status.get("status") != "running":
                logger.info("Experiment completed or stopped")
                break

            # Get evolution data
            evolution_data = get_evolution_data()
            if evolution_data:
                current_generation = len(evolution_data.get("generation_history", []))

                if current_generation > last_generation:
                    last_generation = current_generation

                    # Show progress
                    elapsed = time.time() - start_time
                    best_score = evolution_data.get("best_candidate", {}).get(
                        "score", 0
                    )
                    population_size = len(evolution_data.get("population", []))

                    logger.info(
                        "Evolution progress",
                        generation=current_generation,
                        best_score=f"{best_score:.6f}",
                        population_size=population_size,
                        elapsed_time=f"{elapsed:.1f}s",
                    )

                    # Show generation summary
                    if evolution_data.get("generation_history"):
                        latest_gen = evolution_data["generation_history"][-1]
                        success_rate = (
                            latest_gen["successful_features"]
                            / latest_gen["total_features"]
                            * 100
                        )

                        logger.info(
                            "Generation summary",
                            total_features=latest_gen["total_features"],
                            successful_features=latest_gen["successful_features"],
                            success_rate=f"{success_rate:.1f}%",
                            action_taken=latest_gen["action_taken"],
                        )

            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            break
        except Exception as e:
            logger.error("Error during monitoring", error=str(e))
            time.sleep(check_interval)


def list_presets() -> None:
    """List all available experiment presets."""
    print("\n🧬 Available VULCAN Progressive Evolution Presets:\n")

    for preset_name, preset_data in EXPERIMENT_PRESETS.items():
        print(f"  {preset_name:15} - {preset_data['name']}")
        print(f"  {' ' * 15}   {preset_data['description']}")
        print(f"  {' ' * 15}   Estimated time: {preset_data['estimated_time']}")
        print()


def show_preset_details(preset_name: str) -> None:
    """Show detailed configuration for a preset."""
    if preset_name not in EXPERIMENT_PRESETS:
        print(f"❌ Preset '{preset_name}' not found")
        return

    preset = EXPERIMENT_PRESETS[preset_name]
    print(f"\n🔬 {preset['name']} Configuration:\n")
    print(f"Description: {preset['description']}")
    print(f"Estimated time: {preset['estimated_time']}\n")

    print("Configuration:")
    for key, value in preset["config"].items():
        print(f"  {key:20} = {value}")
    print()


def run_preset(preset_name: str, monitor: bool = True) -> None:
    """Run a preset experiment."""
    if preset_name not in EXPERIMENT_PRESETS:
        print(f"❌ Preset '{preset_name}' not found")
        return

    preset = EXPERIMENT_PRESETS[preset_name]

    print(f"\n🚀 Starting {preset['name']} experiment...")
    print(f"Description: {preset['description']}")
    print(f"Estimated time: {preset['estimated_time']}\n")

    # Check system health
    if not check_system_health():
        print("❌ VULCAN system is not healthy. Please check the backend.")
        return

    # Start experiment
    result = start_experiment(preset["config"])
    if result.get("status") != "success":
        print(f"❌ Failed to start experiment: {result.get('message')}")
        return

    experiment_id = result["data"]["experiment_id"]
    print("✅ Experiment started successfully!")
    print(f"Experiment ID: {experiment_id}")

    if monitor:
        print("\n📊 Monitoring experiment progress (Ctrl+C to stop monitoring)...")
        monitor_experiment(experiment_id)
    else:
        print("\n💡 Use 'python experiment_runner.py monitor' to track progress")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="VULCAN Progressive Evolution Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_runner.py list                    # List all presets
  python experiment_runner.py run quick               # Run quick test
  python experiment_runner.py run standard --monitor  # Run and monitor
  python experiment_runner.py show intensive          # Show preset details
  python experiment_runner.py status                  # Check system status
  python experiment_runner.py stop                    # Stop current experiment
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List all available presets")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show preset details")
    show_parser.add_argument("preset", help="Preset name to show")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment preset")
    run_parser.add_argument("preset", help="Preset name to run")
    run_parser.add_argument(
        "--monitor", action="store_true", help="Monitor experiment progress"
    )
    run_parser.add_argument(
        "--no-monitor", action="store_true", help="Start experiment without monitoring"
    )

    # Status command
    subparsers.add_parser("status", help="Check system status")

    # Stop command
    subparsers.add_parser("stop", help="Stop current experiment")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor current experiment")
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Check interval in seconds (default: 10)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    if args.command == "list":
        list_presets()

    elif args.command == "show":
        show_preset_details(args.preset)

    elif args.command == "run":
        monitor = not args.no_monitor if args.no_monitor else True
        run_preset(args.preset, monitor=monitor)

    elif args.command == "status":
        if not check_system_health():
            print("❌ VULCAN system is not healthy")
            sys.exit(1)

        status = get_system_status()
        if status:
            print("\n📊 VULCAN System Status:")
            print(f"  Status: {status.get('status', 'unknown')}")
            print(f"  Experiments: {status.get('experiments_count', 0)}")
            print(f"  Config loaded: {status.get('config_loaded', False)}")

            components = status.get("components", {})
            print("\n🔧 Components:")
            for component, healthy in components.items():
                status_icon = "✅" if healthy else "❌"
                print(f"  {status_icon} {component}")
        else:
            print("❌ Failed to get system status")

    elif args.command == "stop":
        result = stop_experiment()
        if result.get("status") == "success":
            print("✅ Experiment stop request sent")
        else:
            print(f"❌ Failed to stop experiment: {result.get('message')}")

    elif args.command == "monitor":
        status = get_system_status()
        if status.get("status") == "running":
            print("📊 Monitoring current experiment...")
            monitor_experiment("current", check_interval=args.interval)
        else:
            print("❌ No experiment currently running")


if __name__ == "__main__":
    main()
