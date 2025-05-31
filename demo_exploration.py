#!/usr/bin/env python3
"""Demo script to show VULCAN exploration with visualization."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vulcan.core import VulcanOrchestrator
from vulcan.data.goodreads_loader import GoodreadsDataLoader
from vulcan.types import VulcanConfig


async def run_demo():
    """Run a demo exploration."""
    print("=== VULCAN Exploration Demo ===\n")

    # Create config
    config = VulcanConfig()
    config.mcts.max_iterations = 10

    # Create orchestrator
    orchestrator = VulcanOrchestrator(config)

    # Initialize components
    print("Initializing components...")
    await orchestrator.initialize_components()

    # Create data context
    print("Loading data...")
    loader = GoodreadsDataLoader(
        db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
        splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
        outer_fold=1,
        inner_fold=1,
    )
    data_context = loader.get_data_context()

    print(f"Data loaded: {data_context.n_users} users, {data_context.n_items} items")
    print(f"Starting exploration with {config.mcts.max_iterations} iterations...\n")

    # Start experiment
    experiment_id = await orchestrator.start_experiment(
        experiment_name="Demo Exploration",
        data_context=data_context,
    )

    print(f"Experiment started: {experiment_id}")
    print("\nOpen http://localhost:3001/exploration to see the visualization!")
    print("(Make sure the frontend is running: cd frontend && npm run dev)")

    # Wait for experiment to complete
    while orchestrator.get_status().is_running:
        await asyncio.sleep(1.0)
        print(".", end="", flush=True)

    print("\n\nExploration complete!")

    # Get results
    history = orchestrator.get_experiment_history()
    if history:
        result = history[-1]
        print(f"Best score: {result.best_score:.3f}")
        print(f"Best features: {result.best_features}")

    # Keep the server running for visualization
    print("\nKeeping server running. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(run_demo())
