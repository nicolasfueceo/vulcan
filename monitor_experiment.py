"""Monitor MCTS experiment progress."""

import json
import os
import time
from datetime import datetime


def monitor_experiment():
    """Monitor MCTS experiment progress."""
    print("🔍 Monitoring MCTS experiment progress...")
    print("=" * 60)

    # Files to monitor
    files_to_check = [
        "mcts_experiment_results.json",
        "mcts_tree_data.json",
        "baseline_hyperparams.json",
    ]

    # Check baseline hyperparams
    if os.path.exists("baseline_hyperparams.json"):
        print("✅ Baseline hyperparameters found")
        with open("baseline_hyperparams.json") as f:
            params = json.load(f)
            print(f"   Models tuned: {', '.join(params.keys())}")
    else:
        print("⚠️  No baseline hyperparameters found")

    print("\n📊 Waiting for experiment results...")

    # Monitor for results
    start_time = time.time()
    last_update = None

    while True:
        # Check if results file exists
        if os.path.exists("mcts_experiment_results.json"):
            try:
                with open("mcts_experiment_results.json") as f:
                    results = json.load(f)

                # Check if file was updated
                file_time = os.path.getmtime("mcts_experiment_results.json")
                if last_update is None or file_time > last_update:
                    last_update = file_time

                    print(
                        f"\n[{datetime.now().strftime('%H:%M:%S')}] Experiment update:"
                    )

                    if "results" in results:
                        res = results["results"]
                        print(f"  - Iterations: {res.get('total_iterations', '?')}")
                        print(f"  - Best score: {res.get('best_score', '?')}")
                        print(
                            f"  - Features found: {res.get('best_feature_count', '?')}"
                        )

                        if "tree_stats" in res:
                            stats = res["tree_stats"]
                            print(f"  - Tree nodes: {stats.get('total_nodes', '?')}")
                            print(f"  - Tree depth: {stats.get('max_depth', '?')}")

            except json.JSONDecodeError:
                # File is being written to
                pass

        # Check if tree data exists
        if os.path.exists("mcts_tree_data.json"):
            try:
                with open("mcts_tree_data.json") as f:
                    tree_data = json.load(f)
                    if "nodes" in tree_data:
                        print(
                            f"  - Tree visualization data: {len(tree_data['nodes'])} nodes"
                        )
            except:
                pass

        # Wait before next check
        time.sleep(5)

        # Show elapsed time
        elapsed = time.time() - start_time
        print(
            f"\r⏱️  Elapsed: {int(elapsed // 60)}:{int(elapsed % 60):02d}",
            end="",
            flush=True,
        )


if __name__ == "__main__":
    try:
        monitor_experiment()
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
